# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Remove bad episodes from a recorded IsaacLab HDF5 dataset and re-index what is left.

Recording sessions produce the occasional unusable demo -- the operator fumbled the grasp, the
teleop dropped out, the object spawned somewhere silly. Everything downstream of the recorder
(annotate_demos.py, Mimic's data generator, the LeRobot converters) walks ``demo_0 ... demo_{N-1}``
expecting a contiguous run, so simply deleting a group in the middle is not enough: the survivors
have to be renumbered.

This script does both, and leaves the dataset in the exact state a clean recording would have
produced -- contiguous ``demo_i`` in the original recording order, ``data.attrs["total"]``
recomputed, every other attribute (``env_args``, per-episode ``num_samples``/``success``)
preserved.

Usage:

    # drop demos 3, 7 and 11-13 from the dataset, in place (source file is updated)
    ./isaaclab.sh -p scripts/tools/remove_demos_hdf5.py \
        --dataset_file logs/demos/pickup_pringles_VR_V7.hdf5 \
        --episodes 3 7 11-13

    # same, but see what would happen without touching anything
    ./isaaclab.sh -p scripts/tools/remove_demos_hdf5.py \
        --dataset_file logs/demos/pickup_pringles_VR_V7.hdf5 --episodes 3 7 --dry_run

    # write the cleaned dataset somewhere else, leaving the source untouched
    ./isaaclab.sh -p scripts/tools/remove_demos_hdf5.py \
        --dataset_file logs/demos/pickup_pringles_VR_V7.hdf5 --episodes 3 7 \
        --output logs/demos/pickup_pringles_VR_V7_clean.hdf5

Two ways of doing the removal, because they trade disk against time in opposite directions:

* default ("compact"): the kept episodes are copied into a fresh file, which then atomically
  replaces the source. The removed episodes' bytes are genuinely reclaimed, and the source is
  untouched until the new file is complete -- so an interrupted run costs nothing. It needs free
  disk space roughly equal to the kept data, and copying tens of gigabytes takes minutes.
* ``--in_place``: the groups are unlinked and the survivors renamed inside the existing file. This
  is near-instant and needs no extra disk, but HDF5 does not return the freed bytes to the
  filesystem, so the file stays the same size. Use it when the disk is too full for the copy.
"""

import argparse
import os
import shutil
import sys

import h5py

# Written by the --in_place renaming pass, and never present in a dataset otherwise. Survivors are
# moved aside under these names before being moved back to their final demo_i, because a straight
# rename can collide: removing demo_3 from a 20-demo file renames demo_4 -> demo_3, demo_5 -> demo_4
# and so on, and each of those targets is a name that still exists at the time of the move.
_TEMP_PREFIX = "__reindex_tmp_"


def parse_episode_spec(tokens: list[str]) -> set[int]:
    """Turn ``--episodes`` tokens into the set of demo indices they name.

    Accepts, in any mixture: bare indices (``7``), ``demo_``-prefixed names (``demo_7``), inclusive
    ranges (``11-13``, ``demo_11-13``), and comma-separated groups of any of those (``3,7,11-13``),
    so that whichever way the indices were written down while reviewing the demos is accepted
    without re-typing.
    """
    indices: set[int] = set()

    def parse_one(raw: str) -> int:
        text = raw.strip()
        if text.startswith("demo_"):
            text = text[len("demo_") :]
        try:
            return int(text)
        except ValueError:
            raise ValueError(f"could not read '{raw}' as an episode index (expected e.g. 7, demo_7, or 11-13)")

    for token in tokens:
        for part in token.split(","):
            part = part.strip()
            if not part:
                continue
            # A range, but only when the '-' separates two values: a lone leading '-' is a negative
            # index, which is not a range and is rejected as an index below.
            if "-" in part[1:]:
                start_text, _, end_text = part[1:].partition("-")
                start, end = parse_one(part[0] + start_text), parse_one(end_text)
                if end < start:
                    raise ValueError(f"range '{part}' ends before it starts")
                indices.update(range(start, end + 1))
            else:
                indices.add(parse_one(part))

    if any(index < 0 for index in indices):
        raise ValueError("episode indices must not be negative")
    return indices


def demo_index(name: str) -> int:
    """The integer i of a ``demo_i`` group name."""
    return int(name[len("demo_") :])


def read_episode_order(data_group: h5py.Group) -> list[str]:
    """The dataset's episode names in recording order.

    HDF5 hands group members back in alphabetical order, in which ``demo_10`` sits between
    ``demo_1`` and ``demo_2``. Re-indexing off that ordering would silently shuffle the dataset, so
    sort numerically instead.
    """
    names = list(data_group.keys())
    unexpected = [name for name in names if not name.startswith("demo_") or not name[len("demo_") :].isdigit()]
    if unexpected:
        raise ValueError(
            f"dataset contains group(s) under 'data' that are not demo_<int>: {unexpected}."
            " Refusing to re-index a file whose layout this script does not understand."
        )
    return sorted(names, key=demo_index)


def episode_samples(episode_group: h5py.Group) -> int:
    """The recorded step count of one episode, as ``data.attrs['total']`` sums it."""
    return int(episode_group.attrs.get("num_samples", 0))


def remap_mask_group(src_root: h5py.File, dst_root: h5py.File, rename: dict[str, str]) -> None:
    """Copy a robomimic-style ``mask`` group across, rewriting the episode names it lists.

    Not written by the IsaacLab recorder, but train/valid split files do get added to these datasets
    by robomimic tooling later, and a split that still names ``demo_14`` after re-indexing would
    quietly point at a different episode.
    """
    if "mask" not in src_root:
        return
    dst_mask = dst_root.create_group("mask")
    for key, value in src_root["mask"].attrs.items():
        dst_mask.attrs[key] = value
    for split_name, split in src_root["mask"].items():
        kept = []
        for raw in split[:]:
            name = raw.decode() if isinstance(raw, bytes) else str(raw)
            if name in rename:
                kept.append(rename[name])
        dst_mask.create_dataset(split_name, data=[k.encode() for k in kept])


def compact_rewrite(src_path: str, dst_path: str, keep: list[str], rename: dict[str, str]) -> None:
    """Copy the kept episodes into a new file at ``dst_path``, renumbered.

    ``h5py``'s group copy is a native HDF5 object copy, so chunking, gzip compression and every
    attribute (``num_samples``, ``success``) come across unchanged -- the episodes are bit-identical
    to the originals, only their names differ.
    """
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        for key, value in src.attrs.items():
            dst.attrs[key] = value

        src_data = src["data"]
        dst_data = dst.create_group("data")
        for key, value in src_data.attrs.items():
            dst_data.attrs[key] = value

        total = 0
        for position, name in enumerate(keep):
            new_name = rename[name]
            print(f"  copying {name} -> {new_name}  ({position + 1}/{len(keep)})", flush=True)
            src.copy(src_data[name], dst_data, name=new_name)
            total += episode_samples(src_data[name])
        dst_data.attrs["total"] = total

        remap_mask_group(src, dst, rename)
        for key in src:
            if key not in ("data", "mask"):
                src.copy(src[key], dst, name=key)


def in_place_rewrite(src_path: str, remove: list[str], keep: list[str], rename: dict[str, str]) -> None:
    """Unlink the removed episodes and renumber the survivors inside the existing file."""
    with h5py.File(src_path, "r+") as handle:
        data = handle["data"]
        for name in remove:
            print(f"  deleting {name}", flush=True)
            del data[name]

        # Two passes via _TEMP_PREFIX so a survivor is never renamed onto a name still in use.
        for position, name in enumerate(keep):
            data.move(name, f"{_TEMP_PREFIX}{position}")
        for position, name in enumerate(keep):
            new_name = rename[name]
            data.move(f"{_TEMP_PREFIX}{position}", new_name)
            if new_name != name:
                print(f"  renamed {name} -> {new_name}", flush=True)

        data.attrs["total"] = sum(episode_samples(data[rename[name]]) for name in keep)

        if "mask" in handle:
            print(
                "  NOTE: this file has a 'mask' group (robomimic train/valid splits). --in_place"
                " leaves it alone, so its episode names now refer to re-indexed episodes."
                " Re-generate the splits, or re-run without --in_place, which remaps them."
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove episodes from a recorded IsaacLab HDF5 dataset and re-index the rest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset_file", type=str, required=True, help="Path to the HDF5 dataset to clean up.")
    parser.add_argument(
        "--episodes",
        type=str,
        nargs="+",
        required=True,
        metavar="SPEC",
        help=(
            "Episodes to remove. Accepts indices (7), names (demo_7), inclusive ranges (11-13) and"
            " comma-separated lists (3,7,11-13), in any mixture: --episodes 3 7 11-13"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Write the cleaned dataset here instead of replacing --dataset_file. The source is left"
            " untouched. Cannot be combined with --in_place, which edits the source by definition."
        ),
    )
    parser.add_argument(
        "--in_place",
        action="store_true",
        help=(
            "Edit the source file directly (unlink + rename) instead of copying the kept episodes"
            " into a new file. Near-instant and needs no free disk, but HDF5 does not hand the"
            " removed episodes' bytes back to the filesystem, so the file does not shrink."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be removed and how the survivors would be renumbered, then stop.",
    )
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")
    args = parser.parse_args()

    if args.in_place and args.output:
        print("ERROR: --in_place edits the source file, so it cannot also write to --output. Pass one.")
        return 1

    src_path = args.dataset_file
    if not os.path.isfile(src_path):
        print(f"ERROR: no such dataset file: {src_path}")
        return 1

    try:
        requested = parse_episode_spec(args.episodes)
    except ValueError as error:
        print(f"ERROR: {error}")
        return 1
    if not requested:
        print("ERROR: --episodes did not name any episode.")
        return 1

    with h5py.File(src_path, "r") as handle:
        if "data" not in handle:
            print(f"ERROR: {src_path} has no 'data' group -- is it an IsaacLab dataset?")
            return 1
        try:
            order = read_episode_order(handle["data"])
        except ValueError as error:
            print(f"ERROR: {error}")
            return 1
        samples = {name: episode_samples(handle["data"][name]) for name in order}

    present = {demo_index(name) for name in order}
    missing = sorted(requested - present)
    if missing:
        # Hard error rather than a warning: an index that is not there usually means the indices
        # were read off a different file (or off an already-cleaned one), and quietly removing the
        # subset that does match would delete the wrong episodes.
        print(
            f"ERROR: episode(s) {missing} are not in {src_path}, which holds"
            f" demo_0..demo_{max(present)} ({len(order)} episodes). Nothing was changed."
        )
        return 1
    if len(requested) == len(order):
        print("ERROR: that would remove every episode in the dataset. Nothing was changed.")
        return 1

    remove = [name for name in order if demo_index(name) in requested]
    keep = [name for name in order if demo_index(name) not in requested]
    rename = {name: f"demo_{position}" for position, name in enumerate(keep)}

    removed_samples = sum(samples[name] for name in remove)
    kept_samples = sum(samples[name] for name in keep)
    src_size = os.path.getsize(src_path)

    print(f"Dataset : {src_path}  ({len(order)} episodes, {src_size / 1e9:.2f} GB)")
    print(f"Removing: {len(remove)} episode(s) -- {', '.join(remove)}  ({removed_samples} steps)")
    print(f"Keeping : {len(keep)} episode(s)  ({kept_samples} steps)")
    moved = [(name, rename[name]) for name in keep if rename[name] != name]
    if moved:
        print(f"Re-index: {len(moved)} episode(s) renamed, e.g. " + ", ".join(f"{a}->{b}" for a, b in moved[:5]))
    else:
        print("Re-index: none needed (the removed episodes were at the end)")

    if args.dry_run:
        print("\n--dry_run: nothing was written.")
        return 0

    dst_path = None
    if not args.in_place:
        # Written next to the destination, not in /tmp: os.replace is only atomic within one
        # filesystem, and these files are far too large to want a cross-device copy at the end.
        final_path = args.output if args.output else src_path
        dst_path = final_path + ".partial"
        free_bytes = shutil.disk_usage(os.path.dirname(os.path.abspath(final_path))).free
        # The kept episodes' share of the source, plus 5% for the copy's own overhead. Both files
        # are on disk at once, so replacing the source in place still needs this much headroom.
        total_samples = kept_samples + removed_samples
        kept_fraction = kept_samples / total_samples if total_samples else 1.0
        needed = int(src_size * kept_fraction * 1.05)
        if free_bytes < needed:
            print(
                f"\nERROR: this needs about {needed / 1e9:.2f} GB free to write the new file"
                f" ({free_bytes / 1e9:.2f} GB available). Free some space, or re-run with"
                " --in_place, which rewrites the file's index without copying the episode data"
                " (instant, no extra disk -- but the file does not shrink)."
            )
            return 1

    if not args.yes:
        if not sys.stdin.isatty():
            print("\nERROR: nothing to prompt on (stdin is not a terminal). Re-run with --yes to confirm.")
            return 1
        target = "in place" if args.in_place else (args.output if args.output else f"{src_path} (replaced)")
        answer = input(f"\nProceed? {len(remove)} episode(s) will be dropped, writing {target} [y/N]: ")
        if answer.strip().lower() not in ("y", "yes"):
            print("Aborted. Nothing was changed.")
            return 1

    if args.in_place:
        print("\nRewriting in place...")
        in_place_rewrite(src_path, remove, keep, rename)
        final_path = src_path
    else:
        final_path = args.output if args.output else src_path
        print(f"\nWriting {dst_path} ...")
        try:
            compact_rewrite(src_path, dst_path, keep, rename)
        except BaseException:
            # Including KeyboardInterrupt: a half-written .partial is useless, and leaving it behind
            # on a nearly-full disk is actively harmful.
            if os.path.exists(dst_path):
                os.remove(dst_path)
            raise
        os.replace(dst_path, final_path)

    with h5py.File(final_path, "r") as handle:
        names = read_episode_order(handle["data"])
        expected = [f"demo_{i}" for i in range(len(keep))]
        if names != expected:
            print(f"ERROR: post-write check failed -- {final_path} holds {names}, expected {expected}")
            return 1
        total = int(handle["data"].attrs["total"])

    print(
        f"\nDone. {final_path} now holds {len(names)} episodes, demo_0..demo_{len(names) - 1},"
        f" total={total} steps ({os.path.getsize(final_path) / 1e9:.2f} GB)."
    )
    if args.in_place:
        print(
            "The file size is unchanged because --in_place cannot return freed bytes to the"
            " filesystem. Re-run without --in_place (or use h5repack) to actually shrink it."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
