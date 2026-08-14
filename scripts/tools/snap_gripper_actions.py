# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Quantise the gripper columns of an already-recorded OpenArm joint-space dataset.

Salvages demos recorded by ``record_demos_openarm.py --teleop_device vr_joint_ros2_native``
BEFORE its ``snap_gripper_command`` existed. Those store the VR trigger's analog travel in the
gripper column (0.0 m closed .. 0.044 m open), so a half-squeezed trigger recorded a half-closed
target: ``replay_demos.py`` reproduces it as a gripper that visibly never shuts, and Mimic's
``actions_to_gripper_actions`` maps it to a non-canonical value (-0.18 rather than -1). This
rewrites those two columns to the two endpoints, with the same hysteresis the recorder now
applies live.

Only ``data/*/actions`` columns 7 and 15 are touched. Recorded STATES are left alone: they are
measured positions, and a measured finger stalled against the cube at 0.032 m is what the demo
really looked like, not something to be "corrected".

What that costs: the actions no longer exactly reproduce the states recorded next to them. A
snapped close commands 0.0 where the demo commanded 0.018, so an action replay squeezes harder
than the original grasp did and the object can end up slightly differently placed. That is fine
for what these datasets are for -- ``annotate_demos.py`` and Mimic generation only ever read the
gripper's SIGN, and a firmer grasp makes the action-replay path more likely to reproduce the
demo, not less -- but a bit-exact reproduction of the original episode is gone. Re-recording is
still the better option when it is cheap.

Check --dry_run FIRST. The default thresholds are the live recorder's, which only call a squeeze
past 65%% of the trigger's travel a "close" -- but these demos were recorded when the gripper was
proportional, so a 40%% squeeze may well have been the operator's grip. Snapping that to "open"
would delete the grasp rather than fix it. The dry run prints each episode's deepest squeeze and
suggests a --close_below when it sees one that would be lost.

Usage:

  # inspect what would change, write nothing
  ./isaaclab.sh -p scripts/tools/snap_gripper_actions.py --dataset_file logs/demos/pickup.hdf5 --dry_run

  # write a snapped copy next to the original (default: <name>_snapped.hdf5)
  ./isaaclab.sh -p scripts/tools/snap_gripper_actions.py --dataset_file logs/demos/pickup.hdf5

  # rewrite the file itself
  ./isaaclab.sh -p scripts/tools/snap_gripper_actions.py --dataset_file logs/demos/pickup.hdf5 --in_place

Needs only h5py/numpy -- no Isaac Sim, so plain ``python`` works too.
"""

import argparse
import h5py
import numpy as np
import os
import shutil

# Deliberate duplicates of record_demos_openarm.py's constants (that module parses argv and
# launches an AppLauncher at import time, so it cannot be imported here -- same reason
# replay_demos.py duplicates them). Keep the three in sync.
#   [0:7] left arm | [7] left gripper | [8:15] right arm | [15] right gripper
JOINT_ACTION_DIM = 16
GRIPPER_COLUMNS = (7, 15)
GRIPPER_OPEN_VAL = 0.044
GRIPPER_CLOSED_VAL = 0.0
GRIPPER_SNAP_CLOSE_BELOW = 0.35 * GRIPPER_OPEN_VAL
GRIPPER_SNAP_OPEN_ABOVE = 0.65 * GRIPPER_OPEN_VAL


def snap_gripper_column(raw: np.ndarray, close_below: float, open_above: float) -> np.ndarray:
    """Quantise one episode's gripper column to fully open / fully closed.

    Mirrors record_demos_openarm.py's ``snap_gripper_command``: below the close threshold is
    closed, above the open threshold is open, and a value between them holds the previous state
    (hysteresis, so a trigger resting near the middle doesn't flap the fingers step to step).
    State starts OPEN at the top of every episode, which is where a demo begins anyway -- unlike
    the live recorder, there is no earlier state to inherit here.
    """
    snapped = np.empty_like(raw)
    state = GRIPPER_OPEN_VAL
    for i, value in enumerate(raw):
        if value <= close_below:
            state = GRIPPER_CLOSED_VAL
        elif value >= open_above:
            state = GRIPPER_OPEN_VAL
        snapped[i] = state
    return snapped


def snap_dataset(dataset_file: str, close_below: float, open_above: float, dry_run: bool) -> None:
    """Rewrite (or, with dry_run, just report on) every episode's gripper columns in place."""
    mode = "r" if dry_run else "r+"
    with h5py.File(dataset_file, mode) as file_stream:
        episode_names = list(file_stream["data"].keys())
        if not episode_names:
            print("No episodes found in the dataset.")
            return

        total_changed = 0
        total_rows = 0
        skipped = []
        for episode_name in episode_names:
            episode_group = file_stream["data"][episode_name]
            if "actions" not in episode_group:
                skipped.append(f"{episode_name} (no actions)")
                continue
            actions = episode_group["actions"][:]
            if actions.ndim != 2 or actions.shape[1] != JOINT_ACTION_DIM:
                skipped.append(f"{episode_name} (actions are {actions.shape}, not Nx{JOINT_ACTION_DIM})")
                continue

            snapped = actions.copy()
            for column in GRIPPER_COLUMNS:
                snapped[:, column] = snap_gripper_column(actions[:, column], close_below, open_above)

            changed = int(np.count_nonzero(np.any(snapped != actions, axis=1)))
            total_changed += changed
            total_rows += actions.shape[0]
            deepest = float(actions[:, GRIPPER_COLUMNS].min())
            closed_frames = int(np.count_nonzero(snapped[:, GRIPPER_COLUMNS] == GRIPPER_CLOSED_VAL))
            print(
                f"  {episode_name}: {actions.shape[0]:5} steps, {changed:5} rows rewritten"
                f" | deepest recorded squeeze {deepest:.4f} m"
                f" ({100 * (1 - deepest / GRIPPER_OPEN_VAL):.0f}% of travel)"
                f" -> {closed_frames} closed gripper-frames"
            )
            if closed_frames == 0:
                if deepest >= 0.98 * GRIPPER_OPEN_VAL:
                    # The trigger was never squeezed at all: there is no grasp to recover here,
                    # whatever the episode looks like.
                    print(f"      warning: {episode_name} never squeezes -- drop or re-record it.")
                else:
                    # The operator DID squeeze, just not past the close threshold -- likely,
                    # because these demos were recorded when the gripper was still proportional
                    # and a partial squeeze was all it took to grip. Snapping it away would
                    # delete a real grasp, so say what threshold would keep it.
                    # Suggest a threshold pair, not just --close_below: the two are ordered, so a
                    # close threshold raised above the default open threshold needs that one
                    # moved too or the run just fails validation.
                    suggested_close = min(0.95, deepest / GRIPPER_OPEN_VAL + 0.05)
                    suggested_open = min(1.0, max(0.65, suggested_close + 0.05))
                    print(
                        f"      warning: {episode_name}'s deepest squeeze ({deepest:.4f} m) never"
                        f" reaches the close threshold ({close_below:.4f} m), so its grasp would be"
                        " snapped to OPEN and lost. If that squeeze was a real grip, re-run with"
                        f" --close_below {suggested_close:.2f} --open_above {suggested_open:.2f}"
                        " (and check the result in replay)."
                    )
            if not dry_run:
                episode_group["actions"][:] = snapped

        for note in skipped:
            print(f"  skipped {note}")
        verb = "would rewrite" if dry_run else "rewrote"
        print(f"{verb} {total_changed} of {total_rows} action rows across {len(episode_names)} episodes.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset_file", type=str, required=True, help="Dataset file to snap.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Where to write the snapped copy. Defaults to <dataset_file stem>_snapped.hdf5.",
    )
    parser.add_argument(
        "--in_place",
        action="store_true",
        default=False,
        help="Rewrite --dataset_file itself instead of copying. The original is not recoverable.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Report what would change and write nothing.",
    )
    parser.add_argument(
        "--close_below",
        type=float,
        default=0.35,
        help=(
            "Fraction of the gripper's travel below which a recorded command counts as 'closed'"
            " (default 0.35, i.e. a trigger squeezed past 65%%, matching the live recorder). Raise"
            " it for demos recorded when the gripper was still proportional and a partial squeeze"
            " was enough to grip -- --dry_run reports the deepest squeeze per episode."
        ),
    )
    parser.add_argument(
        "--open_above",
        type=float,
        default=0.65,
        help=(
            "Fraction of travel above which a recorded command counts as 'open' (default 0.65)."
            " Values between the two thresholds hold the previous state -- that gap is the"
            " hysteresis that stops a trigger resting mid-travel from flapping the fingers."
        ),
    )
    args = parser.parse_args()

    if not os.path.exists(args.dataset_file):
        raise FileNotFoundError(f"The dataset file {args.dataset_file} does not exist.")
    if args.in_place and args.output is not None:
        raise ValueError("--in_place rewrites --dataset_file; it cannot be combined with --output.")
    if not 0.0 <= args.close_below < args.open_above <= 1.0:
        raise ValueError(
            "Expected 0 <= --close_below < --open_above <= 1 (fractions of the gripper's travel),"
            f" got --close_below {args.close_below} and --open_above {args.open_above}."
        )
    close_below = args.close_below * GRIPPER_OPEN_VAL
    open_above = args.open_above * GRIPPER_OPEN_VAL
    print(f"Thresholds: closed at <= {close_below:.4f} m, open at >= {open_above:.4f} m")

    if args.dry_run:
        print(f"Dry run over {args.dataset_file} (nothing will be written):")
        snap_dataset(args.dataset_file, close_below, open_above, dry_run=True)
        return

    if args.in_place:
        target = args.dataset_file
    else:
        target = args.output or f"{os.path.splitext(args.dataset_file)[0]}_snapped.hdf5"
        if os.path.exists(target):
            raise FileExistsError(f"{target} already exists -- pass --output explicitly to write elsewhere.")
        # Copy rather than rebuild: an episode's image observations are the bulk of these files and
        # nothing here has any business touching them.
        print(f"Copying {args.dataset_file} -> {target}")
        shutil.copy2(args.dataset_file, target)

    print(f"Snapping gripper columns {GRIPPER_COLUMNS} in {target}:")
    snap_dataset(target, close_below, open_above, dry_run=False)
    print(f"Done. Replay it with: ./isaaclab.sh -p scripts/tools/replay_demos.py --dataset_file {target} ...")


if __name__ == "__main__":
    main()
