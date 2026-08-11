# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bake a mirror reflection into a binary STL collision mesh.

Why this exists
---------------
URDFs often mirror a left/right part by reusing one mesh with a negative scale
factor, e.g.::

    <mesh filename=".../finger.stl" scale="0.001 -0.001 0.001"/>

That works for *visuals* -- renderers handle a negative-determinant transform
fine -- but PhysX does not cook collision geometry for a mirrored (negative
determinant) scale, so the collider is silently dropped and the link passes
straight through everything. In the OpenArm ``v1_camera`` model this is exactly
what left ``openarm_right_right_finger`` (and its left-arm twin) with no
collision at all, so the gripper closed through the cube instead of gripping it.

The fix is to bake the reflection into the mesh itself and reference it with a
positive scale. Reflection reverses triangle orientation, so each triangle's
winding is flipped (v2 <-> v3) to keep normals pointing outward; the stored
facet normal is reflected to match.

Usage::

    ./isaaclab.sh -p scripts/tools/mirror_collision_mesh.py in.stl out.stl --axis y
"""

import argparse
import struct

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def mirror_binary_stl(src: str, dst: str, axis: str = "y") -> int:
    """Reflect *src* about the plane normal to *axis* and write it to *dst*.

    Returns the number of triangles written.
    """
    ax = AXIS_INDEX[axis]
    with open(src, "rb") as f:
        header = f.read(80)
        (n_tri,) = struct.unpack("<I", f.read(4))
        body = f.read(n_tri * 50)
    if len(body) != n_tri * 50:
        raise ValueError(f"{src}: truncated -- expected {n_tri * 50} bytes of triangle data, got {len(body)}")

    out = bytearray()
    for i in range(n_tri):
        rec = body[i * 50 : (i + 1) * 50]
        vals = list(struct.unpack("<12f", rec[:48]))
        attr = rec[48:50]
        normal, v1, v2, v3 = vals[0:3], vals[3:6], vals[6:9], vals[9:12]
        for vec in (normal, v1, v2, v3):
            vec[ax] = -vec[ax]
        # a reflection reverses orientation: swap two vertices so the winding
        # still agrees with the (reflected) outward normal
        out += struct.pack("<12f", *normal, *v1, *v3, *v2) + attr

    with open(dst, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", n_tri))
        f.write(bytes(out))
    return n_tri


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="path to the source binary STL")
    p.add_argument("output", help="path to write the mirrored binary STL")
    p.add_argument("--axis", default="y", choices=sorted(AXIS_INDEX), help="axis to reflect about (default: y)")
    args = p.parse_args()

    n = mirror_binary_stl(args.input, args.output, args.axis)
    print(f"mirrored {n} triangles about {args.axis}: {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
