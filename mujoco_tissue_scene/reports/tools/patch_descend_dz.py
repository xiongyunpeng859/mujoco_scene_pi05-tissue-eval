#!/usr/bin/env python3
"""Add --descend-dz so the grasp height can be swept.

Section 47: the fingers stop 1-8 cm ABOVE the pack's top face and never touch it, so the
grasp height is the single outstanding defect.  Sweep the descend/close/lift waypoint z and
look for the first setting where hand<->target contacts become non-zero.
"""
from pathlib import Path

SRC = Path("/workspace/shared/mujoco_tissue_scene/reports/tools/scripted_pick_place.py")
t = SRC.read_text()
if "descend_dz" in t:
    print("already patched")
    raise SystemExit(0)
old = '    parser.add_argument("--pack-offset-cm", type=float, default=0.0)'
assert t.count(old) == 1, "arg anchor %d" % t.count(old)
t = t.replace(old, old + '\n    parser.add_argument("--descend-dz", type=float, default=0.0)')
old = "            z_here = (ref_z_pack if over_pack else ref_z_box) + dz"
assert t.count(old) == 1, "z anchor %d" % t.count(old)
t = t.replace(old,
    "            z_here = (ref_z_pack if over_pack else ref_z_box) + dz\n"
    "            if name in (\"descend\", \"close\", \"lift\"):\n"
    "                z_here += args.descend_dz")
SRC.write_text(t)
import ast
ast.parse(t)
print("--descend-dz added")
