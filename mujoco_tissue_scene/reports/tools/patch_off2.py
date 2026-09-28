#!/usr/bin/env python3
"""Add an off-centre-grasp offset option (robust anchors only)."""
from pathlib import Path
import shutil
ROOT = Path("/workspace/shared/mujoco_tissue_scene")
SRC = ROOT / "validate_kinematic.py"
text = SRC.read_text()
if "pack_offset_cm" in text:
    print("already patched")
    raise SystemExit(0)
old = '    parser.add_argument("--target-source", choices=["fk", "fused"],\n                        default="fk")'
assert text.count(old) == 1, "argparse anchor %d" % text.count(old)
text = text.replace(old, old + '\n    parser.add_argument("--pack-offset-cm", type=float, default=0.0)')
old = """        box = boxes[0]
        adr, dof = env.box_free[box["name"]]
        cx, cy = target_cm[e]"""
assert text.count(old) == 1, "placement anchor %d" % text.count(old)
new = """        box = boxes[0]
        adr, dof = env.box_free[box["name"]]
        cx, cy = target_cm[e]
        if args.pack_offset_cm:
            # emulate an off-centre grasp: the pack extends ahead of the fingers, so
            # shift it along the direction in which it will be carried
            tpx = (data.xpos[env.tray_body][0] + tcm[0] / 200.0) * 100.0
            tpy = (data.xpos[env.tray_body][1] + tcm[1] / 200.0) * 100.0
            vx, vy = tpx - cx, tpy - cy
            nn = float(np.hypot(vx, vy)) or 1.0
            cx = cx + args.pack_offset_cm * vx / nn
            cy = cy + args.pack_offset_cm * vy / nn"""
text = text.replace(old, new)
shutil.copy(SRC, SRC.with_suffix(".py.bak_before_offset"))
SRC.write_text(text)
import ast
ast.parse(text)
print("pack-offset option added; syntax ok")
