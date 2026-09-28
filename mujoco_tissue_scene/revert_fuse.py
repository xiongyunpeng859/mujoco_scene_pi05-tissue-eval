#!/usr/bin/env python3
"""Make the FK closure the DEFAULT target source again.

Fusing in the detector's centre was tested and is clearly worse: the detector's
box_centre_cm is built from the contact edge plus the ASSUMED footprint depth, so it
carries its own error, while the FK closure point is validated to land on the real pack.
"""
from pathlib import Path
import shutil

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
SRC = ROOT / "validate_kinematic.py"
text = SRC.read_text()
old = '    parser.add_argument("--target-source", choices=["fk", "fused"],\n                        default="fused")'
new = '    parser.add_argument("--target-source", choices=["fk", "fused"],\n                        default="fk")'
assert text.count(old) == 1, text.count(old)
shutil.copy(SRC, SRC.with_suffix(".py.bak_before_fuse_default"))
SRC.write_text(text.replace(old, new))
print("target-source default -> fk")
