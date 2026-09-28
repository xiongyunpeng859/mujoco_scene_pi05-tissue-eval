#!/usr/bin/env python3
"""The target is placed at the FK closure point, which IS where the detector finds
the same pack (they agree to 2.2-2.5 cm).  Placing a distractor at that detection too
puts two packs in the same place; they interpenetrate and are ejected.  Drop any
detection that coincides with the target, and keep only genuinely different packs.
"""
from pathlib import Path
import shutil

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
SRC = ROOT / "validate_kinematic.py"
text = SRC.read_text()
if "duplicate-of-target" in text:
    print("already patched")
    raise SystemExit(0)

# A. yaw mode flag
old = '    parser.add_argument("--yaw", type=float, default=None)\n'
new = ('    parser.add_argument("--yaw", type=float, default=None)\n'
       '    parser.add_argument("--target-yaw", choices=["tray", "closing"],\n'
       '                        default="tray")\n')
assert text.count(old) == 1
text = text.replace(old, new)

# B. keep the closing yaw separate so both are available
old = "    target_cm = {}\n    target_yaw = {}\n"
assert text.count(old) == 1
text = text.replace(old, "    target_cm = {}\n    target_yaw = {}\n    target_yaw_close = {}\n")
old = """            target_yaw[e] = float(np.arctan2(perp[1], perp[0]))
        else:
            target_yaw[e] = float(tyaw)"""
new = """            target_yaw_close[e] = float(np.arctan2(perp[1], perp[0]))
        else:
            target_yaw_close[e] = float(tyaw)"""
assert text.count(old) == 1
text = text.replace(old, new)
old = "    env = sim_env.TissueSceneEnv(config_path=path, dataset=SUCCESS, render=False,\n                                 output_dir=OUT)"
assert text.count(old) == 1
text = text.replace(old,
    "    for e in target_yaw_close:\n"
    "        target_yaw[e] = (float(tyaw) if args.target_yaw == \"tray\"\n"
    "                         else target_yaw_close[e])\n"
    "    env = sim_env.TissueSceneEnv(config_path=path, dataset=SUCCESS, render=False,\n"
    "                                 output_dir=OUT)")

# C. drop detections that are the same physical pack as the target
old = """        bags = align.measure_bags(align.dataset_frame(SUCCESS, e, 0), config)
        for box2, bag in zip(boxes[1:], bags):"""
new = """        bags = align.measure_bags(align.dataset_frame(SUCCESS, e, 0), config)
        # the FK closure point and the detector agree to ~2.5 cm, so a detection
        # that close is the SAME pack -- placing it again overlaps two flexes
        bags = [b for b in bags
                if np.hypot(b["box_centre_cm"][0] - cx,
                            b["box_centre_cm"][1] - cy) > 8.0]   # not duplicate-of-target
        for box2, bag in zip(boxes[1:], bags):"""
assert text.count(old) == 1
text = text.replace(old, new)

shutil.copy(SRC, SRC.with_suffix(".py.bak_before_doubleplace"))
SRC.write_text(text)
import ast
ast.parse(text)
print("patched: duplicate placement dropped, --target-yaw added; syntax ok")
