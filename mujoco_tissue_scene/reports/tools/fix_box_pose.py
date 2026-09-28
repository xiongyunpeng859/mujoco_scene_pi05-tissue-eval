#!/usr/bin/env python3
"""Apply the verified green-box pose.  Both the tray block and measured_layout
feed the builder, so both must be corrected or the old value wins."""
from __future__ import annotations

import shutil
from pathlib import Path

import yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
CFG = ROOT / "configs/scene.yaml"

NEW_CM = (61.39, 39.66)          # 49 clean green-quad fits, residual 1.9 px
NEW_YAW_RAD = -1.4629            # -83.82 deg, folded mod 180 onto the old branch

config = yaml.safe_load(CFG.read_text())
table = config["table"]["size"]
old_center = list(config["tray"]["center"])
old_yaw = float(config["tray"]["yaw"])

shutil.copy(CFG, CFG.with_suffix(".yaml.bak_before_boxpose"))
world = [NEW_CM[0] / 100.0 - table[0] / 2.0, NEW_CM[1] / 100.0 - table[1] / 2.0]
config["tray"]["center"] = [float(world[0]), float(world[1])]
config["tray"]["yaw"] = float(NEW_YAW_RAD)

layout = config.setdefault("measured_layout", {})
layout["tray_center_xy_cm_from_left_bottom"] = [float(NEW_CM[0]), float(NEW_CM[1])]
notes = layout.setdefault("notes", [])
notes.append(
    "CORRECTED 2026: the green box centre was recorded as 71.25cm,40.61cm. That pose "
    "projects 46px away from the box's own green floor in the calibrated image, i.e. it "
    "was never the box. Refitting the floor over 49 clean frames gives "
    "61.39+/-0.20cm, 39.66+/-0.17cm, yaw -83.8+/-1.3deg at 1.9px corner RMS; the old "
    "pose is 9.9cm off in x and 7.4deg off in yaw. scene.py rebuilds tray.center from "
    "measured_layout, which is how the wrong value survived every rebuild.")

CFG.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
print("tray centre  %s -> %s  (world m)" % (old_center, config["tray"]["center"]))
print("tray yaw     %.4f -> %.4f rad" % (old_yaw, config["tray"]["yaw"]))
print("table cm     (%.2f, %.2f) -> (%.2f, %.2f)"
      % ((old_center[0] + table[0] / 2) * 100, (old_center[1] + table[1] / 2) * 100,
         NEW_CM[0], NEW_CM[1]))
