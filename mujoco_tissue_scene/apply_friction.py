#!/usr/bin/env python3
"""Set the hand-bag friction to the value that both holds and releases.

friction 4.0 holds the pinch but drags the bag out of the box on retreat (placed 0/4);
friction 2.5 holds just as well (grasped 4/4) and lets ep0 settle inside (0.92 inside);
below ~2.0 the pinch fails outright (0/4 grasped).  2.5 strictly dominates 4.0.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
CFG = ROOT / "configs/scene.yaml"
NEW = [2.5, 0.1, 0.002]

config = yaml.safe_load(CFG.read_text())
shutil.copy(CFG, CFG.with_suffix(".yaml.bak_before_friction"))
for box in config["boxes"]:
    if box.get("soft_body"):
        box["friction"] = list(NEW)
config.setdefault("boxes_soft_body", {})["friction"] = list(NEW)
evidence = config["boxes_soft_body"].setdefault("evidence", {})
evidence["friction_sweep_2026"] = (
    "grasped/placed over episodes 0,1,3,4: 4.0 -> 4/4, 0/4; 2.5 -> 4/4, 1/4; "
    "1.5 -> 0/4, 0/4; 0.8 -> 0/4, 0/4. Release trace shows why 4.0 fails: the hand "
    "opens with the bag 68% inside the box, then the retreating palm keeps 14-25 "
    "contacts and drags the bag back to 0.187m outside.")
CFG.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
print("friction -> %s  (2.5 dominates 4.0: same grip, real placement)" % NEW)
