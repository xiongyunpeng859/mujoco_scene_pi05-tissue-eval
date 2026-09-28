#!/usr/bin/env python3
"""Restore friction 4.0.  The sweep that moved it to 2.5 ran while the target pack
was being placed twice, so its numbers were invalid; the corrected sweep reverses it.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
CFG = ROOT / "configs/scene.yaml"
config = yaml.safe_load(CFG.read_text())
shutil.copy(CFG, CFG.with_suffix(".yaml.bak_before_fric4"))
for box in config["boxes"]:
    if box.get("soft_body"):
        box["friction"] = [4.0, 0.1, 0.002]
block = config.setdefault("boxes_soft_body", {})
block["friction"] = [4.0, 0.1, 0.002]
ev = block.setdefault("evidence", {})
ev["friction_resweep_after_doubleplace_fix"] = (
    "episodes 0-7, kinematic target, duplicates dropped: "
    "2.5 -> grasped 4/8 placed 0/8; 3.0 -> 6/8, 2/8; 3.5 -> 7/8, 1/8; 4.0 -> 7/8, 2/8. "
    "The earlier sweep that selected 2.5 was run with the target placed twice, so it "
    "was invalid; 4.0 is restored.")
ev.pop("friction_sweep_2026", None)
CFG.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
print("friction restored to 4.0")
