#!/usr/bin/env python3
"""Drop the arm base so the fingers scoop UNDER the pack instead of pinching its side.

Measured with mj_geomDistance (exact skin-to-table gap, so a link-origin offset
cannot fool it): at the frame the hand closes, the closest fingertip surface is
3.2-4.3 cm above the tabletop, while the pack is 6.5 cm tall -- mid-pack, i.e. a side
pinch.  The user's real grasp scoops the four fingers under the pack, which needs the
fingertips at the tabletop.  The hand CAN reach the table (gap 0.0 at the rest pose),
so this is a base-height offset, not a reach limit.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
CFG = ROOT / "configs/scene.yaml"
DROP = float(sys.argv[1]) if len(sys.argv) > 1 else 0.037

config = yaml.safe_load(CFG.read_text())
shutil.copy(CFG, CFG.with_suffix(".yaml.bak_before_basez"))
old = config["arm"]["position"][2]
config["arm"]["position"][2] = float(old - DROP)
config.setdefault("arm", {}).setdefault("base_z_evidence", {})
config["arm"]["base_z_evidence"] = {
    "measured_gap_at_closure_m": [0.0320, 0.0431],
    "method": "mj_geomDistance from the fingertip geoms to black_table",
    "why": "real grasp scoops four fingers UNDER the pack (user); sim pinched its side",
    "drop_m": DROP,
}
CFG.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
print("arm base z %.4f -> %.4f (dropped %.3f m)" % (old, config["arm"]["position"][2], DROP))
