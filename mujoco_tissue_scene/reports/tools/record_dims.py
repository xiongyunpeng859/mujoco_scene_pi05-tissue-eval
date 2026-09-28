#!/usr/bin/env python3
"""Record which tray dimensions are user-measured and which are estimates."""
from __future__ import annotations

import shutil
from pathlib import Path

import yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
CFG = ROOT / "configs/scene.yaml"
config = yaml.safe_load(CFG.read_text())
shutil.copy(CFG, CFG.with_suffix(".yaml.bak_before_dims"))
tray = config["tray"]
tray["size_measured"] = True          # footprint AND height
tray["size_measured_source"] = (
    "21 x 20 x 7.5 cm all measured by the user (confirmed 2026); the footprint was "
    "additionally verified against the green inner floor by PnP at 1.9 px corner RMS.")
tray["wall_thickness"] = 0.008
tray["wall_thickness_measured"] = False
tray["wall_thickness_source"] = (
    "ESTIMATE only: the user has not measured the wall thickness but says it is thin "
    "(2026).  Nothing depends strongly on it because 8 mm is small against a 7.5 cm rim.")
layout = config.setdefault("measured_layout", {})
layout["tray_size_cm_measured"] = [21.0, 20.0, 7.5]
layout.setdefault("notes", []).append(
    "Tray dimensions 21 x 20 x 7.5 cm are ALL user-measured, including the 7.5 cm height, "
    "which had been recorded in the docs without attribution. The 8 mm wall thickness is "
    "an ESTIMATE (the user has not measured it, but says it is thin).")
CFG.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
print("tray dims recorded: size %s (measured), wall %.3f m (estimate)"
      % (tray["size"], tray["wall_thickness"]))
