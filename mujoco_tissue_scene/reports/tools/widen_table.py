#!/usr/bin/env python3
"""The real tabletop is bigger than the assumed 120x75 rectangle.

ep2's pack centre back-projects to x = -4.5 cm and the image (inspect_grip.py)
shows it plainly ON the black table, straddling the projected table boundary.  So
both the filter and the sim's table geom were too tight, and they were throwing
away the very pack the arm grasped.

table.size must NOT change -- the cm<->world conversion uses it -- so widen only
the collision/render geom, and relax the filter by the same margin.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
MARGIN_CM = 15.0
MARGIN_M = MARGIN_CM / 100.0

# 1. config: geom-only margin
cfg_path = ROOT / "configs/scene.yaml"
config = yaml.safe_load(cfg_path.read_text())
shutil.copy(cfg_path, cfg_path.with_suffix(".yaml.bak_before_tablemargin"))
config["table"]["geom_margin_xy"] = [MARGIN_M, MARGIN_M]
cfg_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
print("config: table.geom_margin_xy = [%.2f, %.2f] m" % (MARGIN_M, MARGIN_M))

# 2. scene.py: widen the table geom only
scene_path = ROOT / "scene.py"
text = scene_path.read_text()
anchor = '    geom(world, "black_table", [0, 0, surface-thickness/2],\n         [width/2, depth/2, thickness/2]'
assert text.count(anchor) == 1, "anchor count %d" % text.count(anchor)
new = ('    _mx, _my = config["table"].get("geom_margin_xy", [0.0, 0.0])\n'
       '    geom(world, "black_table", [0, 0, surface-thickness/2],\n'
       '         [width/2 + _mx, depth/2 + _my, thickness/2]')
shutil.copy(scene_path, scene_path.with_suffix(".py.bak_before_tablemargin"))
scene_path.write_text(text.replace(anchor, new))
print("scene.py: table geom widened by %.0f cm per side" % MARGIN_CM)

# 3. align_with_dataset.py: relax the off-table rejection to the same margin
align_path = ROOT / "align_with_dataset.py"
text = align_path.read_text()
old = """        if not (cx - half[0] > 1.0 and cx + half[0] < table_cm[0] - 1.0
                and cy - half[1] > 1.0 and cy + half[1] < table_cm[1] - 1.0):"""
new = """        if not (cx - half[0] > -15.0 and cx + half[0] < table_cm[0] + 15.0
                and cy - half[1] > -15.0 and cy + half[1] < table_cm[1] + 15.0):"""
assert text.count(old) == 1, "filter anchor count %d" % text.count(old)
shutil.copy(align_path, align_path.with_suffix(".py.bak_before_margin"))
align_path.write_text(text.replace(old, new))
print("align_with_dataset.py: off-table margin now +/- %.0f cm" % MARGIN_CM)
