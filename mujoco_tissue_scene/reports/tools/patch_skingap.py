#!/usr/bin/env python3
"""How low does the finger SKIN actually get?

Section 45: the fingertip-midpoint body origins sit at ~13 cm and never come down, yet the
IK targets exactly that height and reaches it.  In a real posture the skin hangs 8-10 cm
below those origins, so it should be at 3-5 cm.  Measure the real thing instead of inferring
it: the minimum distance from the hand's GEOMS to the table geom, per segment, using
mj_geomDistance.
"""
from pathlib import Path

SRC = Path("/workspace/shared/mujoco_tissue_scene/reports/tools/scripted_pick_place.py")
t = SRC.read_text()
done = []


def sub(tag, old, new):
    global t
    n = t.count(old)
    if n == 1:
        t = t.replace(old, new)
        done.append(tag)
    else:
        done.append("%s SKIPPED(%d)" % (tag, n))


sub("geom-ids",
    "        peak, track = 0.0, 0.0\n        worst_vec = None",
    "        table_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, \"black_table\")\n"
    "        hand_gids = [g for g in range(model.ngeom)\n"
    "                     if (model.body(int(model.geom_bodyid[g])).name or \"\").startswith(\"hand_\")]\n"
    "        skin = {}\n"
    "        peak, track = 0.0, 0.0\n        worst_vec = None")

sub("skin-per-step",
    "            _seg = next((w for w, a0, a1 in seg_bounds if a0 <= step_i < a1), \"?\")\n            _n = 0",
    "            _seg = next((w for w, a0, a1 in seg_bounds if a0 <= step_i < a1), \"?\")\n"
    "            _dmin = min(mujoco.mj_geomDistance(model, mdata, _g, table_gid, 1.0, None)\n"
    "                        for _g in hand_gids)\n"
    "            skin[_seg] = min(skin.get(_seg, 9.9), float(_dmin))\n"
    "            _n = 0")

sub("skin-print",
    "        print(\"      hand<->target contacts per segment:",
    "        print(\"      finger-skin lowest gap to the table per segment: \"\n"
    "              + \"  \".join(\"%s %.4f\" % kv for kv in skin.items()))\n"
    "        print(\"      overall lowest skin gap: %.4f m (pack is 0.065 m tall)\"\n"
    "              % min(skin.values()))\n"
    "        print(\"      hand<->target contacts per segment:")

SRC.write_text(t)
import ast
ast.parse(t)
print("patch results:")
for d in done:
    print("   ", d)
