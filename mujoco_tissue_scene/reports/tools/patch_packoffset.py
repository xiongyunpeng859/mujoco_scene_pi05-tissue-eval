#!/usr/bin/env python3
"""Print, frame by frame through the grasp window, where the PACK is versus where the
fingertip midpoint I am aiming at is -- both in table cm.  Section 31/22's images showed
the hand up-and-left of the pack, but the IK residual is only 2-5 mm, so the offset must
live between the target I aim at and the pack itself.  Numbers, not pixels."""
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


sub("rows-init",
    "        first_div = None\n        frames = []",
    "        first_div = None\n        rows = []\n        frames = []")

sub("rows-capture",
    """            if step_i % 24 == 0 or step_i == len(actions) - 1:
                frames.append((step_i, np.asarray(obs["observation.images.top"]).copy()))""",
    """            if step_i % 24 == 0 or step_i == len(actions) - 1:
                frames.append((step_i, np.asarray(obs["observation.images.top"]).copy()))
            if 30 <= step_i <= 110 and step_i % 6 == 0:
                _a = int(model.flex_vertadr[fid])
                _n = int(model.flex_vertnum[fid])
                _pc = mdata.flexvert_xpos[_a:_a + _n].mean(0)
                _tb = [model.body(x).id for x in
                       ("hand_L_thumb_tip", "hand_L_index_tip", "hand_L_middle_tip",
                        "hand_L_ring_tip", "hand_L_pinky_tip")]
                _tm = 0.5 * (mdata.xpos[_tb[0]]
                             + np.mean([mdata.xpos[i] for i in _tb[1:]], axis=0))
                _seg = next((w for w, a0, a1 in seg_bounds if a0 <= step_i < a1), "?")
                rows.append((step_i, _seg,
                             (_pc[0] + ts[0] / 2) * 100.0, (_pc[1] + ts[1] / 2) * 100.0,
                             (_tm[0] + ts[0] / 2) * 100.0, (_tm[1] + ts[1] / 2) * 100.0,
                             float(np.hypot(_pc[0] - _tm[0], _pc[1] - _tm[1])) * 100.0))""")

sub("rows-print",
    "        if first_div is not None:",
    """        if rows:
            print("      step segment     pack(x,y) cm      tip-mid(x,y) cm    gap cm")
            for r in rows:
                print("      %4d %-10s (%6.1f,%6.1f)   (%6.1f,%6.1f)   %6.1f"
                      % (r[0], r[1], r[2], r[3], r[4], r[5], r[6]))
        if first_div is not None:""")

SRC.write_text(t)
import ast
ast.parse(t)
print("patch results:")
for d in done:
    print("   ", d)
