#!/usr/bin/env python3
"""Does the hand EVER touch the target pack?  And how high are the fingers?

Section 26: the fingertip midpoint now reaches the pack to 0.6 cm, yet the pack never
moves.  Count hand<->target contacts per segment and report the fingertip height against
the pack's own height range.
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


sub("init-counters",
    "        first_div = None\n        rows = []\n        frames = []",
    "        first_div = None\n        rows = []\n        hp = {}\n        tipz = []\n"
    "        surf_low = []\n        frames = []")

sub("count-handpack",
    """            err = np.abs(np.asarray(info["tracking_error"]))
            if first_div is None""",
    """            err = np.abs(np.asarray(info["tracking_error"]))
            _seg = next((w for w, a0, a1 in seg_bounds if a0 <= step_i < a1), "?")
            _n = 0
            for _k in range(mdata.ncon):
                _ct = mdata.contact[_k]
                _f = -1
                if int(_ct.flex[0]) == fid:
                    _f = 0
                elif int(_ct.flex[1]) == fid:
                    _f = 1
                if _f < 0:
                    continue
                _o = int(_ct.geom1) if _f == 1 else int(_ct.geom2)
                if _o >= 0 and (model.body(int(model.geom_bodyid[_o])).name
                                or "").startswith("hand_"):
                    _n += 1
            if _n:
                hp[_seg] = max(hp.get(_seg, 0), _n)
            _tb = [model.body(x).id for x in
                   ("hand_L_thumb_tip", "hand_L_index_tip", "hand_L_middle_tip",
                    "hand_L_ring_tip", "hand_L_pinky_tip")]
            _tm = 0.5 * (mdata.xpos[_tb[0]]
                         + np.mean([mdata.xpos[i] for i in _tb[1:]], axis=0))
            tipz.append(float(_tm[2]) - config["table"]["surface_z"])
            _a = int(model.flex_vertadr[fid])
            _nn = int(model.flex_vertnum[fid])
            surf_low.append(float(mdata.flexvert_xpos[_a:_a + _nn][:, 2].min())
                            - config["table"]["surface_z"])
            if first_div is None""")

sub("print-handpack",
    "        if rows:",
    """        print("      hand<->target contacts per segment: %s"
              % ("  ".join("%s %d" % kv for kv in hp.items()) if hp else "NONE EVER"))
        print("      fingertip-mid height above table: min %.3f max %.3f | pack lowest "
              "vertex %.3f m" % (min(tipz), max(tipz), min(surf_low)))
        if rows:""")

SRC.write_text(t)
import ast
ast.parse(t)
print("patch results:")
for d in done:
    print("   ", d)
