#!/usr/bin/env python3
"""At the step of maximum tracking error, name what the arm is touching."""
from pathlib import Path

SRC = Path("/workspace/shared/mujoco_tissue_scene/reports/tools/scripted_pick_place.py")
t = SRC.read_text()
if "worst_pairs" in t:
    print("already patched")
    raise SystemExit(0)

old = """        peak, track = 0.0, 0.0
        worst_vec = None
        frames = []"""
new = """        peak, track = 0.0, 0.0
        worst_vec = None
        worst_step, worst_pairs, worst_clip = -1, [], False
        worst_cmd = None
        frames = []"""
assert t.count(old) == 1, "init %d" % t.count(old)
t = t.replace(old, new)

old = """            if float(err.max()) > track:
                track = float(err.max())
                worst_vec = np.asarray(info["tracking_error"]).copy()"""
new = """            if float(err.max()) > track:
                track = float(err.max())
                worst_vec = np.asarray(info["tracking_error"]).copy()
                worst_step = step_i
                worst_clip = bool(info.get("action_clipped", False))
                worst_cmd = np.asarray(value).copy()
                pairs = {}
                for k in range(mdata.ncon):
                    ct = mdata.contact[k]
                    b1 = model.body(int(model.geom_bodyid[ct.geom1])).name or "world"
                    b2 = model.body(int(model.geom_bodyid[ct.geom2])).name or "world"
                    key = " <-> ".join(sorted([b1, b2]))
                    pairs[key] = pairs.get(key, 0) + 1
                worst_pairs = sorted(pairs.items(), key=lambda kv: -kv[1])[:8]"""
assert t.count(old) == 1, "capture %d" % t.count(old)
t = t.replace(old, new)

old = """        if e < 3 and worst_vec is not None:"""
new = """        if e < 3:
            print("      worst step %d: action_clipped=%s  contacts=%d"
                  % (worst_step, worst_clip, sum(n for _, n in worst_pairs)))
            for key, n in worst_pairs:
                print("         %3d x %s" % (n, key))
            if worst_cmd is not None:
                print("         arm cmd  : %s" % np.round(worst_cmd[:6], 3))
                print("         arm aimed: %s"
                      % np.round(worst_cmd[:6] + worst_vec[:6], 3))
        if e < 3 and worst_vec is not None:"""
assert t.count(old) == 1, "print %d" % t.count(old)
t = t.replace(old, new)

SRC.write_text(t)
import ast
ast.parse(t)
print("contact capture added")
