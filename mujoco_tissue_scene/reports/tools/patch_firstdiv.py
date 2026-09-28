#!/usr/bin/env python3
"""Ease from the arm's ACTUAL post-reset state, and report the FIRST divergence step.

Section 38: joint6 is perfect in isolation, so the section-37 failure is a trajectory
transient.  Two changes: stop easing from a fixed q_ref, and stop looking only at the
worst step -- find the first step where command and achievement part company.
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


sub("ease-from-actual",
    "        actions, prev, seg_bounds = [], np.concatenate([q_ref, hand_open]), []",
    "        # ease from where the arm ACTUALLY is after reset, not from a fixed posture\n"
    "        actions, prev, seg_bounds = [], mdata.qpos[env.joint_adr].copy(), []")

sub("first-div-init",
    """        worst_cmd = None
        frames = []""",
    """        worst_cmd = None
        first_div = None
        frames = []""")

sub("first-div-capture",
    """            err = np.abs(np.asarray(info["tracking_error"]))
            if float(err.max()) > track:""",
    """            err = np.abs(np.asarray(info["tracking_error"]))
            if first_div is None and float(err[:6].max()) > 0.5:
                ach = np.asarray(value) + np.asarray(info["tracking_error"])
                seg = next((w for w, a0, a1 in seg_bounds if a0 <= step_i < a1), "?")
                first_div = (step_i, seg, np.asarray(value)[:6].copy(), ach[:6].copy())
            if float(err.max()) > track:""")

sub("first-div-print",
    """        if e < 3:
            print("      worst step %d""",
    """        if first_div is not None:
            print("      FIRST divergence at step %d (segment %s)" % (first_div[0],
                                                                     first_div[1]))
            print("         cmd : %s" % np.round(first_div[2], 3))
            print("         ach : %s" % np.round(first_div[3], 3))
        else:
            print("      no divergence above 0.5 rad on any arm joint")
        if e < 3:
            print("      worst step %d""")

SRC.write_text(t)
import ast
ast.parse(t)
print("patch results:")
for d in done:
    print("   ", d)
