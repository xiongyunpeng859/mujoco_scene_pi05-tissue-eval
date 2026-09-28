#!/usr/bin/env python3
"""Is the scripted controller's grasp posture plausible, or absurd?

The user requires postures "at least like a posture a human hand could grasp with".  The
deep grasp (--descend-dz -0.20) is a 20 cm deviation from the recorded reference height, so
measure how far the collected grasp-frame arm joints are from the REAL demonstrated grasp
postures in the closure library.  A real grasp posture is the benchmark for "plausible".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import dataset_io

D = ROOT / "outputs/lerobot_sim"
PITCH = [8, 10, 11, 13, 15]
lib = json.loads((ROOT / "outputs/closure_library/library.json").read_text())["grasp"]
lib_q = np.array([e["q"] for e in lib])                       # (108, 6)
lib_x = np.array([e["x_cm"] for e in lib])
lib_y = np.array([e["y_cm"] for e in lib])

# how spread out are the REAL grasp postures themselves?  This is the yardstick: a
# plausible posture should sit inside the cloud of real ones, not outside it.
print("real demonstrated grasp postures (%d), per-joint spread:" % len(lib_q))
for i, nm in enumerate(("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")):
    print("   %-7s mean %+7.3f  std %.3f  range [%+.3f, %+.3f]"
          % (nm, lib_q[:, i].mean(), lib_q[:, i].std(),
             lib_q[:, i].min(), lib_q[:, i].max()))

episodes, order = dataset_io.load(D, fields=("action", "observation.state"))
print()
print("  %-4s %-18s %-28s %9s %9s %9s" %
      ("ep", "pack cm (sampled)", "grasp-frame arm joints", "max dev", "mean dev",
       "inside real range?"))
all_dev = []
for e in order:
    ac = episodes[e]["action"]
    st = episodes[e]["observation.state"]
    closed = (np.abs(ac[:, PITCH]) > 0.35).all(axis=1)
    idx = np.where(closed)[0]
    if not len(idx):
        continue
    f = int(idx[0])
    q = st[f][:6]
    # nearest real grasp posture is unknown for a sampled pack, so compare against the
    # whole real cloud: the distance to the CLOSEST real posture is the fair measure
    d = np.linalg.norm(lib_q - q[None, :], axis=1)
    j = int(np.argmin(d))
    dev = q - lib_q[j]
    inside = all(lib_q[:, i].min() - 0.15 <= q[i] <= lib_q[:, i].max() + 0.15
                 for i in range(6))
    all_dev.append(np.abs(dev).max())
    print("  %-4d %-18s %-28s %9.3f %9.3f %9s"
          % (e, "[sampled]", np.array2string(q, precision=2, suppress_small=True),
             np.abs(dev).max(), np.abs(dev).mean(), "yes" if inside else "NO"))
print()
all_dev = np.array(all_dev)
print("max deviation from the CLOSEST real grasp posture: median %.3f rad, worst %.3f rad"
      % (np.median(all_dev), all_dev.max()))
print("(= %.1f deg median, %.1f deg worst)" % (np.degrees(np.median(all_dev)),
                                               np.degrees(all_dev.max())))
print()
print("verdict: a posture within ~0.2-0.3 rad of a real demonstrated grasp is plausible;")
print("         a deviation of >=1 rad means the arm is doing something the demos never show.")
