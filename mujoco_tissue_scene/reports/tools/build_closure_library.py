#!/usr/bin/env python3
"""Real demo postures: grasp (hand closes) AND release (hand opens).

Each episode yields two real, reachable arm postures:
  * grasp   -- at the first frame the hand is fully closed, over the pack;
  * release -- at the LAST frame the hand is still closed, which is the posture that
               actually put the pack in the box.
The second is the right reference for the box waypoints, where a pack-grasp posture is a
poor seed and where the pack-aligned yaw is meaningless.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reports/tools"))
import dataset_io
from scripted_pick_place import Arm, PITCH, SUCCESS

CFG = ROOT / "configs/scene.yaml"
OUT = ROOT / "outputs/closure_library"


def main() -> int:
    config = yaml.safe_load(CFG.read_text())
    ts = config["table"]["size"]
    tray = config["tray"]
    box_cm = [(tray["center"][0] + ts[0] / 2) * 100.0,
              (tray["center"][1] + ts[1] / 2) * 100.0]
    data, order = dataset_io.load(SUCCESS, fields=("observation.state", "action"))
    arm = Arm(CFG)

    def entry(e, q, hand=None):
        arm.set_arm(q)
        if hand is not None:
            arm.set_hand(hand)
            arm.mujoco.mj_kinematics(arm.model, arm.data)
        g = arm.tip_mid()
        return {"episode": int(e), "x_cm": float((g[0] + ts[0] / 2) * 100.0),
                "y_cm": float((g[1] + ts[1] / 2) * 100.0), "z_m": float(g[2]),
                "q": [float(v) for v in q]}

    grasp, release = [], []
    for e in order:
        st, ac = data[e]["observation.state"], data[e]["action"]
        closed = (np.abs(ac[:, PITCH]) > 0.35).all(axis=1)
        idx = np.where(closed)[0]
        if not len(idx):
            continue
        # the hand pose matters: the fingertip midpoint moves several cm between the
        # open and closed gestures, so record the position WITH the hand actually closed
        grasp.append(entry(e, st[int(idx[0])][:6], ac[int(idx[0])][6:]))
        release.append(entry(e, st[int(idx[-1])][:6], ac[int(idx[-1])][6:]))
    (OUT / "library.json").write_text(json.dumps({"grasp": grasp, "release": release},
                                                 indent=1))
    for name, rows in (("grasp", grasp), ("release", release)):
        xs = np.array([p["x_cm"] for p in rows])
        ys = np.array([p["y_cm"] for p in rows])
        zs = np.array([p["z_m"] for p in rows])
        d = np.hypot(xs - box_cm[0], ys - box_cm[1])
        print("%-8s %3d postures  x %6.1f..%6.1f  y %6.1f..%6.1f  z %.3f..%.3f"
              % (name, len(rows), xs.min(), xs.max(), ys.min(), ys.max(),
                 zs.min(), zs.max()))
        if name == "release":
            print("         distance from the box centre: median %.2f cm  max %.2f cm"
                  % (np.median(d), d.max()))
            j = int(np.argmin(d))
            print("         closest release posture: ep%d at (%.1f, %.1f), %.2f cm from box"
                  % (rows[j]["episode"], rows[j]["x_cm"], rows[j]["y_cm"], d[j]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
