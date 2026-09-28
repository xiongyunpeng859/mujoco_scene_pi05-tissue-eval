#!/usr/bin/env python3
"""Which part of the sampling region can the arm actually reach with a sane grasp posture?

An IK that hits the target exactly when free to choose any posture, but only within
centimetres when forced to stay near a real grasp posture, means the target is at or
beyond the reachable workspace.  Map it over the user's region on a grid.
"""
from __future__ import annotations

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


def main() -> int:
    config = yaml.safe_load(CFG.read_text())
    ts = config["table"]["size"]
    tray = config["tray"]
    region = config["box_randomization"]["region_xy_cm_from_left_bottom"]
    data, order = dataset_io.load(SUCCESS, fields=("observation.state", "action"))
    st, ac = data[0]["observation.state"], data[0]["action"]
    cf = np.where((np.abs(ac[:, PITCH]) > 0.35).all(axis=1))[0]
    q_ref = st[int(cf[0])][:6]
    arm = Arm(CFG)
    arm.set_arm(q_ref)
    ref = arm.tip_mid()
    print("base at table (%.1f, %.1f) cm, z %.3f ; reference grasp height %.3f m (%.1f cm up)"
          % (config["measured_layout"]["arm_mount_xy_cm_from_left_bottom"][0],
             config["measured_layout"]["arm_mount_xy_cm_from_left_bottom"][1],
             config["arm"]["position"][2], ref[2], (ref[2] - config["table"]["surface_z"]) * 100))
    print("sampling region x %s  y %s" % (region["x"], region["y"]))
    print()
    want_yaw = float(tray["yaw"]) + np.pi / 2.0
    xs = np.arange(region["x"][0], region["x"][1] + 1, 4.0)
    ys = np.arange(region["y"][0], region["y"][1] + 1, 4.0)
    print("IK residual at the grasp height, forcing a real grasp posture.")
    print("symbols: . <1cm   o <3cm   x <8cm   X >=8cm (unreachable)")
    print()
    print("      " + "".join("%6.0f" % x for x in xs))
    worst_by_x = {}
    for y in ys:
        row = ""
        for x in xs:
            px = x / 100.0 - ts[0] / 2.0
            py = y / 100.0 - ts[1] / 2.0
            _, err, _ = arm.solve((px, py), ref[2], want_yaw, q_ref, q_ref)
            sym = "." if err < 0.01 else ("o" if err < 0.03 else ("x" if err < 0.08 else "X"))
            row += "%6s" % sym
            worst_by_x.setdefault(x, []).append(err)
        print("y%5.0f%s" % (y, row))
    print()
    print("  mean IK residual by x (cm):")
    for x in xs:
        e = np.array(worst_by_x[x])
        print("     x%5.0f  mean %.4f m   max %.4f m   reachable(<1cm) %d/%d"
              % (x, e.mean(), e.max(), int((e < 0.01).sum()), len(e)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
