#!/usr/bin/env python3
"""Compare named bases on the grasp-independent criterion, and look past the grid edge."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import dataset_io
import scene

SUCCESS = Path("/workspace/shared/new_program_qiuzhi/without_tactile/success_episode/"
               "pick_up_the_tissue_pack_and_place_it_on_the_right_side_"
               "20260908_merged_all_108eps")


def main() -> int:
    import mujoco
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    surface = config["table"]["surface_z"]
    table = config["table"]["size"]
    xml = scene.build(ROOT / "configs/scene.yaml", "/tmp/htp/scene.xml", with_hand=True)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mount = model.body("qiuzhi_arm_mount").id
    palm = model.body("static_omnihand_reference").id
    tray_body = model.body("tray").id
    adr = [model.joint("joint%d" % i).qposadr[0] for i in range(1, 7)]
    mujoco.mj_kinematics(model, data)
    tray_xy = data.xpos[tray_body][:2].copy()

    episodes, order = dataset_io.load(SUCCESS, fields=("observation.state",))
    states = [episodes[e]["observation.state"][:, :6] for e in order[:30]]

    def evaluate(x_cm, y_cm, yaw, skip=60, detail=False):
        model.body_pos[mount] = [x_cm / 100.0 - table[0] / 2.0,
                                 y_cm / 100.0 - table[1] / 2.0, surface]
        model.body_quat[mount] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        per = []
        for rows in states:
            best = 9.9
            for row in rows[skip:]:
                data.qpos[adr] = row
                mujoco.mj_kinematics(model, data)
                p3 = data.xpos[palm]
                if p3[2] < surface + 0.02:
                    continue
                d = float(np.linalg.norm(p3[:2] - tray_xy))
                if d < best:
                    best = d
            per.append(best)
        per = np.array(per)
        return per

    named = [("fit from grasp (18,10,90)", 18.0, 10.0, np.pi / 2),
             ("user tape (25,10,90)", 25.0, 10.0, np.pi / 2),
             ("tray-over best (34,24,60)", 34.0, 24.0, np.radians(60)),
             ("same but yaw 90", 34.0, 24.0, np.pi / 2),
             ("(34,24,60) with skip=0", 34.0, 24.0, np.radians(60))]
    print("  %-26s %10s %10s %12s" % ("base", "mean", "worst", "within 12cm"))
    for label, x, y, yaw in named:
        skip = 0 if "skip=0" in label else 60
        per = evaluate(x, y, yaw, skip=skip)
        print("  %-26s %10.4f %10.4f %12.2f"
              % (label, per.mean(), per.max(), (per < 0.12).mean()))

    print()
    print("  fine sweep past the grid edge (skip=60):")
    print("  %-24s %10s %10s %12s" % ("base", "mean", "worst", "within 12cm"))
    rows = []
    for x in (30.0, 34.0, 38.0, 42.0):
        for y in (18.0, 24.0, 30.0):
            for yaw_deg in (45.0, 60.0, 75.0):
                per = evaluate(x, y, np.radians(yaw_deg))
                rows.append((float(per.mean()), x, y, yaw_deg, float(per.max()),
                             float((per < 0.12).mean())))
    rows.sort()
    for mean_d, x, y, yaw_deg, worst, frac in rows[:10]:
        print("  x=%5.1f y=%5.1f yaw=%5.1f  %10.4f %10.4f %12.2f"
              % (x, y, yaw_deg, mean_d, worst, frac))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
