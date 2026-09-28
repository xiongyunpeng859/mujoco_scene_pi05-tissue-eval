#!/usr/bin/env python3
"""Do the IK's model and the env's model actually agree?

The IK converges to 8.9 mm but the env's fingertip settles 5.4 cm from the same target,
even after the arm stops moving.  That means the two models differ.  Compare the arm base
and a sample FK in both.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reports/tools"))
import action_layout as layout
import dataset_io
import mujoco
import sim_env
from scripted_pick_place import Arm, PITCH, SUCCESS

OUT = ROOT / "outputs/model_compare"
OUT.mkdir(parents=True, exist_ok=True)
CFG = ROOT / "configs/scene.yaml"


def main() -> int:
    config = yaml.safe_load(CFG.read_text())
    p = OUT / "scene.yaml"
    p.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    print("config arm.position          : %s" % config["arm"]["position"])
    print("measured_layout arm mount cm : %s"
          % config["measured_layout"]["arm_mount_xy_cm_from_left_bottom"])
    ts = config["table"]["size"]
    lc = config["measured_layout"]["arm_mount_xy_cm_from_left_bottom"]
    print("  -> that cm converts to world : (%.3f, %.3f)"
          % (lc[0] / 100.0 - ts[0] / 2.0, lc[1] / 100.0 - ts[1] / 2.0))
    print()

    arm = Arm(p)
    env = sim_env.TissueSceneEnv(config_path=p, dataset=SUCCESS, render=False,
                                output_dir=OUT)
    m1, m2 = arm.model, env.model
    for label, m in (("IK model ", m1), ("env model", m2)):
        b = m.body("base_link") if any(m.body(i).name == "base_link"
                                       for i in range(m.nbody)) else None
        print("%s: base_link pos=%s  nq=%d" % (label, None if b is None else b.pos, m.nq))
    # sample the same joint values in both and compare the fingertip midpoint
    data, order = dataset_io.load(SUCCESS, fields=("observation.state", "action"))
    st, ac = data[0]["observation.state"], data[0]["action"]
    cf = np.where((np.abs(ac[:, PITCH]) > 0.35).all(axis=1))[0]
    q16 = np.asarray(st[int(cf[0])], dtype=float)
    adr1 = np.array(layout.joint_ids(m1, mujoco))
    adr2 = np.array(layout.joint_ids(m2, mujoco))
    d1, d2 = mujoco.MjData(m1), mujoco.MjData(m2)
    d1.qpos[adr1] = q16
    d2.qpos[adr2] = q16
    mujoco.mj_kinematics(m1, d1)
    mujoco.mj_kinematics(m2, d2)
    names = ["hand_L_thumb_tip", "hand_L_index_tip", "hand_L_middle_tip",
             "hand_L_ring_tip", "hand_L_pinky_tip"]

    def tip(m, d):
        tb = [m.body(x).id for x in names]
        return 0.5 * (d.xpos[tb[0]] + np.mean([d.xpos[i] for i in tb[1:]], axis=0))

    t1, t2 = tip(m1, d1), tip(m2, d2)
    print()
    print("same 16-dim state, fingertip midpoint:")
    print("   IK model : (%.4f, %.4f, %.4f)" % tuple(t1))
    print("   env model: (%.4f, %.4f, %.4f)" % tuple(t2))
    print("   DIFFERENCE: %.4f m  (%.2f cm)" % (np.linalg.norm(t1 - t2),
                                                np.linalg.norm(t1 - t2) * 100))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
