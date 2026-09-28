#!/usr/bin/env python3
"""Verify the soft-bag result honestly: flex contacts are not geom contacts.

MuJoCo records a flex collision with the flex index in `contact.flex`, not through
geom1/geom2, so the previous "0 contacts" was a counting bug.  This checks, for one
soft setting:
  * whether the bag rests on the table when nothing pushes it (flex-table contact),
  * the real number of hand-bag contacts including flex contacts,
  * the z trace during the carry segment.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import dataset_io                       # noqa: E402
import sim_env                          # noqa: E402

DATASET = Path("/workspace/shared/new_program_qiuzhi/without_tactile/"
               "pi05_normal_recovery_merged_214eps")
OUT = ROOT / "outputs/soft_verify"


def main() -> int:
    import mujoco
    OUT.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    for box in config["boxes"]:
        box.update({"soft_body": True, "young": 1.0e5, "flex_count": [5, 4, 3]})
    path = OUT / "scene.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    surface = config["table"]["surface_z"]
    size = config["boxes"][0]["size"]
    name = config["boxes"][0]["name"]

    episodes, _ = dataset_io.load(DATASET)
    state, action = episodes[0]["observation.state"], episodes[0]["action"]
    env = sim_env.TissueSceneEnv(config_path=path, dataset=DATASET, render=False,
                                 output_dir=OUT)
    model, data, mujoco = env.model, env.data, env.mujoco
    qpos_adr, dof_adr = env.box_free[name]
    bag_body = model.body(name).id
    hand_bodies = {i for i in range(model.nbody)
                   if (model.body(i).name or "").startswith("hand_")}

    # how is a flex represented?
    print("nflex=%d  flex name = %s" %
          (model.nflex, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_FLEX, 0)))

    def flex_contacts():
        """(total contacts, those involving the bag flex, those also with the hand)"""
        total = bag = hand = 0
        for index in range(data.ncon):
            c = data.contact[index]
            total += 1
            flex_ids = {int(c.flex[0]), int(c.flex[1])}
            if 0 in flex_ids:
                bag += 1
                other = int(c.geom1) if int(c.geom2) < 0 else int(c.geom2)
                if other >= 0:
                    body = int(model.geom_bodyid[other])
                    if body in hand_bodies:
                        hand += 1
        return total, bag, hand

    # 1) rest test: put the bag on the table and do nothing
    env.reset(options={"state": state[0], "randomize_objects": False})
    data.qpos[qpos_adr + 0] = -0.35
    data.qpos[qpos_adr + 1] = 0.20
    data.qpos[qpos_adr + 2] = surface + size[2] / 2.0 + 0.002
    data.qpos[qpos_adr + 3:qpos_adr + 7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[dof_adr:dof_adr + 6] = 0.0
    mujoco.mj_forward(model, data)
    z_start = float(data.qpos[qpos_adr + 2])
    data.ctrl[:] = 0.0
    for _ in range(500):                 # 1 s of free settling
        mujoco.mj_step(model, data)
    z_rest = float(data.qpos[qpos_adr + 2])
    total, bag_c, hand_c = flex_contacts()
    print()
    print("rest test: z %.4f -> %.4f  (dropped %.4f m); contacts total=%d flex=%d hand-flex=%d"
          % (z_start, z_rest, z_start - z_rest, total, bag_c, hand_c))
    print("           expected rest z = %.4f (= table + half height)"
          % (surface + size[2] / 2.0))
    print("           -> flex-table contact %s" %
          ("WORKS" if abs(z_rest - (surface + size[2] / 2.0)) < 0.01 else "BROKEN"))

    # 2) carry test with flex-aware counting
    frame = 95
    env.reset(options={"state": state[frame], "randomize_objects": False})
    closed = state[min(frame + 10, len(state) - 1)].copy()
    data.qpos[env.joint_adr] = closed
    mujoco.mj_forward(model, data)
    thumb_pos = data.xpos[model.body("hand_L_thumb_tip").id].copy()
    finger_pos = np.array([data.xpos[model.body("hand_L_%s_tip" % h).id]
                           for h in ("index", "middle", "ring", "pinky")]).mean(0)
    grip = 0.5 * (thumb_pos + finger_pos)
    axis = finger_pos - thumb_pos
    axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    yaw = float(np.arctan2(axis[1], axis[0])) - np.pi / 2.0
    data.qpos[qpos_adr + 0] = grip[0]
    data.qpos[qpos_adr + 1] = grip[1]
    data.qpos[qpos_adr + 2] = surface + size[2] / 2.0 + 0.002
    data.qpos[qpos_adr + 3] = np.cos(yaw / 2.0)
    data.qpos[qpos_adr + 4] = 0.0
    data.qpos[qpos_adr + 5] = 0.0
    data.qpos[qpos_adr + 6] = np.sin(yaw / 2.0)
    data.qvel[dof_adr:dof_adr + 6] = 0.0
    mujoco.mj_forward(model, data)
    z0 = float(data.qpos[qpos_adr + 2])
    print()
    print("carry test from frame %d, bag placed at grip midpoint %s"
          % (frame, np.round(grip, 3).tolist()))
    print("  %-5s %8s %7s %7s %9s %9s" %
          ("step", "bag z", "total", "flex", "hand-flex", "peak |F|"))
    for step in range(frame, frame + 60):
        env.step(action[step])
        total, bag_c, hand_c = flex_contacts()
        peak = 0.0
        for index in range(data.ncon):
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, force)
            peak = max(peak, float(np.linalg.norm(force[:3])))
        if (step - frame) % 6 == 0 or step == frame + 59:
            print("  %-5d %8.4f %7d %7d %9d %8.3f N"
                  % (step, float(data.qpos[qpos_adr + 2]), total, bag_c, hand_c, peak))
    print()
    print("  bag rise = %.4f m" % (float(data.qpos[qpos_adr + 2]) - z0))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
