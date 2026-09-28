#!/usr/bin/env python3
"""Corrected grasp isolation: put the bag at the real grip centre, axis-aligned.

The previous version used the centroid of all five fingertips, but four fingers
sit on one side and the thumb on the other, so that centroid is not where an
object is gripped.  Here the bag goes at the midpoint between the thumb tip and
the four-finger centroid, with its long axis along that line.
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
THUMB = "hand_L_thumb_tip"
FINGERS = ["hand_L_index_tip", "hand_L_middle_tip", "hand_L_ring_tip", "hand_L_pinky_tip"]
FRAMES = (85, 90, 95, 100, 105, 110)
CARRY = 60


def main() -> int:
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    surface = config["table"]["surface_z"]
    size = config["boxes"][0]["size"]
    name = config["boxes"][0]["name"]
    episodes, _ = dataset_io.load(DATASET)
    state, action = episodes[0]["observation.state"], episodes[0]["action"]

    env = sim_env.TissueSceneEnv(dataset=DATASET, render=False)
    model, data, mujoco = env.model, env.data, env.mujoco
    thumb_id = model.body(THUMB).id
    finger_ids = [model.body(f).id for f in FINGERS]
    bag_body = model.body(name).id
    hand_bodies = {i for i in range(model.nbody)
                   if (model.body(i).name or "").startswith("hand_")}
    qpos_adr, dof_adr = env.box_free[name]

    print("=== corrected isolation: bag at the thumb/finger grip midpoint ===")
    print("  %-6s %-30s %8s %8s %9s %9s  %s" %
          ("frame", "grip centre (m)", "aperture", "bag z0", "bag z1", "contacts", "held"))
    best = (-9.9, None)
    for frame in FRAMES:
        env.reset(options={"state": state[frame], "randomize_objects": False})
        closed = state[min(frame + 10, len(state) - 1)].copy()
        data.qpos[env.joint_adr] = closed
        mujoco.mj_forward(model, data)
        thumb = data.xpos[thumb_id].copy()
        fingers = np.array([data.xpos[i] for i in finger_ids]).mean(0)
        grip = 0.5 * (thumb + fingers)
        axis = fingers - thumb
        aperture = float(np.linalg.norm(axis))
        axis = axis / max(aperture, 1e-9)
        yaw = float(np.arctan2(axis[1], axis[0]))

        data.qpos[qpos_adr + 0] = grip[0]
        data.qpos[qpos_adr + 1] = grip[1]
        data.qpos[qpos_adr + 2] = surface + size[2] / 2.0 + 0.0005
        data.qpos[qpos_adr + 3] = np.cos(yaw / 2.0)
        data.qpos[qpos_adr + 4] = 0.0
        data.qpos[qpos_adr + 5] = 0.0
        data.qpos[qpos_adr + 6] = np.sin(yaw / 2.0)
        data.qvel[dof_adr:dof_adr + 6] = 0.0
        mujoco.mj_forward(model, data)
        z0 = float(data.qpos[qpos_adr + 2])

        contacts = 0
        for step in range(frame, min(frame + CARRY, len(action))):
            env.step(action[step])
            for index in range(data.ncon):
                c = data.contact[index]
                bodies = {int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])}
                if bag_body in bodies and (bodies & hand_bodies):
                    contacts += 1
        z1 = float(data.qpos[qpos_adr + 2])
        held = z1 - z0 > 0.02
        if z1 - z0 > best[0]:
            best = (z1 - z0, frame)
        print("  %-6d %-30s %8.3f %8.3f %9.3f %9d  %s"
              % (frame, np.round(grip, 3).tolist(), aperture, z0, z1, contacts,
                 "YES" if held else "no"))

    print()
    print("  best rise %.4f m at frame %s" % (best[0], best[1]))
    if best[0] > 0.02:
        print("  RESULT: the hand CAN hold the bag when the bag is placed at the true")
        print("  grip centre.  The earlier failure was my placement error.")
    else:
        print("  RESULT: still not held.  The fingers close to a ~11 cm aperture, which")
        print("  cannot trap a 12x8.5x6.5 cm rigid box; the real grasp must rely on the")
        print("  tissue pack deforming, which a rigid body cannot reproduce.")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
