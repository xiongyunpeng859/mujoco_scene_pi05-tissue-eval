#!/usr/bin/env python3
"""Measure the TRUE grasp aperture and the grip force the hand actually applies.

Body origins are joint centres, so an earlier "thumb-to-index 11.4 cm" figure may
have been a measurement artefact.  `mj_geomDistance` gives the real surface gap.
Then the contact forces during a real lift segment are compared with the bag's
weight, which is what decides whether a nearly-rigid pack can be carried.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import dataset_io                       # noqa: E402
import hand_control                     # noqa: E402
import sim_env                          # noqa: E402

DATASET = Path("/workspace/shared/new_program_qiuzhi/without_tactile/"
               "pi05_normal_recovery_merged_214eps")


def main() -> int:
    import mujoco
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    size = config["boxes"][0]["size"]
    mass = config["boxes"][0]["mass"]
    weight = mass * 9.81
    episodes, _ = dataset_io.load(DATASET)
    state, action = episodes[0]["observation.state"], episodes[0]["action"]

    env = sim_env.TissueSceneEnv(dataset=DATASET, render=False)
    model, data = env.model, env.data

    def owner(gid):
        return model.body(int(model.geom_bodyid[gid])).name or ""

    contact_geoms = [g for g in range(model.ngeom)
                     if owner(g).startswith("hand_")
                     and (int(model.geom_contype[g]) or int(model.geom_conaffinity[g]))]
    thumb = [g for g in contact_geoms if "thumb" in owner(g)]
    fingers = {h: [g for g in contact_geoms if h in owner(g)]
               for h in ("index", "middle", "ring", "pinky")}
    print("hand contact geoms: thumb %d, fingers %s"
          % (len(thumb), {k: len(v) for k, v in fingers.items()}))
    print("bag %.1f x %.1f x %.1f cm, mass %.0f g -> weight %.3f N"
          % (size[0] * 100, size[1] * 100, size[2] * 100, mass * 1000, weight))
    print()

    print("=== true surface gap between the thumb and each finger ===")
    poses = hand_control.poses()
    for label, vector in (("OPEN", poses["open"]), ("CLOSED", poses["closed"]),
                          ("frame 100 (approaching)", state[100][6:].tolist()),
                          ("frame 119 (real grasp)", state[119][6:].tolist())):
        data.qpos[env.joint_adr] = np.concatenate([state[119][:6], np.asarray(vector)])
        mujoco.mj_forward(model, data)
        gaps = {}
        for hint, geoms in fingers.items():
            best = 1e9
            for a in thumb:
                for b in geoms:
                    best = min(best, mujoco.mj_geomDistance(model, data, a, b, 0.5, None))
            gaps[hint] = round(best * 100, 2)
        print("  %-24s %s cm" % (label, gaps))
    print()

    # --- grip force during the real lift segment, bag at the grip midpoint -----
    data.qpos[env.joint_adr] = state[95]
    mujoco.mj_forward(model, data)
    thumb_pos = data.xpos[model.body("hand_L_thumb_tip").id]
    finger_pos = np.array([data.xpos[model.body(n).id] for n in
                           ("hand_L_index_tip", "hand_L_middle_tip",
                            "hand_L_ring_tip", "hand_L_pinky_tip")]).mean(0)
    grip = 0.5 * (thumb_pos + finger_pos)
    axis = finger_pos - thumb_pos
    axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    yaw = float(np.arctan2(axis[1], axis[0]))
    name = config["boxes"][0]["name"]
    qpos_adr, dof_adr = env.box_free[name]
    surface = config["table"]["surface_z"]
    data.qpos[qpos_adr + 0] = grip[0]
    data.qpos[qpos_adr + 1] = grip[1]
    data.qpos[qpos_adr + 2] = surface + size[2] / 2.0 + 0.001
    data.qpos[qpos_adr + 3] = np.cos(yaw / 2.0)
    data.qpos[qpos_adr + 4:qpos_adr + 7] = 0.0
    data.qvel[dof_adr:dof_adr + 6] = 0.0
    mujoco.mj_forward(model, data)
    z0 = float(data.qpos[qpos_adr + 2])
    bag_body = model.body(name).id
    hand_bodies = {i for i in range(model.nbody)
                   if (model.body(i).name or "").startswith("hand_")}

    print("=== grip force while the real lift segment is driven ===")
    print("  %-6s %8s %9s %13s %13s %9s" %
          ("step", "contacts", "hb-con", "sum |F| hand-bag", "bag weight", "bag z"))
    peak = 0.0
    for step in range(95, 150):
        env.step(action[step])
        total, hb = 0.0, 0
        for index in range(data.ncon):
            c = data.contact[index]
            b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            if bag_body not in (b1, b2):
                continue
            other = b2 if b1 == bag_body else b1
            if other not in hand_bodies:
                continue
            hb += 1
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, force)
            total += float(np.linalg.norm(force[:3]))
        peak = max(peak, total)
        if step % 6 == 0 or step == 149:
            print("  %-6d %8d %9d %12.4f N %12.3f N %9.4f"
                  % (step, data.ncon, hb, total, weight, float(data.qpos[qpos_adr + 2])))

    hand_force = np.abs(data.actuator_force[env.ctrl_ids[6:]]).max()
    limit = model.actuator_forcerange[env.ctrl_ids[6]][1]
    print()
    print("  peak hand-bag contact force : %.4f N   (bag weight %.3f N)" % (peak, weight))
    print("  max hand actuator force     : %.4f N.m (limit %.2f)" % (hand_force, limit))
    print("  bag rise over the segment   : %.4f m" % (float(data.qpos[qpos_adr + 2]) - z0))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
