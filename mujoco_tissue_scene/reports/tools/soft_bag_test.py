#!/usr/bin/env python3
"""Does a deformable bag get held where a rigid one did not?

Same isolation test as grasp_isolate2 (bag placed at the true grip midpoint, then
the real lift segment is driven), but the bag is a volumetric soft grid.  Sweeps a
couple of stiffnesses so the result is not a single lucky setting.
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
OUT = ROOT / "outputs/soft_bag_test"
THUMB = "hand_L_thumb_tip"
FINGERS = ["hand_L_index_tip", "hand_L_middle_tip", "hand_L_ring_tip", "hand_L_pinky_tip"]
FRAMES = (90, 95, 100, 105, 110)
CARRY = 60


def trial(override, state, action, tag):
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    for box in config["boxes"]:
        box.update(override)
    path = OUT / tag / "scene.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    surface = config["table"]["surface_z"]
    size = config["boxes"][0]["size"]
    name = config["boxes"][0]["name"]

    env = sim_env.TissueSceneEnv(config_path=path, dataset=DATASET, render=False,
                                 output_dir=path.parent)
    model, data, mujoco = env.model, env.data, env.mujoco
    nflex = model.nflex
    flex_vertices = int(model.flex_vertadr[0 + 0]) if nflex else 0
    if nflex:
        flex_vertices = int(model.flex_nvert[0])
    thumb_id = model.body(THUMB).id
    finger_ids = [model.body(f).id for f in FINGERS]
    qpos_adr, dof_adr = env.box_free[name]
    hand_bodies = {i for i in range(model.nbody)
                   if (model.body(i).name or "").startswith("hand_")}
    geom_bag = {g for g in range(model.ngeom) if model.geom_bodyid[g] == model.body(name).id}

    best = (-9.9, None, 0)
    for frame in FRAMES:
        env.reset(options={"state": state[frame], "randomize_objects": False})
        closed = state[min(frame + 10, len(state) - 1)].copy()
        data.qpos[env.joint_adr] = closed
        mujoco.mj_forward(model, data)
        thumb = data.xpos[thumb_id].copy()
        fingers = np.array([data.xpos[i] for i in finger_ids]).mean(0)
        grip = 0.5 * (thumb + fingers)
        axis = fingers - thumb
        axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
        yaw = float(np.arctan2(axis[1], axis[0]))

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
        contacts = 0
        for step in range(frame, min(frame + CARRY, len(action))):
            env.step(action[step])
            for index in range(data.ncon):
                c = data.contact[index]
                b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
                if (b1 == model.body(name).id and b2 in hand_bodies) or \
                   (b2 == model.body(name).id and b1 in hand_bodies):
                    contacts += 1
        rise = float(data.qpos[qpos_adr + 2]) - z0
        if rise > best[0]:
            best = (rise, frame, contacts)
    env.close()
    return best, nflex, flex_vertices


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    episodes, _ = dataset_io.load(DATASET)
    state, action = episodes[0]["observation.state"], episodes[0]["action"]

    candidates = [
        ("rigid_reference", {"soft_body": False}),
        ("soft_200k", {"soft_body": True, "young": 200000.0, "flex_count": [6, 4, 3]}),
        ("soft_50k", {"soft_body": True, "young": 50000.0, "flex_count": [6, 4, 3]}),
        ("soft_10k", {"soft_body": True, "young": 10000.0, "flex_count": [6, 4, 3]}),
        ("soft_dense", {"soft_body": True, "young": 50000.0, "flex_count": [8, 5, 4],
                        "flex_radius": 0.003}),
    ]
    print("  %-16s %10s %8s %9s %7s %7s  %s" %
          ("candidate", "best rise", "frame", "contacts", "nflex", "nvert", "result"))
    rows = []
    for tag, override in candidates:
        (rise, frame, contacts), nflex, nvert = trial(override, state, action, tag)
        rows.append((tag, rise, contacts))
        print("  %-16s %10.4f %8s %9d %7d %7d  %s"
              % (tag, rise, frame, contacts, nflex, nvert,
                 "HELD" if rise > 0.02 else "not held"))
    rows.sort(key=lambda r: -r[1])
    print()
    print("  best: %s rise %.4f m (%d contacts)" % (rows[0][0], rows[0][1], rows[0][2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
