#!/usr/bin/env python3
"""Sweep bag collision extent and contact softness: can the hand hold it now?

The fingers only close to ~35 deg, so a full-size rigid box cannot be enclosed.
This shrinks the collision extent and softens the contact, then re-runs the
isolation test (bag placed exactly inside the closed fingers, then the real lift
segment is driven) and reports how far the bag rises.
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
OUT = ROOT / "outputs/bag_collision_sweep"
TIPS = ["hand_L_thumb_tip", "hand_L_index_tip", "hand_L_middle_tip",
        "hand_L_ring_tip", "hand_L_pinky_tip"]
FRAMES = (90, 95, 100, 105)
CARRY = 55


def trial(override, episodes, state, action, tag):
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    for box in config["boxes"]:
        box.update(override)
    path = OUT / tag / "scene.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))

    surface = config["table"]["surface_z"]
    height = config["boxes"][0]["size"][2]
    name = config["boxes"][0]["name"]
    env = sim_env.TissueSceneEnv(config_path=path, dataset=DATASET, render=False,
                                 output_dir=path.parent)
    model, data, mujoco = env.model, env.data, env.mujoco
    tip_ids = [model.body(t).id for t in TIPS]
    bag_body = model.body(name).id
    hand_bodies = {i for i in range(model.nbody)
                   if (model.body(i).name or "").startswith("hand_")}
    qpos_adr, dof_adr = env.box_free[name]

    best_rise, best_contacts, best_frame = -9.9, 0, -1
    for frame in FRAMES:
        env.reset(options={"state": state[frame], "randomize_objects": False})
        closed = state[min(frame + 10, len(state) - 1)].copy()
        data.qpos[env.joint_adr] = closed
        mujoco.mj_forward(model, data)
        centroid = np.array([data.xpos[i] for i in tip_ids]).mean(0)
        data.qpos[qpos_adr + 0] = centroid[0]
        data.qpos[qpos_adr + 1] = centroid[1]
        data.qpos[qpos_adr + 2] = surface + height / 2.0 + 0.0005
        data.qpos[qpos_adr + 3] = 1.0
        data.qpos[qpos_adr + 4:qpos_adr + 7] = 0.0
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
        rise = float(data.qpos[qpos_adr + 2]) - z0
        if rise > best_rise:
            best_rise, best_contacts, best_frame = rise, contacts, frame
    env.close()
    return best_rise, best_contacts, best_frame


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    episodes, _ = dataset_io.load(DATASET)
    state = episodes[0]["observation.state"]
    action = episodes[0]["action"]

    candidates = [
        ("baseline", {}),
        ("scale80", {"collision_scale": [0.80, 0.70, 0.75]}),
        ("scale70", {"collision_scale": [0.70, 0.60, 0.60]}),
        ("scale60", {"collision_scale": [0.60, 0.50, 0.45]}),
        ("soft", {"collision_scale": [0.80, 0.70, 0.75],
                  "solref": [0.04, 1.0], "solimp": [0.85, 0.95, 0.005]}),
        ("soft_grippy", {"collision_scale": [0.80, 0.70, 0.75],
                         "solref": [0.04, 1.0], "solimp": [0.85, 0.95, 0.005],
                         "friction": [2.0, 0.05, 0.001]}),
        ("much_softer", {"collision_scale": [0.70, 0.60, 0.55],
                         "solref": [0.10, 1.0], "solimp": [0.70, 0.85, 0.02],
                         "friction": [2.0, 0.05, 0.001]}),
    ]
    print("  %-14s %10s %10s %8s   %s" % ("candidate", "best rise", "contacts", "frame",
                                          "collision half-extent (cm)"))
    rows = []
    for tag, override in candidates:
        rise, contacts, frame = trial(override, episodes, state, action, tag)
        scale = override.get("collision_scale", [1.0, 1.0, 1.0])
        size = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())["boxes"][0]["size"]
        half = [round(size[i] * 100 * scale[i] / 2, 2) for i in range(3)]
        rows.append((tag, rise, contacts))
        print("  %-14s %10.4f %10d %8d   %s" % (tag, rise, contacts, frame, half))
    rows.sort(key=lambda r: -r[1])
    print()
    print("  best: %s with rise %.4f m, %d contacts" % (rows[0][0], rows[0][1], rows[0][2]))
    print("  (a rise above ~0.02 m means the bag left the table with the hand)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
