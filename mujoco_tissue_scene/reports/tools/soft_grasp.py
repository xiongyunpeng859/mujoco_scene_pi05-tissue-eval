#!/usr/bin/env python3
"""Soft-bag grasp test with the aperture measured at the FINGERTIPS.

Two fixes over the earlier attempts:
  * the aperture comes from the distal (tip/dip) geoms only -- the previous figure
    was the minimum over all geoms, which is dominated by the finger bases;
  * the bag is placed with its 8.5 cm side along the thumb-to-finger line, because
    that is the dimension the fingers span, not the 12 cm length.
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
OUT = ROOT / "outputs/soft_grasp"
FRAMES = (90, 95, 100, 105, 110)
CARRY = 60
DISTAL = ("tip", "dip")


def geoms_of(model, token, distal_only=True):
    out = []
    for g in range(model.ngeom):
        body = model.body(int(model.geom_bodyid[g])).name or ""
        if not body.startswith("hand_") or token not in body:
            continue
        if not (int(model.geom_contype[g]) or int(model.geom_conaffinity[g])):
            continue
        if distal_only and not any(d in body for d in DISTAL):
            continue
        out.append(g)
    return out


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
    thumb = geoms_of(model, "thumb")
    fingers = {h: geoms_of(model, h) for h in ("index", "middle", "ring", "pinky")}
    qpos_adr, dof_adr = env.box_free[name]
    bag_body = model.body(name).id
    hand_bodies = {i for i in range(model.nbody)
                   if (model.body(i).name or "").startswith("hand_")}

    best = (-9.9, None, 0, 0.0)
    for frame in FRAMES:
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
        # bag's local +y (8.5 cm) along the grip axis
        yaw = float(np.arctan2(axis[1], axis[0])) - np.pi / 2.0

        data.qpos[qpos_adr + 0] = grip[0]
        data.qpos[qpos_adr + 1] = grip[1]
        data.qpos[qpos_adr + 2] = surface + size[2] / 2.0 + 0.001
        data.qpos[qpos_adr + 3] = np.cos(yaw / 2.0)
        data.qpos[qpos_adr + 4] = 0.0
        data.qpos[qpos_adr + 5] = 0.0
        data.qpos[qpos_adr + 6] = np.sin(yaw / 2.0)
        data.qvel[dof_adr:dof_adr + 6] = 0.0
        mujoco.mj_forward(model, data)
        z0 = float(data.qpos[qpos_adr + 2])

        contacts, peak = 0, 0.0
        for step in range(frame, min(frame + CARRY, len(action))):
            env.step(action[step])
            for index in range(data.ncon):
                c = data.contact[index]
                b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
                if bag_body in (b1, b2) and (b1 in hand_bodies or b2 in hand_bodies):
                    contacts += 1
                    force = np.zeros(6)
                    mujoco.mj_contactForce(model, data, index, force)
                    peak = max(peak, float(np.linalg.norm(force[:3])))
        rise = float(data.qpos[qpos_adr + 2]) - z0
        if rise > best[0]:
            best = (rise, frame, contacts, peak)
    env.close()
    return best, {h: len(v) for h, v in fingers.items()}, len(thumb)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    episodes, _ = dataset_io.load(DATASET)
    state, action = episodes[0]["observation.state"], episodes[0]["action"]

    # fingertip-only aperture
    env = sim_env.TissueSceneEnv(dataset=DATASET, render=False)
    model, data, mujoco = env.model, env.data, env.mujoco
    thumb = geoms_of(model, "thumb")
    fingers = {h: geoms_of(model, h) for h in ("index", "middle", "ring", "pinky")}
    print("distal (tip/dip) contact geoms: thumb %d, %s"
          % (len(thumb), {h: len(v) for h, v in fingers.items()}))
    poses = hand_control.poses()
    for label, vector in (("OPEN", poses["open"]), ("CLOSED", poses["closed"]),
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
        print("  %-24s fingertip gap %s cm" % (label, gaps))
    env.close()
    print()

    candidates = [
        ("rigid", {"soft_body": False}),
        ("young_3e5", {"soft_body": True, "young": 3.0e5, "flex_count": [5, 4, 3]}),
        ("young_1e5", {"soft_body": True, "young": 1.0e5, "flex_count": [5, 4, 3]}),
        ("young_3e4", {"soft_body": True, "young": 3.0e4, "flex_count": [5, 4, 3]}),
        ("young_1e4", {"soft_body": True, "young": 1.0e4, "flex_count": [5, 4, 3]}),
    ]
    print("  %-12s %10s %6s %9s %10s  %s" %
          ("candidate", "best rise", "frame", "contacts", "peak force", "held"))
    rows = []
    for tag, override in candidates:
        (rise, frame, contacts, peak), _, _ = trial(override, state, action, tag)
        rows.append((tag, rise, contacts, peak))
        print("  %-12s %10.4f %6s %9d %9.3f N  %s"
              % (tag, rise, frame, contacts, peak, "HELD" if rise > 0.02 else "no"))
    rows.sort(key=lambda r: -r[1])
    print()
    print("  best: %s rise %.4f m, %d contacts, peak %.3f N (bag weight 0.491 N)"
          % (rows[0][0], rows[0][1], rows[0][2], rows[0][3]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
