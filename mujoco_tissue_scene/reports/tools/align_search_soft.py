#!/usr/bin/env python3
"""Re-search the alignment now that grasping physically works.

The isolation test proved the soft bag can be held and carried, so a replay that
fails is an alignment failure.  Objective per candidate: how far a bag is lifted
during the real episode, using flex-aware contact counting.

    python align_search_soft.py --stage yaw
    python align_search_soft.py --stage base --yaw A
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import align_with_dataset as align      # noqa: E402
import dataset_io                       # noqa: E402
import sim_env                          # noqa: E402

DATASET = Path("/workspace/shared/new_program_qiuzhi/without_tactile/"
               "pi05_normal_recovery_merged_214eps")
OUT = ROOT / "outputs/align_search_soft"
EPISODE = 0


def replay_candidate(x_cm, y_cm, yaw):
    import mujoco
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    config["measured_layout"]["arm_mount_xy_cm_from_left_bottom"] = [x_cm, y_cm]
    config["arm"]["euler"][2] = yaw
    for box in config["boxes"]:
        box.update({"soft_body": True, "young": 100000.0, "flex_count": [5, 4, 3]})
    tag = "x%05.1f_y%05.1f_a%06.4f" % (x_cm, y_cm, yaw)
    path = OUT / tag / "scene.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))

    episodes, _ = dataset_io.load(DATASET)
    action = episodes[EPISODE]["action"]
    state = episodes[EPISODE]["observation.state"]
    env = sim_env.TissueSceneEnv(config_path=path, dataset=DATASET, render=False,
                                 output_dir=path.parent)
    env.reset(options={"state": state[0], "randomize_objects": False})

    # put the bags where the real first frame shows them
    real_first = align.dataset_frame(DATASET, EPISODE, 0)
    found = align.measure_bags(real_first, config)
    table_cm = [config["table"]["size"][0] * 100.0, config["table"]["size"][1] * 100.0]
    surface = config["table"]["surface_z"]
    for box, bag in zip(config["boxes"], found):
        bx, by = bag["box_centre_cm"]
        byaw = np.radians(bag["yaw_deg"])
        qpos_adr, dof_adr = env.box_free[box["name"]]
        env.data.qpos[qpos_adr + 0] = bx / 100.0 - table_cm[0] / 200.0
        env.data.qpos[qpos_adr + 1] = by / 100.0 - table_cm[1] / 200.0
        env.data.qpos[qpos_adr + 2] = surface + box["size"][2] / 2.0 + 0.002
        env.data.qpos[qpos_adr + 3] = np.cos(byaw / 2.0)
        env.data.qpos[qpos_adr + 4] = 0.0
        env.data.qpos[qpos_adr + 5] = 0.0
        env.data.qpos[qpos_adr + 6] = np.sin(byaw / 2.0)
        env.data.qvel[dof_adr:dof_adr + 6] = 0.0
    env.mujoco.mj_forward(env.model, env.data)

    model, data = env.model, env.data
    hand_bodies = {i for i in range(model.nbody)
                   if (model.body(i).name or "").startswith("hand_")}
    names = list(env.box_free)
    starts = {n: float(data.qpos[env.box_free[n][0] + 2]) for n in names}
    peak = {n: 0.0 for n in names}
    hand_contacts = 0

    for value in action:
        env.step(value)
        for n in names:
            peak[n] = max(peak[n], float(data.qpos[env.box_free[n][0] + 2]) - starts[n])
        for index in range(data.ncon):
            c = data.contact[index]
            # flex contacts carry the flex id in contact.flex, not in geom1/geom2
            if int(c.flex[0]) >= 0 or int(c.flex[1]) >= 0:
                other = int(c.geom1) if int(c.geom2) < 0 else int(c.geom2)
                if other >= 0 and int(model.geom_bodyid[other]) in hand_bodies:
                    hand_contacts += 1
            else:
                b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
                if (b1 in hand_bodies) != (b2 in hand_bodies):
                    hand_contacts += 1

    # containment, flex-aware: sample the flex vertices of each bag
    contained = {}
    for name in names:
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, name + "_soft")
        if index < 0:
            continue
        adr, num = int(model.flex_vertadr[index]), int(model.flex_vertnum[index])
        points = data.flexvert_xpos[adr:adr + num]
        tray = data.xpos[env.tray_body]
        yaw_t = float(config["tray"].get("yaw", 0.0))
        cos_t, sin_t = np.cos(-yaw_t), np.sin(-yaw_t)
        dx, dy = points[:, 0] - tray[0], points[:, 1] - tray[1]
        lx = cos_t * dx - sin_t * dy
        ly = sin_t * dx + cos_t * dy
        inside = ((np.abs(lx) < config["tray"]["size"][0] / 2)
                  & (np.abs(ly) < config["tray"]["size"][1] / 2)
                  & (points[:, 2] > tray[2] + 0.005))
        contained[name] = bool(inside.mean() > 0.3)
    env.close()
    return {"peak_rise": peak, "hand_contacts": hand_contacts, "contained": contained,
            "best_rise": max(peak.values())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["yaw", "base"], default="yaw")
    parser.add_argument("--x", type=float, default=25.0)
    parser.add_argument("--y", type=float, default=10.0)
    parser.add_argument("--yaw", type=float, default=1.90)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.stage == "yaw":
        candidates = [(25.0, 10.0, yaw) for yaw in
                      (1.5708, 1.70, 1.80, 1.90, 2.00, 2.10, 2.25, 2.40)]
    else:
        candidates = [(x, y, args.yaw) for x in (12.0, 20.0, 28.0, 36.0)
                      for y in (0.0, 10.0, 20.0)]

    print("  %-22s %10s %9s %9s  %s" %
          ("candidate", "best rise", "hand con", "in tray", "per-bag rise"))
    rows = []
    for x_cm, y_cm, yaw in candidates:
        result = replay_candidate(x_cm, y_cm, yaw)
        rows.append(((x_cm, y_cm, yaw), result))
        print("  x=%5.1f y=%5.1f a=%.4f %10.4f %9d %9s  %s"
              % (x_cm, y_cm, yaw, result["best_rise"], result["hand_contacts"],
                 any(result["contained"].values()),
                 {k[:9]: round(v, 3) for k, v in result["peak_rise"].items()}))
    def score(item):
        result = item[1]
        return (any(result["contained"].values()), result["best_rise"],
                result["hand_contacts"])
    rows.sort(key=score, reverse=True)
    best = rows[0]
    print()
    print("  best: x=%.1f y=%.1f yaw=%.4f -> rise %.4f m, %d hand contacts, in tray %s"
          % (*best[0], best[1]["best_rise"], best[1]["hand_contacts"],
             any(best[1]["contained"].values())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
