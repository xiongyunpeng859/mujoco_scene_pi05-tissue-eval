#!/usr/bin/env python3
"""Step 1: choose the base yaw by whether the task actually works, not by edges.

The mounting constraint says the bolt pattern is perpendicular to the table edge,
so the yaw should be near pi/2 rather than fitted freely.  This replays a real
successful episode for each candidate and reports the numbers that decide it:

  * closest approach between the hand and each bag,
  * whether the hand and a bag ever touch,
  * whether a bag finishes inside the green box.

    python arm_yaw_task_search.py --stage yaw
    python arm_yaw_task_search.py --stage base --yaw 1.5708
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
OUT = ROOT / "outputs/yaw_task_search"


def run_candidate(x_cm, y_cm, yaw, episode=0):
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    config["measured_layout"]["arm_mount_xy_cm_from_left_bottom"] = [x_cm, y_cm]
    config["arm"]["euler"][2] = yaw
    tag = "x%05.1f_y%05.1f_a%06.4f" % (x_cm, y_cm, yaw)
    path = OUT / tag / "scene.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))

    episodes, _ = dataset_io.load(DATASET)
    action = episodes[episode]["action"]
    state = episodes[episode]["observation.state"]

    env = sim_env.TissueSceneEnv(config_path=path, dataset=DATASET, render=False,
                                 output_dir=path.parent)
    env.reset(options={"state": state[0]})

    # Put the bags where the real first frame shows them.
    real_first = align.dataset_frame(DATASET, episode, 0)
    found = align.measure_bags(real_first, config)
    table_size_cm = [config["table"]["size"][0] * 100.0, config["table"]["size"][1] * 100.0]
    surface = config["table"]["surface_z"]
    for box, bag in zip(config["boxes"], found):
        x_b, y_b = bag["box_centre_cm"]
        yaw_b = np.radians(bag["yaw_deg"])
        qpos_adr, dof_adr = env.box_free[box["name"]]
        env.data.qpos[qpos_adr + 0] = x_b / 100.0 - table_size_cm[0] / 200.0
        env.data.qpos[qpos_adr + 1] = y_b / 100.0 - table_size_cm[1] / 200.0
        env.data.qpos[qpos_adr + 2] = surface + box["size"][2] / 2.0 + 0.001
        env.data.qpos[qpos_adr + 3] = np.cos(yaw_b / 2.0)
        env.data.qpos[qpos_adr + 6] = np.sin(yaw_b / 2.0)
        env.data.qvel[dof_adr:dof_adr + 6] = 0.0
    env.mujoco.mj_forward(env.model, env.data)

    model = env.model
    bag_bodies = {env.model.body(name).id: name for name in env.box_free}
    hand_bodies = [i for i in range(model.nbody)
                   if (model.body(i).name or "").startswith("hand_")]
    closest = {name: 1e9 for name in bag_bodies.values()}
    contacts = 0
    for value in action:
        env.step(value)
        hand_positions = [env.data.xpos[i] for i in hand_bodies]
        for body_id, name in bag_bodies.items():
            centre = env.data.xpos[body_id]
            distance = min(np.linalg.norm(p - centre) for p in hand_positions)
            closest[name] = min(closest[name], distance)
        for index in range(env.data.ncon):
            contact = env.data.contact[index]
            bodies = {int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])}
            if bodies & set(bag_bodies) and bodies & set(hand_bodies):
                contacts += 1
    in_tray = {name: env.success() for name in bag_bodies}
    # per-bag containment check
    tray = env.data.xpos[env.tray_body]
    yaw_t = float(config["tray"].get("yaw", 0.0))
    cos_t, sin_t = np.cos(-yaw_t), np.sin(-yaw_t)
    contained = {}
    for name, (qpos_adr, _) in env.box_free.items():
        position = env.data.qpos[qpos_adr:qpos_adr + 3]
        dx, dy = position[0] - tray[0], position[1] - tray[1]
        lx = cos_t * dx - sin_t * dy
        ly = sin_t * dx + cos_t * dy
        contained[name] = bool(abs(lx) < config["tray"]["size"][0] / 2
                               and abs(ly) < config["tray"]["size"][1] / 2
                               and position[2] > tray[2] + 0.005)
    env.close()
    return {"closest": closest, "contacts": contacts, "contained": contained,
            "min_closest": min(closest.values())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["yaw", "base"], default="yaw")
    parser.add_argument("--x", type=float, default=25.0)
    parser.add_argument("--y", type=float, default=10.0)
    parser.add_argument("--yaw", type=float, default=1.5708)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "yaw":
        candidates = [(args.x, args.y, yaw) for yaw in
                      (1.3678, 1.45, 1.5708, 1.65, 1.75, 1.90)]
    else:
        candidates = [(x, y, args.yaw) for x in (15.0, 25.0, 35.0)
                      for y in (0.0, 10.0, 20.0)]

    print("  %-24s %10s %10s %10s   %s" %
          ("candidate", "closest(m)", "contacts", "in tray", "per-bag closest"))
    rows = []
    for x_cm, y_cm, yaw in candidates:
        result = run_candidate(x_cm, y_cm, yaw)
        rows.append((x_cm, y_cm, yaw, result))
        print("  x=%5.1f y=%5.1f a=%.4f %10.4f %10d %10s   %s"
              % (x_cm, y_cm, yaw, result["min_closest"], result["contacts"],
                 any(result["contained"].values()),
                 {k: round(v, 3) for k, v in result["closest"].items()}))
    rows.sort(key=lambda r: r[3]["min_closest"])
    best = rows[0]
    print()
    print("best by closest approach: x=%.1f y=%.1f yaw=%.4f -> %.4f m, %d contacts, in tray %s"
          % (best[0], best[1], best[2], best[3]["min_closest"], best[3]["contacts"],
             any(best[3]["contained"].values())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
