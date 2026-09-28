#!/usr/bin/env python3
"""Search the base using a continuous criterion: how close the carried bag gets.

"Bag ends in the tray" is 0/1 and noisy.  Once the bag is grasped its path is fixed
by the arm trajectory, so the meaningful, continuous quantity is the closest the
carried bag comes to the tray centre.  This averages that over several real
episodes, which is much better conditioned than counting successes.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import align_with_dataset as align
import dataset_io
import sim_env

DATASET = Path("/workspace/shared/new_program_qiuzhi/without_tactile/"
               "pi05_normal_recovery_merged_214eps")
OUT = ROOT / "outputs/base_release_search"
LIFT_MIN = 0.05
CARRY_MIN = 0.02


def run(x_cm, y_cm, yaw, episodes, frame_cache):
    import mujoco
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    config["measured_layout"]["arm_mount_xy_cm_from_left_bottom"] = [x_cm, y_cm]
    config["arm"]["euler"][2] = yaw
    tag = "x%05.1f_y%05.1f_a%06.4f" % (x_cm, y_cm, yaw)
    path = OUT / tag / "scene.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    env = sim_env.TissueSceneEnv(config_path=path, dataset=DATASET, render=False,
                                 output_dir=path.parent)
    model, data = env.model, env.data
    surface = config["table"]["surface_z"]
    table_cm = [config["table"]["size"][0] * 100.0, config["table"]["size"][1] * 100.0]
    yaw_t = float(config["tray"].get("yaw", 0.0))
    cos_t, sin_t = np.cos(-yaw_t), np.sin(-yaw_t)
    rows = []
    for episode in episodes:
        action = episodes[episode]["action"]
        state = episodes[episode]["observation.state"]
        env.reset(options={"state": state[0], "randomize_objects": False})
        for box, bag in zip(config["boxes"], frame_cache[episode]):
            bx, by = bag["box_centre_cm"]
            byaw = np.radians(bag["yaw_deg"])
            qpos_adr, dof_adr = env.box_free[box["name"]]
            data.qpos[qpos_adr + 0] = bx / 100.0 - table_cm[0] / 200.0
            data.qpos[qpos_adr + 1] = by / 100.0 - table_cm[1] / 200.0
            data.qpos[qpos_adr + 2] = surface + box["size"][2] / 2.0 + 0.002
            data.qpos[qpos_adr + 3] = np.cos(byaw / 2.0)
            data.qpos[qpos_adr + 4:qpos_adr + 7] = 0.0
            data.qvel[dof_adr:dof_adr + 6] = 0.0
        mujoco.mj_forward(model, data)
        names = list(env.box_free)
        fids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, n + "_soft")
                for n in names}
        start_z = {n: float(data.qpos[env.box_free[n][0] + 2]) for n in names}
        tray = data.xpos[env.tray_body].copy()
        peak = {n: 0.0 for n in names}
        nearest = {n: 9.9 for n in names}
        for value in action:
            env.step(value)
            for n in names:
                adr, num = int(model.flex_vertadr[fids[n]]), int(model.flex_vertnum[fids[n]])
                points = data.flexvert_xpos[adr:adr + num]
                centre = points.mean(0)
                rise = float(centre[2]) - start_z[n]
                peak[n] = max(peak[n], rise)
                if rise > CARRY_MIN:
                    nearest[n] = min(nearest[n],
                                     float(np.linalg.norm(centre[:2] - tray[:2])))
        best_name = max(names, key=lambda n: peak[n])
        rows.append({"episode": episode, "bag": best_name,
                     "peak": round(peak[best_name], 4),
                     "nearest": round(nearest[best_name], 4)
                     if nearest[best_name] < 9 else None})
    env.close()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--stage", choices=["xy", "yaw"], default="xy")
    parser.add_argument("--yaw", type=float, default=1.5708)
    parser.add_argument("--best", type=str, default="20,6")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    episodes_all, order = dataset_io.load(DATASET)
    chosen = [e for e in args.episodes if e in episodes_all]
    base_config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    frame_cache = {e: align.measure_bags(align.dataset_frame(DATASET, e, 0), base_config)
                   for e in chosen}
    print("episodes %s ; bags kept per episode: %s"
          % (chosen, {e: len(v) for e, v in frame_cache.items()}))

    if args.stage == "xy":
        candidates = [(x, y, args.yaw) for x in (18.0, 20.0, 22.0) for y in (4.0, 6.0, 8.0)]
    else:
        bx, by = (float(v) for v in args.best.split(","))
        candidates = [(bx, by, yaw) for yaw in (1.45, 1.51, 1.5708, 1.63, 1.69)]

    print()
    print("  %-22s %9s %9s %s" % ("candidate", "grasped", "mean d(tray)", "per-episode peak / nearest"))
    summary = []
    for x_cm, y_cm, yaw in candidates:
        rows = run(x_cm, y_cm, yaw, {e: episodes_all[e] for e in chosen}, frame_cache)
        grasped = [r for r in rows if r["peak"] > LIFT_MIN]
        distances = [r["nearest"] for r in grasped if r["nearest"] is not None]
        mean_d = float(np.mean(distances)) if distances else 9.9
        summary.append({"base": [x_cm, y_cm, yaw], "n_grasped": len(grasped),
                        "mean_nearest": mean_d, "rows": rows})
        print("  x=%5.1f y=%4.1f a=%.4f %9d %9.4f  %s"
              % (x_cm, y_cm, yaw, len(grasped), mean_d,
                 [(r["episode"], r["peak"], r["nearest"]) for r in rows]))
    summary.sort(key=lambda s: (-s["n_grasped"], s["mean_nearest"]))
    best = summary[0]
    print()
    print("  BEST base (%.1f, %.1f) yaw %.4f : grasped %d/%d, mean nearest %.4f m"
          % (best["base"][0], best["base"][1], best["base"][2],
             best["n_grasped"], len(chosen), best["mean_nearest"]))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
