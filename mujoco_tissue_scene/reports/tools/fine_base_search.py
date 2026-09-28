#!/usr/bin/env python3
"""Fine x/y search on the all-success dataset with the corrected criterion.

Lift and containment are measured separately, so a base is scored by
(episodes grasped, episodes grasped AND placed in the box).
"""
from __future__ import annotations

import argparse
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

SUCCESS = Path("/workspace/shared/new_program_qiuzhi/without_tactile/success_episode/"
               "pick_up_the_tissue_pack_and_place_it_on_the_right_side_"
               "20260908_merged_all_108eps")
OUT = ROOT / "outputs/fine_base_search"
LIFT_MIN = 0.05


def evaluate(x_cm, y_cm, yaw, episodes, frame_cache):
    import mujoco
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    config["measured_layout"]["arm_mount_xy_cm_from_left_bottom"] = [x_cm, y_cm]
    config["arm"]["euler"][2] = yaw
    tag = "x%05.1f_y%05.1f" % (x_cm, y_cm)
    path = OUT / tag / "scene.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    env = sim_env.TissueSceneEnv(config_path=path, dataset=SUCCESS, render=False,
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
                adr, num = (int(model.flex_vertadr[fids[n]]),
                            int(model.flex_vertnum[fids[n]]))
                centre = data.flexvert_xpos[adr:adr + num].mean(0)
                rise = float(centre[2]) - start_z[n]
                peak[n] = max(peak[n], rise)
                if rise > 0.02:
                    nearest[n] = min(nearest[n],
                                     float(np.linalg.norm(centre[:2] - tray[:2])))
        inside = False
        for n in names:
            adr, num = int(model.flex_vertadr[fids[n]]), int(model.flex_vertnum[fids[n]])
            points = data.flexvert_xpos[adr:adr + num]
            ddx, ddy = points[:, 0] - tray[0], points[:, 1] - tray[1]
            lx = cos_t * ddx - sin_t * ddy
            ly = sin_t * ddx + cos_t * ddy
            frac = float((((np.abs(lx) < config["tray"]["size"][0] / 2)
                           & (np.abs(ly) < config["tray"]["size"][1] / 2)
                           & (points[:, 2] > tray[2] + 0.005)).mean()))
            if frac > 0.3:
                inside = True
        best = max(names, key=lambda n: peak[n])
        rows.append({"episode": episode, "peak": round(peak[best], 4),
                     "grasped": bool(peak[best] > LIFT_MIN),
                     "placed": bool(inside and peak[best] > LIFT_MIN),
                     "nearest": round(nearest[best], 4) if nearest[best] < 9 else None})
    env.close()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--bases", type=str,
                        default="16,6;16,10;18,6;18,8;18,10;20,6;20,10")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    episodes, order = dataset_io.load(SUCCESS)
    chosen = [e for e in args.episodes if e in episodes]
    base_cfg = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    cache = {e: align.measure_bags(align.dataset_frame(SUCCESS, e, 0), base_cfg)
             for e in chosen}
    print("all-success episodes %s ; bags kept %s"
          % (chosen, {e: len(v) for e, v in cache.items()}))
    print()
    print("  %-12s %9s %9s %11s  %s" %
          ("base", "grasped", "placed", "mean near", "per-episode peak"))
    summary = []
    for spec in args.bases.split(";"):
        x_cm, y_cm = (float(v) for v in spec.split(","))
        rows = evaluate(x_cm, y_cm, 1.5708, {e: episodes[e] for e in chosen}, cache)
        g = sum(1 for r in rows if r["grasped"])
        p = sum(1 for r in rows if r["placed"])
        d = [r["nearest"] for r in rows if r["grasped"] and r["nearest"] is not None]
        md = float(np.mean(d)) if d else 9.9
        summary.append({"base": [x_cm, y_cm], "grasped": g, "placed": p,
                        "mean_nearest": md, "rows": rows})
        print("  (%4.1f,%4.1f) %9d %9d %11.4f  %s"
              % (x_cm, y_cm, g, p, md, [r["peak"] for r in rows]))
    print()
    summary.sort(key=lambda s: (-s["placed"], -s["grasped"], s["mean_nearest"]))
    best = summary[0]
    print("  BEST base (%.1f, %.1f): grasped %d/%d, placed %d/%d, mean nearest %.4f m"
          % (best["base"][0], best["base"][1], best["grasped"], len(chosen),
             best["placed"], len(chosen), best["mean_nearest"]))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
