#!/usr/bin/env python3
"""Validate the simulation on the ALL-SUCCESS dataset, not the mixed recovery one.

The merged dataset contains normal AND recovery episodes, so a bag that does not end
in the tray there may be correct behaviour.  The 108-episode success dataset is the
right ground truth: in every one of those the real robot placed the pack in the box.
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
OUT = ROOT / "outputs/validate_success"
LIFT_MIN = 0.05


def main() -> int:
    import mujoco
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--x", type=float, default=18.0)
    parser.add_argument("--y", type=float, default=8.0)
    parser.add_argument("--yaw", type=float, default=1.5708)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    config["measured_layout"]["arm_mount_xy_cm_from_left_bottom"] = [args.x, args.y]
    config["arm"]["euler"][2] = args.yaw
    path = OUT / "scene.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))

    episodes, order = dataset_io.load(SUCCESS)
    chosen = [e for e in args.episodes if e in episodes] or order[:6]
    print("all-success dataset: %d episodes total; testing %s" % (len(order), chosen))
    print("base (%.1f, %.1f) yaw %.4f" % (args.x, args.y, args.yaw))
    print()

    env = sim_env.TissueSceneEnv(config_path=path, dataset=SUCCESS, render=False,
                                 output_dir=OUT)
    model, data = env.model, env.data
    surface = config["table"]["surface_z"]
    table_cm = [config["table"]["size"][0] * 100.0, config["table"]["size"][1] * 100.0]
    yaw_t = float(config["tray"].get("yaw", 0.0))
    cos_t, sin_t = np.cos(-yaw_t), np.sin(-yaw_t)
    half_x = config["tray"]["size"][0] / 2.0
    half_y = config["tray"]["size"][1] / 2.0

    print("  %-8s %10s %11s %9s %9s %s" %
          ("episode", "peak lift", "nearest", "grasped", "in tray", "bags kept"))
    rows = []
    for episode in chosen:
        action = episodes[episode]["action"]
        state = episodes[episode]["observation.state"]
        env.reset(options={"state": state[0], "randomize_objects": False})
        bags = align.measure_bags(align.dataset_frame(SUCCESS, episode, 0), config)
        for box, bag in zip(config["boxes"], bags):
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
        # lift and containment are INDEPENDENT measurements: a bag can be lifted
        # without ending in the box (that is exactly what we are diagnosing), so
        # never let containment mask the lift.
        best = max(names, key=lambda n: peak[n])
        best_peak = peak[best]
        inside = False
        for n in names:
            adr, num = int(model.flex_vertadr[fids[n]]), int(model.flex_vertnum[fids[n]])
            points = data.flexvert_xpos[adr:adr + num]
            ddx, ddy = points[:, 0] - tray[0], points[:, 1] - tray[1]
            lx = cos_t * ddx - sin_t * ddy
            ly = sin_t * ddx + cos_t * ddy
            frac = float((((np.abs(lx) < half_x) & (np.abs(ly) < half_y)
                           & (points[:, 2] > tray[2] + 0.005)).mean()))
            if frac > 0.3:
                inside = True
        grasped = best_peak > LIFT_MIN
        rows.append({"episode": episode, "peak": round(best_peak, 4),
                     "grasped": grasped, "in_tray": inside,
                     "nearest": round(nearest[best], 4) if best and nearest[best] < 9 else None,
                     "bags": len(bags)})
        print("  %-8d %10.4f %11s %9s %9s %d"
              % (episode, best_peak,
                 ("%.4f" % nearest[best]) if best and nearest[best] < 9 else "-",
                 grasped, inside, len(bags)))
    env.close()
    g = sum(1 for r in rows if r["grasped"])
    t = sum(1 for r in rows if r["in_tray"] and r["grasped"])
    dist = [r["nearest"] for r in rows if r["grasped"] and r["nearest"] is not None]
    print()
    print("  grasped %d/%d ; grasped AND in tray %d/%d ; mean nearest %.4f m"
          % (g, len(rows), t, len(rows), float(np.mean(dist)) if dist else -1))
    (OUT / "result.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
