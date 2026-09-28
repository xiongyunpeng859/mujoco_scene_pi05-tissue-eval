#!/usr/bin/env python3
"""Decide the base by counting successes over many real episodes.

Image matching kept being confounded (the idle second arm, then the arm's colour),
so this uses the task itself as the criterion.  For each candidate base, replay N
real successful episodes and count how many actually grasp the bag and finish with
it inside the box.  Success is strict, so a bag merely pushed along the table does
not count.
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
import align_with_dataset as align      # noqa: E402
import dataset_io                       # noqa: E402
import sim_env                          # noqa: E402

DATASET = Path("/workspace/shared/new_program_qiuzhi/without_tactile/"
               "pi05_normal_recovery_merged_214eps")
OUT = ROOT / "outputs/base_validation"
MIN_LIFT_M = 0.05          # a real grasp, not a nudge


def run_base(x_cm, y_cm, yaw, episodes, frame_cache):
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
    names = list(env.box_free)
    fids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, n + "_soft")
            for n in names}
    yaw_t = float(config["tray"].get("yaw", 0.0))
    cos_t, sin_t = np.cos(-yaw_t), np.sin(-yaw_t)
    half_x = config["tray"]["size"][0] / 2.0
    half_y = config["tray"]["size"][1] / 2.0

    results = []
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
        start_z = {n: float(data.qpos[env.box_free[n][0] + 2]) for n in names}
        tray = data.xpos[env.tray_body].copy()
        peak = {n: 0.0 for n in names}
        for value in action:
            env.step(value)
            for n in names:
                adr, num = int(model.flex_vertadr[fids[n]]), int(model.flex_vertnum[fids[n]])
                centre = data.flexvert_xpos[adr:adr + num].mean(0)
                peak[n] = max(peak[n], float(centre[2]) - start_z[n])
        inside_any, best_name, best_peak = False, None, 0.0
        for n in names:
            adr, num = int(model.flex_vertadr[fids[n]]), int(model.flex_vertnum[fids[n]])
            points = data.flexvert_xpos[adr:adr + num]
            dx, dy = points[:, 0] - tray[0], points[:, 1] - tray[1]
            lx = cos_t * dx - sin_t * dy
            ly = sin_t * dx + cos_t * dy
            frac = float((((np.abs(lx) < half_x) & (np.abs(ly) < half_y)
                           & (points[:, 2] > tray[2] + 0.005)).mean()))
            if frac > 0.3 and peak[n] > best_peak:
                inside_any, best_name, best_peak = True, n, peak[n]
        results.append({"episode": episode, "grasped_lifted": best_peak > MIN_LIFT_M,
                        "in_tray": inside_any, "bag": best_name,
                        "peak_rise": round(best_peak, 4),
                        "success": bool(inside_any and best_peak > MIN_LIFT_M)})
    env.close()
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--bases", type=str,
                        default="18,6;20,6;22,6;20,2;20,10;25,10")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    episodes, order = dataset_io.load(DATASET)
    chosen = order[:args.episodes]
    base_config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    frame_cache = {}
    for episode in chosen:
        frame_cache[episode] = align.measure_bags(
            align.dataset_frame(DATASET, episode, 0), base_config)
    print("episodes %s ; bags detected in first frames: %s"
          % (chosen, {e: len(v) for e, v in frame_cache.items()}))
    print()

    summary = []
    for spec in args.bases.split(";"):
        x_cm, y_cm = (float(v) for v in spec.split(","))
        tag = "x%05.1f_y%05.1f" % (x_cm, y_cm)
        res = run_base(x_cm, y_cm, 1.5708, {e: episodes[e] for e in chosen}, frame_cache)
        wins = sum(1 for r in res if r["success"])
        lifts = [r["peak_rise"] for r in res]
        summary.append({"base": [x_cm, y_cm], "successes": wins, "n": len(res),
                        "lifts": lifts, "detail": res})
        print("  base (%5.1f,%5.1f)  success %d/%d   peak lifts %s"
              % (x_cm, y_cm, wins, len(res), [round(v, 2) for v in lifts]))
    print()
    summary.sort(key=lambda s: (-s["successes"], -max(s["lifts"])))
    best = summary[0]
    print("  BEST base (%.1f, %.1f) -> %d/%d strict successes"
          % (best["base"][0], best["base"][1], best["successes"], best["n"]))
    for s in summary:
        print("    (%5.1f,%5.1f): %d/%d"
              % (s["base"][0], s["base"][1], s["successes"], s["n"]))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
