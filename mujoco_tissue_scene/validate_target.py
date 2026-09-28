#!/usr/bin/env python3
"""Score the TARGET pack: the one the arm actually lifts.

Leftover packs sitting in the tray at episode start are real, so "some pack is in
the tray" is meaningless.  Only the pack that gets lifted can be scored, and it
must end up in the tray.
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
OUT = ROOT / "outputs/validate_target"
LIFT_MIN = 0.05


def main() -> int:
    import mujoco
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, nargs="+", default=list(range(12)))
    parser.add_argument("--x", type=float, default=18.0)
    parser.add_argument("--y", type=float, default=10.0)
    parser.add_argument("--yaw", type=float, default=1.5708)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    config["measured_layout"]["arm_mount_xy_cm_from_left_bottom"] = [args.x, args.y]
    config["arm"]["euler"][2] = args.yaw
    path = OUT / "scene.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    surface = config["table"]["surface_z"]
    ts = config["table"]["size"]
    table_cm = [ts[0] * 100.0, ts[1] * 100.0]
    tray = config["tray"]
    tyaw = float(tray.get("yaw", 0.0))
    cos_t, sin_t = np.cos(-tyaw), np.sin(-tyaw)
    half_x = tray["size"][0] / 2.0
    half_y = tray["size"][1] / 2.0

    dataset, order = dataset_io.load(SUCCESS)
    chosen = [e for e in args.episodes if e in dataset]
    env = sim_env.TissueSceneEnv(config_path=path, dataset=SUCCESS, render=False,
                                 output_dir=OUT)
    model, data = env.model, env.data
    print("base (%.1f, %.1f) yaw %.4f ; %d episodes" % (args.x, args.y, args.yaw,
                                                        len(chosen)))
    print("  %-4s %-8s %8s %9s %8s %8s %9s" %
          ("ep", "detected", "peak", "nearest", "grasped", "target_in", "placed"))
    rows = []
    for episode in chosen:
        action = dataset[episode]["action"]
        state = dataset[episode]["observation.state"]
        env.reset(options={"state": state[0], "randomize_objects": False})
        bags = align.measure_bags(align.dataset_frame(SUCCESS, episode, 0), config)
        for box, bag in zip(config["boxes"], bags):
            bx, by = bag["box_centre_cm"]
            byaw = np.radians(bag["yaw_deg"])
            adr, dof = env.box_free[box["name"]]
            data.qpos[adr + 0] = bx / 100.0 - table_cm[0] / 200.0
            data.qpos[adr + 1] = by / 100.0 - table_cm[1] / 200.0
            data.qpos[adr + 2] = surface + box["size"][2] / 2.0 + 0.002
            data.qpos[adr + 3] = np.cos(byaw / 2.0)
            data.qpos[adr + 4:adr + 7] = 0.0
            data.qvel[dof:dof + 6] = 0.0
        mujoco.mj_forward(model, data)
        names = list(env.box_free)
        fids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, n + "_soft")
                for n in names}
        start_z = {n: float(data.qpos[env.box_free[n][0] + 2]) for n in names}
        tray_pos = data.xpos[env.tray_body].copy()
        peak = {n: 0.0 for n in names}
        nearest = {n: 9.9 for n in names}
        for value in action:
            env.step(value)
            for n in names:
                adr, num = (int(model.flex_vertadr[fids[n]]),
                            int(model.flex_vertnum[fids[n]]))
                c = data.flexvert_xpos[adr:adr + num].mean(0)
                rise = float(c[2]) - start_z[n]
                peak[n] = max(peak[n], rise)
                if rise > 0.02:
                    nearest[n] = min(nearest[n],
                                     float(np.linalg.norm(c[:2] - tray_pos[:2])))
        # the target is whichever pack actually moved upward
        target = max(names, key=lambda n: peak[n])
        adr = int(model.flex_vertadr[fids[target]])
        num = int(model.flex_vertnum[fids[target]])
        pts = data.flexvert_xpos[adr:adr + num]
        ddx, ddy = pts[:, 0] - tray_pos[0], pts[:, 1] - tray_pos[1]
        lx = cos_t * ddx - sin_t * ddy
        ly = sin_t * ddx + cos_t * ddy
        frac = float(((np.abs(lx) < half_x) & (np.abs(ly) < half_y)
                      & (pts[:, 2] > tray_pos[2] + 0.005)).mean())
        grasped = peak[target] > LIFT_MIN
        placed = bool(grasped and frac > 0.3)
        print("  %-4d %-8d %8.4f %9s %8s %8.2f %9s"
              % (episode, len(bags), peak[target],
                 ("%.3f" % nearest[target]) if nearest[target] < 9 else "-",
                 grasped, frac, placed), flush=True)
        rows.append({"episode": episode, "detected": len(bags),
                     "peak": round(peak[target], 4), "grasped": grasped,
                     "target_inside": round(frac, 3), "placed": placed})
    env.close()
    g = sum(r["grasped"] for r in rows)
    p = sum(r["placed"] for r in rows)
    print()
    print("  TARGET PACK: grasped %d/%d   placed %d/%d" % (g, len(rows), p, len(rows)))
    (OUT / "result.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
