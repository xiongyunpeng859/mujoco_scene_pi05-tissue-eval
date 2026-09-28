#!/usr/bin/env python3
"""For each real episode, does SOME bag placement let the real trajectory succeed?

This is the decisive question left.  If a placement exists for every episode, the
simulation reproduces them all and the only thing wrong is my image-based bag
detector.  Success is strict: the bag must be lifted (>5 cm) and end in the box.
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
import dataset_io                       # noqa: E402
import sim_env                          # noqa: E402

DATASET = Path("/workspace/shared/new_program_qiuzhi/without_tactile/"
               "pi05_normal_recovery_merged_214eps")
OUT = ROOT / "outputs/placement_per_episode"
MIN_LIFT = 0.05


def build_env(x_cm, y_cm, yaw, tag):
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    config["measured_layout"]["arm_mount_xy_cm_from_left_bottom"] = [x_cm, y_cm]
    config["arm"]["euler"][2] = yaw
    path = OUT / tag / "scene.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    return config, path


def main() -> int:
    import mujoco
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--base", type=str, default="20,6")
    parser.add_argument("--yaw", type=float, default=1.5708)
    parser.add_argument("--anchor", type=str, default="24,58",
                        help="centre of the placement search, in table cm")
    parser.add_argument("--span", type=float, default=8.0)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    base_x, base_y = (float(v) for v in args.base.split(","))
    anchor_x, anchor_y = (float(v) for v in args.anchor.split(","))
    episodes_all, order = dataset_io.load(DATASET)
    chosen = [e for e in args.episodes if e in episodes_all]

    grid = [(dx, dy, dyaw) for dx, dy, dyaw in
            itertools.product((-args.span, 0.0, args.span), (-args.span, 0.0, args.span),
                              (-45.0, 0.0, 45.0))]

    print("base (%.1f, %.1f) yaw %.4f ; search anchor (%.1f, %.1f) cm, %d placements x %d episodes"
          % (base_x, base_y, args.yaw, anchor_x, anchor_y, len(grid), len(chosen)))
    summary = {}
    for episode in chosen:
        action = episodes_all[episode]["action"]
        state = episodes_all[episode]["observation.state"]
        config, path = build_env(base_x, base_y, args.yaw, "search")
        env = sim_env.TissueSceneEnv(config_path=path, dataset=DATASET, render=False,
                                     output_dir=path.parent)
        model, data = env.model, env.data
        surface = config["table"]["surface_z"]
        table_cm = [config["table"]["size"][0] * 100.0, config["table"]["size"][1] * 100.0]
        name = config["boxes"][0]["name"]
        qpos_adr, dof_adr = env.box_free[name]
        fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, name + "_soft")
        yaw_t = float(config["tray"].get("yaw", 0.0))
        cos_t, sin_t = np.cos(-yaw_t), np.sin(-yaw_t)

        best = None
        rows = []
        for dx, dy, dyaw in grid:
            x_cm, y_cm, yaw_deg = anchor_x + dx, anchor_y + dy, dyaw
            env.reset(options={"state": state[0], "randomize_objects": False})
            yaw_r = np.radians(yaw_deg)
            data.qpos[qpos_adr + 0] = x_cm / 100.0 - table_cm[0] / 200.0
            data.qpos[qpos_adr + 1] = y_cm / 100.0 - table_cm[1] / 200.0
            data.qpos[qpos_adr + 2] = surface + config["boxes"][0]["size"][2] / 2.0 + 0.002
            data.qpos[qpos_adr + 3] = np.cos(yaw_r / 2.0)
            data.qpos[qpos_adr + 4:qpos_adr + 7] = 0.0
            data.qvel[dof_adr:dof_adr + 6] = 0.0
            mujoco.mj_forward(model, data)
            z0 = float(data.qpos[qpos_adr + 2])
            tray = data.xpos[env.tray_body].copy()
            peak = 0.0
            for value in action:
                env.step(value)
                adr, num = int(model.flex_vertadr[fid]), int(model.flex_vertnum[fid])
                centre = data.flexvert_xpos[adr:adr + num].mean(0)
                peak = max(peak, float(centre[2]) - z0)
            adr, num = int(model.flex_vertadr[fid]), int(model.flex_vertnum[fid])
            points = data.flexvert_xpos[adr:adr + num]
            ddx, ddy = points[:, 0] - tray[0], points[:, 1] - tray[1]
            lx = cos_t * ddx - sin_t * ddy
            ly = sin_t * ddx + cos_t * ddy
            frac = float((((np.abs(lx) < config["tray"]["size"][0] / 2)
                           & (np.abs(ly) < config["tray"]["size"][1] / 2)
                           & (points[:, 2] > tray[2] + 0.005)).mean()))
            ok = bool(frac > 0.3 and peak > MIN_LIFT)
            rows.append({"placement": [x_cm, y_cm, yaw_deg], "peak": round(peak, 4),
                         "in_tray_frac": round(frac, 3), "success": ok})
            if best is None or (ok, peak, frac) > (best["success"], best["peak"],
                                                   best["in_tray_frac"]):
                best = rows[-1]
        env.close()
        wins = [r for r in rows if r["success"]]
        summary[episode] = {"best": best, "wins": len(wins), "n": len(rows)}
        print("  episode %-2d : %2d/%d placements succeed   best peak %.3f m, "
              "in-tray %.2f, placement %s"
              % (episode, len(wins), len(rows), best["peak"], best["in_tray_frac"],
                 [round(v, 1) for v in best["placement"]]))

    print()
    good = [e for e, s in summary.items() if s["wins"] > 0]
    print("  episodes with at least one working placement: %d / %d  %s"
          % (len(good), len(chosen), good))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
