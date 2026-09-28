#!/usr/bin/env python3
"""Does a realistic hand-bag friction let the bag stay in the box?"""
from __future__ import annotations

import argparse
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
OUT = ROOT / "outputs/sweep_release"


def run(friction, dataset, wanted):
    import mujoco
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    for box in config["boxes"]:
        if box.get("soft_body"):
            box["friction"] = [friction, 0.1, 0.002]
    d = OUT / ("f%.1f" % friction)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "scene.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    surface = config["table"]["surface_z"]
    table_cm = [config["table"]["size"][0] * 100.0, config["table"]["size"][1] * 100.0]
    yaw_t = float(config["tray"].get("yaw", 0.0))
    cos_t, sin_t = np.cos(-yaw_t), np.sin(-yaw_t)
    half_x, half_y = config["tray"]["size"][0] / 2.0, config["tray"]["size"][1] / 2.0
    env = sim_env.TissueSceneEnv(config_path=path, dataset=SUCCESS, render=False,
                                 output_dir=d)
    model, data = env.model, env.data
    out = []
    for episode in wanted:
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
        tray = data.xpos[env.tray_body].copy()
        peak = {n: 0.0 for n in names}
        for value in action:
            env.step(value)
            for n in names:
                adr, num = (int(model.flex_vertadr[fids[n]]),
                            int(model.flex_vertnum[fids[n]]))
                c = data.flexvert_xpos[adr:adr + num].mean(0)
                peak[n] = max(peak[n], float(c[2]) - start_z[n])
        best = max(names, key=lambda n: peak[n])
        adr = int(model.flex_vertadr[fids[best]])
        num = int(model.flex_vertnum[fids[best]])
        pts = data.flexvert_xpos[adr:adr + num]
        ddx, ddy = pts[:, 0] - tray[0], pts[:, 1] - tray[1]
        lx = cos_t * ddx - sin_t * ddy
        ly = sin_t * ddx + cos_t * ddy
        frac = float(((np.abs(lx) < half_x) & (np.abs(ly) < half_y)
                      & (pts[:, 2] > tray[2] + 0.005)).mean())
        out.append({"episode": episode, "bags_detected": len(bags),
                    "peak": round(peak[best], 4), "grasped": bool(peak[best] > 0.05),
                    "final_inside": round(frac, 3),
                    "placed": bool(peak[best] > 0.05 and frac > 0.3)})
    env.close()
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frictions", type=float, nargs="+",
                        default=[4.0, 2.5, 1.5, 0.8])
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 3, 4])
    args = parser.parse_args()
    dataset, order = dataset_io.load(SUCCESS)
    print("  %-8s %-4s %9s %8s %8s %6s %6s" %
          ("friction", "ep", "peak lift", "grasped", "inside", "placed", "bags"))
    print("", flush=True)
    summary = {}
    for friction in args.frictions:
        rows = run(friction, dataset, args.episodes)
        for r in rows:
            print("  %-8.1f %-4d %9.4f %8s %8.2f %6s %6d"
                  % (friction, r["episode"], r["peak"], r["grasped"],
                     r["final_inside"], r["placed"], r["bags_detected"]), flush=True)
        summary[friction] = (sum(r["grasped"] for r in rows),
                             sum(r["placed"] for r in rows))
        print("  -> friction %.1f : grasped %d/%d  placed %d/%d"
              % (friction, summary[friction][0], len(rows),
                 summary[friction][1], len(rows)), flush=True)
        print("", flush=True)
    print("  SUMMARY", flush=True)
    for f, (g, p) in summary.items():
        print("     friction %-4.1f  grasped %d/%d   placed %d/%d"
              % (f, g, len(args.episodes), p, len(args.episodes)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
