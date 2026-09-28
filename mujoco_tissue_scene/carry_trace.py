#!/usr/bin/env python3
"""Trace the carry with the working grasp: where does the bag end up?

Uses base (20, 10) cm with the soft bag and friction 4.0, and places the bags where
the real first frame shows them.  Reports the bag's path, how long it is held, the
closest approach to the box, and the release point.
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
OUT = ROOT / "outputs/carry_trace"


def main() -> int:
    import mujoco
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x", type=float, default=20.0)
    parser.add_argument("--y", type=float, default=10.0)
    parser.add_argument("--yaw", type=float, default=1.5708)
    parser.add_argument("--episode", type=int, default=0)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    config["measured_layout"]["arm_mount_xy_cm_from_left_bottom"] = [args.x, args.y]
    config["arm"]["euler"][2] = args.yaw
    path = OUT / "scene.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))

    episodes, _ = dataset_io.load(DATASET)
    action, state = episodes[args.episode]["action"], episodes[args.episode]["observation.state"]
    env = sim_env.TissueSceneEnv(config_path=path, dataset=DATASET, render=False,
                                 output_dir=OUT)
    model, data = env.model, env.data
    env.reset(options={"state": state[0], "randomize_objects": False})

    real_first = align.dataset_frame(DATASET, args.episode, 0)
    found = align.measure_bags(real_first, config)
    surface = config["table"]["surface_z"]
    table_cm = [config["table"]["size"][0] * 100.0, config["table"]["size"][1] * 100.0]
    for box, bag in zip(config["boxes"], found):
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
    hand_bodies = {i for i in range(model.nbody)
                   if (model.body(i).name or "").startswith("hand_")}
    flex_ids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, n + "_soft")
                for n in names}
    start_z = {n: float(data.qpos[env.box_free[n][0] + 2]) for n in names}
    tray = data.xpos[env.tray_body].copy()
    half = config["tray"]["size"][0] / 2.0
    print("base (%.1f, %.1f) cm, yaw %.4f   tray centre %s  half %.3f m"
          % (args.x, args.y, args.yaw, np.round(tray, 3).tolist(), half))
    print()
    print("  %-5s %s" % ("step", "  ".join("%-26s" % n for n in names)))
    peak = {n: 0.0 for n in names}
    carried = {n: 0 for n in names}
    min_tray = {n: 9.9 for n in names}
    for step, value in enumerate(action):
        env.step(value)
        if step % 15 == 0 or step == len(action) - 1:
            cells = []
            for n in names:
                p = data.qpos[env.box_free[n][0]:env.box_free[n][0] + 3]
                rise = float(p[2]) - start_z[n]
                cells.append("z%.3f r%+.3f d%.2f" % (p[2], rise,
                                                     np.linalg.norm(p[:2] - tray[:2])))
            print("  %-5d %s" % (step, "  ".join("%-26s" % c for c in cells)))
        for n in names:
            p = data.qpos[env.box_free[n][0]:env.box_free[n][0] + 3]
            rise = float(p[2]) - start_z[n]
            peak[n] = max(peak[n], rise)
            if rise > 0.02:
                carried[n] += 1
                min_tray[n] = min(min_tray[n], float(np.linalg.norm(p[:2] - tray[:2])))

    print()
    for n in names:
        p = data.qpos[env.box_free[n][0]:env.box_free[n][0] + 3]
        inside = (float(np.linalg.norm(p[:2] - tray[:2])) < half
                  and float(p[2]) > surface + 0.01)
        print("  %-16s peak rise %.4f m   carried %3d steps   closest to tray %.4f m"
              "   ends in tray %s"
              % (n, peak[n], carried[n], min_tray[n] if min_tray[n] < 9 else -1, inside))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
