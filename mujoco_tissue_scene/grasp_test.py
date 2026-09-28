#!/usr/bin/env python3
"""Can the simulation actually perform the task?  Replay a real episode and look.

The bags are placed where they were in the real frame, the arm is driven with the
real recorded action sequence, and the outcome is reported per bag.  If a real
successful episode does not also succeed here, the contact parameters -- not the
policy -- are what needs fixing, and collecting data would be pointless.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import align_with_dataset as align      # noqa: E402
import dataset_io                       # noqa: E402
import sim_env                          # noqa: E402

DATASET = Path("/workspace/shared/new_program_qiuzhi/without_tactile/"
               "pi05_normal_recovery_merged_214eps")


def bag_state(env):
    out = {}
    for name, (qpos_adr, _) in env.box_free.items():
        out[name] = np.array(env.data.qpos[qpos_adr:qpos_adr + 3], dtype=float)
    return out


def inside_tray(env, position, margin=0.0):
    tray = env.data.xpos[env.tray_body]
    yaw = float(env.config["tray"].get("yaw", 0.0))
    cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)
    dx, dy = position[0] - tray[0], position[1] - tray[1]
    local_x = cos_y * dx - sin_y * dy
    local_y = sin_y * dx + cos_y * dy
    half_x = env.config["tray"]["size"][0] / 2.0 - margin
    half_y = env.config["tray"]["size"][1] / 2.0 - margin
    return bool(abs(local_x) < half_x and abs(local_y) < half_y
                and position[2] > tray[2] + 0.005)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/grasp_test")
    parser.add_argument("--use-image-bags", action="store_true", default=True)
    parser.add_argument("--no-image-bags", dest="use_image_bags", action="store_false")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    episodes, _ = dataset_io.load(DATASET)
    action = episodes[args.episode]["action"]
    state = episodes[args.episode]["observation.state"]
    print("episode %d: %d steps, %.2f s" % (args.episode, len(action), len(action) / 30.0))

    env = sim_env.TissueSceneEnv(dataset=DATASET, render=True, output_dir=args.out)
    env.reset(options={"state": state[0]})

    if args.use_image_bags:
        real_first = align.dataset_frame(DATASET, args.episode, args.frame)
        cv2.imwrite(str(args.out / "real_first.png"), real_first)
        found = align.measure_bags(real_first, config)
        print("bags measured in the real first frame: %d" % len(found))
        table_size_cm = [config["table"]["size"][0] * 100.0, config["table"]["size"][1] * 100.0]
        surface = config["table"]["surface_z"]
        for box, bag in zip(config["boxes"], found):
            x_cm, y_cm = bag["box_centre_cm"]
            yaw = np.radians(bag["yaw_deg"])
            qpos_adr, dof_adr = env.box_free[box["name"]]
            env.data.qpos[qpos_adr + 0] = x_cm / 100.0 - table_size_cm[0] / 200.0
            env.data.qpos[qpos_adr + 1] = y_cm / 100.0 - table_size_cm[1] / 200.0
            env.data.qpos[qpos_adr + 2] = surface + box["size"][2] / 2.0 + 0.001
            env.data.qpos[qpos_adr + 3] = np.cos(yaw / 2.0)
            env.data.qpos[qpos_adr + 6] = np.sin(yaw / 2.0)
            env.data.qvel[dof_adr:dof_adr + 6] = 0.0
            print("   %-16s <- (%.1f, %.1f) cm yaw %.0f deg" %
                  (box["name"], x_cm, y_cm, bag["yaw_deg"]))
        env.mujoco.mj_forward(env.model, env.data)

    start = bag_state(env)
    cv2.imwrite(str(args.out / "sim_before.png"),
                cv2.cvtColor(env.render("central"), cv2.COLOR_RGB2BGR))
    print()
    print("start positions (world m):")
    for name, position in start.items():
        print("   %-16s %s   inside tray: %s" %
              (name, np.round(position, 3).tolist(), inside_tray(env, position)))

    tracking, contacts = [], []
    for row, value in enumerate(action):
        obs, reward, terminated, truncated, info = env.step(value)
        tracking.append(np.abs(np.array(info["tracking_error"])).max())
        contacts.append(env.data.ncon)
    cv2.imwrite(str(args.out / "sim_after.png"),
                cv2.cvtColor(env.render("central"), cv2.COLOR_RGB2BGR))

    end = bag_state(env)
    print()
    print("end positions (world m):")
    moved = []
    for name, position in end.items():
        delta = np.linalg.norm(position - start[name])
        hit = inside_tray(env, position)
        moved.append(hit)
        print("   %-16s %s   moved %.3f m   inside tray: %s" %
              (name, np.round(position, 3).tolist(), delta, hit))
    print()
    print("max joint tracking error: %.4f rad" % max(tracking))
    print("contacts per step: min %d  median %d  max %d" %
          (min(contacts), int(np.median(contacts)), max(contacts)))
    print("TASK REPRODUCED IN SIM: %s" % any(moved))

    (args.out / "grasp_test.json").write_text(json.dumps({
        "episode": args.episode, "steps": len(action),
        "start": {k: v.tolist() for k, v in start.items()},
        "end": {k: v.tolist() for k, v in end.items()},
        "any_bag_in_tray": bool(any(moved)),
        "max_tracking_error_rad": float(max(tracking)),
        "contacts_median": int(np.median(contacts)),
    }, indent=2))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
