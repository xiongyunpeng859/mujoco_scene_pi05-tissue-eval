#!/usr/bin/env python3
"""Diagnose why a real episode does not reproduce: did the hand ever reach a bag?

Replays the episode and records, every control step:
  * contacts between any hand body and any bag body,
  * the closest approach between the hand bodies and each bag centre,
  * the closest approach between the hand and the tray,
so geometry problems (hand never arrives) can be told apart from contact problems
(hand arrives, bag does not follow).
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/grasp_diag")
    parser.add_argument("--place-bags", action="store_true", default=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    model_dump = None
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    episodes, _ = dataset_io.load(DATASET)
    action = episodes[args.episode]["action"]
    state = episodes[args.episode]["observation.state"]

    env = sim_env.TissueSceneEnv(dataset=DATASET, render=True, output_dir=args.out)
    env.reset(options={"state": state[0]})

    if args.place_bags:
        real_first = align.dataset_frame(DATASET, args.episode, 0)
        found = align.measure_bags(real_first, config)
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
        env.mujoco.mj_forward(env.model, env.data)

    model = env.model
    bag_bodies = {env.model.body(name).id: name for name in env.box_free}
    hand_bodies = [i for i in range(model.nbody)
                   if (model.body(i).name or "").startswith("hand_")]
    tray_body = env.tray_body
    print("hand bodies: %d, bag bodies: %s" % (len(hand_bodies), list(bag_bodies.values())))

    def min_distance(positions, target):
        if not positions:
            return float("nan")
        return float(min(np.linalg.norm(model.body(i).pos if False else pos - target)
                         for i, pos in positions))

    rows, first_contact = [], {}
    hand_bag_contacts = 0
    for step, value in enumerate(action):
        env.step(value)
        hand_positions = [(i, env.data.xpos[i].copy()) for i in hand_bodies]
        row = {"step": step}
        for body_id, name in bag_bodies.items():
            centre = env.data.xpos[body_id].copy()
            distance = min(np.linalg.norm(pos - centre) for _, pos in hand_positions)
            row[name] = distance
            if distance < 0.06 and name not in first_contact:
                first_contact[name] = step
        tray_centre = env.data.xpos[tray_body].copy()
        row["tray"] = min(np.linalg.norm(pos - tray_centre) for _, pos in hand_positions)
        # MuJoCo contacts directly tell us when a finger actually touches a bag.
        for index in range(env.data.ncon):
            contact = env.data.contact[index]
            bodies = {int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])}
            if bodies & set(bag_bodies) and bodies & set(hand_bodies):
                hand_bag_contacts += 1
                row.setdefault("hand_bag_contact", 0)
                row["hand_bag_contact"] += 1
        rows.append(row)

    keys = [k for k in rows[0] if k != "step"]
    print()
    print("  %-6s %s" % ("step", "  ".join("%-16s" % k for k in keys)))
    for row in rows[:: max(1, len(rows) // 12)] + [rows[-1]]:
        print("  %-6d %s" % (row["step"], "  ".join("%-16.4f" % row[k] for k in keys)))
    print()
    print("closest hand->bag approach per bag (m):")
    for key in keys:
        if key == "tray":
            continue
        values = [row[key] for row in rows]
        print("   %-16s min %.4f at step %d" % (key, min(values), int(np.argmin(values))))
    tray_values = [row["tray"] for row in rows]
    print("   %-16s min %.4f at step %d" % ("hand->tray", min(tray_values), int(np.argmin(tray_values))))
    print()
    print("hand-bag contact samples over the episode: %d" % hand_bag_contacts)
    print("first approach under 6 cm: %s" % first_contact)

    (args.out / "diagnosis.json").write_text(json.dumps(
        {"min_distance": {k: min(row[k] for row in rows) for k in keys},
         "hand_bag_contact_samples": hand_bag_contacts,
         "first_approach_under_6cm": first_contact}, indent=2))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
