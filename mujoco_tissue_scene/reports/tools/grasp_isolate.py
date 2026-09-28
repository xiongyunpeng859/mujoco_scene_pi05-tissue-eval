#!/usr/bin/env python3
"""Isolate the grasp: can this hand hold a tissue bag at all?

No reliance on the real bag positions or on base alignment.  For each candidate
frame of a real episode we:

  1. set the arm to that frame's joint state and close the fingers,
  2. read where the closed fingertips actually are, and put the bag exactly there
     (x, y from the fingertip centroid, z resting on the tabletop),
  3. drive the arm through the next N frames of the real episode, which in the
     real recording is the lift-and-carry motion,
  4. report whether the bag leaves the table with the hand.

If no frame can hold the bag, the contact parameters (or the finger geometry) are
what needs fixing, not the alignment.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import dataset_io                       # noqa: E402
import sim_env                          # noqa: E402

DATASET = Path("/workspace/shared/new_program_qiuzhi/without_tactile/"
               "pi05_normal_recovery_merged_214eps")
FINGER_TIPS = ["hand_L_thumb_tip", "hand_L_index_tip", "hand_L_middle_tip",
               "hand_L_ring_tip", "hand_L_pinky_tip"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frames", type=int, nargs="+",
                        default=[95, 100, 105, 110, 115, 120, 125])
    parser.add_argument("--carry", type=int, default=45,
                        help="how many frames of real motion to drive after closing")
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    surface = config["table"]["surface_z"]
    bag_height = config["boxes"][0]["size"][2]
    episodes, _ = dataset_io.load(DATASET)
    action = episodes[args.episode]["action"]
    state = episodes[args.episode]["observation.state"]

    env = sim_env.TissueSceneEnv(dataset=DATASET, render=False)
    model, data, mujoco = env.model, env.data, env.mujoco
    tip_ids = [model.body(name).id for name in FINGER_TIPS]
    bag_name = config["boxes"][0]["name"]
    qpos_adr, dof_adr = env.box_free[bag_name]

    print("=== grasp isolation test: place the bag exactly inside the closed fingers ===")
    print("  %-7s %-28s %8s %8s %9s %9s  %s" %
          ("frame", "closed-fingertip centroid (m)", "tip z", "bag z0", "bag z1",
           "contacts", "held"))
    results = []
    for frame in args.frames:
        if frame + args.carry >= len(action):
            continue
        env.reset(options={"state": state[frame], "randomize_objects": False})
        # close the fingers using the real values from a later (grasping) frame
        closed = state[min(frame + 10, len(state) - 1)].copy()
        data.qpos[env.joint_adr] = closed
        mujoco.mj_forward(model, data)
        tips = np.array([data.xpos[i] for i in tip_ids])
        centroid = tips.mean(0)
        low = tips[:, 2].min()

        # put the bag where the fingertips are, resting on the table
        data.qpos[qpos_adr + 0] = centroid[0]
        data.qpos[qpos_adr + 1] = centroid[1]
        data.qpos[qpos_adr + 2] = surface + bag_height / 2.0 + 0.0005
        data.qpos[qpos_adr + 3] = 1.0
        data.qpos[qpos_adr + 4:qpos_adr + 7] = 0.0
        data.qvel[dof_adr:dof_adr + 6] = 0.0
        mujoco.mj_forward(model, data)
        z0 = float(data.qpos[qpos_adr + 2])

        # drive the real lift-and-carry segment
        contacts = 0
        bag_body = model.body(bag_name).id
        hand_bodies = {i for i in range(model.nbody)
                       if (model.body(i).name or "").startswith("hand_")}
        for step in range(frame, min(frame + args.carry, len(action))):
            env.step(action[step])
            for index in range(data.ncon):
                contact = data.contact[index]
                bodies = {int(model.geom_bodyid[contact.geom1]),
                          int(model.geom_bodyid[contact.geom2])}
                if bag_body in bodies and (bodies & hand_bodies):
                    contacts += 1
        z1 = float(data.qpos[qpos_adr + 2])
        held = z1 > surface + bag_height / 2.0 + 0.02
        results.append((frame, held, z1 - z0, contacts))
        print("  %-7d %-28s %8.3f %8.3f %9.3f %9d  %s"
              % (frame, np.round(centroid, 3).tolist(), low, z0, z1, contacts,
                 "YES" if held else "no"))

    print()
    if any(r[1] for r in results):
        best = [r for r in results if r[1]]
        print("  RESULT: the hand CAN hold the bag (frames %s)."
              % [r[0] for r in best])
        print("  So the grasp itself works; what is missing is getting the hand to the")
        print("  bag's real position -- i.e. alignment, not contact parameters.")
    else:
        print("  RESULT: no frame could hold the bag.")
        print("  The fingers close but the bag does not come with them, so the problem")
        print("  is the grasp itself: contact parameters or finger geometry.")
        lift = max(r[2] for r in results)
        print("  best bag rise: %.4f m (needs > 0.02 m to count as held)" % lift)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
