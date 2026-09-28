#!/usr/bin/env python3
"""Measure the hand's grip geometry against the bag it is supposed to hold."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import action_layout as layout          # noqa: E402
import dataset_io                       # noqa: E402
import hand_control                     # noqa: E402
import sim_env                          # noqa: E402

DATASET = Path("/workspace/shared/new_program_qiuzhi/without_tactile/"
               "pi05_normal_recovery_merged_214eps")
TIPS = {"thumb": "hand_L_thumb_tip", "index": "hand_L_index_tip",
        "middle": "hand_L_middle_tip", "ring": "hand_L_ring_tip",
        "pinky": "hand_L_pinky_tip"}


def main() -> int:
    import mujoco
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    size = [v * 100.0 for v in config["boxes"][0]["size"]]
    episodes, _ = dataset_io.load(DATASET)
    state = episodes[0]["observation.state"]

    env = sim_env.TissueSceneEnv(dataset=DATASET, render=False)
    model, data = env.model, env.data
    palm = model.body("static_omnihand_reference").id
    print("bag size: %.1f x %.1f x %.1f cm" % tuple(size))
    print()
    print("=== finger joint limits vs the values the real data uses ===")
    print("  %-26s %9s %9s %10s %10s" % ("joint", "lo(deg)", "hi(deg)",
                                          "open(deg)", "data max"))
    poses = hand_control.poses()
    for index, name in enumerate(hand_control.HAND_JOINTS):
        joint = model.joint("hand_" + name)
        lo, hi = np.degrees(model.jnt_range[joint.id])
        column = state[:, 6 + index]
        print("  %-26s %9.1f %9.1f %10.1f %10.1f"
              % (name, lo, hi, np.degrees(poses["open"][index]), np.degrees(column.max())))
    print()

    for label, vector in (("hand OPEN (recorded gesture)", poses["open"]),
                          ("hand CLOSED (recorded gesture)", poses["closed"]),
                          ("hand at the real grasp frame 119", state[119][6:].tolist())):
        data.qpos[env.joint_adr] = np.concatenate([state[119][:6], np.asarray(vector)])
        mujoco.mj_forward(model, data)
        points = {k: data.xpos[model.body(v).id].copy() for k, v in TIPS.items()}
        thumb = points["thumb"]
        print("=== %s ===" % label)
        for finger in ("index", "middle", "ring", "pinky"):
            gap = np.linalg.norm(points[finger] - thumb) * 100
            print("   thumb-to-%-7s %6.2f cm" % (finger, gap))
        allpts = np.array(list(points.values()))
        span = allpts.max(0) - allpts.min(0)
        print("   fingertip cluster span  %.2f x %.2f x %.2f cm"
              % (span[0] * 100, span[1] * 100, span[2] * 100))
        rel = allpts - data.xpos[palm]
        print("   fingertips relative to palm, mean %s cm"
              % np.round(rel.mean(0) * 100, 2).tolist())
        print()
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
