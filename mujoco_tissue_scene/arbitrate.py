#!/usr/bin/env python3
"""Tie-breaker: render the same real frame under both candidate bases and compare.

No fitting involved.  Real frame of a success episode at the moment the arm is over
the box, rendered under base A (the grasp fit) and base B (the kinematic fit).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
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
OUT = ROOT / "outputs/arbitrate"
CANDIDATES = [("A_graspfit_18_10_90", 18.0, 10.0, 1.5708),
              ("B_kinematic_34_24_60", 34.0, 24.0, 1.0472)]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    episodes, order = dataset_io.load(SUCCESS)
    episode = order[0]
    frames = [120, 150, 165]
    tiles = []
    for index in frames:
        real = align.dataset_frame(SUCCESS, episode, index)
        row = [real]
        for label, x, y, yaw in CANDIDATES:
            config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
            config["measured_layout"]["arm_mount_xy_cm_from_left_bottom"] = [x, y]
            config["arm"]["euler"][2] = yaw
            path = OUT / label / "scene.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
            env = sim_env.TissueSceneEnv(config_path=path, dataset=SUCCESS, render=True,
                                         output_dir=path.parent)
            env.reset(options={"state": episodes[episode]["observation.state"][index],
                               "randomize_objects": False})
            sim = cv2.cvtColor(env.render("central"), cv2.COLOR_RGB2BGR)
            env.close()
            cv2.putText(sim, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 255, 255), 1)
            if index == 150:
                cv2.imwrite(str(OUT / ("sim_%s_t%d.png" % (label, index))), sim)
            row.append(sim)
        cv2.putText(row[0], "REAL t=%d" % index, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 255), 1)
        tiles.append(np.hstack(row))
    grid = np.vstack(tiles)
    cv2.imwrite(str(OUT / "triple.png"), grid)
    print("wrote %s   (columns: REAL | base A (18,10,90) | base B (34,24,60))" % (OUT / "triple.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
