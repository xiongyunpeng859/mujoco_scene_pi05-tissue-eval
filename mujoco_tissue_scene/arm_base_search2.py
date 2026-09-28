#!/usr/bin/env python3
"""Search the arm base placement using a mid-episode frame.

The episode start pose is nearly all-zero, so it barely constrains the base.  The
middle of the episode has the arm reaching across to the left, which exercises the
whole kinematic chain, so match against that frame instead.

    python arm_base_search2.py --stage yaw          # sweep base yaw at fixed x,y
    python arm_base_search2.py --stage xy --yaw A   # sweep x,y at the chosen yaw
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import align_with_dataset as align      # noqa: E402
import dataset_io                       # noqa: E402
import sim_env                          # noqa: E402

DATASET = Path("/workspace/shared/new_program_qiuzhi/without_tactile/"
               "pi05_normal_recovery_merged_214eps")
OUT = ROOT / "outputs/arm_base_search2"
# frame indices chosen where the arm is far from the start pose
EPISODE, FRAMES = 0, (119, 90, 150)


def edges(image):
    return cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0), 40, 110)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["yaw", "xy"], default="yaw")
    parser.add_argument("--x", type=float, default=95.0)
    parser.add_argument("--y", type=float, default=15.0)
    parser.add_argument("--yaw", type=float, default=None)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    episodes, _ = dataset_io.load(DATASET)
    states = [episodes[EPISODE]["observation.state"][frame] for frame in FRAMES]
    reals = [align.dataset_frame(DATASET, EPISODE, frame) for frame in FRAMES]
    for frame, image in zip(FRAMES, reals):
        cv2.imwrite(str(OUT / ("real_f%03d.png" % frame)), image)
    real_edges = [edges(image) for image in reals]
    kernel = np.ones((5, 5), np.uint8)

    def evaluate(x_cm, y_cm, yaw):
        config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
        config["measured_layout"]["arm_mount_xy_cm_from_left_bottom"] = [x_cm, y_cm]
        config["arm"]["euler"][2] = yaw
        tag = "x%05.1f_y%05.1f_a%06.4f" % (x_cm, y_cm, yaw)
        path = OUT / ("scene_%s.yaml" % tag)
        path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
        env = sim_env.TissueSceneEnv(config_path=path, dataset=DATASET, render=True,
                                    output_dir=OUT / tag)
        forward, backward = [], []
        for index, state in enumerate(states):
            env.reset(options={"state": state, "randomize_objects": False})
            sim_bgr = cv2.cvtColor(env.render("central"), cv2.COLOR_RGB2BGR)
            if index == 0:
                cv2.imwrite(str(OUT / ("sim_%s.png" % tag)), sim_bgr)
            sim_edges = edges(sim_bgr)
            forward.append(100.0 * ((real_edges[index] > 0) & (cv2.dilate(sim_edges, kernel) > 0)).sum()
                           / max(int((real_edges[index] > 0).sum()), 1))
            backward.append(100.0 * ((sim_edges > 0) & (cv2.dilate(real_edges[index], kernel) > 0)).sum()
                            / max(int((sim_edges > 0).sum()), 1))
        env.close()
        return float(np.mean(forward)), float(np.mean(backward))

    if args.stage == "yaw":
        candidates = [(args.x, args.y, yaw) for yaw in
                      (1.3678, 1.7738, 2.1, 2.5, 2.9, 3.3, 3.7, 4.1, 4.5, 4.9, 5.3, 5.7)]
    else:
        assert args.yaw is not None, "--yaw is required for the xy stage"
        candidates = [(x, y, args.yaw) for x in (80.0, 87.0, 95.0, 103.0, 110.0)
                      for y in (-5.0, 5.0, 15.0, 25.0)]

    print("  %-30s %11s %11s" % ("candidate (x, y, yaw)", "real->sim", "sim->real"))
    rows = []
    for x_cm, y_cm, yaw in candidates:
        forward, backward = evaluate(x_cm, y_cm, yaw)
        rows.append((x_cm, y_cm, yaw, forward, backward))
        print("  x=%6.1f y=%5.1f yaw=%.4f      %11.2f %11.2f" % (x_cm, y_cm, yaw, forward, backward))
    rows.sort(key=lambda r: -(r[3] + r[4]))
    best = rows[0]
    print()
    print("best: x=%.1f  y=%.1f  yaw=%.4f  (%.2f / %.2f)"
          % (best[0], best[1], best[2], best[3], best[4]))
    print("      world [%.4f, %.4f]" % (best[0] / 100 - 0.6, best[1] / 100 - 0.375))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
