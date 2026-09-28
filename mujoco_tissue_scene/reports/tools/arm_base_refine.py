#!/usr/bin/env python3
"""Refine the arm-base placement around the mirrored solution."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import align_with_dataset as align      # noqa: E402
import sim_env                          # noqa: E402
import dataset_io                       # noqa: E402

DATASET = Path("/workspace/shared/new_program_qiuzhi/without_tactile/"
               "pi05_normal_recovery_merged_214eps")
OUT = ROOT / "outputs/arm_base_refine"


def edges(image):
    return cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0),
                     40, 110)


def score(real_edges, sim_bgr, kernel):
    sim_edges = edges(sim_bgr)
    forward = 100.0 * ((real_edges > 0) & (cv2.dilate(sim_edges, kernel) > 0)).sum() / \
        max(int((real_edges > 0).sum()), 1)
    real_near = cv2.dilate(real_edges, kernel) > 0
    backward = 100.0 * ((sim_edges > 0) & real_near).sum() / max(int((sim_edges > 0).sum()), 1)
    return float(forward), float(backward)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = [(0, 0), (0, 60), (1, 0)]
    reals = [align.dataset_frame(DATASET, episode, frame) for episode, frame in frames]
    states = []
    episodes, _ = dataset_io.load(DATASET)
    for episode, frame in frames:
        states.append(episodes[episode]["observation.state"][frame])
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
            f, b = score(real_edges[index], sim_bgr, kernel)
            forward.append(f)
            backward.append(b)
        env.close()
        return float(np.mean(forward)), float(np.mean(backward))

    print("  %-32s %11s %11s" % ("candidate (x_cm, y_cm, yaw_rad)", "real->sim", "sim->real"))
    results = []
    for x_cm in (92.0, 95.0, 98.0):
        for y_cm in (5.0, 10.0, 15.0):
            for yaw in (1.72, 1.7738, 1.83):
                f, b = evaluate(x_cm, y_cm, yaw)
                results.append((x_cm, y_cm, yaw, f, b))
                print("  x=%5.1f y=%5.1f yaw=%.4f        %11.2f %11.2f" % (x_cm, y_cm, yaw, f, b))
    results.sort(key=lambda r: -(r[3] + r[4]))
    print()
    best = results[0]
    print("best: x=%.1f cm  y=%.1f cm  yaw=%.4f rad  (%.2f / %.2f)"
          % (best[0], best[1], best[2], best[3], best[4]))
    print("      -> world position [%.4f, %.4f]"
          % (best[0] / 100.0 - 0.6, best[1] / 100.0 - 0.375))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
