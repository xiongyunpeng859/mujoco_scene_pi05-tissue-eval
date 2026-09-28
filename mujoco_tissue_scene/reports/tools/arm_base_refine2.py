#!/usr/bin/env python3
"""Focused second pass on the arm base: y trend at the best x, then x and yaw."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import align_with_dataset as align      # noqa: E402
import arm_base_refine as refine        # noqa: E402
import dataset_io                       # noqa: E402
import sim_env                          # noqa: E402

DATASET = refine.DATASET
OUT = ROOT / "outputs/arm_base_refine2"
FRAMES = [(0, 0), (0, 60), (1, 0)]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    reals = [align.dataset_frame(DATASET, episode, frame) for episode, frame in FRAMES]
    episodes, _ = dataset_io.load(DATASET)
    states = [episodes[episode]["observation.state"][frame] for episode, frame in FRAMES]
    real_edges = [refine.edges(image) for image in reals]
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
            f, b = refine.score(real_edges[index], sim_bgr, kernel)
            forward.append(f)
            backward.append(b)
        env.close()
        return float(np.mean(forward)), float(np.mean(backward))

    candidates = [(95.0, y, 1.7738) for y in (15.0, 20.0, 25.0, 30.0)]
    candidates += [(x, 20.0, 1.7738) for x in (93.0, 94.0, 96.0, 97.0)]
    candidates += [(95.0, 20.0, yaw) for yaw in (1.70, 1.75, 1.80, 1.85)]
    print("  %-30s %11s %11s" % ("candidate", "real->sim", "sim->real"))
    rows = []
    for x_cm, y_cm, yaw in candidates:
        forward, backward = evaluate(x_cm, y_cm, yaw)
        rows.append((x_cm, y_cm, yaw, forward, backward))
        print("  x=%5.1f y=%5.1f yaw=%.4f      %11.2f %11.2f" % (x_cm, y_cm, yaw, forward, backward))
    rows.sort(key=lambda r: -(r[3] + r[4]))
    best = rows[0]
    print()
    print("best: x=%.1f cm  y=%.1f cm  yaw=%.4f rad  (%.2f / %.2f)"
          % (best[0], best[1], best[2], best[3], best[4]))
    print("      -> world position [%.4f, %.4f]"
          % (best[0] / 100 - 0.6, best[1] / 100 - 0.375))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
