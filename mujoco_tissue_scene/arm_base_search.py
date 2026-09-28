#!/usr/bin/env python3
"""Decide which arm-base placement matches a real dataset frame.

The tape measurement in the config says the base sits 25 cm from the table's
"left", but the real episode frames show the controlled arm entering from the
bottom RIGHT.  So render the recorded start pose at several base placements and
yaw signs and let the image decide.
"""
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

DATASET = Path("/workspace/shared/new_program_qiuzhi/without_tactile/"
               "pi05_normal_recovery_merged_214eps")
OUT = ROOT / "outputs/arm_base_search"


def edge_score(real, sim_bgr):
    def edges(image):
        return cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0),
                         40, 110)
    real_edges, sim_edges = edges(real), edges(sim_bgr)
    kernel = np.ones((5, 5), np.uint8)
    matched = int(((real_edges > 0) & (cv2.dilate(sim_edges, kernel) > 0)).sum())
    return 100.0 * matched / max(int((real_edges > 0).sum()), 1)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    real = align.dataset_frame(DATASET, 0, 0)
    import dataset_io
    episodes, _ = dataset_io.load(DATASET)
    state = episodes[0]["observation.state"][0]
    cv2.imwrite(str(OUT / "real.png"), real)

    base = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    candidates = [
        ("current  x=25  yaw=+1.368", 25.0, 1.3677924),
        ("mirrored x=95  yaw=+1.368", 95.0, 1.3677924),
        ("mirrored x=95  yaw=-1.368", 95.0, -1.3677924),
        ("mirrored x=95  yaw=+1.774", 95.0, 3.14159265 - 1.3677924),
        ("current  x=25  yaw=-1.368", 25.0, -1.3677924),
    ]
    print("  %-28s %12s %12s" % ("candidate", "real->sim %", "sim->real %"))
    rows = []
    for label, x_cm, yaw in candidates:
        config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
        config["measured_layout"]["arm_mount_xy_cm_from_left_bottom"] = [x_cm, 10.0]
        config["arm"]["euler"][2] = yaw
        path = OUT / ("scene_%s.yaml" % label.split()[1].replace("=", ""))
        path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))

        env = sim_env.TissueSceneEnv(config_path=path, dataset=DATASET, render=True,
                                    apply_distortion=True,
                                    output_dir=OUT / label.split()[1].replace("=", ""))
        obs, info = env.reset(options={"state": state, "randomize_objects": False})
        sim = env.render("central")
        sim_bgr = cv2.cvtColor(sim, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(OUT / ("sim_%s.png" % label.split()[1].replace("=", ""))), sim_bgr)
        forward = edge_score(real, sim_bgr)
        backward = edge_score(sim_bgr, real)
        rows.append((label, forward, backward))
        print("  %-28s %12.2f %12.2f" % (label, forward, backward))
        env.close()

    rows.sort(key=lambda r: -(r[1] + r[2]))
    print()
    print("best: %s" % rows[0][0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
