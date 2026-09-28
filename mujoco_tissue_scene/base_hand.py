#!/usr/bin/env python3
"""Pin the base by matching the HAND, which is the sharpest feature available.

Every episode starts from the same arm pose, so the median of many first frames shows
the hand as a bright silver blob against the dark table -- a small, high-contrast,
easily segmented target.  Projecting the simulated hand at q = 0 and matching that
blob is a sharp 3-DOF fit, unlike whole-arm shape or colour comparisons.

Projection is analytic (no rendering), so a dense grid costs nothing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import dataset_io                       # noqa: E402
import measure_layout                   # noqa: E402
import scene                            # noqa: E402

SUCCESS = Path("/workspace/shared/new_program_qiuzhi/without_tactile/success_episode/"
               "pick_up_the_tissue_pack_and_place_it_on_the_right_side_"
               "20260908_merged_all_108eps")
OUT = ROOT / "outputs/base_hand"
HAND_POINTS = ["static_omnihand_reference", "hand_L_thumb_tip", "hand_L_index_tip",
               "hand_L_middle_tip", "hand_L_ring_tip", "hand_L_pinky_tip"]


def main() -> int:
    import mujoco
    OUT.mkdir(parents=True, exist_ok=True)
    consensus = cv2.imread(str(OUT.parent / "base_arm_mask/consensus.png"))
    if consensus is None:
        raise SystemExit("consensus image missing")
    grey = cv2.cvtColor(consensus, cv2.COLOR_BGR2GRAY)
    # the hand is the bright silver blob in the lower-left; the tray is further right
    region = np.zeros_like(grey, bool)
    region[230:400, 0:170] = True
    blob = (grey > 140) & region
    blob = cv2.morphologyEx(blob.astype(np.uint8), cv2.MORPH_OPEN,
                            np.ones((3, 3), np.uint8)) > 0
    ys, xs = np.where(blob)
    if len(xs) < 50:
        raise SystemExit("hand blob not found (only %d px)" % len(xs))
    real_centre = np.array([xs.mean(), ys.mean()])
    print("real hand blob: %d px, centroid (%.1f, %.1f), bbox x %d..%d y %d..%d"
          % (len(xs), real_centre[0], real_centre[1], xs.min(), xs.max(),
             ys.min(), ys.max()))
    overlay = consensus.copy()
    cv2.circle(overlay, tuple(real_centre.astype(int)), 4, (0, 0, 255), -1)
    cv2.rectangle(overlay, (xs.min(), ys.min()), (xs.max(), ys.max()), (0, 255, 0), 1)
    cv2.imwrite(str(OUT / "hand_blob.png"), overlay)

    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    surface = config["table"]["surface_z"]
    table = config["table"]["size"]
    ref = measure_layout.TableFrame(config)
    xml = scene.build(ROOT / "configs/scene.yaml", OUT / "scene.xml", with_hand=True)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mount = model.body("qiuzhi_arm_mount").id
    ids = [model.body(n).id for n in HAND_POINTS]
    adr = [model.joint("joint%d" % i).qposadr[0] for i in range(1, 7)]

    def image_error(x_cm, y_cm, yaw):
        model.body_pos[mount] = [x_cm / 100.0 - table[0] / 2.0,
                                 y_cm / 100.0 - table[1] / 2.0, surface]
        model.body_quat[mount] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        data.qpos[adr] = 0.0
        mujoco.mj_kinematics(model, data)
        projected = [ref.project(data.xpos[i]) for i in ids]
        centre = np.mean(projected, axis=0)
        return float(np.linalg.norm(centre - real_centre)), centre

    print()
    print("  %-30s %10s %s" % ("base / yaw", "px error", "projected hand centre"))
    rows = []
    for x_cm in np.arange(12.0, 48.1, 2.0):
        for y_cm in np.arange(-4.0, 32.1, 2.0):
            for yaw in np.radians(np.arange(0.0, 181.0, 7.5)):
                error, centre = image_error(float(x_cm), float(y_cm), float(yaw))
                rows.append({"x": float(x_cm), "y": float(y_cm), "yaw": float(yaw),
                             "px_error": error,
                             "centre": [float(centre[0]), float(centre[1])]})
    rows.sort(key=lambda r: r["px_error"])
    for r in rows[:12]:
        print("  x=%5.1f y=%5.1f yaw=%6.1f (%5.1f deg) %10.1f  (%.0f, %.0f)"
              % (r["x"], r["y"], r["yaw"], np.degrees(r["yaw"]), r["px_error"],
                 r["centre"][0], r["centre"][1]))
    best = rows[0]
    print()
    print("  BEST x=%.1f y=%.1f yaw=%.1f deg -> %.1f px" %
          (best["x"], best["y"], np.degrees(best["yaw"]), best["px_error"]))
    for label, x, y, deg in (("grasp fit (18,10,90)", 18.0, 10.0, 90.0),
                             ("tape (25,10,90)", 25.0, 10.0, 90.0),
                             ("kinematic (34,24,60)", 34.0, 24.0, 60.0),
                             ("overlap (42,20,80)", 42.0, 20.0, 80.0)):
        error, _ = image_error(x, y, np.radians(deg))
        print("  %-24s -> %6.1f px" % (label, error))
    (OUT / "result.json").write_text(json.dumps(rows[:400], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
