#!/usr/bin/env python3
"""Base search scored only where the simulated arm actually is.

The whole-image difference is dominated by the static scene.  The arm mask is
obtained by rendering the same scene with the arm body removed (identical for every
candidate, so it is computed once), and the score is the difference inside that mask
only -- so the arm's position is what decides.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import base_consensus as bc             # noqa: E402
import sim_env                          # noqa: E402

OUT = ROOT / "outputs/base_arm_mask"
IDLE = bc.IDLE


def render_state(config_path, state, out_dir, drop_arm=False):
    """Render the central camera.  drop_arm uses scene.build's own with_hand=False
    switch, which leaves no dangling joints for the equality or actuator sections."""
    import scene as scene_module
    if drop_arm:
        xml = scene_module.build(config_path, Path(out_dir) / "scene_noarm.xml",
                                 with_hand=False)
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(xml))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        with mujoco.Renderer(model, height=480, width=640) as renderer:
            renderer.update_scene(data, camera="central")
            return cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
    env = sim_env.TissueSceneEnv(config_path=config_path, dataset=bc.SUCCESS_DATASET,
                                 render=True, output_dir=out_dir)
    env.reset(options={"state": state, "randomize_objects": False})
    image = cv2.cvtColor(env.render("central"), cv2.COLOR_RGB2BGR)
    env.close()
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--stage", choices=["coarse", "fine"], default="coarse")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    median, states, count = bc.consensus_first_frames(bc.SUCCESS_DATASET, args.limit)
    median = median.astype(np.uint8)
    start_state = np.median(states, axis=0)
    cv2.imwrite(str(OUT / "consensus.png"), median)
    print("consensus from %d first frames; arm pose spread %.4f rad"
          % (count, float(np.abs(states - start_state).max(0)[:6].max())))

    grey_real = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY).astype(np.float32)
    x0, x1, y0, y1 = IDLE
    idle_mask = np.zeros(grey_real.shape, bool)
    idle_mask[y0:y1, x0:x1] = True

    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    base_path = OUT / "base.yaml"
    base_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    no_arm = render_state(base_path, start_state, OUT, drop_arm=True)
    cv2.imwrite(str(OUT / "no_arm.png"), no_arm)
    print("rendered the arm-free scene (cached for all candidates)")

    if args.stage == "coarse":
        candidates = [(x, y, yaw) for x in (14.0, 18.0, 22.0, 26.0)
                      for y in (2.0, 6.0, 10.0, 14.0)
                      for yaw in (1.45, 1.5708, 1.70)]
    else:
        candidates = [(x, y, 1.5708) for x in np.arange(16.0, 23.1, 1.0)
                      for y in np.arange(2.0, 10.1, 1.0)]

    print()
    print("  %-24s %11s %10s" % ("candidate", "arm-mask err", "arm px"))
    rows = []
    for x_cm, y_cm, yaw in candidates:
        cfg = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
        cfg["measured_layout"]["arm_mount_xy_cm_from_left_bottom"] = [x_cm, y_cm]
        cfg["arm"]["euler"][2] = yaw
        tag = "x%05.1f_y%05.1f_a%06.4f" % (x_cm, y_cm, yaw)
        path = OUT / tag / "scene.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
        with_arm = render_state(path, start_state, path.parent)
        difference = np.abs(with_arm.astype(np.float32) - no_arm.astype(np.float32)).max(2)
        mask = (difference > 25) & (~idle_mask)
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        if mask.sum() < 200:
            error = 999.0
        else:
            sim_grey = cv2.cvtColor(with_arm, cv2.COLOR_BGR2GRAY).astype(np.float32)
            error = float(np.abs(grey_real[mask] - sim_grey[mask]).mean())
        rows.append((x_cm, y_cm, yaw, error, int(mask.sum())))
        print("  x=%5.1f y=%5.1f a=%.4f %11.2f %10d" % (x_cm, y_cm, yaw, error, mask.sum()))

    rows.sort(key=lambda r: r[3])
    print()
    best = rows[0]
    print("  best: x=%.1f y=%.1f yaw=%.4f -> arm-mask error %.2f"
          % (best[0], best[1], best[2], best[3]))
    print("  runners-up: %s" % [(round(r[0], 1), round(r[1], 1), round(r[2], 3),
                                 round(r[3], 2)) for r in rows[1:4]])
    (OUT / "result.json").write_text(json.dumps(
        [{"x_cm": r[0], "y_cm": r[1], "yaw": r[2], "arm_mask_error": r[3],
          "arm_pixels": r[4]} for r in rows], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
