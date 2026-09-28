#!/usr/bin/env python3
"""Quantitative multi-frame arbitration between two candidate bases.

The arm's rendered colour now matches the real one (dark links), which was what broke
earlier image comparisons.  So the arm's silhouette can be compared directly: the
score is the grey difference between the real frame and the render, measured only
where the simulated arm actually is (the difference between a render with the arm and
one without), averaged over several frames of a real episode.
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
import dataset_io                       # noqa: E402
import scene                            # noqa: E402
import sim_env                          # noqa: E402

SUCCESS = Path("/workspace/shared/new_program_qiuzhi/without_tactile/success_episode/"
               "pick_up_the_tissue_pack_and_place_it_on_the_right_side_"
               "20260908_merged_all_108eps")
OUT = ROOT / "outputs/arbitrate_quant"
FRAMES = (60, 90, 120, 150, 180, 210)
IDLE = (400, 640, 240, 480)


def render_no_arm(config_path, out_dir):
    xml = scene.build(config_path, Path(out_dir) / "noarm.xml", with_hand=False)
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        renderer.update_scene(data, camera="central")
        return cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)


def render_with_arm(config_path, out_dir, state):
    env = sim_env.TissueSceneEnv(config_path=config_path, dataset=SUCCESS, render=True,
                                 output_dir=out_dir)
    env.reset(options={"state": state, "randomize_objects": False})
    image = cv2.cvtColor(env.render("central"), cv2.COLOR_RGB2BGR)
    env.close()
    return image


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    episodes, order = dataset_io.load(SUCCESS)
    episode = order[0]
    states = episodes[episode]["observation.state"]
    reals = {f: align.dataset_frame(SUCCESS, episode, f) for f in FRAMES}
    for f, img in reals.items():
        cv2.imwrite(str(OUT / ("real_%03d.png" % f)), img)
    grey_real = {f: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
                 for f, img in reals.items()}
    x0, x1, y0, y1 = IDLE

    candidates = [("A_graspfit_18_10_90", 18.0, 10.0, 1.5708),
                  ("B_kinematic_34_24_60", 34.0, 24.0, 1.0472),
                  ("C_tape_25_10_90", 25.0, 10.0, 1.5708)]
    print("  %-24s %s" % ("candidate", "  ".join("t=%-4d" % f for f in FRAMES) + "   mean"))
    summary = []
    for label, bx, by, byaw in candidates:
        config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
        config["measured_layout"]["arm_mount_xy_cm_from_left_bottom"] = [bx, by]
        config["arm"]["euler"][2] = byaw
        cdir = OUT / label
        cdir.mkdir(parents=True, exist_ok=True)
        path = cdir / "scene.yaml"
        path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
        no_arm = render_no_arm(path, cdir)
        errors = []
        for f in FRAMES:
            with_arm = render_with_arm(path, cdir, states[f])
            difference = np.abs(with_arm.astype(np.float32)
                                - no_arm.astype(np.float32)).max(2)
            mask = (difference > 20)
            mask[y0:y1, x0:x1] = False
            mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            if mask.sum() < 100:
                errors.append(float("nan"))
                continue
            sim_grey = cv2.cvtColor(with_arm, cv2.COLOR_BGR2GRAY).astype(np.float32)
            errors.append(float(np.abs(grey_real[f][mask] - sim_grey[mask]).mean()))
            if f == 150:
                cv2.imwrite(str(cdir / "sim_150.png"), with_arm)
        mean_error = float(np.nanmean(errors))
        summary.append((label, mean_error, errors))
        print("  %-24s %s   %.2f"
              % (label, "  ".join("%-6.1f" % e for e in errors), mean_error))
    summary.sort(key=lambda s: s[1])
    print()
    print("  best base by arm-silhouette agreement: %s (%.2f)" % (summary[0][0], summary[0][1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
