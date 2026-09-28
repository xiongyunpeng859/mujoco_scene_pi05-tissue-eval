#!/usr/bin/env python3
"""Does the simulated arm actually look like the real one?

Renders the same episode frame with the arm and with the arm removed, so the arm's
pixels are known exactly, then reports what colour those pixels are in both the
render and the real frame.  Also saves a crop for visual comparison.
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
import scene
import sim_env

SUCCESS = Path("/workspace/shared/new_program_qiuzhi/without_tactile/success_episode/"
               "pick_up_the_tissue_pack_and_place_it_on_the_right_side_"
               "20260908_merged_all_108eps")
OUT = ROOT / "outputs/arm_look"


def main() -> int:
    import mujoco
    OUT.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    episodes, order = dataset_io.load(SUCCESS)
    episode = order[0]
    state = episodes[episode]["observation.state"]

    for frame in (120, 150, 180):
        real = align.dataset_frame(SUCCESS, episode, frame)
        env = sim_env.TissueSceneEnv(dataset=SUCCESS, render=True, output_dir=OUT)
        env.reset(options={"state": state[frame], "randomize_objects": False})
        with_arm = cv2.cvtColor(env.render("central"), cv2.COLOR_RGB2BGR)
        env.close()
        xml = scene.build(ROOT / "configs/scene.yaml", OUT / "noarm.xml", with_hand=False)
        model = mujoco.MjModel.from_xml_path(str(xml))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        with mujoco.Renderer(model, height=480, width=640) as renderer:
            renderer.update_scene(data, camera="central")
            no_arm = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)

        diff = np.abs(with_arm.astype(np.float32) - no_arm.astype(np.float32)).max(2)
        mask = diff > 20
        # split the mask into the light part (hand/housing) and the rest (links)
        sim_grey = cv2.cvtColor(with_arm, cv2.COLOR_BGR2GRAY)
        real_grey = cv2.cvtColor(real, cv2.COLOR_BGR2GRAY)
        print("frame %d: sim-arm pixels %d" % (frame, mask.sum()))
        if mask.sum():
            print("   sim  arm: mean BGR %s   grey p10/p50/p90 = %.0f/%.0f/%.0f"
                  % (np.round(with_arm[mask].mean(0), 1).tolist(),
                     *np.percentile(sim_grey[mask], [10, 50, 90])))
            print("   real arm: mean BGR %s   grey p10/p50/p90 = %.0f/%.0f/%.0f"
                  % (np.round(real[mask].mean(0), 1).tolist(),
                     *np.percentile(real_grey[mask], [10, 50, 90])))
            print("   -> real objects in the sim arm mask: %.0f%% are bright (>50)"
                  % (100 * (real_grey[mask] > 50).mean()))
        cv2.imwrite(str(OUT / ("side_%d.png" % frame)), np.hstack([real, with_arm]))
    print("saved side_*.png (REAL | SIM) in %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
