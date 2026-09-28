#!/usr/bin/env python3
"""Is the -5.5 cm x offset a DETECTOR bias or a CAMERA misalignment?

`validate3.py` compared a pack's TRUE sampled position against what measure_bags reported
from the rendered frame and found x consistently ~5 cm low.  Those two explanations have very
different consequences:

  * detector bias  -> the rendered world is consistent; only the DETECTOR's centre estimate is
                      off (it uses the contact edge + an assumed depth), and any controller
                      that uses the TRUE pose is unaffected;
  * camera offset  -> the rendered images themselves are inconsistent with the world geometry,
                      which WOULD corrupt any policy trained on them.

Separate them: place the pack at a known world position, project that exact position into the
image with the calibrated TableFrame, and see whether the pack's actual pixels are there.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import align_with_dataset as align
import dataset_io
import measure_layout
import mujoco
import sim_env

SUCCESS = Path("/workspace/shared/new_program_qiuzhi/without_tactile/success_episode/"
               "pick_up_the_tissue_pack_and_place_it_on_the_right_side_"
               "20260908_merged_all_108eps")
OUT = ROOT / "outputs/camera_vs_detector"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    p = OUT / "scene.yaml"
    p.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    ts = config["table"]["size"]
    surface = config["table"]["surface_z"]
    table = measure_layout.TableFrame(config)
    data, order = dataset_io.load(SUCCESS, fields=("observation.state",))
    env = sim_env.TissueSceneEnv(config_path=p, dataset=SUCCESS, render=True,
                                output_dir=OUT)
    model, mdata = env.model, env.data

    true_cm = np.array([28.5, 63.6])          # where we put the pack
    for trial in (true_cm, true_cm + np.array([-10.0, 0.0])):
        env.reset(options={"state": data[0]["observation.state"][0],
                           "randomize_objects": False})
        for name in env.box_free:
            adr, dof = env.box_free[name]
            mdata.qpos[adr + 0], mdata.qpos[adr + 1] = -2.0, -2.0
            mdata.qpos[adr + 2] = surface + 0.04
            mdata.qpos[adr + 3] = 1.0
            mdata.qpos[adr + 4:adr + 7] = 0.0
            mdata.qvel[dof:dof + 6] = 0.0
        tgt = config["boxes"][0]
        adr, dof = env.box_free[tgt["name"]]
        mdata.qpos[adr + 0] = trial[0] / 100.0 - ts[0] / 2.0
        mdata.qpos[adr + 1] = trial[1] / 100.0 - ts[1] / 2.0
        mdata.qpos[adr + 2] = surface + tgt["size"][2] / 2.0 + 0.002
        mdata.qpos[adr + 3] = 1.0
        mdata.qpos[adr + 4:adr + 7] = 0.0
        mdata.qvel[dof:dof + 6] = 0.0
        mujoco.mj_forward(model, mdata)
        for _ in range(30):
            env.step(np.asarray(data[0]["observation.state"][0], dtype=float))

        img = np.asarray(env.observation()["observation.images.top"])
        # (a) where the pack REALLY is, projected through the calibrated frame
        fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, tgt["name"] + "_soft")
        a = int(model.flex_vertadr[fid])
        n = int(model.flex_vertnum[fid])
        centroid = mdata.flexvert_xpos[a:a + n].mean(0)
        proj = table.project(centroid)
        # (b) where the pack's PIXELS actually are (blue dominance)
        grey = img.astype(np.int16)
        blue = grey[:, :, 2] - np.maximum(grey[:, :, 0], grey[:, :, 1])
        ys, xs = np.where(blue > 10)
        pix = np.array([xs.mean(), ys.mean()]) if len(xs) else np.array([np.nan, np.nan])
        # (c) what measure_bags reports
        bags = align.measure_bags(img, config)
        det = bags[0]["box_centre_cm"] if bags else None
        print("true pack (%.1f, %.1f) cm" % (trial[0], trial[1]))
        print("   flex centroid table cm : (%.2f, %.2f)"
              % ((centroid[0] + ts[0] / 2) * 100, (centroid[1] + ts[1] / 2) * 100))
        print("   projected pixel        : (%.1f, %.1f)" % (proj[0], proj[1]))
        print("   pack PIXELS centroid   : (%.1f, %.1f)   -> projection error %.1f px"
              % (pix[0], pix[1], float(np.linalg.norm(pix - proj))))
        print("   measure_bags centre    : %s   -> offset from truth %s"
              % (None if det is None else "(%.1f, %.1f)" % (det[0], det[1]),
                 "n/a" if det is None else "(%+.1f, %+.1f) cm"
                 % (det[0] - trial[0], det[1] - trial[1])))
        print()
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
