#!/usr/bin/env python3
"""Is the pack actually where I put it, and does it render?"""
import sys
from pathlib import Path

import numpy as np
import yaml
import cv2

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import align_with_dataset as align
import dataset_io
import sim_env

SUCCESS = Path("/workspace/shared/new_program_qiuzhi/without_tactile/success_episode/"
               "pick_up_the_tissue_pack_and_place_it_on_the_right_side_"
               "20260908_merged_all_108eps")
OUT = ROOT / "outputs/check_flex_render"
import mujoco
OUT.mkdir(parents=True, exist_ok=True)
config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
path = OUT / "scene.yaml"
path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
surface = config["table"]["surface_z"]
ts = config["table"]["size"]
tcm = [ts[0] * 100.0, ts[1] * 100.0]

data, order = dataset_io.load(SUCCESS, fields=("observation.state",))
env = sim_env.TissueSceneEnv(config_path=path, dataset=SUCCESS, render=True,
                             output_dir=OUT)
model, mdata = env.model, env.data
env.reset(options={"state": data[0]["observation.state"][0], "randomize_objects": False})

print("flex bodies in the model:")
for name in env.box_free:
    fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, name + "_soft")
    print("   %-18s flex id %s  box_free=%s" % (name, fid, env.box_free[name]))

bags = align.measure_bags(align.dataset_frame(SUCCESS, 0, 0), config)
print("\ndetections:", [b["box_centre_cm"] for b in bags])
for box, bag in zip(config["boxes"], bags):
    bx, by = bag["box_centre_cm"]
    adr, dof = env.box_free[box["name"]]
    mdata.qpos[adr + 0] = bx / 100.0 - tcm[0] / 200.0
    mdata.qpos[adr + 1] = by / 100.0 - tcm[1] / 200.0
    mdata.qpos[adr + 2] = surface + box["size"][2] / 2.0 + 0.002
    mdata.qpos[adr + 3] = 1.0
    mdata.qpos[adr + 4:adr + 7] = 0.0
    mdata.qvel[dof:dof + 6] = 0.0
mujoco.mj_forward(model, mdata)

for name in env.box_free:
    fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, name + "_soft")
    adr = int(model.flex_vertadr[fid])
    num = int(model.flex_vertnum[fid])
    c = mdata.flexvert_xpos[adr:adr + num].mean(0)
    print("   %-18s flex centroid world (%.3f, %.3f, %.3f) = table cm (%.1f, %.1f)"
          % (name, c[0], c[1], c[2], (c[0] + tcm[0] / 200.0) * 100.0,
             (c[1] + tcm[1] / 200.0) * 100.0))
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    print("      geom '%s' exists: %s   flex has %d verts, %d elems"
          % (name, gid, num, int(model.flex_elemnum[fid])))

for camera in ("top", "left"):
    try:
        img = env.render(camera)
        arr = np.asarray(img)
        cv2.imwrite(str(OUT / ("render_%s.png" % camera)),
                    cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        print("rendered %s: shape %s  mean %.1f" % (camera, arr.shape, arr.mean()))
    except Exception as exc:
        print("render %s failed: %s" % (camera, exc))
env.close()
print("wrote", OUT)
