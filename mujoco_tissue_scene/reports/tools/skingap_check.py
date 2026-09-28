#!/usr/bin/env python3
"""Why did every segment report a 0.0000 skin gap?  Print per-geom distances.

Printed at the reset state and then with the arm commanded to a raised posture, so at least
one of the two must be far from the table.  If a geom reports 0 in both, it is not fingertip
skin and the statistic was polluted by it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import dataset_io
import sim_env

SUCCESS = Path("/workspace/shared/new_program_qiuzhi/without_tactile/success_episode/"
               "pick_up_the_tissue_pack_and_place_it_on_the_right_side_"
               "20260908_merged_all_108eps")
OUT = ROOT / "outputs/skingap_check"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    import mujoco
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    p = OUT / "scene.yaml"
    p.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    data, order = dataset_io.load(SUCCESS, fields=("observation.state",))
    start = np.asarray(data[0]["observation.state"][0], dtype=float).copy()
    env = sim_env.TissueSceneEnv(config_path=p, dataset=SUCCESS, render=False,
                                output_dir=OUT)
    model, mdata = env.model, env.data
    env.reset(options={"state": start, "randomize_objects": False})
    tg = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "black_table")
    print("table geom id %d name=%s" % (tg, model.geom(tg).name))
    hand_gids = [g for g in range(model.ngeom)
                 if (model.body(int(model.geom_bodyid[g])).name or "").startswith("hand_")]
    print("%d geoms whose body starts with 'hand_'" % len(hand_gids))
    print()
    print("  %-32s %-26s %8s %10s" % ("geom", "body", "contype", "dist to table"))
    mujoco.mj_forward(model, mdata)
    for g in hand_gids:
        d = mujoco.mj_geomDistance(model, mdata, g, tg, 2.0, None)
        print("  %-32s %-26s %8d %10.4f"
              % (model.geom(g).name, model.body(int(model.geom_bodyid[g])).name,
                 model.geom_contype[g], d))
    # now raise the arm well clear of the table and repeat
    up = start.copy()
    up[2] += 0.4                       # lift joint3
    up[1] -= 0.3
    for _ in range(90):
        env.step(up)
    print()
    print("  after commanding a raised posture (joint2 -0.3, joint3 +0.4):")
    print("  %-32s %-26s %10s" % ("geom", "body", "dist to table"))
    for g in hand_gids:
        d = mujoco.mj_geomDistance(model, mdata, g, tg, 2.0, None)
        print("  %-32s %-26s %10.4f"
              % (model.geom(g).name, model.body(int(model.geom_bodyid[g])).name, d))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
