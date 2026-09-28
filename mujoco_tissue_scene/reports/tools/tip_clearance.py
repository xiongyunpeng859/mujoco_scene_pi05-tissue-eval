#!/usr/bin/env python3
"""Exact fingertip-geom-to-table distance, so a body-origin offset cannot fool us.

The body origin of a finger link is at its joint, not at the skin.  mj_geomDistance
measures the real surface gap between the finger geoms and the table geom.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import action_layout as layout
import dataset_io
import scene

SUCCESS = Path("/workspace/shared/new_program_qiuzhi/without_tactile/success_episode/"
               "pick_up_the_tissue_pack_and_place_it_on_the_right_side_"
               "20260908_merged_all_108eps")
OUT = ROOT / "outputs/tip_clearance"
PITCH = [8, 10, 11, 13, 15]


def main() -> int:
    import mujoco
    OUT.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(scene.build(
        ROOT / "configs/scene.yaml", OUT / "ref.xml", with_hand=True)))
    data = mujoco.MjData(model)
    ra = np.array(layout.joint_ids(model, mujoco))
    table_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "black_table")
    hand_gids = []
    for g in range(model.ngeom):
        b = model.body(int(model.geom_bodyid[g])).name or ""
        if b.startswith("hand_") and ("_tip" in b or "_dip" in b or "_distal" in b):
            hand_gids.append(g)
    print("table geom id %d ; %d fingertip geoms on the hand" % (table_gid, len(hand_gids)))
    if not hand_gids:
        print("no tip geoms matched; body names:")
        for b in range(model.nbody):
            n = model.body(b).name or ""
            if n.startswith("hand_") and ("tip" in n or "dip" in n):
                print("   ", n)
        return 1
    for g in hand_gids[:3]:
        print("   geom %d body %s rbound %.4f"
              % (g, model.body(int(model.geom_bodyid[g])).name, model.geom_rbound[g]))

    dataset, order = dataset_io.load(SUCCESS)
    print()
    print("  %-4s %14s %8s %14s" %
          ("ep", "min gap (m)", "at frm", "gap at close frm"))
    gaps_all = []
    for e in range(8):
        st, ac = dataset[e]["observation.state"], dataset[e]["action"]
        f = np.where((np.abs(ac[:, PITCH]) > 0.35).all(axis=1))[0]
        cf = int(f[0]) if len(f) else -1
        best, at = 9.9, -1
        gap_at_close = float("nan")
        for t in range(len(st)):
            data.qpos[ra] = st[t]
            mujoco.mj_kinematics(model, data)
            d = min(mujoco.mj_geomDistance(model, data, g, table_gid, 1.0, None)
                    for g in hand_gids)
            if d < best:
                best, at = float(d), t
            if t == cf:
                gap_at_close = float(d)
        gaps_all.append(best)
        print("  %-4d %14.4f %8d %14.4f" % (e, best, at, gap_at_close))
    gaps_all = np.array(gaps_all)
    print()
    print("  smallest fingertip-to-table gap anywhere: median %.4f m  min %.4f  max %.4f"
          % (np.median(gaps_all), gaps_all.min(), gaps_all.max()))
    print("  (a scoop needs this to reach ~0.000-0.010 m)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
