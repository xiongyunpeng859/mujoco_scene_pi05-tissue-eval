#!/usr/bin/env python3
"""How low does the hand ever get?  A scoop grasp needs the fingertips at the
tabletop at the grasp moment.  If the lowest fingertip is 10 cm up, the arm base
is mounted too high and the hand can never reach the pack.
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
OUT = ROOT / "outputs/base_height"
THUMB = "hand_L_thumb_tip"
FINGERS = ["hand_L_index_tip", "hand_L_middle_tip", "hand_L_ring_tip", "hand_L_pinky_tip"]
PITCH = [8, 10, 11, 13, 15]


def main() -> int:
    import mujoco
    OUT.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    surface = config["table"]["surface_z"]
    print("arm base position in config: %s" % config["arm"]["position"])
    print("table surface_z: %.4f" % surface)
    print()
    model = mujoco.MjModel.from_xml_path(str(scene.build(
        ROOT / "configs/scene.yaml", OUT / "ref.xml", with_hand=True)))
    data = mujoco.MjData(model)
    ra = np.array(layout.joint_ids(model, mujoco))
    fingers = [model.body(n).id for n in FINGERS]
    thumb = model.body(THUMB).id
    dataset, order = dataset_io.load(SUCCESS)
    print("  %-4s %10s %8s %10s %8s %10s" %
          ("ep", "min tip z", "at frm", "tip z@close", "close frm", "tip lowest - table"))
    lows = []
    for e in range(12):
        st, ac = dataset[e]["observation.state"], dataset[e]["action"]
        f = np.where((np.abs(ac[:, PITCH]) > 0.35).all(axis=1))[0]
        zs = []
        for t in range(len(st)):
            data.qpos[ra] = st[t]
            mujoco.mj_kinematics(model, data)
            zs.append(float(np.mean([data.xpos[i][2] for i in fingers])))
        zs = np.array(zs)
        lo = int(np.argmin(zs))
        cf = int(f[0]) if len(f) else -1
        lows.append(zs[lo] - surface)
        print("  %-4d %10.4f %8d %10.4f %8d %10.4f"
              % (e, zs[lo], lo, (zs[cf] if cf >= 0 else float("nan")), cf,
                 zs[lo] - surface))
    lows = np.array(lows)
    print()
    print("  lowest fingertip above the tabletop: median %.4f m  min %.4f  max %.4f"
          % (np.median(lows), lows.min(), lows.max()))
    print("  a scoop grasp needs the fingertips at roughly 0.000-0.010 m.")
    print("  implied base-height correction: %.4f m (lower the base by this)"
          % float(np.median(lows) - 0.005))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
