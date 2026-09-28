#!/usr/bin/env python3
"""Is this a SCOOP (fingers under the pack) or a side PINCH?

The user: four nearly-straight fingers scoop UNDER the pack and the thumb closes on
top; the pack never touches the palm.  A scoop is supported by gravity and cannot
slip, a side pinch is held only by friction and can.  So compare the fingertip height
at the closure frame with the pack's own height range.
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
OUT = ROOT / "outputs/grasp_geometry"
THUMB = "hand_L_thumb_tip"
FINGERS = ["hand_L_index_tip", "hand_L_middle_tip", "hand_L_ring_tip", "hand_L_pinky_tip"]
MIDS = ["hand_L_index_dip", "hand_L_middle_dip", "hand_L_ring_dip", "hand_L_pinky_dip"]
PITCH = [8, 10, 11, 13, 15]


def main() -> int:
    import mujoco
    OUT.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    surface = config["table"]["surface_z"]
    pack_h = config["boxes"][0]["size"][2]
    model = mujoco.MjModel.from_xml_path(str(scene.build(
        ROOT / "configs/scene.yaml", OUT / "ref.xml", with_hand=True)))
    data = mujoco.MjData(model)
    ra = np.array(layout.joint_ids(model, mujoco))
    thumb = model.body(THUMB).id
    fingers = [model.body(n).id for n in FINGERS]
    mids = [model.body(n).id for n in MIDS]
    dataset, order = dataset_io.load(SUCCESS)
    print("table surface z %.3f m; pack occupies %.3f .. %.3f m above the table"
          % (surface, 0.0, pack_h))
    print()
    print("  %-4s %8s %8s %8s %8s | %s" %
          ("ep", "tip z", "mid z", "thumb z", "spread",
           "verdict (tip height above the tabletop)"))
    for e in range(12):
        st, ac = dataset[e]["observation.state"], dataset[e]["action"]
        f = np.where((np.abs(ac[:, PITCH]) > 0.35).all(axis=1))[0]
        if not len(f):
            continue
        data.qpos[ra] = st[int(f[0])]
        mujoco.mj_kinematics(model, data)
        tips = np.array([data.xpos[i] for i in fingers])
        tip = tips.mean(0)
        mid = np.array([data.xpos[i] for i in mids]).mean(0)
        th = data.xpos[thumb]
        spread = float(np.linalg.norm(tip[:2] - th[:2]))
        h = tip[2] - surface
        # where is the fingertip relative to the pack volume (0..pack_h)?
        if h < 0.002:
            verdict = "ON/under the table (%.1f mm) -> scoop" % (h * 1000)
        elif h < pack_h * 0.35:
            verdict = "%.1f cm up: inside the pack's lower third -> scoop-like" % (h * 100)
        elif h < pack_h:
            verdict = "%.1f cm up: mid-pack -> SIDE PINCH (friction only)" % (h * 100)
        else:
            verdict = "%.1f cm up: ABOVE the pack -> would miss it" % (h * 100)
        print("  %-4d %8.3f %8.3f %8.3f %8.3f | %s"
              % (e, h, mid[2] - surface, th[2] - surface, spread, verdict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
