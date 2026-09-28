#!/usr/bin/env python3
"""Zoom the real frame around the FK closure point for the episodes whose grip
lands off the assumed tabletop."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml
import cv2

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import action_layout as layout
import align_with_dataset as align
import dataset_io
import measure_layout
import scene

SUCCESS = Path("/workspace/shared/new_program_qiuzhi/without_tactile/success_episode/"
               "pick_up_the_tissue_pack_and_place_it_on_the_right_side_"
               "20260908_merged_all_108eps")
OUT = ROOT / "outputs/inspect_grip"
THUMB = "hand_L_thumb_tip"
FINGERS = ["hand_L_index_tip", "hand_L_middle_tip", "hand_L_ring_tip", "hand_L_pinky_tip"]
PITCH = [8, 10, 11, 13, 15]


def main() -> int:
    import mujoco
    OUT.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    table = measure_layout.TableFrame(config)
    ts = config["table"]["size"]
    model = mujoco.MjModel.from_xml_path(str(scene.build(
        ROOT / "configs/scene.yaml", OUT / "ref.xml", with_hand=True)))
    data = mujoco.MjData(model)
    ra = np.array(layout.joint_ids(model, mujoco))
    thumb, fingers = model.body(THUMB).id, [model.body(n).id for n in FINGERS]
    dataset, order = dataset_io.load(SUCCESS)

    for episode in (2, 5, 11, 0):
        st, ac = dataset[episode]["observation.state"], dataset[episode]["action"]
        f = np.where((np.abs(ac[:, PITCH]) > 0.35).all(axis=1))[0]
        t = int(f[0]) if len(f) else 0
        data.qpos[ra] = st[t]
        mujoco.mj_kinematics(model, data)
        g = 0.5 * (data.xpos[thumb]
                   + np.mean([data.xpos[i] for i in fingers], axis=0))
        gp = table.project([g[0], g[1], table.surface]).astype(int)
        img = align.dataset_frame(SUCCESS, episode, 0)
        vis = img.copy()
        cv2.drawMarker(vis, tuple(gp), (255, 0, 0), cv2.MARKER_CROSS, 30, 3)
        # table rectangle, projected
        corners = [(-ts[0] / 2, -ts[1] / 2), (ts[0] / 2, -ts[1] / 2),
                   (ts[0] / 2, ts[1] / 2), (-ts[0] / 2, ts[1] / 2)]
        pts = np.array([table.project([c[0], c[1], table.surface]) for c in corners],
                       np.int32)
        cv2.polylines(vis, [pts], True, (0, 255, 255), 2)
        cv2.imwrite(str(OUT / ("ep%02d_full.png" % episode)), vis)
        x0, y0 = max(0, gp[0] - 150), max(0, gp[1] - 150)
        crop = vis[y0:y0 + 320, x0:x0 + 320]
        if crop.size:
            big = cv2.resize(crop, None, fx=2.4, fy=2.4, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(OUT / ("ep%02d_gripzoom.png" % episode)), big)
        print("ep%-3d closure t=%-4d grip px %s  table cm (%.1f, %.1f)"
              % (episode, t, gp, (g[0] + ts[0] / 2) * 100, (g[1] + ts[1] / 2) * 100))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
