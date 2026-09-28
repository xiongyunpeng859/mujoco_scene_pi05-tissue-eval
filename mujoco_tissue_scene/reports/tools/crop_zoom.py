#!/usr/bin/env python3
"""Zoom the raw frame around the box and around the FK closure point."""
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
OUT = ROOT / "outputs/detector_debug"
PITCH = [8, 10, 11, 13, 15]
THUMB = "hand_L_thumb_tip"
FINGERS = ["hand_L_index_tip", "hand_L_middle_tip", "hand_L_ring_tip", "hand_L_pinky_tip"]


def main() -> int:
    import mujoco
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    table = measure_layout.TableFrame(config)
    dataset, order = dataset_io.load(SUCCESS)
    ref = mujoco.MjModel.from_xml_path(str(scene.build(
        ROOT / "configs/scene.yaml", OUT / "ref.xml", with_hand=True)))
    rd = mujoco.MjData(ref)
    ra = np.array(layout.joint_ids(ref, mujoco))
    thumb, fingers = ref.body(THUMB).id, [ref.body(n).id for n in FINGERS]

    for episode in (2, 0):
        frame = align.dataset_frame(SUCCESS, episode, 0)
        # where the box is, in pixels
        tray = config["tray"]
        tyaw = float(tray["yaw"])
        hx = tray["size"][0] / 2.0 - tray["wall_thickness"]
        hy = tray["size"][1] / 2.0 - tray["wall_thickness"]
        local = np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]])
        rot = np.array([[np.cos(tyaw), -np.sin(tyaw)], [np.sin(tyaw), np.cos(tyaw)]])
        c = np.array(tray["center"][:2])
        quad = np.array([table.project([*(rot @ k + c), table.surface + 0.008])
                         for k in local])
        x0, y0 = quad.min(0).astype(int) - 40
        x1, y1 = quad.max(0).astype(int) + 40
        crop = frame[max(0, y0):y1, max(0, x0):x1]
        if crop.size:
            big = cv2.resize(crop, None, fx=2.6, fy=2.6, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(OUT / ("ep%02d_zoom_box.png" % episode)), big)
        # where the fingers closed
        st, ac = dataset[episode]["observation.state"], dataset[episode]["action"]
        f = np.where(np.abs(ac[:, PITCH]).mean(1) > 0.35)[0]
        info = "step %s" % (f[0] if len(f) else "none")
        if len(f):
            rd.qpos[ra] = st[int(f[0])]
            mujoco.mj_kinematics(ref, rd)
            g = 0.5 * (rd.xpos[thumb] + np.mean([rd.xpos[i] for i in fingers], axis=0))
            gp = table.project([g[0], g[1], table.surface]).astype(int)
            print("ep%d FK closure px %s -> table cm (%.1f, %.1f)  %s"
                  % (episode, gp, (g[0] + config["table"]["size"][0] / 2) * 100,
                     (g[1] + config["table"]["size"][1] / 2) * 100, info))
            crop2 = frame[max(0, gp[1] - 120):gp[1] + 120,
                          max(0, gp[0] - 120):gp[0] + 120]
            if crop2.size:
                big2 = cv2.resize(crop2, None, fx=2.6, fy=2.6,
                                  interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(OUT / ("ep%02d_zoom_grip.png" % episode)), big2)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
