#!/usr/bin/env python3
"""Open my eyes: at the real release frame, was the real hand over the box?

The recorded trajectory is a human demonstration that closed the loop visually, so
whatever the real hand did at release is ground truth.  Project the sim's hand position
and the box onto the real frame at that instant and look.
"""
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
OUT = ROOT / "outputs/release_look"
THUMB = "hand_L_thumb_tip"
FINGERS = ["hand_L_index_tip", "hand_L_middle_tip", "hand_L_ring_tip", "hand_L_pinky_tip"]
PITCH = [8, 10, 11, 13, 15]


def main() -> int:
    import mujoco
    OUT.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    table = measure_layout.TableFrame(config)
    ts = config["table"]["size"]
    tray = config["tray"]
    model = mujoco.MjModel.from_xml_path(str(scene.build(
        ROOT / "configs/scene.yaml", OUT / "ref.xml", with_hand=True)))
    data = mujoco.MjData(model)
    ra = np.array(layout.joint_ids(model, mujoco))
    thumb = model.body(THUMB).id
    fingers = [model.body(n).id for n in FINGERS]
    dataset, order = dataset_io.load(SUCCESS)

    for e in (0, 11, 16):
        st, ac = dataset[e]["observation.state"], dataset[e]["action"]
        closed = (np.abs(ac[:, PITCH]) > 0.35).all(axis=1)
        idx = np.where(closed)[0]
        open_start = int(idx[0])
        open_end = int(idx[-1])
        release = min(open_end + 1, len(ac) - 1)
        # also the last frame of the carry, while still closed
        print("ep%-3d closed frames %d..%d ; release frame %d of %d"
              % (e, open_start, open_end, release, len(ac)))
        for label, fr in (("carry_end", open_end), ("release", release)):
            img = align.dataset_frame(SUCCESS, e, fr)
            vis = img.copy()
            # box: inner floor of the tray, from the config pose
            tyaw = float(tray["yaw"])
            hx = tray["size"][0] / 2.0 - tray["wall_thickness"]
            hy = tray["size"][1] / 2.0 - tray["wall_thickness"]
            local = np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]])
            rot = np.array([[np.cos(tyaw), -np.sin(tyaw)], [np.sin(tyaw), np.cos(tyaw)]])
            c = np.array(tray["center"][:2])
            quad = np.array([table.project([*(rot @ k + c), table.surface + 0.008])
                             for k in local], np.int32)
            cv2.polylines(vis, [quad], True, (255, 0, 255), 2)
            # sim hand position at this frame
            data.qpos[ra] = st[fr]
            mujoco.mj_kinematics(model, data)
            g = 0.5 * (data.xpos[thumb]
                       + np.mean([data.xpos[i] for i in fingers], axis=0))
            gp = table.project([g[0], g[1], table.surface]).astype(int)
            cv2.drawMarker(vis, tuple(gp), (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
            hx_cm = (g[0] + ts[0] / 2) * 100.0
            hy_cm = (g[1] + ts[1] / 2) * 100.0
            bx_cm = (c[0] + ts[0] / 2) * 100.0
            by_cm = (c[1] + ts[1] / 2) * 100.0
            d = np.hypot(hx_cm - bx_cm, hy_cm - by_cm)
            cv2.putText(vis, "hand (%.1f,%.1f) d=%.1fcm" % (hx_cm, hy_cm, d),
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            cv2.imwrite(str(OUT / ("ep%02d_%s.png" % (e, label))), vis)
            print("      %-9s hand table cm (%.1f, %.1f)  box (%.1f, %.1f)  d=%.1f cm"
                  % (label, hx_cm, hy_cm, bx_cm, by_cm, d))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
