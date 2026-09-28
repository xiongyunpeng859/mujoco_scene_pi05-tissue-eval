#!/usr/bin/env python3
"""Draw the detector's seeds, its accepted/rejected fits, the box and the FK
closure point onto the real frame so the failure is visible instead of inferred."""
from __future__ import annotations

import argparse
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
THUMB = "hand_L_thumb_tip"
FINGERS = ["hand_L_index_tip", "hand_L_middle_tip", "hand_L_ring_tip", "hand_L_pinky_tip"]
PITCH = [8, 10, 11, 13, 15]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 2])
    args = parser.parse_args()
    import mujoco
    OUT.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    tray, table_size = config["tray"], config["table"]["size"]
    table = measure_layout.TableFrame(config)
    dataset, order = dataset_io.load(SUCCESS)

    ref = mujoco.MjModel.from_xml_path(str(scene.build(
        ROOT / "configs/scene.yaml", OUT / "ref.xml", with_hand=True)))
    rd = mujoco.MjData(ref)
    ra = np.array(layout.joint_ids(ref, mujoco))
    thumb, fingers = ref.body(THUMB).id, [ref.body(n).id for n in FINGERS]

    for episode in args.episodes:
        frame = align.dataset_frame(SUCCESS, episode, 0)
        vis = frame.copy()
        # table outline (config table is centred on the world origin)
        corners = [(-table_size[0] / 2, -table_size[1] / 2),
                   (table_size[0] / 2, -table_size[1] / 2),
                   (table_size[0] / 2, table_size[1] / 2),
                   (-table_size[0] / 2, table_size[1] / 2)]
        pts = np.array([table.project([c[0], c[1], table.surface]) for c in corners],
                       np.int32)
        cv2.polylines(vis, [pts], True, (0, 255, 255), 2)
        cv2.putText(vis, "table", tuple(pts[0] + [4, -6]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # green box, both the pose we use and a fresh fit
        tyaw = float(tray.get("yaw", 0.0))
        half_x = tray["size"][0] / 2.0 - tray["wall_thickness"]
        half_y = tray["size"][1] / 2.0 - tray["wall_thickness"]
        local = np.array([[half_x, half_y], [-half_x, half_y],
                          [-half_x, -half_y], [half_x, -half_y]])
        rot = np.array([[np.cos(tyaw), -np.sin(tyaw)], [np.sin(tyaw), np.cos(tyaw)]])
        cfg_c = np.array([tray["center"][0], tray["center"][1]])
        quad = np.array([table.project([*(rot @ c + cfg_c), table.surface + 0.008])
                         for c in local], np.int32)
        cv2.polylines(vis, [quad], True, (255, 0, 255), 2)
        cv2.putText(vis, "box(used)", tuple(quad[0] + [4, -6]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        # FK closure point
        st, ac = dataset[episode]["observation.state"], dataset[episode]["action"]
        f = np.where(np.abs(ac[:, PITCH]).mean(1) > 0.35)[0]
        if len(f):
            rd.qpos[ra] = st[int(f[0])]
            mujoco.mj_kinematics(ref, rd)
            g = 0.5 * (rd.xpos[thumb] + np.mean([rd.xpos[i] for i in fingers], axis=0))
            gp = table.project([g[0], g[1], table.surface])
            cv2.drawMarker(vis, tuple(np.int32(gp)), (255, 0, 0),
                           cv2.MARKER_CROSS, 26, 3)
            cv2.putText(vis, "FK closure", tuple(np.int32(gp) + [6, 18]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # detector seeds and outcomes
        blue = measure_layout.blue_dominance(frame)
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bag_size = config["boxes"][0]["size"]
        table_cm = [table_size[0] * 100.0, table_size[1] * 100.0]
        tcx = (tray["center"][0] + table_size[0] / 2.0) * 100.0
        tcy = (tray["center"][1] + table_size[1] / 2.0) * 100.0
        info = []
        for i, seed in enumerate(measure_layout.find_bag_seeds(frame)):
            sx, sy = int(seed[0]), int(seed[1])
            cv2.circle(vis, (sx, sy), 6, (0, 165, 255), 2)
            cv2.putText(vis, "s%d" % i, (sx + 7, sy - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            points = measure_layout.contact_points(grey, blue, seed)
            if len(points) < 4:
                info.append("s%d few-points" % i)
                continue
            r = measure_layout.bag_from_contact(table, points, bag_size)
            if not r:
                info.append("s%d no-fit" % i)
                continue
            cx, cy = r["box_centre_cm"]
            half = [bag_size[0] * 50.0, bag_size[1] * 50.0]
            ok = (cx - half[0] > 1.0 and cx + half[0] < table_cm[0] - 1.0
                  and cy - half[1] > 1.0 and cy + half[1] < table_cm[1] - 1.0)
            dx, dy = cx - tcx, cy - tcy
            lx = np.cos(-tyaw) * dx - np.sin(-tyaw) * dy
            ly = np.sin(-tyaw) * dx + np.cos(-tyaw) * dy
            intray = abs(lx) < tray["size"][0] * 50.0 + 2.0 and abs(ly) < tray["size"][1] * 50.0 + 2.0
            lbl = "KEEP" if (ok and not intray) else ("offtab" if not ok else "inbox")
            col = (0, 255, 0) if lbl == "KEEP" else (0, 0, 255)
            p = table.project([cx / 100.0 - table_size[0] / 2.0,
                               cy / 100.0 - table_size[1] / 2.0, table.surface])
            cv2.circle(vis, tuple(np.int32(p)), 8, col, 2)
            cv2.putText(vis, "%s %.0f,%.0f" % (lbl, cx, cy),
                        tuple(np.int32(p) + [10, -10]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, col, 1)
            info.append("s%d %s (%.1f,%.1f)" % (i, lbl, cx, cy))
        print("ep%-3d  %s" % (episode, " | ".join(info)))
        cv2.imwrite(str(OUT / ("ep%02d_annotated.png" % episode)), vis)
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
