#!/usr/bin/env python3
"""Where exactly does the bag detector lose the real pack?"""
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

SUCCESS = Path("/workspace/shared/new_program_qiuzhi/without_tactile/success_episode/"
               "pick_up_the_tissue_pack_and_place_it_on_the_right_side_"
               "20260908_merged_all_108eps")


def main() -> int:
    import cv2
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    tray = config["tray"]
    table_cm = [config["table"]["size"][0] * 100.0, config["table"]["size"][1] * 100.0]
    tcx = (tray["center"][0] + table_cm[0] / 200.0) * 100.0
    tcy = (tray["center"][1] + table_cm[1] / 200.0) * 100.0
    thx, thy = tray["size"][0] * 50.0, tray["size"][1] * 50.0
    dataset, order = dataset_io.load(SUCCESS)
    print("  %-3s %6s %6s %6s %6s | %s" %
          ("ep", "seeds", "fewpts", "nofit", "kept", "why each seed was dropped"))
    for episode in range(6):
        frame = align.dataset_frame(SUCCESS, episode, 0)
        table = measure_layout.TableFrame(config)
        blue = measure_layout.blue_dominance(frame)
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bag_size = config["boxes"][0]["size"]
        seeds = list(measure_layout.find_bag_seeds(frame))
        why, few, nofit, kept = [], 0, 0, 0
        for seed in seeds:
            points = measure_layout.contact_points(grey, blue, seed)
            if len(points) < 4:
                few += 1
                why.append("seed@%s few-points(%d)" % (np.round(seed[:2]).astype(int),
                                                       len(points)))
                continue
            result = measure_layout.bag_from_contact(table, points, bag_size)
            if not result:
                nofit += 1
                why.append("seed@%s no-fit" % np.round(seed[:2]).astype(int))
                continue
            cx, cy = result["box_centre_cm"]
            half = [bag_size[0] * 50.0, bag_size[1] * 50.0]
            if not (cx - half[0] > 1.0 and cx + half[0] < table_cm[0] - 1.0
                    and cy - half[1] > 1.0 and cy + half[1] < table_cm[1] - 1.0):
                why.append("seed@%s off-table(%.1f,%.1f)"
                           % (np.round(seed[:2]).astype(int), cx, cy))
                continue
            dx, dy = cx - tcx, cy - tcy
            tyaw = float(tray.get("yaw", 0.0))
            lx = np.cos(-tyaw) * dx - np.sin(-tyaw) * dy
            ly = np.sin(-tyaw) * dx + np.cos(-tyaw) * dy
            if abs(lx) < thx + 2.0 and abs(ly) < thy + 2.0:
                why.append("seed@%s inside-tray(%.1f,%.1f)"
                           % (np.round(seed[:2]).astype(int), cx, cy))
                continue
            kept += 1
            why.append("seed@%s KEPT(%.1f,%.1f)" % (np.round(seed[:2]).astype(int),
                                                    cx, cy))
        print("  %-3d %6d %6d %6d %6d | %s"
              % (episode, len(seeds), few, nofit, kept, "; ".join(why[:4])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
