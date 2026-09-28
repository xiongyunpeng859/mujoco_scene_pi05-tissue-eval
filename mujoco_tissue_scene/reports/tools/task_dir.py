#!/usr/bin/env python3
"""Measure where the packs are at the first and last frame of each episode.

If the pack moves from the table (left) to the green tray, the task is
place-INTO-tray and the scene's direction is right."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml
import cv2

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import align_with_dataset as align
import dataset_io
import measure_layout

SUCCESS = Path("/workspace/shared/new_program_qiuzhi/without_tactile/success_episode/"
               "pick_up_the_tissue_pack_and_place_it_on_the_right_side_"
               "20260908_merged_all_108eps")


def all_bags(frame, config):
    """Every detection, with no in-tray / off-table filtering at all."""
    table = measure_layout.TableFrame(config)
    blue = measure_layout.blue_dominance(frame)
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    size = config["boxes"][0]["size"]
    out = []
    for seed in measure_layout.find_bag_seeds(frame):
        pts = measure_layout.contact_points(grey, blue, seed)
        if len(pts) < 4:
            continue
        r = measure_layout.bag_from_contact(table, pts, size)
        if r:
            out.append(r["box_centre_cm"])
    return out


def main() -> int:
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    tray = config["tray"]
    ts = config["table"]["size"]
    box_cm = [(tray["center"][0] + ts[0] / 2) * 100.0,
              (tray["center"][1] + ts[1] / 2) * 100.0]
    tyaw = float(tray["yaw"])
    hx = tray["size"][0] / 2.0 - tray["wall_thickness"]
    hy = tray["size"][1] / 2.0 - tray["wall_thickness"]
    data, order = dataset_io.load(SUCCESS)
    print("green tray centre (%.1f, %.1f) cm, inner half-extent (%.1f, %.1f) cm"
          % (box_cm[0], box_cm[1], hx * 100, hy * 100))
    print()
    print("  %-3s %-6s | %s" % ("ep", "frame", "all detections (table cm), with in-tray flag"))
    for episode in range(8):
        n = len(data[episode]["action"])
        for label, fi in (("first", 0), ("last", n - 1)):
            frame = align.dataset_frame(SUCCESS, episode, fi)
            bags = all_bags(frame, config)
            parts = []
            for cx, cy in bags:
                dx, dy = cx - box_cm[0], cy - box_cm[1]
                lx = np.cos(-tyaw) * dx - np.sin(-tyaw) * dy
                ly = np.sin(-tyaw) * dx + np.cos(-tyaw) * dy
                intray = abs(lx) < hx * 100 and abs(ly) < hy * 100
                parts.append("(%.0f,%.0f)%s" % (cx, cy, "[TRAY]" if intray else ""))
            print("  %-3d %-6s | %s" % (episode, label,
                                        " ".join(parts) if parts else "none"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
