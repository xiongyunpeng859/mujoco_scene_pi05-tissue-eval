#!/usr/bin/env python3
"""Two candidate green-box poses disagree by 10 cm.  Score both on the same quad."""
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


def observed_quad(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.morphologyEx(cv2.inRange(hsv, (40, 120, 110), (80, 255, 255)),
                            cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        return None
    index = 1 + int(np.argmax(stats[1:, 4]))
    blob = (labels == index).astype(np.uint8)
    contour = max(cv2.findContours(blob, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
    quad = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    quad = quad.reshape(-1, 2).astype(float)
    if len(quad) != 4:
        quad = cv2.boxPoints(cv2.minAreaRect(contour)).astype(float)
    c = quad.mean(0)
    return quad[np.argsort(np.arctan2(quad[:, 1] - c[1], quad[:, 0] - c[0]))], float(
        cv2.contourArea(contour))


def score(table, tray, quad, x_cm, y_cm, yaw):
    floor_z = table.surface + 0.008
    half_x = tray["size"][0] / 2.0 - tray["wall_thickness"]
    half_y = tray["size"][1] / 2.0 - tray["wall_thickness"]
    local = np.array([[half_x, half_y], [-half_x, half_y],
                      [-half_x, -half_y], [half_x, -half_y]])
    rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    xy = table.from_cm(x_cm, y_cm)[:2]
    model = np.array([table.project([*(rot @ corner + xy), floor_z]) for corner in local])
    best = None
    for reverse in (False, True):
        for shift in range(4):
            obs = np.roll(quad[::-1] if reverse else quad, -shift, axis=0)
            rms = float(np.sqrt(((model - obs) ** 2).sum(1).mean()))
            if best is None or rms < best[0]:
                best = (rms, reverse, shift)
    return best


def main() -> int:
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    tray = config["tray"]
    table = measure_layout.TableFrame(config)
    config_cm = [(tray["center"][0] + config["table"]["size"][0] / 2.0) * 100.0,
                 (tray["center"][1] + config["table"]["size"][1] / 2.0) * 100.0]
    print("config pose : (%.2f, %.2f) cm  yaw %.4f rad (%.1f deg)"
          % (config_cm[0], config_cm[1], tray["yaw"], np.degrees(tray["yaw"])))
    print()
    print("  %-4s %-8s | %-34s | %-34s" % ("ep", "frame", "config pose residual px",
                                           "fresh fit residual px"))
    for e in range(4):
        for fi in (0, 10):
            image = align.dataset_frame(SUCCESS, e, fi)
            got = observed_quad(image)
            if got is None:
                continue
            quad, area = got
            r_cfg = score(table, tray, quad, config_cm[0], config_cm[1], tray["yaw"])
            fit = measure_layout.fit_tray(table, image, tray)
            if fit:
                fc = fit["centre_cm_from_left_bottom"]
                r_new = score(table, tray, quad, fc[0], fc[1],
                              np.radians(fit["yaw_deg"]))
                print("  %-4d %-8d | %6.2f px (flip %d)              | (%6.2f,%6.2f) yaw %6.1f "
                      "-> %6.2f px (flip %d)  area %d px^2"
                      % (e, fi, r_cfg[0], r_cfg[1], fc[0], fc[1], fit["yaw_deg"],
                         r_new[0], r_new[1], int(area)))
            else:
                print("  %-4d %-8d | %6.2f px                       | fit failed"
                      % (e, fi, r_cfg[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
