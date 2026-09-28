#!/usr/bin/env python3
"""Re-fit the green box with the user's constraints: axis-parallel yaw and 22.5 x 19.5 cm.

The user states the box's edges are parallel to the table's, and its footprint is
22.5 x 19.5 cm (height 7.5 cm, already used).  My config had yaw -83.8 deg, i.e. ~6 deg off
axis-parallel -- a few degrees of yaw error is exactly what a near-square footprint lets a
corner fit absorb.  So LOCK the yaw to a table axis and re-solve only the centre, then verify
the residual against the box's own green floor.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reports/tools"))
import align_with_dataset as align
import measure_layout
from box_pose_check import observed_quad
from scipy.optimize import least_squares

SUCCESS = Path("/workspace/shared/new_program_qiuzhi/without_tactile/success_episode/"
               "pick_up_the_tissue_pack_and_place_it_on_the_right_side_"
               "20260908_merged_all_108eps")
CFG = ROOT / "configs/scene.yaml"
YAW = -np.pi / 2.0            # edges parallel to the table: local x -> world y


def residuals(table, tray, quad, xy):
    floor_z = table.surface + 0.008
    hx = tray["size"][0] / 2.0 - tray["wall_thickness"]
    hy = tray["size"][1] / 2.0 - tray["wall_thickness"]
    local = np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]])
    rot = np.array([[np.cos(YAW), -np.sin(YAW)], [np.sin(YAW), np.cos(YAW)]])
    model = np.array([table.project([*(rot @ c + xy), floor_z]) for c in local])
    best = None
    for reverse in (False, True):
        for shift in range(4):
            obs = np.roll(quad[::-1] if reverse else quad, -shift, axis=0)
            r = (model - obs).ravel()
            c = float((r ** 2).mean())
            if best is None or c < best[0]:
                best = (c, r, reverse, shift)
    return best


def main() -> int:
    config = yaml.safe_load(CFG.read_text())
    table = measure_layout.TableFrame(config)
    tray = dict(config["tray"])
    shutil.copy(CFG, CFG.with_suffix(".yaml.bak_before_axisbox"))

    # user's measurements: footprint 22.5 x 19.5 cm, height 7.5 cm, wall thickness estimated
    tray["size"] = [0.225, 0.195, 0.075]
    tray["yaw"] = float(YAW)
    x0 = np.array([tray["center"][0], tray["center"][1]])

    centres, rmss = [], []
    for e in range(20):
        for fi in (0, 10, 20):
            try:
                image = align.dataset_frame(SUCCESS, e, fi)
            except Exception:
                continue
            got = observed_quad(image)
            if got is None:
                continue
            quad, area = got
            if area < 5000:
                continue
            res = least_squares(lambda xy: residuals(table, tray, quad, xy)[1],
                                x0, method="lm", max_nfev=400)
            r = residuals(table, tray, quad, res.x)
            rms = float(np.sqrt((r[1] ** 2).mean()))
            if rms < 5.0:
                centres.append(res.x)
                rmss.append(rms)
    if not centres:
        print("no clean frames; aborting")
        return 1
    centres = np.array(centres)
    cm = [(c[0] + config["table"]["size"][0] / 2) * 100.0 for c in centres]
    cm_y = [(c[1] + config["table"]["size"][1] / 2) * 100.0 for c in centres]
    print("frames used: %d   mean corner RMS %.2f px" % (len(centres), np.mean(rmss)))
    print("centre x: median %.2f cm (std %.2f)   y: median %.2f cm (std %.2f)"
          % (np.median(cm), np.std(cm), np.median(cm_y), np.std(cm_y)))
    print()
    print("  comparison on one clean frame:")
    image = align.dataset_frame(SUCCESS, 0, 0)
    quad, _ = observed_quad(image)
    for label, (t, xy) in (("OLD 21x20 yaw -83.8", (config["tray"], x0)),
                           ("NEW 22.5x19.5 yaw -90.0",
                            (tray, np.array([np.median(cm) / 100.0
                                             - config["table"]["size"][0] / 2,
                                             np.median(cm_y) / 100.0
                                             - config["table"]["size"][1] / 2])))):
        r = residuals(table, t, quad, xy)
        print("     %-26s corner RMS %.2f px" % (label, np.sqrt(r[0])))
    new_center = [float(np.median(cm) / 100.0 - config["table"]["size"][0] / 2),
                  float(np.median(cm_y) / 100.0 - config["table"]["size"][1] / 2)]
    config["tray"]["size"] = tray["size"]
    config["tray"]["yaw"] = float(YAW)
    config["tray"]["center"] = new_center
    config["tray"]["size_measured"] = True
    config["tray"]["size_measured_source"] = (
        "22.5 x 19.5 x 7.5 cm all measured by the user (2026, corrected from an earlier "
        "21 x 20 x 7.5); the user also states the box edges are PARALLEL to the table edges, "
        "so the yaw is locked to -90 deg instead of the image-fitted -83.8 deg -- a near-square "
        "footprint lets a corner fit absorb several degrees of yaw error.")
    config["measured_layout"]["tray_center_xy_cm_from_left_bottom"] = [
        float(round(np.median(cm), 2)), float(round(np.median(cm_y), 2))]
    config["measured_layout"]["tray_size_cm"] = [22.5, 19.5]
    config["measured_layout"]["tray_size_cm_measured"] = [22.5, 19.5, 7.5]
    config["measured_layout"].setdefault("notes", []).append(
        "Tray corrected 2026: footprint 21x20 -> 22.5x19.5 cm and yaw -83.8 -> exactly -90 deg "
        "(edges parallel to the table, per the user, who can see the cell). Re-solved the "
        "centre with the yaw LOCKED, so the fit can no longer trade yaw against translation.")
    CFG.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    print()
    print("config updated: size %s  yaw %.4f  centre %s"
          % (config["tray"]["size"], config["tray"]["yaw"], config["tray"]["center"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
