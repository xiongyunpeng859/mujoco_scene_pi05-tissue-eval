#!/usr/bin/env python3
"""Local LM refine of the green box pose -- 8 starts per frame instead of 5600."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import least_squares

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import align_with_dataset as align
import measure_layout
from box_pose_check import observed_quad, score

SUCCESS = Path("/workspace/shared/new_program_qiuzhi/without_tactile/success_episode/"
               "pick_up_the_tissue_pack_and_place_it_on_the_right_side_"
               "20260908_merged_all_108eps")
START_CM = (61.44, 39.61)          # from the global fits on clean frames


def refine(table, tray, quad, start_cm, start_yaw):
    floor_z = table.surface + 0.008
    half_x = tray["size"][0] / 2.0 - tray["wall_thickness"]
    half_y = tray["size"][1] / 2.0 - tray["wall_thickness"]
    local = np.array([[half_x, half_y], [-half_x, half_y],
                      [-half_x, -half_y], [half_x, -half_y]])
    best = None
    for reverse in (False, True):
        for shift in range(4):
            obs = np.roll(quad[::-1] if reverse else quad, -shift, axis=0)

            def residuals(p):
                rot = np.array([[np.cos(p[2]), -np.sin(p[2])],
                                [np.sin(p[2]), np.cos(p[2])]])
                model = np.array([table.project([*(rot @ c + p[:2]), floor_z])
                                  for c in local])
                return (model - obs).ravel()

            try:
                r = least_squares(residuals, [*table.from_cm(*start_cm)[:2], start_yaw],
                                  method="lm", max_nfev=400)
            except Exception:
                continue
            rms = float(np.sqrt((r.fun ** 2).mean()))
            if best is None or rms < best[0]:
                best = (rms, r)
    return best


def main() -> int:
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    tray, table = config["tray"], measure_layout.TableFrame(config)
    start_yaw = float(tray["yaw"])
    xs, ys, yaws, rms_all = [], [], [], []
    print("  %-4s %-6s %8s %8s %9s %8s %9s" %
          ("ep", "frame", "x cm", "y cm", "yaw deg", "rms px", "area"))
    kept = 0
    for e in range(20):
        for fi in (0, 5, 10, 15, 20):
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
            best = refine(table, tray, quad, START_CM, start_yaw)
            if best is None or best[0] > 5.0:
                continue
            r = best[1]
            cm = table.to_cm(r.x[:2])
            folded = ((np.degrees(r.x[2]) + 90.0) % 180.0) - 90.0
            xs.append(cm[0]); ys.append(cm[1]); yaws.append(folded)
            rms_all.append(best[0])
            kept += 1
            if kept <= 16:
                print("  %-4d %-6d %8.2f %8.2f %9.2f %8.2f %9d"
                      % (e, fi, cm[0], cm[1], folded, best[0], int(area)))
    xs, ys, yaws = np.array(xs), np.array(ys), np.array(yaws)
    print()
    print("  frames kept: %d   mean corner rms %.2f px" % (kept, np.mean(rms_all)))
    print("  x   : median %7.2f  std %5.2f  spread %5.2f  (min %.2f max %.2f)"
          % (np.median(xs), xs.std(), xs.max() - xs.min(), xs.min(), xs.max()))
    print("  y   : median %7.2f  std %5.2f  spread %5.2f  (min %.2f max %.2f)"
          % (np.median(ys), ys.std(), ys.max() - ys.min(), ys.min(), ys.max()))
    print("  yaw : median %7.2f  std %5.2f  (mod 180, nearly-square box)"
          % (np.median(yaws), yaws.std()))
    new_cm = [float(np.median(xs)), float(np.median(ys))]
    print()
    print("  candidate poses on ep0 frame 0 (observed quad fixed):")
    quad, _ = observed_quad(align.dataset_frame(SUCCESS, 0, 0))
    old_cm = [(tray["center"][0] + config["table"]["size"][0] / 2.0) * 100.0,
              (tray["center"][1] + config["table"]["size"][1] / 2.0) * 100.0]
    for label, (px, py, pyaw) in (
            ("OLD config          ", (old_cm[0], old_cm[1], tray["yaw"])),
            ("NEW xy, old yaw     ", (new_cm[0], new_cm[1], tray["yaw"])),
            ("NEW xy, fitted yaw  ", (new_cm[0], new_cm[1],
                                      np.radians(float(np.median(yaws)))))):
        r = score(table, tray, quad, px, py, pyaw)
        print("     %s (%6.2f,%6.2f) yaw %7.2f -> %6.2f px  (shift %+.2f,%+.2f cm)"
              % (label, px, py, np.degrees(pyaw), r[0], px - old_cm[0], py - old_cm[1]))
    print()
    print("  CORRECTED centre: (%.2f, %.2f) cm from left-bottom" % (new_cm[0], new_cm[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
