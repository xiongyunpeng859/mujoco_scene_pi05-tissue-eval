#!/usr/bin/env python3
"""Robustly re-measure the green box centre over many clean frames."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import align_with_dataset as align
import measure_layout
from box_pose_check import observed_quad, score

SUCCESS = Path("/workspace/shared/new_program_qiuzhi/without_tactile/success_episode/"
               "pick_up_the_tissue_pack_and_place_it_on_the_right_side_"
               "20260908_merged_all_108eps")


def main() -> int:
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    tray = config["tray"]
    table = measure_layout.TableFrame(config)
    old_cm = [(tray["center"][0] + config["table"]["size"][0] / 2.0) * 100.0,
              (tray["center"][1] + config["table"]["size"][1] / 2.0) * 100.0]

    samples = []
    for e in range(16):
        for fi in (0, 10, 20):
            try:
                image = align.dataset_frame(SUCCESS, e, fi)
            except Exception:
                continue
            got = observed_quad(image)
            if got is None:
                continue
            quad, area = got
            fit = measure_layout.fit_tray(table, image, tray)
            if not fit or fit["corner_rms_px"] > 5.0 or area < 5000:
                continue
            cx, cy = fit["centre_cm_from_left_bottom"]
            yaw = fit["yaw_deg"]
            folded = ((yaw + 90.0) % 180.0) - 90.0        # square box: mod-180 symmetry
            samples.append((e, fi, cx, cy, folded, fit["corner_rms_px"], int(area)))
        if len(samples) >= 24:
            break

    print("clean green-quad fits: %d" % len(samples))
    print("  %-4s %-6s %8s %8s %9s %8s %9s" %
          ("ep", "frame", "x cm", "y cm", "yaw deg", "rms px", "area"))
    for s in samples[:14]:
        print("  %-4d %-6d %8.2f %8.2f %9.2f %8.2f %9d" % s)
    xs = np.array([s[2] for s in samples])
    ys = np.array([s[3] for s in samples])
    yaws = np.array([s[4] for s in samples])
    print()
    print("  x   : mean %7.2f  median %7.2f  std %5.2f  spread %5.2f"
          % (xs.mean(), np.median(xs), xs.std(), xs.max() - xs.min()))
    print("  y   : mean %7.2f  median %7.2f  std %5.2f  spread %5.2f"
          % (ys.mean(), np.median(ys), ys.std(), ys.max() - ys.min()))
    print("  yaw : mean %7.2f  median %7.2f  std %5.2f   (mod 180)" % (
        yaws.mean(), np.median(yaws), yaws.std()))
    new_cm = [float(np.median(xs)), float(np.median(ys))]
    print()
    print("  candidate poses, scored on the median frame:")
    for label, (px, py, pyaw) in (
            ("OLD config            ", (old_cm[0], old_cm[1], tray["yaw"])),
            ("NEW xy + old yaw      ", (new_cm[0], new_cm[1], tray["yaw"])),
            ("NEW xy + fitted yaw   ", (new_cm[0], new_cm[1],
                                        np.radians(float(np.median(yaws)))))):
        image = align.dataset_frame(SUCCESS, 0, 0)
        quad, _ = observed_quad(image)
        r = score(table, tray, quad, px, py, pyaw)
        print("     %s (%7.2f,%7.2f) yaw %7.2f  ->  %6.2f px"
              % (label, px, py, np.degrees(pyaw), r[0]))
    print()
    print("  median-absolute-deviation of individual fits from the median pose:")
    print("     dx %.2f cm, dy %.2f cm" % (np.median(np.abs(xs - new_cm[0])),
                                            np.median(np.abs(ys - new_cm[1]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
