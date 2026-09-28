#!/usr/bin/env python3
"""Measure the pack's real footprint edge instead of assuming it.

bag_from_contact() reports contact_edge_measured_cm -- the genuinely measured length
of the pack's visible contact edge -- and then SNAPS it to whichever assumed side
(12 or 8.5 cm) is nearer, deriving the depth from the assumed area.  So the assumed
12 x 8.5 cm footprint is testable against the measurement.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import align_with_dataset as align

SUCCESS = Path("/workspace/shared/new_program_qiuzhi/without_tactile/success_episode/"
               "pick_up_the_tissue_pack_and_place_it_on_the_right_side_"
               "20260908_merged_all_108eps")


def main() -> int:
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    sx = config["boxes"][0]["size"][0] * 100.0
    sy = config["boxes"][0]["size"][1] * 100.0
    print("assumed footprint: %.1f x %.1f cm (area %.1f cm^2)" % (sx, sy, sx * sy))
    print()
    print("  %-4s %10s %10s %10s %8s %8s" %
          ("ep", "measured", "snapped to", "depth used", "rms cm", "px"))
    measured, used, rms = [], [], []
    for e in range(36):
        frame = align.dataset_frame(SUCCESS, e, 0)
        for bag in align.measure_bags(frame, config):
            m = bag["contact_edge_measured_cm"]
            u = bag["contact_edge_used_cm"]
            measured.append(m)
            used.append(u)
            rms.append(bag["contact_line_rms_cm"])
            if e < 10:
                print("  %-4d %10.2f %10.2f %10.2f %8.2f %8d"
                      % (e, m, u, bag["estimated_depth_cm"],
                         bag["contact_line_rms_cm"], bag["contact_pixel_count"]))
    measured = np.array(measured)
    used = np.array(used)
    print()
    print("  detections measured: %d" % len(measured))
    print("  measured contact edge : median %.2f cm  mean %.2f  std %.2f  "
          "range %.2f..%.2f" % (np.median(measured), measured.mean(), measured.std(),
                                measured.min(), measured.max()))
    print("  contact-line RMS      : median %.2f cm (fit quality)" % np.median(rms))
    print()
    print("  how often the snap chose each assumed side:")
    print("     snapped to %.1f cm : %d" % (sx, int((used == sx).sum())))
    print("     snapped to %.1f cm : %d" % (sy, int((used == sy).sum())))
    print()
    # If the real long side were 12 cm, measurements of it should cluster at 12.
    near12 = measured[np.abs(measured - sx) < 1.5]
    near85 = measured[np.abs(measured - sy) < 1.5]
    other = measured[(np.abs(measured - sx) >= 1.5) & (np.abs(measured - sy) >= 1.5)]
    print("  measurements within 1.5 cm of 12.0 : %d  (median %.2f)"
          % (len(near12), np.median(near12) if len(near12) else float("nan")))
    print("  measurements within 1.5 cm of  8.5 : %d  (median %.2f)"
          % (len(near85), np.median(near85) if len(near85) else float("nan")))
    print("  measurements matching NEITHER      : %d  (median %.2f)"
          % (len(other), np.median(other) if len(other) else float("nan")))
    print()
    print("  distribution of all measured edges (2 cm bins):")
    hist, edges = np.histogram(measured, bins=np.arange(4, 18, 2))
    for h, lo in zip(hist, edges[:-1]):
        print("     %4.0f-%4.0f cm : %s%d" % (lo, lo + 2, "#" * h, h))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
