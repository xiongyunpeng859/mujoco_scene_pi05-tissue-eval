#!/usr/bin/env python3
"""Compare the appearance of a real central frame with the matching sim render.

    python match_appearance.py --real /tmp/frame.png \
        --sim outputs/measured_placement/central.png

Samples a few regions whose identity is unambiguous in both images (wall, empty
tabletop, the green box floor, one tissue bag) and prints the mean RGB of each
together with the difference, plus the whole-image agreement.  Use it to iterate
on `materials:` / `lighting:` in configs/scene.yaml.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# (name, x0, x1, y0, y1, expected) -- pixel boxes valid for the 640x480 central view.
REGIONS = [
    ("wall",          250, 430,  40, 140),
    ("table",         430, 500, 380, 450),
    ("tray_floor",    330, 370, 300, 360),
    ("bag",           100, 135, 195, 220),
]


def sample(image, box):
    x0, x1, y0, y1 = box
    patch = image[y0:y1, x0:x1].reshape(-1, 3).astype(np.float64)
    return patch.mean(0)          # BGR order


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--sim", type=Path, required=True)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--annotate", type=Path, default=None)
    args = parser.parse_args()

    real = cv2.imread(str(args.real))
    sim = cv2.imread(str(args.sim))
    if real is None or sim is None:
        raise SystemExit("cannot read one of the images")
    if real.shape != sim.shape:
        sim = cv2.resize(sim, (real.shape[1], real.shape[0]))

    report = {"real": str(args.real), "sim": str(args.sim), "regions": {}}
    print("region        real RGB            sim RGB             delta (sim-real)")
    print("-" * 74)
    for name, *box in REGIONS:
        r = sample(real, box)[::-1]        # to RGB for readability
        s = sample(sim, box)[::-1]
        delta = s - r
        report["regions"][name] = {
            "box": box,
            "real_rgb": [round(float(v), 1) for v in r],
            "sim_rgb": [round(float(v), 1) for v in s],
            "delta_rgb": [round(float(v), 1) for v in delta],
            "real_luma": round(float(r.mean()), 1),
            "sim_luma": round(float(s.mean()), 1),
        }
        print("%-12s %-19s %-19s %s  (luma %.0f -> %.0f)" %
              (name, np.round(r, 1).tolist(), np.round(s, 1).tolist(),
               np.round(delta, 1).tolist(), r.mean(), s.mean()))

    real_grey = cv2.cvtColor(real, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sim_grey = cv2.cvtColor(sim, cv2.COLOR_BGR2GRAY).astype(np.float32)
    diff = np.abs(real_grey - sim_grey)
    report["mean_abs_grey_diff"] = round(float(diff.mean()), 2)
    report["rmse_grey"] = round(float(np.sqrt((diff ** 2).mean())), 2)
    print("-" * 74)
    print("whole image: mean|diff| = %.1f   RMSE = %.1f (0-255)" %
          (report["mean_abs_grey_diff"], report["rmse_grey"]))

    if args.annotate:
        canvas = np.hstack([real, sim])
        for name, *box in REGIONS:
            x0, x1, y0, y1 = box
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 255, 255), 1)
            cv2.rectangle(canvas, (x0 + real.shape[1], y0), (x1 + real.shape[1], y1),
                          (0, 255, 255), 1)
            cv2.putText(canvas, name, (x0, y0 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(args.annotate), canvas)
        print("annotated -> %s" % args.annotate)

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print("wrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
