#!/usr/bin/env python3
"""Side-by-side and overlay comparison between a real camera frame and the sim render.

Usage
-----
    python sim_real_align.py --real /tmp/now_12.png \
        --sim outputs/calibrated_20260918/central.png \
        --out outputs/alignment_20260918

Writes real.png, sim.png, overlay.png (real with the sim's edges burned in red)
and alignment.json.  Both images must come from the same camera and resolution;
the sim render is produced by `scene.py --render`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def edge_mask(image, low=40, high=110):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    grey = cv2.GaussianBlur(grey, (5, 5), 0)
    return cv2.Canny(grey, low, high)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--sim", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--blend", type=float, default=0.5)
    args = parser.parse_args()

    real = cv2.imread(str(args.real))
    sim = cv2.imread(str(args.sim))
    if real is None or sim is None:
        raise SystemExit("cannot read one of the input images")
    if real.shape[:2] != sim.shape[:2]:
        sim = cv2.resize(sim, (real.shape[1], real.shape[0]), interpolation=cv2.INTER_AREA)

    args.out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out / "real.png"), real)
    cv2.imwrite(str(args.out / "sim.png"), sim)

    real_grey = cv2.cvtColor(real, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sim_grey = cv2.cvtColor(sim, cv2.COLOR_BGR2GRAY).astype(np.float32)
    absdiff = np.abs(real_grey - sim_grey)

    real_edges = edge_mask(real)
    sim_edges = edge_mask(sim)
    # An edge counts as aligned when it lands within 2 px of the other image's edges.
    kernel = np.ones((5, 5), np.uint8)
    real_near = cv2.dilate(real_edges, kernel) > 0
    sim_near = cv2.dilate(sim_edges, kernel) > 0
    real_hits = int((real_edges > 0).sum())
    sim_hits = int((sim_edges > 0).sum())
    matched_real = int(((real_edges > 0) & sim_near).sum())
    matched_sim = int(((sim_edges > 0) & real_near).sum())

    blend = cv2.addWeighted(real, 1.0 - args.blend, sim, args.blend, 0.0)
    cv2.imwrite(str(args.out / "blend.png"), blend)

    overlay = real.copy()
    overlay[sim_near] = (0.45 * overlay[sim_near] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
    overlay[real_edges > 0] = (0, 255, 0)
    cv2.putText(overlay, "green = real edges   red = sim edges (2px dilated)", (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(args.out / "overlay.png"), overlay)

    report = {
        "real": str(args.real),
        "sim": str(args.sim),
        "resolution": [int(real.shape[1]), int(real.shape[0])],
        "mean_abs_grey_diff": round(float(absdiff.mean()), 3),
        "rmse_grey": round(float(np.sqrt((absdiff ** 2).mean())), 3),
        "real_edge_pixels": real_hits,
        "sim_edge_pixels": sim_hits,
        "real_edges_matching_sim_within_2px_percent": round(100.0 * matched_real / max(real_hits, 1), 2),
        "sim_edges_matching_real_within_2px_percent": round(100.0 * matched_sim / max(sim_hits, 1), 2),
        "note": "The arm is usually in a different joint configuration in the real frame, "
                "so treat arm edges as expected mismatch; table, tray and objects should align.",
    }
    (args.out / "alignment.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
