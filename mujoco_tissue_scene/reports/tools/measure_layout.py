#!/usr/bin/env python3
"""Measure tabletop object poses from a real central-camera frame.

The central camera is calibrated (intrinsics + extrinsics), so a single frame is
enough to recover where objects sit on the table, in centimetres from the table's
left / near corner.  No tape measure needed.

    python measure_layout.py --frame /tmp/now_bags.png --annotate /tmp/measured.png
    python measure_layout.py --frame /tmp/now_12.png --mode tray

Two detectors are provided:

* ``bags``  - the blue tissue bags.  Blue-dominance (B - max(R,G)) isolates the
  printed artwork from the neutral grey wall and the black tabletop.  The bag's
  position then comes from its *contact line* with the table (the bright-to-dark
  transition in each image column), which is unaffected by the wall clipping the
  top of bags placed near the table's far edge.
* ``tray``  - the green box.  Its green inner floor is a known rectangle, so a
  standard planar pose fit gives centre and yaw.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares


def make_projector(K, R_cam_from_world, camera_pos):
    def project(point):
        cam = R_cam_from_world @ (np.asarray(point, float) - camera_pos)
        if cam[2] <= 1e-6:
            return np.array([1e4, 1e4])
        return np.array([K[0, 0] * cam[0] / cam[2] + K[0, 2],
                         K[1, 1] * cam[1] / cam[2] + K[1, 2]])
    return project


def make_backprojector(K_inv, R_cam_from_world, camera_pos):
    def backproject(u, v, z_world):
        direction = R_cam_from_world.T @ (K_inv @ np.array([u, v, 1.0]))
        direction = direction / np.linalg.norm(direction)
        if abs(direction[2]) < 1e-9:
            return None
        scale = (z_world - camera_pos[2]) / direction[2]
        if scale <= 0:
            return None
        return camera_pos + scale * direction
    return backproject


class TableFrame:
    """Conversions between pixels, world metres and table centimetres."""

    def __init__(self, config):
        self.surface = float(config["table"].get("surface_z", 0.75))
        self.size_cm = [value * 100.0 for value in config["table"]["size"][:2]]
        camera = config["cameras"]["central"]
        intr = camera["intrinsics"]
        self.K = np.array([[intr["fx"], 0.0, intr["cx"]],
                           [0.0, intr["fy"], intr["cy"]], [0.0, 0.0, 1.0]])
        self.K_inv = np.linalg.inv(self.K)
        self.camera_pos = np.array([float(value) for value in camera["position"]])
        axes = [float(value) for value in camera["xyaxes"]]
        x_axis, y_axis = np.array(axes[:3]), np.array(axes[3:])
        # Columns of A are the camera's own axes in world coordinates, so
        # world -> camera is A.T; then flip to the OpenCV convention (+z forward, y down).
        A = np.column_stack([x_axis, y_axis, np.cross(x_axis, y_axis)])
        self.R = np.diag([1.0, -1.0, -1.0]) @ A.T
        self.project = make_projector(self.K, self.R, self.camera_pos)
        self.backproject = make_backprojector(self.K_inv, self.R, self.camera_pos)

    def to_cm(self, point):
        return np.array([(point[0] + self.size_cm[0] / 200.0) * 100.0,
                         (point[1] + self.size_cm[1] / 200.0) * 100.0])

    def from_cm(self, x_cm, y_cm, z_extra=0.0):
        return np.array([x_cm / 100.0 - self.size_cm[0] / 200.0,
                         y_cm / 100.0 - self.size_cm[1] / 200.0,
                         self.surface + z_extra])

    def camera_xy_cm(self):
        return self.to_cm(self.camera_pos)


def blue_dominance(image):
    blue = image[:, :, 0].astype(np.int16)
    rest = np.maximum(image[:, :, 1], image[:, :, 2]).astype(np.int16)
    return blue - rest


def find_bag_seeds(image, min_area=500, dominance=10):
    mask = (blue_dominance(image) > dominance).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    height, width = image.shape[:2]
    seeds = []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < min_area:
            continue
        if x <= 0 or y <= 0 or x + w >= width or y + h >= height:
            continue                      # arms run out of frame; bags never do
        seeds.append((int(x), int(y), int(w), int(h), int(area)))
    return sorted(seeds, key=lambda s: -s[4])


def contact_points(grey, blue, seed, reach=30, floor_window=16, drop=30,
                   bright=140, dark=110, min_blue=5):
    """Contact line where the bag meets the dark tabletop.

    The table's far edge is also a bright-to-dark transition, so the search window
    is anchored just above the bottom of the blue seed and every column must carry
    blue artwork above the transition.  Bags sitting near the far edge are then
    measured by their footprint rather than by their (wall-clipped) silhouette.
    """
    x, y, w, h, _ = seed
    low = y + h - floor_window
    high = min(grey.shape[0] - 1, y + h + drop)
    points = []
    for column in range(max(0, x - reach), min(grey.shape[1], x + w + reach)):
        found = None
        for candidate in range(max(1, low), high):
            if grey[candidate, column] > bright and grey[candidate + 1, column] < dark:
                found = candidate
        if found is None:
            continue
        if blue[max(0, low):found + 1, column].max() <= min_blue:
            continue
        points.append((column, found))
    return points


def robust_line(cloud, iterations=6, tolerance=2.5):
    """Split a 2D point cloud into inliers of its dominant line."""
    inliers = np.ones(len(cloud), bool)
    if len(cloud) < 3:
        return inliers, None, None
    for _ in range(iterations):
        centre = cloud[inliers].mean(0)
        _, _, vh = np.linalg.svd(cloud[inliers] - centre)
        direction = vh[0]
        offset = cloud - centre
        residual = np.abs(offset[:, 0] * direction[1] - offset[:, 1] * direction[0])
        limit = max(tolerance, 1.2 * residual[inliers].std())
        updated = residual <= limit
        if updated.sum() < 3:
            break
        if np.array_equal(updated, inliers):
            break
        inliers = updated
    return inliers, direction, centre


def bag_from_contact(table, points, bag_size):
    """Footprint of a bag whose visible contact edge is measured on the tabletop."""
    world = [table.backproject(u, v, table.surface + 0.002) for u, v in points]
    cloud = np.array([table.to_cm(p) for p in world if p is not None])
    if len(cloud) < 3:
        return None
    inliers, edge_dir, centre = robust_line(cloud)
    if edge_dir is None or inliers.sum() < 3:
        return None
    cloud, points = cloud[inliers], [p for p, k in zip(points, inliers) if k]
    centre = cloud.mean(0)
    length = float(np.ptp((cloud - centre) @ edge_dir))
    if length < 3.0:
        return None
    normal = np.array([-edge_dir[1], edge_dir[0]])
    if normal @ (centre - table.camera_xy_cm()) < 0:
        normal = -normal
    sx, sy = bag_size[0] * 100.0, bag_size[1] * 100.0
    # The visible contact edge must be one whole side of the bag, so snap its
    # measured length to whichever side is closer and take the other as the depth.
    near_side = sx if abs(length - sx) <= abs(length - sy) else sy
    depth = sx * sy / near_side
    box_centre = centre + normal * depth / 2.0
    # Local +x runs along the 12cm side: it is the contact edge itself when that
    # edge is the long side, otherwise it points away from the camera.
    yaw = float(np.arctan2(edge_dir[1], edge_dir[0]) if near_side == sx
                else np.arctan2(normal[1], normal[0]))
    half = near_side / 2.0
    corners = [centre + edge_dir * along + normal * across
               for along in (-half, half) for across in (0.0, depth)]
    residual = np.abs((cloud - centre) @ normal)
    return {
        "contact_centre_cm": [round(float(centre[0]), 2), round(float(centre[1]), 2)],
        "box_centre_cm": [round(float(box_centre[0]), 2), round(float(box_centre[1]), 2)],
        "yaw_deg": round(float(np.degrees(yaw)), 1),
        "contact_edge_measured_cm": round(length, 2),
        "contact_edge_used_cm": round(near_side, 2),
        "estimated_depth_cm": round(depth, 2),
        "contact_line_rms_cm": round(float(np.sqrt((residual ** 2).mean())), 2),
        "footprint_corners_cm": [[round(float(c[0]), 2), round(float(c[1]), 2)] for c in corners],
        "contact_pixel_count": len(points),
    }


def fit_tray(table, image, tray):
    """Planar pose of the green box from its green inner floor."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.morphologyEx(cv2.inRange(hsv, (40, 120, 110), (80, 255, 255)),
                            cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        return None
    index = 1 + int(np.argmax(stats[1:, 4]))
    blob = (labels == index).astype(np.uint8)
    contour = max(cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
                  key=cv2.contourArea)
    quad = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    quad = quad.reshape(-1, 2).astype(float)
    if len(quad) != 4:
        quad = cv2.boxPoints(cv2.minAreaRect(contour)).astype(float)
    centroid = quad.mean(0)
    quad = quad[np.argsort(np.arctan2(quad[:, 1] - centroid[1], quad[:, 0] - centroid[0]))]

    floor_z = table.surface + 0.008
    half_x = tray["size"][0] / 2.0 - tray["wall_thickness"]
    half_y = tray["size"][1] / 2.0 - tray["wall_thickness"]
    local = np.array([[half_x, half_y], [-half_x, half_y], [-half_x, -half_y], [half_x, -half_y]])

    best = None
    for reverse in (False, True):
        for shift in range(4):
            observed = np.roll(quad[::-1] if reverse else quad, -shift, axis=0)

            def residuals(parameters):
                yaw = parameters[2]
                rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
                model = np.array([table.project([*(rotation @ corner + parameters[:2]), floor_z])
                                  for corner in local])
                return (model - observed).ravel()

            for x_cm in np.arange(-10.0, 110.0, 10.0):
                for y_cm in np.arange(0.0, 75.0, 10.0):
                    for yaw in np.arange(-1.6, 1.6, 0.4):
                        start = [*table.from_cm(x_cm, y_cm)[:2], yaw]
                        try:
                            result = least_squares(residuals, start, method="lm", max_nfev=600)
                        except Exception:
                            continue
                        if best is None or result.cost < best[0]:
                            best = (result.cost, result, reverse, shift, observed)
    if best is None:
        return None
    _, result, reverse, shift, observed = best
    yaw = float(result.x[2])
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    model = np.array([table.project([*(rotation @ corner + result.x[:2]), floor_z])
                      for corner in local])
    residual = np.linalg.norm(model - observed, axis=1)
    corners = [table.to_cm([*(rotation @ corner + result.x[:2]), floor_z]) for corner in local]
    return {
        "centre_cm_from_left_bottom": [round(float(v), 2) for v in table.to_cm(result.x[:2])],
        "yaw_deg": round(float(np.degrees(yaw)), 1),
        "corner_rms_px": round(float(np.sqrt((residual ** 2).mean())), 2),
        "inner_floor_corners_cm": [[round(float(c[0]), 2), round(float(c[1]), 2)] for c in corners],
        "quad_px": [[int(u), int(v)] for u, v in quad],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frame", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/scene.yaml"))
    parser.add_argument("--mode", choices=["bags", "tray", "both"], default="bags")
    parser.add_argument("--annotate", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--min-area", type=int, default=500)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    table = TableFrame(config)
    image = cv2.imread(str(args.frame))
    if image is None:
        raise SystemExit(f"cannot read {args.frame}")
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blue = blue_dominance(image)
    annotated = image.copy() if args.annotate else None

    report = {"frame": str(args.frame), "table_size_cm": table.size_cm}

    if args.mode in ("bags", "both"):
        bag_size = config["boxes"][0]["size"]
        report["bag_size_cm"] = bag_size
        bags = []
        for seed in find_bag_seeds(image, min_area=args.min_area):
            points = contact_points(grey, blue, seed)
            if len(points) < 4:
                continue
            result = bag_from_contact(table, points, bag_size)
            if result is None:
                continue
            result["seed_bbox_px"] = list(seed[:4])
            bags.append(result)
            if annotated is not None:
                for u, v in points:
                    cv2.circle(annotated, (u, v), 2, (0, 255, 255), -1)
                quad = np.array([table.project(table.from_cm(*corner))[:2]
                                 for corner in result["footprint_corners_cm"]])
                cv2.polylines(annotated, [quad.astype(np.int32)], True, (255, 0, 255), 2)
                label = "(%.1f, %.1f)" % tuple(result["box_centre_cm"])
                x, y, w, h, _ = seed
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 220, 0), 1)
                cv2.putText(annotated, label, (x - 10, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        report["bags"] = bags
        if bags:
            centres = np.array([b["box_centre_cm"] for b in bags])
            corners = np.array([c for b in bags for c in b["footprint_corners_cm"]])
            report["bag_centre_bbox_cm"] = {
                "x": [round(float(centres[:, 0].min()), 2), round(float(centres[:, 0].max()), 2)],
                "y": [round(float(centres[:, 1].min()), 2), round(float(centres[:, 1].max()), 2)],
            }
            report["bag_footprint_bbox_cm"] = {
                "x": [round(float(corners[:, 0].min()), 2), round(float(corners[:, 0].max()), 2)],
                "y": [round(float(corners[:, 1].min()), 2), round(float(corners[:, 1].max()), 2)],
            }

    if args.mode in ("tray", "both"):
        report["tray"] = fit_tray(table, image, dict(config["tray"]))

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if annotated is not None:
        cv2.imwrite(str(args.annotate), annotated)
        print(f"annotated -> {args.annotate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
