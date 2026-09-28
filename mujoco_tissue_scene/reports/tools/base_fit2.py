#!/usr/bin/env python3
"""Clean re-fit of the arm base pose.

The first attempt (base_kinematic_fit.py) was wrecked by the detector's off-table
filter, which discarded the very pack each episode grasped and left a pack in the tray
70 cm away as the only candidate.  With that filter fixed, re-run the same purely
kinematic constraint: the fingertip midpoint at the frame the hand closes must coincide
with a pack that the detector can actually see.

Outliers are excluded (>0.25 m) instead of dragging the median, and the base transform
is planar so FK is evaluated once per episode and the candidate base applied
analytically -- no model rebuild per grid point.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import action_layout as layout
import align_with_dataset as align
import dataset_io
import scene

SUCCESS = Path("/workspace/shared/new_program_qiuzhi/without_tactile/success_episode/"
               "pick_up_the_tissue_pack_and_place_it_on_the_right_side_"
               "20260908_merged_all_108eps")
OUT = ROOT / "outputs/base_fit2"
THUMB = "hand_L_thumb_tip"
FINGERS = ["hand_L_index_tip", "hand_L_middle_tip", "hand_L_ring_tip", "hand_L_pinky_tip"]
PITCH = [8, 10, 11, 13, 15]
CFG = ROOT / "configs/scene.yaml"


def main() -> int:
    import mujoco
    n_eps = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    OUT.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(CFG.read_text())
    table = config["table"]["size"]
    lc = config["measured_layout"]
    x0_cm, y0_cm = lc["arm_mount_xy_cm_from_left_bottom"]
    yaw0 = float(config["arm"]["euler"][2])
    c0 = np.array([x0_cm / 100.0 - table[0] / 2.0, y0_cm / 100.0 - table[1] / 2.0])
    model = mujoco.MjModel.from_xml_path(str(scene.build(
        CFG, OUT / "ref.xml", with_hand=True)))
    data = mujoco.MjData(model)
    ra = np.array(layout.joint_ids(model, mujoco))
    thumb, fingers = model.body(THUMB).id, [model.body(n).id for n in FINGERS]
    dataset, order = dataset_io.load(SUCCESS)

    samples = []
    for e in range(n_eps):
        if e not in dataset:
            continue
        st, ac = dataset[e]["observation.state"], dataset[e]["action"]
        f = np.where((np.abs(ac[:, PITCH]) > 0.35).all(axis=1))[0]
        if not len(f):
            continue
        data.qpos[ra] = st[int(f[0])]
        mujoco.mj_kinematics(model, data)
        g = 0.5 * (data.xpos[thumb]
                   + np.mean([data.xpos[i] for i in fingers], axis=0))
        packs = align.measure_bags(align.dataset_frame(SUCCESS, e, 0), config)
        if not packs:
            continue
        pts = np.array([[p["box_centre_cm"][0] / 100.0 - table[0] / 2.0,
                         p["box_centre_cm"][1] / 100.0 - table[1] / 2.0]
                        for p in packs])
        samples.append({"episode": e, "g0": g[:2].copy(), "packs": pts})
    print("episodes with both a closure frame and a detected pack: %d" % len(samples))

    def residuals(x_cm, y_cm, yaw):
        c = np.array([x_cm / 100.0 - table[0] / 2.0, y_cm / 100.0 - table[1] / 2.0])
        d = yaw - yaw0
        rot = np.array([[np.cos(d), -np.sin(d)], [np.sin(d), np.cos(d)]])
        out = []
        for s in samples:
            gp = rot @ (s["g0"] - c0) + c
            out.append(float(np.min(np.linalg.norm(s["packs"] - gp, axis=1))))
        return np.array(out)

    def score(x_cm, y_cm, yaw):
        r = residuals(x_cm, y_cm, yaw)
        keep = r < 0.25                      # exclude detector misses, do not let them lead
        return (float(np.median(r[keep])) if keep.sum() >= 5 else 9.9,
                float(keep.mean()), int(keep.sum()), r)

    best = None
    for x_cm in np.arange(4.0, 32.01, 1.0):
        for y_cm in np.arange(-2.0, 22.01, 1.0):
            for yaw in np.arange(1.22, 1.93, 0.035):
                med, frac, n, _ = score(x_cm, y_cm, yaw)
                if best is None or med < best[0]:
                    best = (med, frac, n, x_cm, y_cm, yaw)
    med, frac, n, bx, by, byaw = best
    for _ in range(3):
        for dx in np.arange(-0.8, 0.81, 0.2):
            for dy in np.arange(-0.8, 0.81, 0.2):
                for dyaw in np.arange(-0.03, 0.031, 0.0075):
                    m2, f2, n2, _ = score(bx + dx, by + dy, byaw + dyaw)
                    if m2 < med:
                        med, frac, n, bx, by, byaw = m2, f2, n2, bx + dx, by + dy, byaw + dyaw
    cur = score(x0_cm, y0_cm, yaw0)
    print()
    print("  CURRENT base (%.1f, %.1f) yaw %.4f (%.2f deg):" %
          (x0_cm, y0_cm, yaw0, np.degrees(yaw0)))
    print("     median residual %.4f m  inliers %d/%d (%.0f%%)  mean %.4f"
          % (cur[0], cur[2], len(samples), cur[1] * 100, cur[3].mean()))
    print("  FITTED base  (%.2f, %.2f) yaw %.4f (%.2f deg):" %
          (bx, by, byaw, np.degrees(byaw)))
    print("     median residual %.4f m  inliers %d/%d (%.0f%%)  mean %.4f"
          % (med, n, len(samples), frac * 100, score(bx, by, byaw)[3].mean()))
    print()
    print("  shift from current: dx %+.2f cm  dy %+.2f cm  dyaw %+.2f deg"
          % (bx - x0_cm, by - y0_cm, np.degrees(byaw - yaw0)))
    print()
    print("  per-episode residual (m): current                    fitted")
    rcur, rfit = cur[3], score(bx, by, byaw)[3]
    for s, a, b in zip(samples, rcur, rfit):
        flag = "  <-- detector miss" if a > 0.25 else ""
        print("     ep%-4d %8.4f   %8.4f%s" % (s["episode"], a, b, flag))
    (OUT / "result.json").write_text(json.dumps(
        {"current": {"x_cm": x0_cm, "y_cm": y0_cm, "yaw": yaw0,
                     "median": cur[0], "inliers": cur[2]},
         "fitted": {"x_cm": bx, "y_cm": by, "yaw": byaw, "median": med, "inliers": n},
         "n_samples": len(samples)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
