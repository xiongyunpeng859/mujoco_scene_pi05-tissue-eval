#!/usr/bin/env python3
"""Lock the arm base pose from one purely kinematic constraint:

    where the sim's fingers CLOSE  ==  where the pack actually IS in the image.

The pack cannot move before it is grasped, so these two must coincide.  The FK side
depends on the base pose; the image side does not.  That makes this a direct
measurement with no tunable threshold -- unlike the proxies that were retracted.

The base transform is planar, so FK is evaluated once per episode and the candidate
base is applied analytically; no model rebuild per grid point.
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
OUT = ROOT / "outputs/base_kinematic_fit"
THUMB = "hand_L_thumb_tip"
FINGERS = ["hand_L_index_tip", "hand_L_middle_tip", "hand_L_ring_tip", "hand_L_pinky_tip"]
PITCH = [8, 10, 11, 13, 15]
CFG = ROOT / "configs/scene.yaml"


def main() -> int:
    import mujoco
    OUT.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(CFG.read_text())
    table = config["table"]["size"]
    layout_cfg = config["measured_layout"]
    x0_cm, y0_cm = layout_cfg["arm_mount_xy_cm_from_left_bottom"]
    yaw0 = float(config["arm"]["euler"][2])
    c0 = np.array([x0_cm / 100.0 - table[0] / 2.0, y0_cm / 100.0 - table[1] / 2.0])
    print("reference base (%.1f, %.1f) cm  yaw %.4f rad" % (x0_cm, y0_cm, yaw0))

    model = mujoco.MjModel.from_xml_path(str(scene.build(
        CFG, OUT / "ref.xml", with_hand=True)))
    data = mujoco.MjData(model)
    ra = np.array(layout.joint_ids(model, mujoco))
    thumb, fingers = model.body(THUMB).id, [model.body(n).id for n in FINGERS]
    dataset, order = dataset_io.load(SUCCESS)

    episodes = list(range(int(sys.argv[1]) if len(sys.argv) > 1 else 24))
    samples = []
    for e in episodes:
        if e not in dataset:
            continue
        st, ac = dataset[e]["observation.state"], dataset[e]["action"]
        f = np.where(np.abs(ac[:, PITCH]).mean(1) > 0.35)[0]
        if not len(f):
            continue
        data.qpos[ra] = st[int(f[0])]
        mujoco.mj_kinematics(model, data)
        g = 0.5 * (data.xpos[thumb]
                   + np.mean([data.xpos[i] for i in fingers], axis=0))
        packs = align.measure_bags(align.dataset_frame(SUCCESS, e, 0), config)
        pts = np.array([[p["box_centre_cm"][0] / 100.0 - table[0] / 2.0,
                         p["box_centre_cm"][1] / 100.0 - table[1] / 2.0]
                        for p in packs])
        if not len(pts):
            continue
        samples.append({"episode": e, "g0": g[:2].copy(), "packs": pts,
                        "closure_frame": int(f[0])})
    print("episodes with both a closure frame and a detected pack: %d" % len(samples))

    def evaluate(x_cm, y_cm, yaw):
        c = np.array([x_cm / 100.0 - table[0] / 2.0, y_cm / 100.0 - table[1] / 2.0])
        d = yaw - yaw0
        rot = np.array([[np.cos(d), -np.sin(d)], [np.sin(d), np.cos(d)]])
        res = []
        for s in samples:
            gp = rot @ (s["g0"] - c0) + c
            res.append(float(np.min(np.linalg.norm(s["packs"] - gp, axis=1))))
        res = np.array(res)
        return res

    best = None
    for x_cm in np.arange(2.0, 34.01, 1.0):
        for y_cm in np.arange(-2.0, 24.01, 1.0):
            for yaw in np.arange(1.15, 1.99, 0.035):
                res = evaluate(x_cm, y_cm, yaw)
                score = float(np.median(res))
                if best is None or score < best[0]:
                    best = (score, x_cm, y_cm, yaw, res)
    score, bx, by, byaw, res = best
    for _ in range(3):
        for dx in np.arange(-0.8, 0.81, 0.2):
            for dy in np.arange(-0.8, 0.81, 0.2):
                for dyaw in np.arange(-0.03, 0.031, 0.0075):
                    r = evaluate(bx + dx, by + dy, byaw + dyaw)
                    s = float(np.median(r))
                    if s < score:
                        score, bx, by, byaw, res = s, bx + dx, by + dy, byaw + dyaw, r
    print()
    print("  BEST base from FK-closure == image-pack:")
    print("     x %.2f cm   y %.2f cm   yaw %.4f rad (%.2f deg)"
          % (bx, by, byaw, np.degrees(byaw)))
    print("     median residual %.4f m   mean %.4f m   worst %.4f m   within 5cm %d/%d"
          % (score, res.mean(), res.max(), int((res < 0.05).sum()), len(res)))
    print()
    print("  per-episode residual (m), current config vs fitted:")
    cur = evaluate(x0_cm, y0_cm, yaw0)
    print("     %-5s %10s %10s" % ("ep", "current", "fitted"))
    for s, a, b in zip(samples, cur, res):
        print("     %-5d %10.4f %10.4f" % (s["episode"], a, b))
    print()
    print("  current config: median %.4f  mean %.4f  worst %.4f  within 5cm %d/%d"
          % (np.median(cur), cur.mean(), cur.max(), int((cur < 0.05).sum()), len(cur)))
    (OUT / "result.json").write_text(json.dumps(
        {"fitted": {"x_cm": bx, "y_cm": by, "yaw": byaw, "median": score,
                    "mean": float(res.mean()), "worst": float(res.max())},
         "current": {"x_cm": x0_cm, "y_cm": y0_cm, "yaw": yaw0,
                     "median": float(np.median(cur)), "mean": float(cur.mean())},
         "per_episode": [{"episode": s["episode"], "current": float(a),
                          "fitted": float(b), "closure_frame": s["closure_frame"]}
                         for s, a, b in zip(samples, cur, res)]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
