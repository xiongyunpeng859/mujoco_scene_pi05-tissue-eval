#!/usr/bin/env python3
"""Final base refinement with a smooth, threshold-free criterion.

In every successful episode the real robot put the pack in the box, so the grip point
-- where the object is held -- must arrive well inside the box.  The mean (and worst)
minimum grip-to-box-centre distance over many episodes is continuous, needs no height
band, no bag detection and no grasping, and runs at pure-kinematics speed.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import dataset_io                       # noqa: E402
import scene                            # noqa: E402

SUCCESS = Path("/workspace/shared/new_program_qiuzhi/without_tactile/success_episode/"
               "pick_up_the_tissue_pack_and_place_it_on_the_right_side_"
               "20260908_merged_all_108eps")
OUT = ROOT / "outputs/base_refine_final"
THUMB = "hand_L_thumb_tip"
FINGERS = ["hand_L_index_tip", "hand_L_middle_tip", "hand_L_ring_tip", "hand_L_pinky_tip"]


def main() -> int:
    import mujoco
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--skip", type=int, default=40)
    parser.add_argument("--stage", choices=["coarse", "fine"], default="coarse")
    parser.add_argument("--best", type=str, default=None)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    surface = config["table"]["surface_z"]
    table = config["table"]["size"]
    half = config["tray"]["size"][0] / 2.0
    xml = scene.build(ROOT / "configs/scene.yaml", OUT / "scene.xml", with_hand=True)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mount = model.body("qiuzhi_arm_mount").id
    thumb = model.body(THUMB).id
    fingers = [model.body(n).id for n in FINGERS]
    tray_body = model.body("tray").id
    adr = [model.joint("joint%d" % i).qposadr[0] for i in range(1, 7)]
    mujoco.mj_kinematics(model, data)
    tray_world = data.xpos[tray_body][:2].copy()

    episodes, order = dataset_io.load(SUCCESS, fields=("observation.state",))
    states = [episodes[e]["observation.state"][:, :6] for e in order[:args.episodes]]
    print("episodes=%d  box half-size %.3f m  (baseline base (18,10) yaw 90deg)"
          % (len(states), half))

    def evaluate(x_cm, y_cm, yaw):
        model.body_pos[mount] = [x_cm / 100.0 - table[0] / 2.0,
                                 y_cm / 100.0 - table[1] / 2.0, surface]
        model.body_quat[mount] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        per = []
        for rows in states:
            best = 9.9
            for row in rows[args.skip:]:
                data.qpos[adr] = row
                mujoco.mj_kinematics(model, data)
                grip = 0.5 * (data.xpos[thumb]
                              + np.mean([data.xpos[i] for i in fingers], axis=0))
                d = float(np.linalg.norm(grip[:2] - tray_world))
                if d < best:
                    best = d
            per.append(best)
        per = np.array(per)
        return per

    if args.stage == "coarse":
        xs = np.arange(12.0, 26.1, 2.0)
        ys = np.arange(4.0, 18.1, 2.0)
        yaws = np.radians(np.arange(75.0, 106.0, 5.0))
    else:
        bx, by, byaw = (float(v) for v in args.best.split(","))
        xs = bx + np.arange(-1.0, 1.05, 0.5)
        ys = by + np.arange(-1.0, 1.05, 0.5)
        yaws = byaw + np.radians(np.arange(-3.0, 3.1, 1.0))

    rows = []
    for x_cm, y_cm, yaw in itertools.product(xs, ys, yaws):
        per = evaluate(float(x_cm), float(y_cm), float(yaw))
        rows.append({"x": float(x_cm), "y": float(y_cm), "yaw": float(yaw),
                     "mean": float(per.mean()), "worst": float(per.max()),
                     "frac_inside": float((per < half).mean()),
                     "frac_8cm": float((per < 0.08).mean())})
    rows.sort(key=lambda r: (-r["frac_inside"], -r["frac_8cm"], r["mean"]))
    print()
    print("  %-30s %9s %9s %9s %8s" %
          ("base / yaw", "mean", "worst", "<half", "<8cm"))
    for r in rows[:14]:
        print("  x=%5.1f y=%5.1f yaw=%6.1f (%5.1f deg) %9.4f %9.4f %9.2f %8.2f"
              % (r["x"], r["y"], r["yaw"], np.degrees(r["yaw"]), r["mean"], r["worst"],
                 r["frac_inside"], r["frac_8cm"]))
    best = rows[0]
    print()
    print("  BEST x=%.1f y=%.1f yaw=%.1f deg -> mean %.4f m, worst %.4f m, "
          "%.0f%% inside the box radius (%.0f%% within 8cm)"
          % (best["x"], best["y"], np.degrees(best["yaw"]), best["mean"], best["worst"],
             100 * best["frac_inside"], 100 * best["frac_8cm"]))
    (OUT / ("result_%s.json" % args.stage)).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
