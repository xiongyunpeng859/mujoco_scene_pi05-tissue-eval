#!/usr/bin/env python3
"""Fix the base with a grasp-independent, bag-independent criterion.

In every successful episode the real arm MUST bring its hand over the tray, or the
pack could not be placed.  So the minimum palm-to-tray distance over an episode is a
purely kinematic criterion: it involves no grasping and no bag detection, so it
cannot be biased by them.

The scene is built once and the base is moved by writing model.body_pos/body_quat at
run time, so the whole grid costs seconds instead of an hour.
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
OUT = ROOT / "outputs/hand_tray_fast"


def main() -> int:
    import mujoco
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--stage", choices=["coarse", "fine"], default="coarse")
    parser.add_argument("--best", type=str, default=None)
    parser.add_argument("--skip", type=int, default=60,
                        help="ignore the folded start pose; the hand must be over the "
                             "box when it PLACES, which is mid-episode")
    parser.add_argument("--require-high", action="store_true", default=True,
                        help="also require the palm to be above the box rim")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    surface = config["table"]["surface_z"]
    table = config["table"]["size"]
    xml = scene.build(ROOT / "configs/scene.yaml", OUT / "scene.xml", with_hand=True)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mount = model.body("qiuzhi_arm_mount").id
    palm = model.body("static_omnihand_reference").id
    tray_body = model.body("tray").id
    adr = [model.joint("joint%d" % i).qposadr[0] for i in range(1, 7)]
    mujoco.mj_kinematics(model, data)
    tray_xy = data.xpos[tray_body][:2].copy()

    episodes, order = dataset_io.load(SUCCESS, fields=("observation.state",))
    chosen = order[:args.episodes]
    states = [episodes[e]["observation.state"][:, :6] for e in chosen]
    print("episodes=%d  tray xy (world) = %s  base will be moved at run time"
          % (len(chosen), np.round(tray_xy, 4).tolist()))

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
                p3 = data.xpos[palm]
                if p3[2] < surface + 0.02:
                    continue
                d = float(np.linalg.norm(p3[:2] - tray_xy))
                if d < best:
                    best = d
            per.append(best)
        per = np.array(per)
        return float(per.mean()), float(per.max()), float((per < 0.12).mean()), per

    if args.stage == "coarse":
        xs = np.arange(8.0, 34.1, 2.0)
        ys = np.arange(-4.0, 24.1, 2.0)
        yaws = np.radians(np.arange(45.0, 136.0, 7.5))
    else:
        bx, by, byaw = (float(v) for v in args.best.split(","))
        xs = bx + np.arange(-3.0, 3.1, 1.0)
        ys = by + np.arange(-3.0, 3.1, 1.0)
        yaws = byaw + np.radians(np.arange(-10.0, 10.1, 2.5))

    rows = []
    for x_cm, y_cm, yaw in itertools.product(xs, ys, yaws):
        mean_d, worst, frac, per = evaluate(float(x_cm), float(y_cm), float(yaw))
        rows.append({"x": float(x_cm), "y": float(y_cm), "yaw": float(yaw),
                     "mean": mean_d, "worst": worst, "frac_within_12cm": frac})
    rows.sort(key=lambda r: r["mean"])
    print()
    print("  %-26s %10s %10s %12s" % ("base / yaw", "mean d", "worst", "within 12cm"))
    for r in rows[:10]:
        print("  x=%5.1f y=%5.1f yaw=%6.1f (%5.1f deg) %10.4f %10.4f %12.2f"
              % (r["x"], r["y"], r["yaw"], np.degrees(r["yaw"]), r["mean"], r["worst"],
                 r["frac_within_12cm"]))
    best = rows[0]
    print()
    print("  BEST x=%.1f y=%.1f yaw=%.1f deg -> mean %.4f m, worst %.4f m, "
          "%.0f%% of episodes bring the hand within 12 cm of the box"
          % (best["x"], best["y"], np.degrees(best["yaw"]), best["mean"],
             best["worst"], 100 * best["frac_within_12cm"]))
    user = [r for r in rows if abs(r["x"] - 25) < 0.1 and abs(r["y"] - 10) < 0.1
            and abs(np.degrees(r["yaw"]) - 90) < 0.1]
    if user:
        print("  user's tape value (25,10)/90deg -> mean %.4f m, worst %.4f m, "
              "%.0f%% within 12 cm"
              % (user[0]["mean"], user[0]["worst"], 100 * user[0]["frac_within_12cm"]))
    (OUT / ("result_%s.json" % args.stage)).write_text(json.dumps(rows[:200], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
