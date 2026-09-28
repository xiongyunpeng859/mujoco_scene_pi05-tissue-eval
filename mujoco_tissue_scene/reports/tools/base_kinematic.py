#!/usr/bin/env python3
"""Pin the arm base with a purely kinematic criterion.

In every one of the 108 successful episodes the real arm must bring its hand over
the box, so the minimum palm-to-box distance over an episode is a necessary
condition on the base that involves no grasping and no bag detection.

A height band is required too: the palm must be in a *placing* pose just above the
box rim, which also rules out the trivial answer of parking the base on the box (the
folded start pose sits much higher).

The scene is built once and the base is moved by writing body_pos/body_quat, so the
whole grid costs seconds.
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
OUT = ROOT / "outputs/base_kinematic"


def main() -> int:
    import mujoco
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--skip", type=int, default=50, help="ignore the folded start")
    parser.add_argument("--stage", choices=["coarse", "fine"], default="coarse")
    parser.add_argument("--best", type=str, default=None)
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
    # The object is held at the GRIP POINT -- halfway between the thumb tip and the
    # centroid of the four fingertips -- not at the palm base, so this is the point
    # that must arrive over the box.
    thumb = model.body("hand_L_thumb_tip").id
    fingers = [model.body("hand_L_%s_tip" % f).id
               for f in ("index", "middle", "ring", "pinky")]
    mujoco.mj_kinematics(model, data)
    tray_xy = data.xpos[tray_body][:2].copy()

    episodes, order = dataset_io.load(SUCCESS, fields=("observation.state",))
    states = [episodes[e]["observation.state"][:, :6] for e in order[:args.episodes]]
    z_lo, z_hi = surface + 0.01, surface + 0.14
    print("episodes=%d  box centre %s  placing height band %.3f..%.3f"
          % (len(states), np.round(tray_xy, 4).tolist(), z_lo, z_hi))

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
                if not (z_lo <= grip[2] <= z_hi):
                    continue
                d = float(np.linalg.norm(grip[:2] - tray_xy))
                if d < best:
                    best = d
            per.append(best)
        per = np.array(per)
        return per

    if args.stage == "coarse":
        xs = np.arange(14.0, 46.1, 2.0)
        ys = np.arange(0.0, 34.1, 2.0)
        yaws = np.radians(np.arange(30.0, 121.0, 7.5))
    else:
        bx, by, byaw = (float(v) for v in args.best.split(","))
        xs = bx + np.arange(-3.0, 3.1, 1.0)
        ys = by + np.arange(-3.0, 3.1, 1.0)
        yaws = byaw + np.radians(np.arange(-9.0, 9.1, 1.5))

    rows = []
    for x_cm, y_cm, yaw in itertools.product(xs, ys, yaws):
        per = evaluate(float(x_cm), float(y_cm), float(yaw))
        rows.append({"x": float(x_cm), "y": float(y_cm), "yaw": float(yaw),
                     "mean": float(per.mean()), "worst": float(per.max()),
                     "frac12": float((per < 0.12).mean()),
                     "frac6": float((per < 0.06).mean())})
    # a good base satisfies every episode, then is close on average
    rows.sort(key=lambda r: (-r["frac12"], -r["frac6"], r["worst"], r["mean"]))
    print()
    print("  %-30s %9s %9s %8s %8s" %
          ("base / yaw", "mean", "worst", "<12cm", "<6cm"))
    for r in rows[:14]:
        print("  x=%5.1f y=%5.1f yaw=%6.1f (%5.1f deg) %9.4f %9.4f %8.2f %8.2f"
              % (r["x"], r["y"], r["yaw"], np.degrees(r["yaw"]), r["mean"], r["worst"],
                 r["frac12"], r["frac6"]))
    best = rows[0]
    print()
    print("  BEST x=%.1f y=%.1f yaw=%.1f deg -> mean %.4f, worst %.4f, "
          "%.0f%% within 12cm, %.0f%% within 6cm"
          % (best["x"], best["y"], np.degrees(best["yaw"]), best["mean"], best["worst"],
             100 * best["frac12"], 100 * best["frac6"]))
    for label, x, y, yaw in (("previous (34,24,60)", 34.0, 24.0, np.radians(60)),
                             ("grasp fit (18,10,90)", 18.0, 10.0, np.radians(90)),
                             ("tape (25,10,90)", 25.0, 10.0, np.radians(90)),
                             ("(34,24,90)", 34.0, 24.0, np.radians(90))):
        per = evaluate(x, y, yaw)
        print("  %-22s mean %.4f worst %.4f  <12cm %.2f"
              % (label, per.mean(), per.max(), (per < 0.12).mean()))
    (OUT / ("result_%s.json" % args.stage)).write_text(json.dumps(rows[:300], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
