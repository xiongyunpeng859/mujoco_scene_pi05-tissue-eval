#!/usr/bin/env python3
"""Align the base with a grasp-independent criterion: does the hand go over the box?

In every successful episode the real arm must bring its hand over the tray, or the
pack could not be placed.  So the minimum distance from the palm to the tray centre
over an episode is a purely kinematic, continuous criterion that needs no grasp and
no bag detection -- which makes a full grid search cheap.

Only mj_kinematics is used, so thousands of candidates over dozens of episodes run
in a minute.
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
OUT = ROOT / "outputs/hand_tray_search"


def main() -> int:
    import mujoco
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--stage", choices=["coarse", "fine"], default="coarse")
    parser.add_argument("--best", type=str, default=None)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    tray = config["tray"]
    size = config["table"]["size"]
    tray_cm = [(tray["center"][0] + size[0] / 2) * 100.0,
               (tray["center"][1] + size[1] / 2) * 100.0]
    episodes, order = dataset_io.load(SUCCESS, fields=("observation.state",))
    chosen = order[:args.episodes]
    states = [episodes[e]["observation.state"] for e in chosen]
    print("tray centre (table cm) = (%.2f, %.2f) ; episodes %d" %
          (tray_cm[0], tray_cm[1], len(chosen)))

    # build once with a placeholder base; only the arm mount transform changes per
    # candidate, so re-parent by rebuilding is simplest and still fast enough.
    base_config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())

    def evaluate(x_cm, y_cm, yaw):
        cfg = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
        cfg["measured_layout"]["arm_mount_xy_cm_from_left_bottom"] = [x_cm, y_cm]
        cfg["arm"]["euler"][2] = yaw
        path = OUT / "cand.yaml"
        path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
        xml = scene.build(path, OUT / "cand.xml", with_hand=True)
        model = mujoco.MjModel.from_xml_path(str(xml))
        data = mujoco.MjData(model)
        adr = [model.joint("joint%d" % i).qposadr[0] for i in range(1, 7)]
        palm = model.body("static_omnihand_reference").id
        tray_body = model.body("tray").id
        mujoco.mj_forward(model, data)
        tray_xy = data.xpos[tray_body][:2].copy()
        distances = []
        for state in states:
            best = 9.9
            for row in state:
                data.qpos[adr] = row[:6]
                mujoco.mj_kinematics(model, data)
                best = min(best, float(np.linalg.norm(data.xpos[palm][:2] - tray_xy)))
            distances.append(best)
        return float(np.mean(distances)), distances

    if args.stage == "coarse":
        xs = np.arange(10.0, 32.1, 2.0)
        ys = np.arange(0.0, 22.1, 2.0)
        yaws = np.radians(np.arange(60.0, 121.0, 7.5))
    else:
        bx, by, byaw = (float(v) for v in args.best.split(","))
        xs = bx + np.arange(-1.5, 1.6, 0.5)
        ys = by + np.arange(-1.5, 1.6, 0.5)
        yaws = byaw + np.radians(np.arange(-6.0, 6.1, 1.5))

    print("evaluating %d candidates" % (len(xs) * len(ys) * len(yaws)))
    rows = []
    for x_cm, y_cm, yaw in itertools.product(xs, ys, yaws):
        mean_d, per = evaluate(float(x_cm), float(y_cm), float(yaw))
        rows.append({"x": float(x_cm), "y": float(y_cm), "yaw": float(yaw),
                     "mean_hand_tray": mean_d,
                     "worst_episode": float(max(per)),
                     "frac_within_12cm": float(np.mean([d < 0.12 for d in per]))})
    rows.sort(key=lambda r: r["mean_hand_tray"])
    print()
    print("  %-24s %12s %12s %10s" %
          ("base / yaw", "mean d(hand,box)", "worst", "within 12cm"))
    for r in rows[:12]:
        print("  x=%5.1f y=%5.1f yaw=%6.1f  %12.4f %12.4f %10.2f"
              % (r["x"], r["y"], np.degrees(r["yaw"]), r["mean_hand_tray"],
                 r["worst_episode"], r["frac_within_12cm"]))
    best = rows[0]
    print()
    print("  BEST: x=%.1f y=%.1f yaw=%.4f rad (%.1f deg) -> mean %.4f m, worst %.4f m"
          % (best["x"], best["y"], best["yaw"], np.degrees(best["yaw"]),
             best["mean_hand_tray"], best["worst_episode"]))
    ref = [r for r in rows if abs(r["x"] - 25) < 0.1 and abs(r["y"] - 10) < 0.1
           and abs(np.degrees(r["yaw"]) - 90) < 0.1]
    if ref:
        print("  (user's tape base 25,10 / 90deg -> mean %.4f m)"
              % ref[0]["mean_hand_tray"])
    (OUT / "result_%s.json" % args.stage).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
