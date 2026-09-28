#!/usr/bin/env python3
"""Decouple the base from the bag detector.

The grasped pack's true position is known from the robot's own finger closure
(bag_truth.py).  Placing it there instead of at the detector's estimate removes the
detector from the loop, so a base search can no longer be skewed by it -- which is
what made earlier searches prefer a wrong base.
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
import action_layout as layout          # noqa: E402
import align_with_dataset as align      # noqa: E402
import dataset_io                       # noqa: E402
import scene                            # noqa: E402
import sim_env                          # noqa: E402

SUCCESS = Path("/workspace/shared/new_program_qiuzhi/without_tactile/success_episode/"
               "pick_up_the_tissue_pack_and_place_it_on_the_right_side_"
               "20260908_merged_all_108eps")
OUT = ROOT / "outputs/decoupled"
THUMB = "hand_L_thumb_tip"
FINGERS = ["hand_L_index_tip", "hand_L_middle_tip", "hand_L_ring_tip", "hand_L_pinky_tip"]
PITCH_DIMS = [8, 10, 11, 13, 15]
CLOSED = 0.35
LIFT_MIN = 0.05


def main() -> int:
    import mujoco
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--bases", type=str, default="18,10;20,12;22,14;24,16")
    parser.add_argument("--yaws", type=str, default="1.5708;1.4835;1.3963;1.3090")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    episodes, order = dataset_io.load(SUCCESS)
    chosen = [e for e in args.episodes if e in episodes]
    config0 = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    surface = config0["table"]["surface_z"]
    table = config0["table"]["size"]
    tray = config0["tray"]
    half_x = tray["size"][0] / 2.0
    half_y = tray["size"][1] / 2.0

    # 1) ground truth for the grasped pack, computed once with a reference base
    ref_xml = scene.build(ROOT / "configs/scene.yaml", OUT / "ref.xml", with_hand=True)
    ref = mujoco.MjModel.from_xml_path(str(ref_xml))
    refdata = mujoco.MjData(ref)
    thumb = ref.body(THUMB).id
    fingers = [ref.body(n).id for n in FINGERS]
    refadr = np.array(layout.joint_ids(ref, mujoco))
    truth = {}
    for e in chosen:
        action = episodes[e]["action"]
        state = episodes[e]["observation.state"]
        closed = np.abs(action[:, PITCH_DIMS]).mean(1)
        frames = np.where(closed > CLOSED)[0]
        if len(frames) == 0:
            continue
        refdata.qpos[refadr] = state[int(frames[0])]
        mujoco.mj_kinematics(ref, refdata)
        grip = 0.5 * (refdata.xpos[thumb]
                      + np.mean([refdata.xpos[i] for i in fingers], axis=0))
        truth[e] = np.array([(grip[0] + table[0] / 2) * 100.0,
                             (grip[1] + table[1] / 2) * 100.0])
    print("ground-truth pack positions (table cm), from the finger closure:")
    for e in chosen:
        if e in truth:
            det = align.measure_bags(align.dataset_frame(SUCCESS, e, 0), config0)
            nearest = None
            if det:
                c = np.array([b["box_centre_cm"] for b in det])
                nearest = c[int(np.argmin(np.linalg.norm(c - truth[e], axis=1)))]
            print("   ep%-3d truth (%5.1f, %5.1f)   detector %s"
                  % (e, truth[e][0], truth[e][1],
                     "none" if nearest is None else "(%5.1f, %5.1f)" % tuple(nearest)))
    print()

    bases = [tuple(float(v) for v in s.split(",")) for s in args.bases.split(";")]
    yaws = [float(v) for v in args.yaws.split(";")]
    print("  %-24s %9s %9s  %s" % ("base / yaw", "grasped", "placed", "per-episode lift"))
    rows = []
    for (bx, by), byaw in zip(bases, yaws):
        config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
        config["measured_layout"]["arm_mount_xy_cm_from_left_bottom"] = [bx, by]
        config["arm"]["euler"][2] = byaw
        path = OUT / ("x%05.1f_y%05.1f_a%06.4f" % (bx, by, byaw)) / "scene.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
        env = sim_env.TissueSceneEnv(config_path=path, dataset=SUCCESS, render=False,
                                     output_dir=path.parent)
        model, data = env.model, env.data
        yaw_t = float(config["tray"].get("yaw", 0.0))
        cos_t, sin_t = np.cos(-yaw_t), np.sin(-yaw_t)
        lifts, verdicts = [], []
        for e in chosen:
            action = episodes[e]["action"]
            state = episodes[e]["observation.state"]
            env.reset(options={"state": state[0], "randomize_objects": False})
            boxes = config["boxes"]
            # the grasped pack goes to its measured truth; the others stay detected
            det = align.measure_bags(align.dataset_frame(SUCCESS, e, 0), config)
            placed_target = False
            for index, box in enumerate(boxes):
                if index == 0 and e in truth:
                    px, py = truth[e]
                elif index < len(det):
                    px, py = det[index]["box_centre_cm"]
                else:
                    continue
                qpos_adr, dof_adr = env.box_free[box["name"]]
                data.qpos[qpos_adr + 0] = px / 100.0 - table[0] / 2.0
                data.qpos[qpos_adr + 1] = py / 100.0 - table[1] / 2.0
                data.qpos[qpos_adr + 2] = surface + box["size"][2] / 2.0 + 0.002
                data.qpos[qpos_adr + 3] = 1.0
                data.qpos[qpos_adr + 4:qpos_adr + 7] = 0.0
                data.qvel[dof_adr:dof_adr + 6] = 0.0
            mujoco.mj_forward(model, data)
            names = list(env.box_free)
            fids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, n + "_soft")
                    for n in names}
            start_z = {n: float(data.qpos[env.box_free[n][0] + 2]) for n in names}
            tray_pos = data.xpos[env.tray_body].copy()
            peak = {n: 0.0 for n in names}
            for value in action:
                env.step(value)
                for n in names:
                    adr, num = (int(model.flex_vertadr[fids[n]]),
                                int(model.flex_vertnum[fids[n]]))
                    peak[n] = max(peak[n],
                                  float(data.flexvert_xpos[adr:adr + num].mean(0)[2])
                                  - start_z[n])
            inside = False
            for n in names:
                adr, num = int(model.flex_vertadr[fids[n]]), int(model.flex_vertnum[fids[n]])
                points = data.flexvert_xpos[adr:adr + num]
                dx, dy = points[:, 0] - tray_pos[0], points[:, 1] - tray_pos[1]
                lx = cos_t * dx - sin_t * dy
                ly = sin_t * dx + cos_t * dy
                frac = float((((np.abs(lx) < half_x) & (np.abs(ly) < half_y)
                               & (points[:, 2] > tray_pos[2] + 0.005)).mean()))
                if frac > 0.3:
                    inside = True
            best = max(names, key=lambda n: peak[n])
            lifts.append(round(peak[best], 3))
            verdicts.append(bool(inside and peak[best] > LIFT_MIN))
        env.close()
        g = sum(1 for l in lifts if l > LIFT_MIN)
        p = sum(verdicts)
        rows.append({"base": [bx, by, byaw], "lift": lifts, "grasped": g, "placed": p})
        print("  x=%5.1f y=%5.1f a=%.4f %9d %9d  %s" % (bx, by, byaw, g, p, lifts))
    print()
    rows.sort(key=lambda r: (-r["placed"], -r["grasped"]))
    b = rows[0]
    print("  BEST base (%.1f, %.1f) yaw %.4f : grasped %d/%d, placed %d/%d"
          % (b["base"][0], b["base"][1], b["base"][2], b["grasped"], len(chosen),
             b["placed"], len(chosen)))
    (OUT / "result.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
