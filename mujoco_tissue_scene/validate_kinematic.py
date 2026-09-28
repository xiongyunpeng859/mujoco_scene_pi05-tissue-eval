#!/usr/bin/env python3
"""Reproduction rate using the VALIDATED kinematic target estimate.

The target pack's pose is inverted from the sim itself: the fingertip midpoint at
the frame the hand closes.  inspect_grip.py shows that point landing on the real
pack in the image (ep2), and it agrees with the image detector to 2.2-2.5 cm where
the detector works (ep0, ep1).  So it is a direct measurement, not a proxy.

Other packs go where the detector puts them; packs with no detection are parked out
of the workspace so leftover packs cannot block the tray.
"""
from __future__ import annotations

import argparse
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
import sim_env

SUCCESS = Path("/workspace/shared/new_program_qiuzhi/without_tactile/success_episode/"
               "pick_up_the_tissue_pack_and_place_it_on_the_right_side_"
               "20260908_merged_all_108eps")
OUT = ROOT / "outputs/validate_kinematic"
THUMB = "hand_L_thumb_tip"
FINGERS = ["hand_L_index_tip", "hand_L_middle_tip", "hand_L_ring_tip", "hand_L_pinky_tip"]
PITCH = [8, 10, 11, 13, 15]
LIFT_MIN = 0.05


def main() -> int:
    import mujoco
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, nargs="+", default=list(range(12)))
    parser.add_argument("--x", type=float, default=None)
    parser.add_argument("--y", type=float, default=None)
    parser.add_argument("--yaw", type=float, default=None)
    parser.add_argument("--target-yaw", choices=["tray", "closing"],
                        default="tray")
    parser.add_argument("--target-source", choices=["fk", "fused"],
                        default="fk")
    parser.add_argument("--pack-offset-cm", type=float, default=0.0)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    if args.x is not None:
        config["measured_layout"]["arm_mount_xy_cm_from_left_bottom"] = [args.x, args.y]
    if args.yaw is not None:
        config["arm"]["euler"][2] = args.yaw
    path = OUT / "scene.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    surface = config["table"]["surface_z"]
    ts = config["table"]["size"]
    tcm = [ts[0] * 100.0, ts[1] * 100.0]
    tray = config["tray"]
    tyaw = float(tray.get("yaw", 0.0))
    cos_t, sin_t = np.cos(-tyaw), np.sin(-tyaw)
    half_x, half_y = tray["size"][0] / 2.0, tray["size"][1] / 2.0

    dataset, order = dataset_io.load(SUCCESS)
    chosen = [e for e in args.episodes if e in dataset]

    # FK: invert each episode's target pack pose from the closure frame
    ref = mujoco.MjModel.from_xml_path(str(scene.build(
        path, OUT / "ref.xml", with_hand=True)))
    rd = mujoco.MjData(ref)
    ra = np.array(layout.joint_ids(ref, mujoco))
    thumb = ref.body(THUMB).id
    fingers = [ref.body(n).id for n in FINGERS]
    target_cm = {}
    target_yaw = {}
    target_source = {}
    all_packs = {}
    target_yaw_close = {}
    for e in chosen:
        st, ac = dataset[e]["observation.state"], dataset[e]["action"]
        f = np.where((np.abs(ac[:, PITCH]) > 0.35).all(axis=1))[0]
        if not len(f):
            continue
        rd.qpos[ra] = st[int(f[0])]
        mujoco.mj_kinematics(ref, rd)
        finger_mid = np.mean([rd.xpos[i] for i in fingers], axis=0)
        g = 0.5 * (rd.xpos[thumb] + finger_mid)
        target_cm[e] = np.array([(g[0] + ts[0] / 2) * 100.0,
                                 (g[1] + ts[1] / 2) * 100.0])
        # the image gives the pack's real CENTRE, capturing an off-centre grasp
        packs = align.measure_bags(align.dataset_frame(SUCCESS, e, 0), config)
        all_packs[e] = packs
        if args.target_source == "fused" and packs:
            dist = [np.hypot(p["box_centre_cm"][0] - target_cm[e][0],
                             p["box_centre_cm"][1] - target_cm[e][1]) for p in packs]
            j = int(np.argmin(dist))
            if dist[j] < 15.0:
                target_cm[e] = np.array(packs[j]["box_centre_cm"], dtype=float)
                target_source[e] = "image (+%.1f cm)" % dist[j]
            else:
                target_source[e] = "FK closure"
        else:
            target_source[e] = "FK closure"
        # pinch closes across the pack, so the long axis is perpendicular to it
        closing = finger_mid[:2] - rd.xpos[thumb][:2]
        norm = float(np.linalg.norm(closing))
        if norm > 1e-6:
            perp = np.array([-closing[1], closing[0]]) / norm
            target_yaw_close[e] = float(np.arctan2(perp[1], perp[0]))
        else:
            target_yaw_close[e] = float(tyaw)

    for e in target_yaw_close:
        target_yaw[e] = (float(tyaw) if args.target_yaw == "tray"
                         else target_yaw_close[e])
    env = sim_env.TissueSceneEnv(config_path=path, dataset=SUCCESS, render=False,
                                 output_dir=OUT)
    model, data = env.model, env.data
    print("  %-4s %-18s %8s %8s %8s %9s %8s %8s  %s" %
          ("ep", "target cm", "yaw", "peak", "nearest", "grasped", "in-tray", "placed", "source"))
    rows = []
    for e in chosen:
        if e not in target_cm:
            continue
        action, state = dataset[e]["action"], dataset[e]["observation.state"]
        env.reset(options={"state": state[0], "randomize_objects": False})
        boxes = config["boxes"]
        for box in boxes:                       # park everything first
            adr, dof = env.box_free[box["name"]]
            data.qpos[adr + 0], data.qpos[adr + 1] = -2.0, -2.0
            data.qpos[adr + 2] = surface + box["size"][2] / 2.0
            data.qpos[adr + 3] = 1.0
            data.qpos[adr + 4:adr + 7] = 0.0
            data.qvel[dof:dof + 6] = 0.0
        # target at the inverted pose, long axis along the carry direction
        box = boxes[0]
        adr, dof = env.box_free[box["name"]]
        cx, cy = target_cm[e]
        if args.pack_offset_cm:
            # emulate an off-centre grasp: the pack extends ahead of the fingers, so
            # shift it along the direction in which it will be carried
            tpx = (data.xpos[env.tray_body][0] + tcm[0] / 200.0) * 100.0
            tpy = (data.xpos[env.tray_body][1] + tcm[1] / 200.0) * 100.0
            vx, vy = tpx - cx, tpy - cy
            nn = float(np.hypot(vx, vy)) or 1.0
            cx = cx + args.pack_offset_cm * vx / nn
            cy = cy + args.pack_offset_cm * vy / nn
        data.qpos[adr + 0] = cx / 100.0 - tcm[0] / 200.0
        data.qpos[adr + 1] = cy / 100.0 - tcm[1] / 200.0
        data.qpos[adr + 2] = surface + box["size"][2] / 2.0 + 0.002
        data.qpos[adr + 3] = np.cos(target_yaw[e] / 2.0)
        data.qpos[adr + 4:adr + 6] = 0.0
        data.qpos[adr + 6] = np.sin(target_yaw[e] / 2.0)
        data.qvel[dof:dof + 6] = 0.0
        # other packs at their detected positions
        bags = all_packs.get(e) or []
        # the FK closure point and the detector agree to ~2.5 cm, so a detection
        # that close is the SAME pack -- placing it again overlaps two flexes
        bags = [b for b in bags
                if np.hypot(b["box_centre_cm"][0] - cx,
                            b["box_centre_cm"][1] - cy) > 8.0]   # not duplicate-of-target
        for box2, bag in zip(boxes[1:], bags):
            bx, by = bag["box_centre_cm"]
            adr, dof = env.box_free[box2["name"]]
            data.qpos[adr + 0] = bx / 100.0 - tcm[0] / 200.0
            data.qpos[adr + 1] = by / 100.0 - tcm[1] / 200.0
            data.qpos[adr + 2] = surface + box2["size"][2] / 2.0 + 0.002
            data.qpos[adr + 3] = 1.0
            data.qpos[adr + 4:adr + 7] = 0.0
            data.qvel[dof:dof + 6] = 0.0
        mujoco.mj_forward(model, data)
        names = list(env.box_free)
        fids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, n + "_soft")
                for n in names}
        start_z = {n: float(data.qpos[env.box_free[n][0] + 2]) for n in names}
        tray_pos = data.xpos[env.tray_body].copy()
        peak = {n: 0.0 for n in names}
        nearest = {n: 9.9 for n in names}
        for value in action:
            env.step(value)
            for n in names:
                a2, num = (int(model.flex_vertadr[fids[n]]),
                           int(model.flex_vertnum[fids[n]]))
                c = data.flexvert_xpos[a2:a2 + num].mean(0)
                rise = float(c[2]) - start_z[n]
                peak[n] = max(peak[n], rise)
                if rise > 0.02:
                    nearest[n] = min(nearest[n],
                                     float(np.linalg.norm(c[:2] - tray_pos[:2])))
        best = max(names, key=lambda n: peak[n])
        a2 = int(model.flex_vertadr[fids[best]])
        num = int(model.flex_vertnum[fids[best]])
        pts = data.flexvert_xpos[a2:a2 + num]
        ddx, ddy = pts[:, 0] - tray_pos[0], pts[:, 1] - tray_pos[1]
        lx = cos_t * ddx - sin_t * ddy
        ly = sin_t * ddx + cos_t * ddy
        frac = float(((np.abs(lx) < half_x) & (np.abs(ly) < half_y)
                      & (pts[:, 2] > tray_pos[2] + 0.005)).mean())
        grasped = peak[best] > LIFT_MIN
        placed = bool(grasped and frac > 0.3)
        print("  %-4d (%6.1f,%6.1f) y%6.1f %8.4f %8s %9s %8.2f %8s  %s"
              % (e, cx, cy, np.degrees(target_yaw[e]), peak[best],
                 ("%.3f" % nearest[best]) if nearest[best] < 9 else "-",
                 grasped, frac, placed, target_source.get(e, "")), flush=True)
        rows.append({"episode": e, "target_cm": [round(float(cx), 2), round(float(cy), 2)],
                     "peak": round(peak[best], 4), "grasped": grasped,
                     "inside": round(frac, 3), "placed": placed})
    env.close()
    g = sum(r["grasped"] for r in rows)
    p = sum(r["placed"] for r in rows)
    print()
    print("  KINEMATIC TARGET: grasped %d/%d   placed %d/%d" % (g, len(rows), p, len(rows)))
    (OUT / "result.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
