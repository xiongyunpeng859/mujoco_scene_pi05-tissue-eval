#!/usr/bin/env python3
"""Where in the carry does the pack stop following the hand?

For each episode: when does hand contact begin and end, where is the pack at the
last contact, and how far is that from the box?  A pack released far from the box
means the carry failed; a pack released at the box means the release failed.
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
OUT = ROOT / "outputs/carry_audit"
THUMB = "hand_L_thumb_tip"
FINGERS = ["hand_L_index_tip", "hand_L_middle_tip", "hand_L_ring_tip", "hand_L_pinky_tip"]
PITCH = [8, 10, 11, 13, 15]


def main() -> int:
    import mujoco
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, nargs="+",
                        default=[0, 2, 3, 6, 9, 11])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    path = OUT / "scene.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    surface = config["table"]["surface_z"]
    ts = config["table"]["size"]
    tcm = [ts[0] * 100.0, ts[1] * 100.0]
    tray = config["tray"]
    tyaw = float(tray.get("yaw", 0.0))
    dataset, order = dataset_io.load(SUCCESS)
    chosen = [e for e in args.episodes if e in dataset]

    ref = mujoco.MjModel.from_xml_path(str(scene.build(
        path, OUT / "ref.xml", with_hand=True)))
    rd = mujoco.MjData(ref)
    ra = np.array(layout.joint_ids(ref, mujoco))
    thumb, fingers = ref.body(THUMB).id, [ref.body(n).id for n in FINGERS]
    target_cm = {}
    for e in chosen:
        st, ac = dataset[e]["observation.state"], dataset[e]["action"]
        f = np.where((np.abs(ac[:, PITCH]) > 0.35).all(axis=1))[0]
        if not len(f):
            continue
        rd.qpos[ra] = st[int(f[0])]
        mujoco.mj_kinematics(ref, rd)
        g = 0.5 * (rd.xpos[thumb] + np.mean([rd.xpos[i] for i in fingers], axis=0))
        target_cm[e] = np.array([(g[0] + ts[0] / 2) * 100.0,
                                 (g[1] + ts[1] / 2) * 100.0])

    env = sim_env.TissueSceneEnv(config_path=path, dataset=SUCCESS, render=False,
                                 output_dir=OUT)
    model, data = env.model, env.data
    print("  %-4s %6s %7s %7s | %-16s %-16s %8s %8s" %
          ("ep", "grip", "first", "last", "pack at last contact",
           "box centre", "d_last", "d_final"))
    rows = []
    for e in chosen:
        if e not in target_cm:
            continue
        action, state = dataset[e]["action"], dataset[e]["observation.state"]
        env.reset(options={"state": state[0], "randomize_objects": False})
        boxes = config["boxes"]
        for box in boxes:
            adr, dof = env.box_free[box["name"]]
            data.qpos[adr + 0], data.qpos[adr + 1] = -2.0, -2.0
            data.qpos[adr + 2] = surface + box["size"][2] / 2.0
            data.qpos[adr + 3] = 1.0
            data.qpos[adr + 4:adr + 7] = 0.0
            data.qvel[dof:dof + 6] = 0.0
        box = boxes[0]
        adr, dof = env.box_free[box["name"]]
        cx, cy = target_cm[e]
        data.qpos[adr + 0] = cx / 100.0 - tcm[0] / 200.0
        data.qpos[adr + 1] = cy / 100.0 - tcm[1] / 200.0
        data.qpos[adr + 2] = surface + box["size"][2] / 2.0 + 0.002
        data.qpos[adr + 3] = np.cos(tyaw / 2.0)
        data.qpos[adr + 4:adr + 6] = 0.0
        data.qpos[adr + 6] = np.sin(tyaw / 2.0)
        data.qvel[dof:dof + 6] = 0.0
        bags = align.measure_bags(align.dataset_frame(SUCCESS, e, 0), config)
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
        fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, box["name"] + "_soft")
        tray_pos = data.xpos[env.tray_body].copy()
        box_cm = [(tray_pos[0] + tcm[0] / 200.0) * 100.0,
                  (tray_pos[1] + tcm[1] / 200.0) * 100.0]
        start_z = float(data.qpos[env.box_free[box["name"]][0] + 2])
        contact_steps, centres, lifts = [], [], []
        for i, value in enumerate(action):
            env.step(value)
            adr = int(model.flex_vertadr[fid])
            num = int(model.flex_vertnum[fid])
            pts = data.flexvert_xpos[adr:adr + num]
            c = pts.mean(0)
            centres.append(c.copy())
            lifts.append(float(c[2]) - start_z)
            hit = 0
            for k in range(data.ncon):
                ct = data.contact[k]
                if int(ct.flex[0]) == fid or int(ct.flex[1]) == fid:
                    other = int(ct.geom1) if int(ct.geom2) < 0 else int(ct.geom2)
                    if other >= 0 and (model.body(int(model.geom_bodyid[other])).name
                                       or "").startswith("hand_"):
                        hit += 1
            if hit:
                contact_steps.append(i)
        centres = np.array(centres)
        fmt = lambda c: "(%6.1f,%6.1f)" % ((c[0] + tcm[0] / 200.0) * 100.0,
                                           (c[1] + tcm[1] / 200.0) * 100.0)
        if contact_steps:
            last = contact_steps[-1]
            at_last = centres[last]
            d_last = float(np.linalg.norm(
                at_last[:2] - np.array([tray_pos[0], tray_pos[1]])))
            print("  %-4d %6.1f %7d %7d | %-16s %-16s %8.3f %8.3f"
                  % (e, float(np.max(lifts)), contact_steps[0], last,
                     fmt(at_last), fmt(np.array([tray_pos[0], tray_pos[1]])),
                     d_last,
                     float(np.linalg.norm(centres[-1][:2]
                                          - np.array([tray_pos[0], tray_pos[1]])))),
                  flush=True)
            rows.append({"episode": e, "grip_cm": [round(float(cx), 2), round(float(cy), 2)],
                         "peak_lift": round(float(np.max(lifts)), 4),
                         "first_contact": contact_steps[0], "last_contact": last,
                         "contact_frames": len(contact_steps),
                         "pack_at_last_contact_cm": [round(float((at_last[0] + tcm[0] / 200.0) * 100.0), 2),
                                                     round(float((at_last[1] + tcm[1] / 200.0) * 100.0), 2)],
                         "d_at_last_contact": round(d_last, 4)})
        else:
            print("  %-4d %6.3f %7s %7s | never contacted the hand"
                  % (e, float(np.max(lifts)), "-", "-"), flush=True)
    env.close()
    (OUT / "result.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
