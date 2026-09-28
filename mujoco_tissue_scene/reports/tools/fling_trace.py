#!/usr/bin/env python3
"""When is the pack launched?  Trace height, hand contact and aperture together.

ep16 reaches 0.010 m from the box centre yet ends 0.00 inside with a 0.342 m lift,
which is a throw rather than a carry.  Comparing it with a success separates
"squeezed out at closure" from "lost during the carry".
"""
from __future__ import annotations

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
OUT = ROOT / "outputs/fling_trace"
THUMB = "hand_L_thumb_tip"
FINGERS = ["hand_L_index_tip", "hand_L_middle_tip", "hand_L_ring_tip", "hand_L_pinky_tip"]
PITCH = [8, 10, 11, 13, 15]


def main() -> int:
    import mujoco
    OUT.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    path = OUT / "scene.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    surface = config["table"]["surface_z"]
    ts = config["table"]["size"]
    tcm = [ts[0] * 100.0, ts[1] * 100.0]
    tyaw = float(config["tray"].get("yaw", 0.0))
    dataset, order = dataset_io.load(SUCCESS)

    ref = mujoco.MjModel.from_xml_path(str(scene.build(
        path, OUT / "ref.xml", with_hand=True)))
    rd = mujoco.MjData(ref)
    ra = np.array(layout.joint_ids(ref, mujoco))
    thumb, fingers = ref.body(THUMB).id, [ref.body(n).id for n in FINGERS]
    target_cm = {}
    for e in (0, 11, 16):
        st, ac = dataset[e]["observation.state"], dataset[e]["action"]
        f = np.where((np.abs(ac[:, PITCH]) > 0.35).all(axis=1))[0]
        rd.qpos[ra] = st[int(f[0])]
        mujoco.mj_kinematics(ref, rd)
        g = 0.5 * (rd.xpos[thumb] + np.mean([rd.xpos[i] for i in fingers], axis=0))
        target_cm[e] = np.array([(g[0] + ts[0] / 2) * 100.0,
                                 (g[1] + ts[1] / 2) * 100.0])

    env = sim_env.TissueSceneEnv(config_path=path, dataset=SUCCESS, render=False,
                                 output_dir=OUT)
    model, data = env.model, env.data
    for e in (0, 16, 11):
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
        bags = [b for b in bags
                if np.hypot(b["box_centre_cm"][0] - cx,
                            b["box_centre_cm"][1] - cy) > 8.0]
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
        start_z = float(data.qpos[env.box_free[box["name"]][0] + 2])
        print()
        print("=== ep%d  target (%.1f, %.1f) cm ===" % (e, cx, cy))
        print("  %5s %8s %7s %7s %9s %8s" %
              ("step", "rise m", "d_box", "contact", "aperture", "vz m/s"))
        for i, value in enumerate(action):
            env.step(value)
            a2 = int(model.flex_vertadr[fid])
            num = int(model.flex_vertnum[fid])
            pts = data.flexvert_xpos[a2:a2 + num]
            c = pts.mean(0)
            rise = float(c[2]) - start_z
            db = float(np.linalg.norm(c[:2] - tray_pos[:2]))
            hit = 0
            for k in range(data.ncon):
                ct = data.contact[k]
                if int(ct.flex[0]) == fid or int(ct.flex[1]) == fid:
                    other = int(ct.geom1) if int(ct.geom2) < 0 else int(ct.geom2)
                    if other >= 0 and (model.body(int(model.geom_bodyid[other])).name
                                       or "").startswith("hand_"):
                        hit += 1
            if i % 4 == 0 and -0.01 < rise or (i % 4 == 0 and hit):
                ap = float(np.linalg.norm(data.xpos[thumb]
                                          - np.mean([data.xpos[f] for f in fingers], 0)))
                if i % 4 == 0:
                    print("  %5d %8.3f %7.3f %7d %9.3f %8.3f"
                          % (i, rise, db, hit, ap, float(c[2])))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
