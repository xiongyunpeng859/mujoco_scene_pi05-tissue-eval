#!/usr/bin/env python3
"""Where does the bag end up, and why -- placement accuracy or a slip mid-carry?"""
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
import sim_env

SUCCESS = Path("/workspace/shared/new_program_qiuzhi/without_tactile/success_episode/"
               "pick_up_the_tissue_pack_and_place_it_on_the_right_side_"
               "20260908_merged_all_108eps")
OUT = ROOT / "outputs/final_position"
THUMB = "hand_L_thumb_tip"
FINGERS = ["hand_L_index_tip", "hand_L_middle_tip", "hand_L_ring_tip", "hand_L_pinky_tip"]
PITCH = [8, 10, 11, 13, 15]


def main() -> int:
    import mujoco
    OUT.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    surface = config["table"]["surface_z"]
    table = config["table"]["size"]
    tray = config["tray"]
    half = tray["size"][0] / 2.0
    episodes, order = dataset_io.load(SUCCESS)
    chosen = order[:6]

    ref = mujoco.MjModel.from_xml_path(str(scene.build(
        ROOT / "configs/scene.yaml", OUT / "ref.xml", with_hand=True)))
    rd = mujoco.MjData(ref)
    ra = np.array(layout.joint_ids(ref, mujoco))
    thumb, fingers = ref.body(THUMB).id, [ref.body(n).id for n in FINGERS]
    truth = {}
    for e in chosen:
        st, ac = episodes[e]["observation.state"], episodes[e]["action"]
        f = np.where(np.abs(ac[:, PITCH]).mean(1) > 0.35)[0]
        if len(f) == 0:
            continue
        rd.qpos[ra] = st[int(f[0])]
        mujoco.mj_kinematics(ref, rd)
        g = 0.5 * (rd.xpos[thumb] + np.mean([rd.xpos[i] for i in fingers], axis=0))
        truth[e] = np.array([(g[0] + table[0] / 2) * 100.0, (g[1] + table[1] / 2) * 100.0])

    path = OUT / "scene.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    env = sim_env.TissueSceneEnv(config_path=path, dataset=SUCCESS, render=False,
                                 output_dir=OUT)
    model, data = env.model, env.data
    yaw_t = float(tray.get("yaw", 0.0))
    cos_t, sin_t = np.cos(-yaw_t), np.sin(-yaw_t)
    print("  %-4s %9s %-24s %11s %10s %s" %
          ("ep", "peak lift", "bag final, table cm", "d(box ctr)", "overlap", "held at end"))
    rows = []
    for e in chosen:
        if e not in truth:
            continue
        action = episodes[e]["action"]
        env.reset(options={"state": episodes[e]["observation.state"][0],
                           "randomize_objects": False})
        box = config["boxes"][0]
        qpos_adr, dof_adr = env.box_free[box["name"]]
        data.qpos[qpos_adr + 0] = truth[e][0] / 100.0 - table[0] / 2.0
        data.qpos[qpos_adr + 1] = truth[e][1] / 100.0 - table[1] / 2.0
        data.qpos[qpos_adr + 2] = surface + box["size"][2] / 2.0 + 0.002
        data.qpos[qpos_adr + 3] = 1.0
        data.qpos[qpos_adr + 4:qpos_adr + 7] = 0.0
        data.qvel[dof_adr:dof_adr + 6] = 0.0
        mujoco.mj_forward(model, data)
        fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, box["name"] + "_soft")
        start_z = float(data.qpos[qpos_adr + 2])
        tray_pos = data.xpos[env.tray_body].copy()
        peak, held_last, dmin = 0.0, 0, 9.9
        for value in action:
            env.step(value)
            adr, num = int(model.flex_vertadr[fid]), int(model.flex_vertnum[fid])
            pts = data.flexvert_xpos[adr:adr + num]
            c = pts.mean(0)
            peak = max(peak, float(c[2]) - start_z)
            dmin = min(dmin, float(np.linalg.norm(c[:2] - tray_pos[:2])))
            held = False
            for i in range(data.ncon):
                ct = data.contact[i]
                if int(ct.flex[0]) == fid or int(ct.flex[1]) == fid:
                    other = int(ct.geom1) if int(ct.geom2) < 0 else int(ct.geom2)
                    if other >= 0 and (model.body(int(model.geom_bodyid[other])).name
                                       or "").startswith("hand_"):
                        held = True
            held_last = int(held)
        adr, num = int(model.flex_vertadr[fid]), int(model.flex_vertnum[fid])
        pts = data.flexvert_xpos[adr:adr + num]
        c = pts.mean(0)
        final_cm = [(c[0] + table[0] / 2) * 100.0, (c[1] + table[1] / 2) * 100.0]
        d = float(np.linalg.norm(c[:2] - tray_pos[:2]))
        dx, dy = pts[:, 0] - tray_pos[0], pts[:, 1] - tray_pos[1]
        lx = cos_t * dx - sin_t * dy
        ly = sin_t * dx + cos_t * dy
        frac = float((((np.abs(lx) < half) & (np.abs(ly) < tray["size"][1] / 2)
                       & (pts[:, 2] > tray_pos[2] + 0.005)).mean()))
        rows.append({"episode": e, "peak": peak, "final_cm": final_cm,
                     "final_d": d, "overlap": frac, "closest": dmin,
                     "held_at_end": bool(held_last)})
        print("  %-4d %9.3f (%5.1f, %5.1f)          %11.4f %10.2f %s"
              % (e, peak, final_cm[0], final_cm[1], d, frac,
                 "yes" if held_last else "no"))
    env.close()
    print()
    print("  closest the bag ever came to the box centre, and where it ended:")
    for r in rows:
        print("     ep%-3d closest %.3f m   final %.3f m   overlap %.2f   held at end %s"
              % (r["episode"], r["closest"], r["final_d"], r["overlap"],
                 r["held_at_end"]))
    (OUT / "result.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
