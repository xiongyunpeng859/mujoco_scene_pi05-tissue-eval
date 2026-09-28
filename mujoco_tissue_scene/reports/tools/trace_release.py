#!/usr/bin/env python3
"""When the hand opens over the box, does the bag stay in or follow the hand out?"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import align_with_dataset as align
import dataset_io
import sim_env

SUCCESS = Path("/workspace/shared/new_program_qiuzhi/without_tactile/success_episode/"
               "pick_up_the_tissue_pack_and_place_it_on_the_right_side_"
               "20260908_merged_all_108eps")
OUT = ROOT / "outputs/trace_release"
THUMB = "hand_L_thumb_tip"
FINGERS = ["hand_L_index_tip", "hand_L_middle_tip", "hand_L_ring_tip", "hand_L_pinky_tip"]


def main() -> int:
    import mujoco
    OUT.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    path = OUT / "scene.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    surface = config["table"]["surface_z"]
    table_cm = [config["table"]["size"][0] * 100.0, config["table"]["size"][1] * 100.0]
    yaw_t = float(config["tray"].get("yaw", 0.0))
    cos_t, sin_t = np.cos(-yaw_t), np.sin(-yaw_t)
    half_x = config["tray"]["size"][0] / 2.0
    half_y = config["tray"]["size"][1] / 2.0

    episodes, order = dataset_io.load(SUCCESS)
    env = sim_env.TissueSceneEnv(config_path=path, dataset=SUCCESS, render=False,
                                 output_dir=OUT)
    model, data = env.model, env.data
    thumb = model.body(THUMB).id
    fingers = [model.body(n).id for n in FINGERS]

    for episode in (0, 1, 4):
        action = episodes[episode]["action"]
        state = episodes[episode]["observation.state"]
        env.reset(options={"state": state[0], "randomize_objects": False})
        bags = align.measure_bags(align.dataset_frame(SUCCESS, episode, 0), config)
        for box, bag in zip(config["boxes"], bags):
            bx, by = bag["box_centre_cm"]
            byaw = np.radians(bag["yaw_deg"])
            adr, dof = env.box_free[box["name"]]
            data.qpos[adr + 0] = bx / 100.0 - table_cm[0] / 200.0
            data.qpos[adr + 1] = by / 100.0 - table_cm[1] / 200.0
            data.qpos[adr + 2] = surface + box["size"][2] / 2.0 + 0.002
            data.qpos[adr + 3] = np.cos(byaw / 2.0)
            data.qpos[adr + 4:adr + 7] = 0.0
            data.qvel[dof:dof + 6] = 0.0
        mujoco.mj_forward(model, data)
        names = list(env.box_free)
        fids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, n + "_soft")
                for n in names}
        tray = data.xpos[env.tray_body].copy()
        print()
        print("=== episode %d : %d steps, box at (%.1f, %.1f) cm, %d bags detected ==="
              % (episode, len(action), (tray[0] + table_cm[0] / 200.0) * 100.0,
                 (tray[1] + table_cm[1] / 200.0) * 100.0, len(bags)))
        print("  %5s %8s %8s %7s %7s %6s %8s %8s" %
              ("step", "d_box", "bag cm", "inX", "inY", "contact", "aperture", "held"))
        history = []
        for i, value in enumerate(action):
            env.step(value)
            line = None
            for n in names:
                adr, num = (int(model.flex_vertadr[fids[n]]),
                            int(model.flex_vertnum[fids[n]]))
                pts = data.flexvert_xpos[adr:adr + num]
                c = pts.mean(0)
                d = float(np.linalg.norm(c[:2] - tray[:2]))
                if d > 0.20:
                    continue
                ddx, ddy = pts[:, 0] - tray[0], pts[:, 1] - tray[1]
                lx = cos_t * ddx - sin_t * ddy
                ly = sin_t * ddx + cos_t * ddy
                frac = float(((np.abs(lx) < half_x) & (np.abs(ly) < half_y)
                              & (pts[:, 2] > tray[2] + 0.005)).mean())
                contact = 0
                for k in range(data.ncon):
                    ct = data.contact[k]
                    if int(ct.flex[0]) == fids[n] or int(ct.flex[1]) == fids[n]:
                        other = int(ct.geom1) if int(ct.geom2) < 0 else int(ct.geom2)
                        if other >= 0 and (model.body(int(model.geom_bodyid[other])).name
                                           or "").startswith("hand_"):
                            contact += 1
                line = (i, d, (c[0] + table_cm[0] / 200.0) * 100.0,
                        (c[1] + table_cm[1] / 200.0) * 100.0, float(np.abs(lx).min()),
                        float(np.abs(ly).min()), contact, frac)
                history.append(line)
            if line and i % 3 == 0:
                ap = float(np.linalg.norm(data.xpos[thumb]
                                          - np.mean([data.xpos[f] for f in fingers], 0)))
                print("  %5d %8.3f %8.2f %7.3f %7.3f %6d %8.3f %8.2f"
                      % (line[0], line[1], line[3], line[4], line[5], line[6], ap,
                         line[7]))
        if history:
            best = min(history, key=lambda r: r[1])
            last = history[-1]
            print("  closest approach : step %d  d %.3f m  inside-fraction %.2f  hand contacts %d"
                  % (best[0], best[1], best[7], best[6]))
            print("  final near-box   : step %d  d %.3f m  inside-fraction %.2f  hand contacts %d"
                  % (last[0], last[1], last[7], last[6]))
            after = [r for r in history if r[0] > best[0]]
            if after:
                print("  after the closest approach the bag %s the box (d %.3f -> %.3f m)"
                      % ("leaves" if after[-1][1] > best[1] + 0.02 else "stays near",
                         best[1], after[-1][1]))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
