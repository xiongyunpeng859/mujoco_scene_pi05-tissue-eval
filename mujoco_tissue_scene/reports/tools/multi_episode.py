#!/usr/bin/env python3
"""Cross-validate the working alignment on several real successful episodes.

One episode finishing in the tray could be luck.  This replays several episodes with
the same base placement and reports, for each, whether the bag is carried and ends
inside the box.  Also compares the measured base (25, 10) as a control.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import align_with_dataset as align      # noqa: E402
import dataset_io                       # noqa: E402
import sim_env                          # noqa: E402

DATASET = Path("/workspace/shared/new_program_qiuzhi/without_tactile/"
               "pi05_normal_recovery_merged_214eps")
OUT = ROOT / "outputs/multi_episode"
CANDIDATES = [(20.0, 6.0, 1.5708), (20.0, 10.0, 1.5708), (25.0, 10.0, 1.5708)]
EPISODES = (0, 1, 2, 3, 4)


def run(x_cm, y_cm, yaw, episode, episodes, frames):
    import mujoco
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    config["measured_layout"]["arm_mount_xy_cm_from_left_bottom"] = [x_cm, y_cm]
    config["arm"]["euler"][2] = yaw
    tag = "x%05.1f_y%05.1f_a%06.4f" % (x_cm, y_cm, yaw)
    path = OUT / tag / "scene.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))

    action = episodes[episode]["action"]
    state = episodes[episode]["observation.state"]
    env = sim_env.TissueSceneEnv(config_path=path, dataset=DATASET, render=False,
                                 output_dir=path.parent)
    model, data = env.model, env.data
    env.reset(options={"state": state[0], "randomize_objects": False})

    real_first = align.dataset_frame(DATASET, episode, 0)
    found = align.measure_bags(real_first, config)
    surface = config["table"]["surface_z"]
    table_cm = [config["table"]["size"][0] * 100.0, config["table"]["size"][1] * 100.0]
    for box, bag in zip(config["boxes"], found):
        bx, by = bag["box_centre_cm"]
        byaw = np.radians(bag["yaw_deg"])
        qpos_adr, dof_adr = env.box_free[box["name"]]
        data.qpos[qpos_adr + 0] = bx / 100.0 - table_cm[0] / 200.0
        data.qpos[qpos_adr + 1] = by / 100.0 - table_cm[1] / 200.0
        data.qpos[qpos_adr + 2] = surface + box["size"][2] / 2.0 + 0.002
        data.qpos[qpos_adr + 3] = np.cos(byaw / 2.0)
        data.qpos[qpos_adr + 4:qpos_adr + 7] = 0.0
        data.qvel[dof_adr:dof_adr + 6] = 0.0
    mujoco.mj_forward(model, data)

    names = list(env.box_free)
    fids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, n + "_soft")
            for n in names}
    start_z = {n: float(data.qpos[env.box_free[n][0] + 2]) for n in names}
    tray = data.xpos[env.tray_body].copy()
    yaw_t = float(config["tray"].get("yaw", 0.0))
    cos_t, sin_t = np.cos(-yaw_t), np.sin(-yaw_t)
    peak = {n: 0.0 for n in names}
    min_tray = {n: 9.9 for n in names}
    for value in action:
        env.step(value)
        for n in names:
            adr, num = int(model.flex_vertadr[fids[n]]), int(model.flex_vertnum[fids[n]])
            centre = data.flexvert_xpos[adr:adr + num].mean(0)
            peak[n] = max(peak[n], float(centre[2]) - start_z[n])
            min_tray[n] = min(min_tray[n], float(np.linalg.norm(centre[:2] - tray[:2])))
    verdict = {}
    for n in names:
        adr, num = int(model.flex_vertadr[fids[n]]), int(model.flex_vertnum[fids[n]])
        points = data.flexvert_xpos[adr:adr + num]
        dx, dy = points[:, 0] - tray[0], points[:, 1] - tray[1]
        lx = cos_t * dx - sin_t * dy
        ly = sin_t * dx + cos_t * dy
        inside = ((np.abs(lx) < config["tray"]["size"][0] / 2)
                  & (np.abs(ly) < config["tray"]["size"][1] / 2)
                  & (points[:, 2] > tray[2] + 0.005))
        verdict[n] = bool(inside.mean() > 0.3)
    env.close()
    name = config["boxes"][0]["name"]
    return {"peak": peak[name], "min_tray": min_tray[name],
            "in_tray": any(verdict.values()), "verdict": verdict,
            "bags_found": len(found)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    episodes, order = dataset_io.load(DATASET)
    available = [e for e in EPISODES if e in episodes]
    print("episodes: %s" % available)
    summary = []
    for x_cm, y_cm, yaw in CANDIDATES:
        print()
        print("=== base (%.1f, %.1f) cm  yaw %.4f ===" % (x_cm, y_cm, yaw))
        print("  %-8s %10s %11s %9s  %s" %
              ("episode", "peak rise", "min d(tray)", "in tray", "bags found"))
        successes = 0
        for episode in available:
            r = run(x_cm, y_cm, yaw, episode, episodes, None)
            successes += int(r["in_tray"])
            print("  %-8d %10.4f %11.4f %9s  %d"
                  % (episode, r["peak"], r["min_tray"], r["in_tray"], r["bags_found"]))
        print("  --> %d / %d episodes finish with a bag in the box"
              % (successes, len(available)))
        summary.append({"base": [x_cm, y_cm, yaw], "successes": successes,
                        "total": len(available)})
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print()
    best = max(summary, key=lambda s: s["successes"])
    print("best base %s : %d / %d" % (best["base"], best["successes"], best["total"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
