#!/usr/bin/env python3
"""Can ANY bag placement make the real trajectory complete the task?

The isolation test showed the soft bag can be held; the full replay only nudges it.
The pinch spans 8-9 cm while the pack is 12x8.5 cm, so the bag's placement and yaw
decide whether the fingers engage.  This searches the target bag's placement around
the measured point and reports the best outcome, which answers whether the
simulation can reproduce the task end to end at all.
"""
from __future__ import annotations

import itertools
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
OUT = ROOT / "outputs/bag_placement_search"
EPISODE = 0


def main() -> int:
    import mujoco
    OUT.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    for box in config["boxes"]:
        box.update({"soft_body": True, "young": 100000.0, "flex_count": [5, 4, 3]})
    config["arm"]["euler"][2] = 1.5708
    config["measured_layout"]["arm_mount_xy_cm_from_left_bottom"] = [25.0, 10.0]
    path = OUT / "scene.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))

    episodes, _ = dataset_io.load(DATASET)
    action, state = episodes[EPISODE]["action"], episodes[EPISODE]["observation.state"]
    real_first = align.dataset_frame(DATASET, EPISODE, 0)
    found = align.measure_bags(real_first, config)
    print("bags measured in the real first frame: %d" % len(found))
    for b in found:
        print("   centre %s yaw %.0f" % (b["box_centre_cm"], b["yaw_deg"]))
    anchors = [b["box_centre_cm"] for b in found] + [[25.9, 58.3], [12.8, 61.8], [20.0, 60.0]]
    print()

    env = sim_env.TissueSceneEnv(config_path=path, dataset=DATASET, render=False,
                                 output_dir=OUT)
    model, data = env.model, env.data
    names = list(env.box_free)
    surface = config["table"]["surface_z"]
    table_cm = [config["table"]["size"][0] * 100.0, config["table"]["size"][1] * 100.0]
    hand_bodies = {i for i in range(model.nbody)
                   if (model.body(i).name or "").startswith("hand_")}
    target = config["boxes"][0]["name"]
    qpos_adr, dof_adr = env.box_free[target]

    def place_target(x_cm, y_cm, yaw_deg):
        yaw = np.radians(yaw_deg)
        data.qpos[qpos_adr + 0] = x_cm / 100.0 - table_cm[0] / 200.0
        data.qpos[qpos_adr + 1] = y_cm / 100.0 - table_cm[1] / 200.0
        data.qpos[qpos_adr + 2] = surface + config["boxes"][0]["size"][2] / 2.0 + 0.002
        data.qpos[qpos_adr + 3] = np.cos(yaw / 2.0)
        data.qpos[qpos_adr + 4] = 0.0
        data.qpos[qpos_adr + 5] = 0.0
        data.qpos[qpos_adr + 6] = np.sin(yaw / 2.0)
        data.qvel[dof_adr:dof_adr + 6] = 0.0

    def replay(x_cm, y_cm, yaw_deg):
        env.reset(options={"state": state[0], "randomize_objects": False})
        place_target(x_cm, y_cm, yaw_deg)
        env.mujoco.mj_forward(model, data)
        z0 = float(data.qpos[qpos_adr + 2])
        peak, contacts = 0.0, 0
        for value in action:
            env.step(value)
            peak = max(peak, float(data.qpos[qpos_adr + 2]) - z0)
            for index in range(data.ncon):
                c = data.contact[index]
                if int(c.flex[0]) >= 0 or int(c.flex[1]) >= 0:
                    other = int(c.geom1) if int(c.geom2) < 0 else int(c.geom2)
                    if other >= 0 and int(model.geom_bodyid[other]) in hand_bodies \
                            and int(c.flex[0]) == 0 or int(c.flex[1]) == 0:
                        contacts += 1
        # containment of the target flex vertices
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, target + "_soft")
        adr, num = int(model.flex_vertadr[index]), int(model.flex_vertnum[index])
        points = data.flexvert_xpos[adr:adr + num]
        tray = data.xpos[env.tray_body]
        yaw_t = float(config["tray"].get("yaw", 0.0))
        cos_t, sin_t = np.cos(-yaw_t), np.sin(-yaw_t)
        dx, dy = points[:, 0] - tray[0], points[:, 1] - tray[1]
        lx = cos_t * dx - sin_t * dy
        ly = sin_t * dx + cos_t * dy
        inside = ((np.abs(lx) < config["tray"]["size"][0] / 2)
                  & (np.abs(ly) < config["tray"]["size"][1] / 2)
                  & (points[:, 2] > tray[2] + 0.005))
        return peak, contacts, bool(inside.mean() > 0.3)

    print("  %-24s %10s %9s %9s" % ("target bag placement", "peak rise", "hand con",
                                    "in tray"))
    rows = []
    anchor = anchors[0]
    for dx, dy, dyaw in itertools.product((-6.0, 0.0, 6.0), (-6.0, 0.0, 6.0),
                                          (-45.0, 0.0, 45.0, 90.0)):
        x_cm, y_cm = anchor[0] + dx, anchor[1] + dy
        yaw_deg = found[0]["yaw_deg"] + dyaw if found else dyaw
        peak, contacts, in_tray = replay(x_cm, y_cm, yaw_deg)
        rows.append((x_cm, y_cm, yaw_deg, peak, contacts, in_tray))
        flag = "  <== IN TRAY" if in_tray else ""
        print("  x=%5.1f y=%5.1f yaw=%6.1f %10.4f %9d %9s%s"
              % (x_cm, y_cm, yaw_deg, peak, contacts, in_tray, flag))
    env.close()
    rows.sort(key=lambda r: (r[5], r[3]), reverse=True)
    print()
    best = rows[0]
    print("  best: x=%.1f y=%.1f yaw=%.1f -> rise %.4f m, %d contacts, in tray %s"
          % (best[0], best[1], best[2], best[3], best[4], best[5]))
    wins = [r for r in rows if r[5]]
    print("  placements that reach the tray: %d / %d" % (len(wins), len(rows)))
    (OUT / "result.json").write_text(json.dumps(
        [{"x_cm": r[0], "y_cm": r[1], "yaw_deg": r[2], "peak_rise": r[3],
          "hand_contacts": r[4], "in_tray": r[5]} for r in rows], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
