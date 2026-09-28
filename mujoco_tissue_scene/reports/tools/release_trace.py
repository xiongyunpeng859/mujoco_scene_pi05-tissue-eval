#!/usr/bin/env python3
"""Where is the bag released relative to the tray?

The full replay lifts the bag 11 cm but it never lands in the box.  This traces the
bag and the tray through the episode to see whether release happens off the tray
(an alignment problem) or on it (a physics/holding problem).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import dataset_io                       # noqa: E402
import sim_env                          # noqa: E402

DATASET = Path("/workspace/shared/new_program_qiuzhi/without_tactile/"
               "pi05_normal_recovery_merged_214eps")
OUT = ROOT / "outputs/release_trace"
PLACEMENT = (31.9, 52.3, -39.3)          # the best bag placement found so far


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
    action, state = episodes[0]["action"], episodes[0]["observation.state"]
    env = sim_env.TissueSceneEnv(config_path=path, dataset=DATASET, render=False,
                                 output_dir=OUT)
    model, data = env.model, env.data
    surface = config["table"]["surface_z"]
    table_cm = [config["table"]["size"][0] * 100.0, config["table"]["size"][1] * 100.0]
    target = config["boxes"][0]["name"]
    qpos_adr, dof_adr = env.box_free[target]
    hand_bodies = {i for i in range(model.nbody)
                   if (model.body(i).name or "").startswith("hand_")}
    flex_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, target + "_soft")

    env.reset(options={"state": state[0], "randomize_objects": False})
    x_cm, y_cm, yaw_deg = PLACEMENT
    yaw = np.radians(yaw_deg)
    data.qpos[qpos_adr + 0] = x_cm / 100.0 - table_cm[0] / 200.0
    data.qpos[qpos_adr + 1] = y_cm / 100.0 - table_cm[1] / 200.0
    data.qpos[qpos_adr + 2] = surface + config["boxes"][0]["size"][2] / 2.0 + 0.002
    data.qpos[qpos_adr + 3] = np.cos(yaw / 2.0)
    data.qpos[qpos_adr + 4:qpos_adr + 7] = 0.0
    data.qvel[dof_adr:dof_adr + 6] = 0.0
    env.mujoco.mj_forward(model, data)
    z0 = float(data.qpos[qpos_adr + 2])

    tray = data.xpos[env.tray_body].copy()
    print("tray centre (world) %s   bag rest z %.4f" % (np.round(tray, 3).tolist(), z0))
    print()
    print("  %-5s %8s %8s %9s %9s %8s %9s" %
          ("step", "bag x", "bag y", "bag z", "d(tray)", "handcon", "carried"))
    carried_frames = []
    for step, value in enumerate(action):
        env.step(value)
        position = data.qpos[qpos_adr:qpos_adr + 3].copy()
        horizontal = float(np.linalg.norm(position[:2] - tray[:2]))
        contacts = 0
        for index in range(data.ncon):
            c = data.contact[index]
            if int(c.flex[0]) == flex_id or int(c.flex[1]) == flex_id:
                other = int(c.geom1) if int(c.geom2) < 0 else int(c.geom2)
                if other >= 0 and int(model.geom_bodyid[other]) in hand_bodies:
                    contacts += 1
        carried = position[2] > z0 + 0.02
        if carried:
            carried_frames.append((step, horizontal, position[2] - z0, contacts))
        if step % 10 == 0 or step in (len(action) - 1,):
            print("  %-5d %8.3f %8.3f %9.4f %9.4f %8d %9s"
                  % (step, position[0], position[1], position[2], horizontal,
                     contacts, "yes" if carried else "no"))

    print()
    if carried_frames:
        best = min(carried_frames, key=lambda r: r[1])
        print("  carried for %d steps (%.2f s)" %
              (len(carried_frames), len(carried_frames) / 30.0))
        print("  closest the CARRIED bag ever got to the tray centre: %.4f m (step %d)"
              % (best[1], best[0]))
        print("  tray half-size: %.3f x %.3f m" %
              (config["tray"]["size"][0] / 2, config["tray"]["size"][1] / 2))
        if best[1] > config["tray"]["size"][0] / 2:
            print("  => released OFF the tray: alignment, not holding.")
        else:
            print("  => carried OVER the tray but did not stay: holding/release physics.")
    else:
        print("  the bag was never carried in this configuration")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
