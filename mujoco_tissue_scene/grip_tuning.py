#!/usr/bin/env python3
"""Is the grip strong enough to survive the real trajectory's dynamics?

The bag is grasped and lifted but lost mid-carry, so this sweeps the two levers that
set grip security: the hand's position gain (how hard the fingers press) and the
hand-to-bag friction.  Objective: how far the carried bag gets toward the tray.
"""
from __future__ import annotations

import itertools
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
OUT = ROOT / "outputs/grip_tuning"
PLACEMENT = (31.9, 52.3, -39.3)


def build(hand_kp, hand_force, friction, young, tag):
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    config["arm"]["euler"][2] = 1.5708
    config["measured_layout"]["arm_mount_xy_cm_from_left_bottom"] = [25.0, 10.0]
    config["hand"]["actuator"] = {"kp": hand_kp, "kv": 0.1, "force": hand_force}
    for box in config["boxes"]:
        box.update({"soft_body": True, "young": young, "flex_count": [5, 4, 3],
                    "friction": friction})
    path = OUT / tag / "scene.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    return config, path


def main() -> int:
    import mujoco
    OUT.mkdir(parents=True, exist_ok=True)
    episodes, _ = dataset_io.load(DATASET)
    action, state = episodes[0]["action"], episodes[0]["observation.state"]

    print("  %-26s %9s %9s %9s %8s %9s" %
          ("candidate", "carry st", "min d(tr)", "max rise", "final d", "in tray"))
    rows = []
    for hand_kp, friction, young in itertools.product((3.0, 15.0, 60.0),
                                                      ([1.5, 0.05, 0.001], [4.0, 0.1, 0.002]),
                                                      (1.0e5, 3.0e5)):
        tag = "kp%g_f%g_y%g" % (hand_kp, friction[0], young)
        config, path = build(hand_kp, 3.0, friction, young, tag)
        env = sim_env.TissueSceneEnv(config_path=path, dataset=DATASET, render=False,
                                     output_dir=path.parent)
        model, data = env.model, env.data
        surface = config["table"]["surface_z"]
        table_cm = [config["table"]["size"][0] * 100.0, config["table"]["size"][1] * 100.0]
        name = config["boxes"][0]["name"]
        qpos_adr, dof_adr = env.box_free[name]
        hand_bodies = {i for i in range(model.nbody)
                       if (model.body(i).name or "").startswith("hand_")}
        flex_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, name + "_soft")

        env.reset(options={"state": state[0], "randomize_objects": False})
        x_cm, y_cm, yaw_deg = PLACEMENT
        yaw = np.radians(yaw_deg)
        data.qpos[qpos_adr + 0] = x_cm / 100.0 - table_cm[0] / 200.0
        data.qpos[qpos_adr + 1] = y_cm / 100.0 - table_cm[1] / 200.0
        data.qpos[qpos_adr + 2] = surface + config["boxes"][0]["size"][2] / 2.0 + 0.002
        data.qpos[qpos_adr + 3] = np.cos(yaw / 2.0)
        data.qpos[qpos_adr + 4:qpos_adr + 7] = 0.0
        data.qvel[dof_adr:dof_adr + 6] = 0.0
        mujoco.mj_forward(model, data)
        z0 = float(data.qpos[qpos_adr + 2])
        tray = data.xpos[env.tray_body].copy()

        carry, min_tray, peak = 0, 9.9, 0.0
        for value in action:
            env.step(value)
            position = data.qpos[qpos_adr:qpos_adr + 3].copy()
            rise = float(position[2]) - z0
            peak = max(peak, rise)
            if rise > 0.02:
                carry += 1
                min_tray = min(min_tray, float(np.linalg.norm(position[:2] - tray[:2])))
        final = float(np.linalg.norm(data.qpos[qpos_adr:qpos_adr + 2] - tray[:2]))
        tray_half = config["tray"]["size"][0] / 2.0
        in_tray = final < tray_half and float(data.qpos[qpos_adr + 2]) > surface + 0.01
        rows.append((tag, carry, min_tray, peak, final, in_tray))
        print("  %-26s %9d %9.4f %9.4f %8.4f %9s"
              % (tag, carry, min_tray if min_tray < 9 else -1, peak, final, in_tray))
        env.close()

    rows.sort(key=lambda r: (r[5], -r[2] if r[2] < 9 else -9, r[3]), reverse=True)
    print()
    best = rows[0]
    print("  best: %s -> carried %d steps, closest to tray %.4f m, rise %.4f m"
          % (best[0], best[1], best[2], best[3]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
