#!/usr/bin/env python3
"""Fine search to close the last few centimetres to the tray.

The grasp now works (bag lifted 14 cm, carried 2.9 s to within ~5 cm of the tray
edge), so the objective is the final approach: closest distance to the tray centre
during the carry, and whether the bag ends inside.
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
OUT = ROOT / "outputs/finish_search"
EPISODE = 0


def main() -> int:
    import mujoco
    OUT.mkdir(parents=True, exist_ok=True)
    episodes, _ = dataset_io.load(DATASET)
    action, state = episodes[EPISODE]["action"], episodes[EPISODE]["observation.state"]
    base_config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    real_first = align.dataset_frame(DATASET, EPISODE, 0)
    found = align.measure_bags(real_first, base_config)

    def one(x_cm, y_cm, yaw):
        config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
        config["measured_layout"]["arm_mount_xy_cm_from_left_bottom"] = [x_cm, y_cm]
        config["arm"]["euler"][2] = yaw
        tag = "x%05.1f_y%05.1f_a%06.4f" % (x_cm, y_cm, yaw)
        path = OUT / tag / "scene.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))

        env = sim_env.TissueSceneEnv(config_path=path, dataset=DATASET, render=False,
                                     output_dir=path.parent)
        model, data = env.model, env.data
        env.reset(options={"state": state[0], "randomize_objects": False})
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
        flex_ids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, n + "_soft")
                    for n in names}
        start_z = {n: float(data.qpos[env.box_free[n][0] + 2]) for n in names}
        tray = data.xpos[env.tray_body].copy()
        yaw_t = float(config["tray"].get("yaw", 0.0))
        cos_t, sin_t = np.cos(-yaw_t), np.sin(-yaw_t)
        half_x = config["tray"]["size"][0] / 2.0
        half_y = config["tray"]["size"][1] / 2.0

        best = {n: {"min_tray": 9.9, "peak": 0.0} for n in names}
        for value in action:
            env.step(value)
            for n in names:
                adr, num = int(model.flex_vertadr[flex_ids[n]]), int(model.flex_vertnum[flex_ids[n]])
                points = data.flexvert_xpos[adr:adr + num]
                centre = points.mean(0)
                best[n]["peak"] = max(best[n]["peak"], float(centre[2]) - start_z[n])
                best[n]["min_tray"] = min(best[n]["min_tray"],
                                          float(np.linalg.norm(centre[:2] - tray[:2])))
        # final containment, flex aware
        verdict = {}
        for n in names:
            adr, num = int(model.flex_vertadr[flex_ids[n]]), int(model.flex_vertnum[flex_ids[n]])
            points = data.flexvert_xpos[adr:adr + num]
            dx, dy = points[:, 0] - tray[0], points[:, 1] - tray[1]
            lx = cos_t * dx - sin_t * dy
            ly = sin_t * dx + cos_t * dy
            inside = ((np.abs(lx) < half_x) & (np.abs(ly) < half_y)
                      & (points[:, 2] > tray[2] + 0.005))
            verdict[n] = bool(inside.mean() > 0.3)
        env.close()
        return best, verdict

    print("  %-22s %11s %10s %10s %9s" %
          ("base / yaw", "min d(tray)", "peak rise", "carried?", "in tray"))
    rows = []
    for x_cm, y_cm in itertools.product((16.0, 18.0, 20.0, 22.0, 24.0), (6.0, 10.0, 14.0)):
        best, verdict = one(x_cm, y_cm, 1.5708)
        name = base_config["boxes"][0]["name"]
        b = best[name]
        carried = b["peak"] > 0.05
        rows.append((x_cm, y_cm, b["min_tray"], b["peak"], carried, verdict[name]))
        print("  x=%5.1f y=%5.1f a=1.5708 %11.4f %10.4f %10s %9s%s"
              % (x_cm, y_cm, b["min_tray"], b["peak"], carried, verdict[name],
                 "   <== IN TRAY" if verdict[name] else ""))

    rows.sort(key=lambda r: (r[5], -r[2] if r[2] < 9 else 0, r[3]), reverse=True)
    print()
    best = rows[0]
    print("  best: x=%.1f y=%.1f -> closest %.4f m, rise %.4f m, in tray %s"
          % (best[0], best[1], best[2], best[3], best[5]))
    print("  placements/ bases that finish inside the tray: %d / %d"
          % (sum(1 for r in rows if r[5]), len(rows)))
    (OUT / "result.json").write_text(json.dumps(
        [{"x_cm": r[0], "y_cm": r[1], "min_tray": r[2], "peak_rise": r[3],
          "carried": r[4], "in_tray": r[5]} for r in rows], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
