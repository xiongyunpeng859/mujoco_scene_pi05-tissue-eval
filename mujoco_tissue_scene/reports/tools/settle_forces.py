#!/usr/bin/env python3
"""Is the leftover offset saturation, or a soft spring?

Command ONE real grasp posture (from the library), let the arm settle, then report per
arm joint: commanded, achieved, error, applied torque, and the torque as a fraction of its
ceiling.  If any joint sits at ~1.0 of its ceiling the offset is saturation; if every joint
is far below its ceiling yet still short, it is something else.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import action_layout as layout
import dataset_io
import sim_env

SUCCESS = Path("/workspace/shared/new_program_qiuzhi/without_tactile/success_episode/"
               "pick_up_the_tissue_pack_and_place_it_on_the_right_side_"
               "20260908_merged_all_108eps")
OUT = ROOT / "outputs/settle_forces"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    import mujoco
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    p = OUT / "scene.yaml"
    p.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    lib = json.loads((ROOT / "outputs/closure_library/library.json").read_text())
    entry = lib["grasp"][0]

    data, order = dataset_io.load(SUCCESS, fields=("observation.state", "action"))
    start = np.asarray(data[0]["observation.state"][0], dtype=float).copy()
    hand_closed = np.asarray(data[0]["action"][0][6:], dtype=float)
    env = sim_env.TissueSceneEnv(config_path=p, dataset=SUCCESS, render=False,
                                output_dir=OUT)
    model, mdata = env.model, env.data
    env.reset(options={"state": start, "randomize_objects": False})

    cmd = start.copy()
    cmd[:6] = np.asarray(entry["q"], dtype=float)
    for _ in range(120):                       # settle
        env.step(cmd)
    ach = np.asarray(mdata.qpos[env.joint_adr], dtype=float)
    aids = layout.actuator_ids(model, mujoco)
    print("commanded a REAL closure posture (library ep%d at (%.1f, %.1f) cm)"
          % (entry["episode"], entry["x_cm"], entry["y_cm"]))
    print()
    print("  %-10s %10s %10s %9s %12s %10s" %
          ("joint", "commanded", "achieved", "error", "torque", "torque/ceil"))
    for i in range(6):
        aid = aids[i]
        f = float(mdata.actuator_force[aid])
        lo, hi = model.actuator_forcerange[aid]
        ceil = max(abs(lo), abs(hi)) or 1.0
        print("  %-10s %10.4f %10.4f %+9.4f %12.3f %10.2f"
              % (layout.SIM_ARM_JOINTS[i], cmd[i], ach[i], ach[i] - cmd[i], f,
                 abs(f) / ceil))
    print()
    print("  tip-mid commanded-not-checked; look at whether any torque/ceil is near 1.0")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
