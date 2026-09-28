#!/usr/bin/env python3
"""Does joint6 follow its own command, and in the right direction?

Section 37: joint6 is commanded +0.235, sits at -3.165 (past its own -3.0107 limit), with
no clipping and no contact.  So test the joint in isolation: hold everything else fixed,
step joint6 up then down, and watch where it actually goes.
"""
from __future__ import annotations

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
OUT = ROOT / "outputs/joint6_test"
OUT.mkdir(parents=True, exist_ok=True)
DIM = 5                      # joint6 is dataset dim 5


def main() -> int:
    import mujoco
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    p = OUT / "scene.yaml"
    p.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    data, order = dataset_io.load(SUCCESS, fields=("observation.state",))
    env = sim_env.TissueSceneEnv(config_path=p, dataset=SUCCESS, render=False,
                                output_dir=OUT)
    model, mdata = env.model, env.data
    base = np.asarray(data[0]["observation.state"][0], dtype=float).copy()
    env.reset(options={"state": base, "randomize_objects": False})
    print("joint names  : %s" % layout.DATASET_NAMES)
    print("commanding dim %d = %s" % (DIM, layout.DATASET_NAMES[DIM]))
    print("joint6 range : %s" % model.jnt_range[model.joint("joint6").id])
    print()
    # settle on the nominal command first
    for _ in range(40):
        env.step(base)
    q0 = float(mdata.qpos[env.joint_adr[DIM]])
    print("after settling, joint6 = %+.4f (commanded %+.4f)" % (q0, base[DIM]))
    print()
    print("  %-10s %12s %12s %10s" % ("commanded", "achieved", "change", "follows?"))
    for delta in (0.3, -0.3, 0.6, -0.6):
        cmd = base.copy()
        cmd[DIM] = float(np.clip(q0 + delta, *model.jnt_range[model.joint("joint6").id]))
        for _ in range(45):
            env.step(cmd)
        q1 = float(mdata.qpos[env.joint_adr[DIM]])
        change = q1 - q0
        want = cmd[DIM] - q0
        ok = "yes" if abs(change - want) < 0.05 else (
            "OPPOSITE SIGN" if change * want < 0 else "no")
        print("  %+10.4f %12.4f %+12.4f %10s" % (cmd[DIM], q1, change, ok))
        # return to nominal
        for _ in range(45):
            env.step(base)
        q0 = float(mdata.qpos[env.joint_adr[DIM]])
    print()
    print("if every row says 'yes' the joint is fine in isolation and the section-37")
    print("failure comes from the trajectory, not the actuator.")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
