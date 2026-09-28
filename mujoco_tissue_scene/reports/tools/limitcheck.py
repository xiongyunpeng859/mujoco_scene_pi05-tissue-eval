#!/usr/bin/env python3
"""Is the env clipping my IK commands?

scripted_pick_place solves IK within the MODEL's jnt_range.  sim_env.step() clips the
action to its OWN limits_lo/limits_hi first.  If those are tighter (e.g. taken from the
observed dataset range rather than the joint range), valid IK commands are silently
clipped and the arm is asked for something else -- which would explain a tracking error
that never shrinks.
"""
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import action_layout as layout
import dataset_io
import mujoco
import sim_env

SUCCESS = Path("/workspace/shared/new_program_qiuzhi/without_tactile/success_episode/"
               "pick_up_the_tissue_pack_and_place_it_on_the_right_side_"
               "20260908_merged_all_108eps")
OUT = ROOT / "outputs/limit_check"
OUT.mkdir(parents=True, exist_ok=True)
config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
p = OUT / "scene.yaml"
p.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
data, order = dataset_io.load(SUCCESS, fields=("observation.state",))

env = sim_env.TissueSceneEnv(config_path=p, dataset=SUCCESS, render=False, output_dir=OUT)
model = env.model
jids = layout.joint_ids(model, mujoco)
lo, hi = np.asarray(env.limits_lo), np.asarray(env.limits_hi)
print("  %-24s %10s %10s | %10s %10s | %s" %
      ("dim", "env lo", "env hi", "model lo", "model hi", "env narrower?"))
narrow = 0
for i, adr in enumerate(jids):
    jid = None
    for j in range(model.njnt):
        if model.jnt_qposadr[j] == adr:
            jid = j
            break
    mlo, mhi = (model.jnt_range[jid][0], model.jnt_range[jid][1]) if jid is not None \
        else (float("nan"), float("nan"))
    is_narrow = (lo[i] > mlo + 1e-6) or (hi[i] < mhi - 1e-6)
    narrow += int(is_narrow)
    print("  %-24s %10.4f %10.4f | %10.4f %10.4f | %s"
          % (layout.DATASET_NAMES[i], lo[i], hi[i], mlo, mhi,
             "YES" if is_narrow else "no"))
print()
print("dims where the env limit is NARROWER than the joint range: %d of 16" % narrow)

# also: does the real dataset ever exceed the env limits?
st = np.array([data[e]["observation.state"][0] for e in order])
out_lo = (st < lo[None, :] - 1e-6).sum(0)
out_hi = (st > hi[None, :] - 1e-6).sum(0)
print("\nreal episode start states outside the env limits (per dim):")
for i in range(16):
    if out_lo[i] or out_hi[i]:
        print("   %-24s below %d, above %d" % (layout.DATASET_NAMES[i], out_lo[i], out_hi[i]))
print("   (no output above = every real start state is inside the env limits)")
env.close()
