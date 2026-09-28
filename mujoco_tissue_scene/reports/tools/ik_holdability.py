#!/usr/bin/env python3
"""Is the IK's OWN solution holdable, and is it in collision?

Section 42: a real library posture holds to 0.012 rad with force=800, but the IK's solution
for a sampled pack leaves the fingertip 5.4 cm short.  So command the IK's solution itself,
settle, and report per-joint error, torque/ceiling, the fingertip error, and any ARM contact.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reports/tools"))
import action_layout as layout
import dataset_io
import sim_env
from scripted_pick_place import Arm, PITCH, SUCCESS

OUT = ROOT / "outputs/ik_holdability"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    import mujoco
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    p = OUT / "scene.yaml"
    p.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    ts = config["table"]["size"]
    lib = json.loads((ROOT / "outputs/closure_library/library.json").read_text())
    grasp = lib["grasp"]
    lx = np.array([e["x_cm"] for e in grasp])
    ly = np.array([e["y_cm"] for e in grasp])

    data, order = dataset_io.load(SUCCESS, fields=("observation.state", "action"))
    start = np.asarray(data[0]["observation.state"][0], dtype=float).copy()

    # the pack position that failed in sections 40-42
    pack_cm = np.array([29.8, 66.0])
    px = pack_cm[0] / 100.0 - ts[0] / 2.0
    py = pack_cm[1] / 100.0 - ts[1] / 2.0

    arm = Arm(p)
    near = int(np.argmin(np.hypot(lx - pack_cm[0], ly - pack_cm[1])))
    entry = grasp[near]
    seed = np.array(entry["q"], dtype=float)
    arm.set_arm(seed)
    ref_z = float(entry["z_m"])
    sol, err, moved = arm.solve((px, py), ref_z, 0.0, seed, seed)
    print("library seed ep%d at (%.1f, %.1f) cm; IK residual %.4f m; joint move %.3f rad"
          % (entry["episode"], entry["x_cm"], entry["y_cm"], err, moved))
    print("  seed joints : %s" % np.round(seed, 3))
    print("  IK solution : %s" % np.round(sol, 3))
    print("  delta       : %s" % np.round(sol - seed, 3))

    env = sim_env.TissueSceneEnv(config_path=p, dataset=SUCCESS, render=False,
                                output_dir=OUT)
    model, mdata = env.model, env.data
    # put the pack at the same place so contacts are meaningful
    env.reset(options={"state": start, "randomize_objects": False})
    tgt = config["boxes"][0]
    adr, dof = env.box_free[tgt["name"]]
    mdata.qpos[adr + 0] = px
    mdata.qpos[adr + 1] = py
    mdata.qpos[adr + 2] = config["table"]["surface_z"] + tgt["size"][2] / 2.0 + 0.002
    mdata.qpos[adr + 3] = 1.0
    mdata.qpos[adr + 4:adr + 7] = 0.0
    mdata.qvel[dof:dof + 6] = 0.0
    mujoco.mj_forward(model, mdata)

    cmd = start.copy()
    cmd[:6] = sol
    for _ in range(150):
        env.step(cmd)
    ach = np.asarray(mdata.qpos[env.joint_adr], dtype=float)
    aids = layout.actuator_ids(model, mujoco)
    print()
    print("  %-9s %10s %10s %9s %10s" %
          ("joint", "commanded", "achieved", "error", "torq/ceil"))
    worst = 0.0
    for i in range(6):
        aid = aids[i]
        f = float(mdata.actuator_force[aid])
        lo, hi = model.actuator_forcerange[aid]
        ceil = max(abs(lo), abs(hi)) or 1.0
        worst = max(worst, abs(ach[i] - cmd[i]))
        print("  %-9s %10.4f %10.4f %+9.4f %10.2f"
              % (layout.SIM_ARM_JOINTS[i], cmd[i], ach[i], ach[i] - cmd[i], abs(f) / ceil))
    # where did the fingertip actually end up?
    names = ["hand_L_thumb_tip", "hand_L_index_tip", "hand_L_middle_tip",
             "hand_L_ring_tip", "hand_L_pinky_tip"]
    tb = [model.body(x).id for x in names]
    tm = 0.5 * (mdata.xpos[tb[0]] + np.mean([mdata.xpos[i] for i in tb[1:]], axis=0))
    tm_cm = np.array([(tm[0] + ts[0] / 2) * 100.0, (tm[1] + ts[1] / 2) * 100.0])
    print()
    print("  fingertip settled at (%.1f, %.1f) cm, target (%.1f, %.1f) cm -> error %.2f cm"
          % (tm_cm[0], tm_cm[1], pack_cm[0], pack_cm[1],
             float(np.linalg.norm(tm_cm - pack_cm))))
    print("  worst joint error %.4f rad" % worst)
    print()
    arm_geoms = {g for g in range(model.ngeom)
                 if (model.body(int(model.geom_bodyid[g])).name or "").startswith(
                     ("link", "base", "hand_"))}
    arm_ct = []
    for k in range(mdata.ncon):
        ct = mdata.contact[k]
        if int(ct.flex[0]) >= 0 or int(ct.flex[1]) >= 0:
            continue
        g1, g2 = int(ct.geom1), int(ct.geom2)
        if g1 in arm_geoms or g2 in arm_geoms:
            arm_ct.append("%s <-> %s" % (model.geom(g1).name, model.geom(g2).name))
    print("  ARM contacts at the settled posture: %d" % len(arm_ct))
    for c in sorted(set(arm_ct))[:10]:
        print("     %s" % c)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
