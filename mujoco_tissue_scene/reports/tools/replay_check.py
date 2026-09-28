#!/usr/bin/env python3
"""Replay real dataset trajectories in the simulation and measure fidelity.

    python replay_check.py --dataset <path> --episodes 0 1 2 [--sweep]

The dataset stores absolute joint-position targets at 30 Hz for all 16 joints.
Feeding them to the scene is the only honest test of whether the simulated arm
would follow the motions the policy was trained on, so this reports:

  * per-joint tracking error between the simulated state and the commanded target,
  * the same error against the real recorded next state (state[t+1]),
  * the resulting link6 (flange) Cartesian error, which is what actually moves
    the wrist camera and the grasp,
  * how often an actuator saturates its force limit.

With --sweep it also searches position-actuator gains for the smallest Cartesian
error, so the choice is data-driven rather than guessed.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import action_layout as layout          # noqa: E402
import dataset_io                       # noqa: E402
import scene                            # noqa: E402


def configure(model, mujoco, arm_kp, arm_kv, arm_force, hand_kp, hand_kv, hand_force,
              arm_damping=None):
    """Rewrite position-actuator gains in place (position: gain=kp, bias=-kp,-kv)."""
    for index in range(model.nu):
        if index < 6:
            kp, kv, force = arm_kp, arm_kv, arm_force
        else:
            kp, kv, force = hand_kp, hand_kv, hand_force
        model.actuator_gainprm[index, 0] = kp
        model.actuator_biasprm[index, 0] = 0.0
        model.actuator_biasprm[index, 1] = -kp
        model.actuator_biasprm[index, 2] = -kv
        model.actuator_forcerange[index] = (-force, force)
    if arm_damping is not None:
        for name in layout.SIM_ARM_JOINTS:
            model.dof_damping[model.joint(name).dofadr[0]] = arm_damping
    return arm_damping


def flange_reference(model, data, mujoco, joint_adr, states, body_id):
    """Flange pose that the real joint states correspond to, evaluated in this model."""
    reference = np.zeros((len(states), 3))
    for row, state in enumerate(states):
        data.qpos[joint_adr] = state
        mujoco.mj_forward(model, data)
        reference[row] = data.xpos[body_id]
    return reference


def replay(model, data, mujoco, ctrl_ids, joint_adr, body_id, action, start_state,
           substeps, settle=0.0):
    mujoco.mj_resetData(model, data)
    data.qpos[joint_adr] = start_state
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.ctrl[ctrl_ids] = start_state
    mujoco.mj_forward(model, data)
    for _ in range(int(round(settle / model.opt.timestep))):
        mujoco.mj_step(model, data)

    frames = len(action)
    sim_state = np.zeros((frames, 16))
    sim_ee = np.zeros((frames, 3))
    saturation = np.zeros(16)
    limits = np.array([model.actuator_forcerange[i][1] for i in ctrl_ids])
    for row in range(frames):
        data.ctrl[ctrl_ids] = action[row]
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        sim_state[row] = data.qpos[joint_adr]
        sim_ee[row] = data.xpos[body_id]
        saturation += np.abs(data.actuator_force[ctrl_ids]) >= 0.995 * limits
    return sim_state, sim_ee, saturation / max(frames, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/replay_check"))
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--settle", type=float, default=1.0)
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    import mujoco
    args.output_dir.mkdir(parents=True, exist_ok=True)
    xml = scene.build(Path(scene.ROOT) / "configs/scene.yaml", args.output_dir / "scene.xml",
                      with_hand=True)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    ctrl_ids = np.array(layout.actuator_ids(model, mujoco))
    joint_adr = np.array(layout.joint_ids(model, mujoco))
    body_id = model.body("link6").id
    substeps = int(round((1.0 / args.control_hz) / model.opt.timestep))
    print("control %g Hz -> %d physics substeps of %.4f s" %
          (args.control_hz, substeps, model.opt.timestep))
    print("actuators %d, mapped joints %d, flange body id %d" % (model.nu, len(joint_adr), body_id))

    episodes, order = dataset_io.load(args.dataset)
    chosen = [e for e in args.episodes if e in episodes] or order[:3]
    print("episodes: %s" % chosen)

    prepared = []
    for index in chosen:
        action = episodes[index]["action"]
        state = episodes[index]["observation.state"]
        reference = flange_reference(model, data, mujoco, joint_adr, state, body_id)
        prepared.append((index, action, state, reference))
        print("  ep %-3d frames=%d  duration=%.2fs" % (index, len(action), len(action) / args.control_hz))

    def evaluate(arm_kp, arm_kv, arm_force, hand_kp, hand_kv, hand_force,
                 arm_damping=None, verbose=False):
        configure(model, mujoco, arm_kp, arm_kv, arm_force, hand_kp, hand_kv, hand_force,
                  arm_damping)
        rows = []
        for index, action, state, reference in prepared:
            sim_state, sim_ee, saturation = replay(
                model, data, mujoco, ctrl_ids, joint_adr, body_id, action, state[0],
                substeps, settle=args.settle)
            target = action                              # commanded at each step
            next_state = np.vstack([state[1:], state[-1:]])
            # action[t] produces the state the real robot showed at t+1, so the
            # honest flange comparison is against reference at t+1 as well.
            reference_next = np.vstack([reference[1:], reference[-1:]])
            ee_err = np.linalg.norm(sim_ee - reference_next, axis=1)
            ee_err_same = np.linalg.norm(sim_ee - reference, axis=1)
            rows.append({
                "episode": index,
                "tracking_vs_target": {
                    "rms": float(np.sqrt(((sim_state - target) ** 2).mean())),
                    "max": float(np.abs(sim_state - target).max()),
                    "per_joint_rms": [round(float(v), 5) for v in
                                      np.sqrt(((sim_state - target) ** 2).mean(0))],
                },
                "tracking_vs_real_next_state": {
                    "rms": float(np.sqrt(((sim_state - next_state) ** 2).mean())),
                    "max": float(np.abs(sim_state - next_state).max()),
                    "per_joint_rms": [round(float(v), 5) for v in
                                      np.sqrt(((sim_state - next_state) ** 2).mean(0))],
                },
                "flange_error_m": {"rms": float(np.sqrt((ee_err ** 2).mean())),
                                   "max": float(ee_err.max()),
                                   "final": float(ee_err[-1])},
                "flange_error_same_index_m": {"rms": float(np.sqrt((ee_err_same ** 2).mean()))},
                "force_saturation_fraction": [round(float(v), 4) for v in saturation],
            })
            if verbose:
                print("  ep %-3d  track_target=%.5f  track_real_next=%.5f rad  "
                      "flange(t+1) rms=%.4f max=%.4f final=%.4f m  sat_arm=%s" %
                      (index, rows[-1]["tracking_vs_target"]["rms"],
                       rows[-1]["tracking_vs_real_next_state"]["rms"],
                       rows[-1]["flange_error_m"]["rms"], rows[-1]["flange_error_m"]["max"],
                       rows[-1]["flange_error_m"]["final"],
                       [round(float(v), 3) for v in saturation[:6]]))
        return rows

    baseline = {"arm_kp": 100.0, "arm_kv": 20.0, "arm_force": 80.0,
                "hand_kp": 3.0, "hand_kv": 0.1, "hand_force": 1.5}
    print()
    print("=== baseline gains (current scene): kp=100 kv=20 force=80 damping=1 ===")
    rows = evaluate(**baseline, arm_damping=1.0, verbose=True)
    report = {"baseline": dict(baseline, arm_damping=1.0), "episodes": rows}

    if args.sweep:
        print()
        print("=== sweep: arm kp / kv / joint damping (force=80, no saturation) ===")
        print("  %6s %6s %8s %12s %11s %11s %8s" %
              ("kp", "kv", "damping", "track_real", "flange_rms", "flange_max", "sat"))
        results = []
        for kp, kv, damping in itertools.product((100.0, 300.0, 1000.0),
                                                (10.0, 20.0, 50.0),
                                                (0.0, 0.1, 1.0)):
            rows = evaluate(kp, kv, baseline["arm_force"], baseline["hand_kp"],
                            baseline["hand_kv"], baseline["hand_force"], arm_damping=damping)
            tracking = float(np.mean([r["tracking_vs_real_next_state"]["rms"] for r in rows]))
            flange = float(np.mean([r["flange_error_m"]["rms"] for r in rows]))
            flange_max = float(np.max([r["flange_error_m"]["max"] for r in rows]))
            sat = float(np.mean([np.mean(r["force_saturation_fraction"][:6]) for r in rows]))
            results.append({"arm_kp": kp, "arm_kv": kv, "arm_damping": damping,
                            "track_real_rms": tracking, "flange_rms": flange,
                            "flange_max": flange_max, "arm_saturation": sat})
            print("  %6.0f %6.0f %8.2f %12.5f %11.4f %11.4f %8.3f%s" %
                  (kp, kv, damping, tracking, flange, flange_max, sat,
                   "  <-- saturating" if sat > 0.01 else ""))
        results.sort(key=lambda r: r["flange_rms"])
        report["sweep"] = results
        print()
        print("  best by flange RMS : kp=%.0f kv=%.0f damping=%.2f -> flange_rms=%.4f m, max=%.4f, sat=%.3f"
              % (results[0]["arm_kp"], results[0]["arm_kv"], results[0]["arm_damping"],
                 results[0]["flange_rms"], results[0]["flange_max"], results[0]["arm_saturation"]))
        by_track = sorted(results, key=lambda r: r["track_real_rms"])
        print("  best by joint RMS  : kp=%.0f kv=%.0f damping=%.2f -> track_real=%.5f rad, flange_rms=%.4f"
              % (by_track[0]["arm_kp"], by_track[0]["arm_kv"], by_track[0]["arm_damping"],
                 by_track[0]["track_real_rms"], by_track[0]["flange_rms"]))

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
