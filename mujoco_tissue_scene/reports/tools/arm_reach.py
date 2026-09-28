#!/usr/bin/env python3
"""Geometry of the supplied URDF: how far can this arm actually reach?

Answers purely from the URDF (no fitting):
  * joint axes, limits, and any MuJoCo `ref` offset that would shift q = 0,
  * the kinematic geometry of the chain,
  * the maximum reachable distance of the flange and of the palm,
  * which candidate base placements can physically reach the bags and the tray.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import action_layout as layout          # noqa: E402
import scene                            # noqa: E402

SURFACE = 0.75
TABLE = [1.2, 0.75]


def to_world(x_cm, y_cm, z=0.0):
    return np.array([x_cm / 100.0 - TABLE[0] / 2.0,
                     y_cm / 100.0 - TABLE[1] / 2.0,
                     SURFACE + z])


def main() -> int:
    import mujoco

    xml = scene.build(Path(scene.ROOT) / "configs/scene.yaml", "/tmp/arm_reach/scene.xml",
                      with_hand=True)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    base = model.body("qiuzhi_arm_mount").id
    palm_body = model.body("static_omnihand_reference").id
    flange_body = model.body("link6").id

    print("=== arm joints exactly as MuJoCo imported them ===")
    print("  %-8s %-16s %22s %20s %9s" % ("joint", "axis", "lo..hi (rad)", "lo..hi (deg)", "ref"))
    for name in layout.SIM_ARM_JOINTS:
        joint = model.joint(name)
        lo, hi = model.jnt_range[joint.id]
        print("  %-8s %-16s %9.4f..%8.4f %8.1f..%8.1f %9.4f"
              % (name, np.round(model.jnt_axis[joint.id], 3).tolist(), lo, hi,
                 np.degrees(lo), np.degrees(hi), model.qpos0[joint.qposadr[0]]))

    adr = [model.joint(n).qposadr[0] for n in layout.SIM_ARM_JOINTS]
    limits_lo = np.array([model.jnt_range[model.joint(n).id][0] for n in layout.SIM_ARM_JOINTS])
    limits_hi = np.array([model.jnt_range[model.joint(n).id][1] for n in layout.SIM_ARM_JOINTS])

    print()
    print("=== geometry at q = 0 (all joints at zero) ===")
    mujoco.mj_resetData(model, data)
    mujoco.mj_kinematics(model, data)
    origin = data.xpos[base].copy()
    for name in ("link1", "link2", "link3", "link4", "link5", "link6"):
        offset = data.xpos[model.body(name).id] - origin
        print("  %-6s offset %-32s |r| = %.4f m"
              % (name, np.round(offset, 4).tolist(), np.linalg.norm(offset)))
    print("  palm   offset %-32s |r| = %.4f m"
          % (np.round(data.xpos[palm_body] - origin, 4).tolist(),
             np.linalg.norm(data.xpos[palm_body] - origin)))

    print()
    print("=== maximum reach, sampling the joint limits 40000 times ===")
    rng = np.random.default_rng(0)
    flange, palm = [], []
    for _ in range(40000):
        data.qpos[adr] = rng.uniform(limits_lo, limits_hi)
        mujoco.mj_kinematics(model, data)
        flange.append(np.linalg.norm(data.xpos[flange_body] - origin))
        palm.append(np.linalg.norm(data.xpos[palm_body] - origin))
    flange, palm = np.array(flange), np.array(palm)
    print("  flange(link6) : max %.4f  p99 %.4f  median %.4f m"
          % (flange.max(), np.percentile(flange, 99), np.median(flange)))
    print("  palm          : max %.4f  p99 %.4f  median %.4f m"
          % (palm.max(), np.percentile(palm, 99), np.median(palm)))

    targets = {
        "bag far-left (12.8, 61.8)": to_world(12.8, 61.8, 0.035),
        "bag target   (25.9, 58.3)": to_world(25.9, 58.3, 0.035),
        "bag 3rd      (47.9, 41.3)": to_world(47.9, 41.3, 0.035),
        "tray centre  (71.3, 40.6)": to_world(71.3, 40.6, 0.06),
    }
    bases = [("near edge  (25, 0)", 25.0, 0.0),
             ("near edge  (25, -10)", 25.0, -10.0),
             ("left edge  (0, 10)", 0.0, 10.0),
             ("left edge  (-10, 10)", -10.0, 10.0),
             ("as configured (25, 10)", 25.0, 10.0),
             ("mirrored (95, 15)", 95.0, 15.0)]
    print()
    print("=== distance from each candidate base to each target, vs the max reach ===")
    print("  reach limit: flange %.3f m / palm %.3f m" % (flange.max(), palm.max()))
    print("  %-24s %s" % ("base", "  ".join("%-26s" % k for k in targets)))
    for name, x_cm, y_cm in bases:
        base_world = to_world(x_cm, y_cm, -0.10)      # base sits below the surface
        row = []
        for target in targets.values():
            gap = np.linalg.norm(target - base_world)
            row.append("%7.3f m %s" % (gap, "OK " if gap <= flange.max() else "TOO FAR"))
        print("  %-24s %s" % (name, "  ".join("%-26s" % v for v in row)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
