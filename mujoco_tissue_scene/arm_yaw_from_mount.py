#!/usr/bin/env python3
"""Derive the base yaw from the mounting constraint instead of fitting it.

The arm is bolted perpendicular to the table edge it is mounted on, so its yaw is
NOT a free parameter: it is whichever rotation points the arm's reach envelope
into the table.  This computes that envelope from the URDF alone.

    python arm_yaw_from_mount.py            # report the derived yaw
    python arm_yaw_from_mount.py --check    # also test task-level agreement
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import action_layout as layout          # noqa: E402
import scene                            # noqa: E402


def reach_envelope():
    import mujoco
    xml = scene.build(Path(scene.ROOT) / "configs/scene.yaml", "/tmp/arm_yaw/scene.xml",
                      with_hand=True)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    base = model.body("qiuzhi_arm_mount").id
    palm = model.body("static_omnihand_reference").id
    adr = [model.joint(n).qposadr[0] for n in layout.SIM_ARM_JOINTS]
    lo = np.array([model.jnt_range[model.joint(n).id][0] for n in layout.SIM_ARM_JOINTS])
    hi = np.array([model.jnt_range[model.joint(n).id][1] for n in layout.SIM_ARM_JOINTS])
    rng = np.random.default_rng(1)
    # Local frame of the base: the mount body carries the arm position/euler, so
    # measure the palm in the mount's own frame by resetting qpos and reading xpos.
    mujoco.mj_resetData(model, data)
    mujoco.mj_kinematics(model, data)
    origin = data.xpos[base].copy()
    samples = []
    for _ in range(120000):
        data.qpos[adr] = rng.uniform(lo, hi)
        mujoco.mj_kinematics(model, data)
        offset = data.xpos[palm] - origin
        samples.append(offset)
    return np.array(samples)


def main() -> int:
    samples = reach_envelope()
    local_xy = samples[:, :2]
    radius = np.linalg.norm(local_xy, axis=1)
    azimuth = np.degrees(np.arctan2(local_xy[:, 1], local_xy[:, 0]))

    print("=== reach envelope in the arm's own base frame ===")
    print("  max radius overall : %.4f m" % radius.max())
    print("  radius percentiles : p50 %.3f  p90 %.3f  p99 %.3f m"
          % tuple(np.percentile(radius, [50, 90, 99])))
    print()
    print("  azimuth (deg)   max radius (m)   p99 radius (m)")
    bins = np.arange(-180, 181, 15)
    best_azimuth, best_radius = None, -1.0
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (azimuth >= low) & (azimuth < high)
        if mask.sum() < 20:
            continue
        peak = radius[mask].max()
        print("   %6.0f..%4.0f   %12.4f   %12.4f" % (low, high, peak,
                                                     np.percentile(radius[mask], 99)))
        if peak > best_radius:
            best_radius, best_azimuth = peak, 0.5 * (low + high)

    print()
    print("  direction of maximum reach, in the base frame: %.1f deg "
          "(%.4f rad), reach %.4f m"
          % (best_azimuth, np.radians(best_azimuth), best_radius))

    # Mounted on the near edge (y = 0) facing into the table: that direction must
    # become world +y, i.e. 90 deg.  A base-frame direction phi under euler z = t
    # lands at phi + t, so t = 90 - phi.
    yaw = np.radians(90.0 - best_azimuth)
    while yaw > np.pi:
        yaw -= 2 * np.pi
    while yaw <= -np.pi:
        yaw += 2 * np.pi
    print()
    print("=== derived base yaw (mounted on the near edge, facing +y) ===")
    print("  yaw = %.4f rad = %.1f deg" % (yaw, np.degrees(yaw)))
    print("  config currently uses 1.3678 rad = 78.4 deg")
    print("  difference = %.1f deg" % (np.degrees(yaw) - 78.4))

    # Reach available toward the bags once the base is rotated that way.
    rotated = np.stack([local_xy[:, 0] * np.cos(yaw) - local_xy[:, 1] * np.sin(yaw),
                        local_xy[:, 0] * np.sin(yaw) + local_xy[:, 1] * np.cos(yaw)], axis=1)
    for label, (x_cm, y_cm, z_cm) in {
            "bag far-left (12.8, 61.8)": (12.8, 61.8, 3.5),
            "bag target   (25.9, 58.3)": (25.9, 58.3, 3.5),
            "tray centre  (71.3, 40.6)": (71.3, 40.6, 6.0)}.items():
        base_xy = np.array([0.25 - 0.6, 0.10 - 0.375])
        want = np.array([x_cm / 100.0 - 0.6, y_cm / 100.0 - 0.375]) - base_xy
        closest = np.linalg.norm(rotated - want, axis=1).min()
        print("  %-28s nearest palm sample %.3f m away" % (label, closest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
