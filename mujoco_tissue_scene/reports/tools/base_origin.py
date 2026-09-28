#!/usr/bin/env python3
"""Is the URDF's base_link origin offset from the physical base?

The user measured the physical base centre at (25, 10) cm.  The fit wants (18, 10),
i.e. 7 cm along table x, with the base yawed by 90 degrees -- so the discrepancy is
along the base frame's local y.  If the URDF's base_link origin is not at the base
plate's centre, placing that origin at the measured point shifts the whole arm.

Everything here comes from the supplied meshes, so it settles the 7 cm without any
new measurement.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import scene                            # noqa: E402


def main() -> int:
    import mujoco
    out = ROOT / "outputs/base_origin"
    out.mkdir(parents=True, exist_ok=True)
    xml = scene.build(ROOT / "configs/scene.yaml", out / "scene.xml", with_hand=True)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    mount = model.body("qiuzhi_arm_mount")
    mount_pos = data.xpos[mount.id].copy()
    mount_rot = data.xmat[mount.id].reshape(3, 3).copy()
    print("mount body at %s" % np.round(mount_pos, 4).tolist())

    print()
    print("geoms on each base-chain body, with the mesh AABB centre in the BODY frame")
    print("  %-8s %-28s %-30s %s" % ("body", "geom type / mesh", "aabb centre (body frame)",
                                     "|offset| cm"))
    report = {}
    for body_name in ("base_link", "link1", "link2", "link3"):
        body = model.body(body_name)
        if body.id < 0:
            continue
        body_pos = data.xpos[body.id].copy()
        body_rot = data.xmat[body.id].reshape(3, 3).copy()
        for gid in range(model.ngeom):
            if int(model.geom_bodyid[gid]) != body.id:
                continue
            if int(model.geom_contype[gid]) != 0:
                continue                      # visual meshes only
            gtype = int(model.geom_type[gid])
            did = int(model.geom_dataid[gid])
            if gtype != 7 or did < 0:
                continue
            adr, num = int(model.mesh_vertadr[did]), int(model.mesh_vertnum[did])
            verts = model.mesh_vert[adr:adr + num].astype(np.float64)
            geom_pos = data.geom_xpos[gid].copy()
            geom_rot = data.geom_xmat[gid].reshape(3, 3).copy()
            world = verts @ geom_rot.T + geom_pos
            # express in the mount body frame
            local = (world - mount_pos) @ mount_rot
            lo, hi = local.min(0), local.max(0)
            centre = 0.5 * (lo + hi)
            size = hi - lo
            report[body_name] = {"aabb_centre_mount_local": centre.tolist(),
                                 "aabb_size": size.tolist()}
            print("  %-8s %-28s %-30s %.2f"
                  % (body_name, "mesh %d (%d verts)" % (did, num),
                     np.round(centre, 4).tolist(), np.linalg.norm(centre) * 100))
            print("           aabb size %s m" % np.round(size, 4).tolist())

    print()
    yaw = float(yaml.safe_load((ROOT / "configs/scene.yaml").read_text())["arm"]["euler"][2])
    print("base yaw = %.4f rad (%.1f deg)" % (yaw, np.degrees(yaw)))
    if "base_link" in report:
        centre = np.array(report["base_link"]["aabb_centre_mount_local"])
        # a shift of the URDF origin by (dx, dy) in the mount-local frame moves the
        # arm on the table by the same rotated amount
        world_shift = np.array([[np.cos(yaw), -np.sin(yaw)],
                                [np.sin(yaw), np.cos(yaw)]]) @ centre[:2]
        print("if the physical base centre is the mesh AABB centre, then placing the "
              "URDF origin at your measured (25, 10) puts the true centre at "
              "(%.1f, %.1f) cm" % (25.0 + world_shift[0] * 100,
                                   10.0 + world_shift[1] * 100))
        print("   -> a base-frame offset of %s m appears on the table as %s m"
              % (np.round(centre[:2], 4).tolist(), np.round(world_shift, 4).tolist()))
        print()
        print("The fit asked for x = 18 instead of 25, i.e. the arm must sit 7 cm "
              "further -x than your measurement.")
        residual = 25.0 - 18.0 + world_shift[0] * 100
        print("   residual after accounting for this offset: %.2f cm along table x"
              % residual)
        print("   (near zero means the 7 cm was purely a base-frame origin offset)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
