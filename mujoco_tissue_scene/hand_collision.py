#!/usr/bin/env python3
"""Inspect the hand's COLLISION geometry: is it a real gripping surface?

A grasp fails if the collision shapes are degenerate (e.g. contact only at the
fingertips) or if the fingers are convex hulls that cannot form a V.  This dumps
every hand/arm geom and renders the collision shapes on their own.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import scene                            # noqa: E402

TYPES = {0: "plane", 1: "hfield", 2: "sphere", 3: "capsule", 4: "ellipsoid",
         5: "cylinder", 6: "box", 7: "mesh"}


def main() -> int:
    import mujoco
    out = ROOT / "outputs/hand_collision"
    out.mkdir(parents=True, exist_ok=True)
    xml = scene.build(ROOT / "configs/scene.yaml", out / "scene.xml", with_hand=True)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    hand_bodies = {i: model.body(i).name for i in range(model.nbody)
                   if (model.body(i).name or "").startswith("hand_")}
    arm_bodies = {i: model.body(i).name for i in range(model.nbody)
                  if (model.body(i).name or "").startswith("link")}

    def dump(title, bodies):
        print("=== %s ===" % title)
        print("  %-34s %-8s %-26s %4s %4s %5s %5s  %s" %
              ("geom", "type", "size", "con", "aff", "condim", "group", "mesh(v/f)"))
        collision = 0
        for gid in range(model.ngeom):
            body = int(model.geom_bodyid[gid])
            if body not in bodies:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            gtype = model.geom_type[gid]
            size = model.geom_size[gid]
            mesh = ""
            if gtype == 7:
                did = model.geom_dataid[gid]
                if did >= 0:
                    mesh = "%d/%d" % (model.mesh_vertnum[did], model.mesh_facenum[did])
            active = int(model.geom_contype[gid]) or int(model.geom_conaffinity[gid])
            if active:
                collision += 1
            print("  %-34s %-8s %-26s %4d %4d %5d %5d  %s" %
                  (name[:34], TYPES.get(int(gtype), "?"),
                   np.round(size, 4).tolist(), int(model.geom_contype[gid]),
                   int(model.geom_conaffinity[gid]), int(model.geom_condim[gid]),
                   int(model.geom_group[gid]), mesh))
        print("  --> geoms with active contact: %d / %d" %
              (collision, sum(1 for g in range(model.ngeom)
                              if int(model.geom_bodyid[g]) in bodies)))
        print()
        return collision

    n_hand = dump("HAND geoms", hand_bodies)
    n_arm = dump("ARM geoms", arm_bodies)

    # Render: collision-only and visual-only, from the central camera.
    import cv2
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        for label, groups in (("visual", [1]), ("collision", [3]), ("both", [1, 3])):
            option = mujoco.MjvOption()
            option.geomgroup[:] = 0
            for g in groups:
                option.geomgroup[g] = 1
            for camera in ("central", "left_wrist"):
                renderer.update_scene(data, camera=camera, scene_option=option)
                image = renderer.render()
                cv2.imwrite(str(out / ("%s_%s.png" % (label, camera))),
                            cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            print("rendered %s" % label)

    print()
    print("hand contact geoms: %d   arm contact geoms: %d" % (n_hand, n_arm))
    print("images in %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
