import sys
from pathlib import Path
import mujoco
sys.path.insert(0, "/workspace/shared/mujoco_tissue_scene")
import scene
m = mujoco.MjModel.from_xml_path(str(scene.build(
    Path("/workspace/shared/mujoco_tissue_scene/configs/scene.yaml"),
    Path("/tmp/_wcm.xml"), with_hand=True)))
print("geoms on body wrist_camera_mount:")
for g in range(m.ngeom):
    b = m.body(int(m.geom_bodyid[g])).name or "?"
    if b == "wrist_camera_mount":
        print("   %-30s type=%-8s contype=%d conaffinity=%d size=%s"
              % (m.geom(g).name, mujoco.mjtGeom(m.geom_type[g]).name,
                 m.geom_contype[g], m.geom_conaffinity[g], m.geom_size[g]))
