#!/usr/bin/env python3
"""Do my IK joint addresses match the action layout's order?"""
import sys
from pathlib import Path

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import action_layout as layout
import scene

import mujoco

model = mujoco.MjModel.from_xml_path(str(scene.build(
    ROOT / "configs/scene.yaml", Path("/tmp/_chk.xml"), with_hand=True)))

jids = layout.joint_ids(model, mujoco)
print("DATASET_NAMES      :", layout.DATASET_NAMES)
print("SIM_ARM_JOINTS     :", layout.SIM_ARM_JOINTS)
print()
print("layout.joint_ids() first 6 qpos addresses:", list(jids[:6]))
mine = [model.joint(n).qposadr[0] for n in layout.SIM_ARM_JOINTS]
print("my arm_qadr (from SIM_ARM_JOINTS)        :", mine)
print("MATCH:", list(jids[:6]) == mine)
print()
print("joint names in dataset order:")
for i, adr in enumerate(jids):
    # find which joint owns this qpos address
    owner = None
    for j in range(model.njnt):
        if model.jnt_qposadr[j] == adr:
            owner = model.joint(j).name
            break
    print("   dim %2d  qposadr %3d  joint %-28s  %s"
          % (i, adr, owner, layout.DATASET_NAMES[i]))
print()
aids = layout.actuator_ids(model)
print("actuator ids in dataset order:")
for i, aid in enumerate(aids):
    print("   dim %2d  actuator %-28s  transmission joint %-24s  %s"
          % (i, model.actuator(aid).name,
             model.joint(int(model.actuator_trnid[aid][0])).name,
             layout.DATASET_NAMES[i]))
print()
print("arm joint ranges:")
for n in layout.SIM_ARM_JOINTS:
    j = model.joint(n)
    print("   %-12s range %s  qposadr %d"
          % (n, model.jnt_range[j.id], j.qposadr[0]))
