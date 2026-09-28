import tempfile
import unittest
from pathlib import Path

import scene
import mujoco
import numpy as np
import hand_control


class HandContactTests(unittest.TestCase):
    def test_binary_motion_and_object_contact(self):
        with tempfile.TemporaryDirectory() as directory:
            model=mujoco.MjModel.from_xml_path(str(scene.build(output=Path(directory)/"scene.xml")))
            data,_=scene.validate(model,seconds=.5)
            indices=[model.joint("hand_"+name).qposadr[0] for name in hand_control.HAND_JOINTS]
            opening=data.qpos[indices].copy()
            hand_control.command(model,data,"closed")
            for _ in range(1000):
                mujoco.mj_step(model,data)
            mujoco.mj_forward(model,data)
            closed=data.qpos[indices].copy()
            np.testing.assert_allclose(closed,hand_control.poses()["closed"],atol=.06)
            self.assertGreater(np.linalg.norm(closed-opening),1)
            np.testing.assert_allclose(closed[[3,6,8]],0,atol=.01)
            hand_control.command(model,data,"open")
            for _ in range(1000):
                mujoco.mj_step(model,data)
            np.testing.assert_allclose(data.qpos[indices],hand_control.poses()["open"],atol=.06)
            # Deliberate overlap is a collision detector probe, NOT a grasp demo.
            finger=model.body("hand_L_index_dip").id
            geom_id=next(i for i in range(model.ngeom)
                         if model.geom_bodyid[i]==finger and model.geom_contype[i]==2)
            mujoco.mj_forward(model,data)
            target=model.body("target").id
            adr=model.joint("target_free").qposadr[0]
            data.qpos[adr:adr+3]=data.geom_xpos[geom_id]
            data.qpos[adr+3:adr+7]=[1,0,0,0]
            mujoco.mj_forward(model,data)
            contacts=[contact for contact in data.contact
                      if target in {model.geom_bodyid[contact.geom1],model.geom_bodyid[contact.geom2]}
                      and (model.geom_contype[contact.geom1]==2 or model.geom_contype[contact.geom2]==2)]
            self.assertGreater(len(contacts),0)
