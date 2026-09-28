import tempfile
import unittest
from pathlib import Path

import scene
import mujoco
import numpy as np


class SceneTests(unittest.TestCase):
    def test_measured_table_corner_coordinates(self):
        np.testing.assert_allclose(
            scene.table_corner_cm_to_world_m([25.0, 10.0], [120.0, 75.0]),
            [-0.35, -0.275],
        )
        settings = {
            "table": {"size": [1.0, 1.0, 0.04]},
            "tray": {"center": [0.0, 0.0], "size": [0.0, 0.0, 0.045]},
            "arm": {"position": [0.0, 0.0, 0.75]},
            "measured_layout": {
                "enabled": True,
                "table_size_cm": [120.0, 75.0],
                "arm_mount_xy_cm_from_left_bottom": [25.0, 10.0],
                "tray_lower_left_xy_cm_from_left_bottom": [63.0, 30.0],
                "tray_size_cm": [21.0, 20.0],
            },
        }
        scene.apply_measured_layout(settings)
        np.testing.assert_allclose(settings["table"]["size"][:2], [1.20, 0.75])
        np.testing.assert_allclose(settings["arm"]["position"][:2], [-0.35, -0.275])
        np.testing.assert_allclose(settings["tray"]["size"][:2], [0.21, 0.20])
        np.testing.assert_allclose(settings["tray"]["center"], [0.135, 0.025])

    def test_physics_and_static_hand(self):
        with tempfile.TemporaryDirectory() as directory:
            path=scene.build(output=Path(directory)/"scene.xml")
            model=mujoco.MjModel.from_xml_path(str(path))
            data,report=scene.validate(model,seconds=1)
            self.assertEqual(model.nq,43)  # 21 box + 6 arm + 16 linked hand joints
            self.assertEqual(model.nu,16)
            hand = model.body("static_omnihand_reference").id
            parents=[]
            while hand:
                parents.append(model.body(hand).name)
                hand=int(model.body_parentid[hand])
            self.assertIn("link6",parents)
            before=data.xpos[model.body("static_omnihand_reference").id].copy()
            camera_id=model.camera("left_wrist").id
            # At the recorded initial arm posture, camera below palm and upright.
            self.assertLess(data.cam_xpos[camera_id,2],data.xpos[model.body("static_omnihand_reference").id,2])
            self.assertGreater(data.cam_xmat[camera_id].reshape(3,3)[2,1],.8)
            camera_before=data.cam_xpos[camera_id].copy()
            camera_rotation_before=data.cam_xmat[camera_id].copy()
            data.qpos[model.joint("joint1").qposadr[0]] += .2
            mujoco.mj_forward(model,data)
            after=data.xpos[model.body("static_omnihand_reference").id].copy()
            self.assertGreater(np.linalg.norm(after-before),.01)
            self.assertGreater(np.linalg.norm(data.cam_xpos[camera_id]-camera_before),.01)
            self.assertGreater(np.linalg.norm(data.cam_xmat[camera_id]-camera_rotation_before),.01)
            self.assertEqual(model.body(model.cam_bodyid[camera_id]).name,"wrist_camera_mount")
            # The calibrated hand-eye transform T_eef_camera is expressed in link6,
            # so the bracket and camera hang off the flange, not off the hand body.
            self.assertEqual(model.body(model.cam_bodyid[camera_id]).name,"wrist_camera_mount")
            mount_parent=int(model.body_parentid[model.body("wrist_camera_mount").id])
            self.assertEqual(model.body(mount_parent).name,"link6")
            self.assertEqual(model.geom("wrist_camera_support").bodyid[0],model.body("link6").id)
            self.assertFalse(report["robot_control_ready"])
            self.assertGreater(data.xpos[model.body("target").id,2],.77)
            self.assertGreaterEqual(data.ncon,1)

    def test_grasp_start(self):
        with tempfile.TemporaryDirectory() as directory:
            path=scene.build(output=Path(directory)/"scene.xml",with_hand=False,target_on_table=True)
            model=mujoco.MjModel.from_xml_path(str(path))
            data=mujoco.MjData(model)
            mujoco.mj_forward(model,data)
            expected = scene.target_on_table_xy(
                scene.apply_measured_layout(
                    scene.yaml.safe_load((scene.ROOT / "configs/scene.yaml").read_text())
                )
            )
            np.testing.assert_allclose(data.xpos[model.body("target").id,:2],expected,atol=.005)


if __name__=="__main__":
    unittest.main()
