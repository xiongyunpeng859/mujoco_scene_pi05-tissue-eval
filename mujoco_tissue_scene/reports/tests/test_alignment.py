import unittest
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET

import scene
import numpy as np
import mujoco
import cv2
from align_real_dataset import set_pose
from fit_central_camera import tray_corners
from smoke_chain import SimulationAdapter
import hand_control
import yaml


class AlignmentTests(unittest.TestCase):
    def test_intrinsic_projection(self):
        config={"intrinsics":{"resolution":[640,480],"fx":394.4324,"fy":393.4235,"cx":318.933,"cy":250.699}}
        attrs=scene.camera_intrinsics(config)
        attributes=" ".join(f'{key}="{value}"' for key,value in attrs.items())
        model=mujoco.MjModel.from_xml_string(f'<mujoco><visual><global offwidth="640" offheight="480"/></visual><worldbody><camera name="c" pos="0 0 1" {attributes}/><geom type="sphere" size=".025" rgba="1 0 0 1"/></worldbody></mujoco>')
        data=mujoco.MjData(model)
        mujoco.mj_forward(model,data)
        renderer=mujoco.Renderer(model,480,640)
        try:
            renderer.update_scene(data,camera="c")
            image=renderer.render().astype(int)
            ys,xs=np.where((image[:,:,0]>image[:,:,1]+20)&(image[:,:,0]>image[:,:,2]+20))
            self.assertAlmostEqual(xs.mean()+.5,318.933,delta=1)
            self.assertAlmostEqual(ys.mean()+.5,250.699,delta=1)
        finally:
            renderer.close()

    def test_world_camera_accepts_full_intrinsics(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = yaml.safe_load((scene.ROOT / "configs/scene.yaml").read_text())
            settings["cameras"]["central"]["intrinsics"] = {
                "resolution": [640, 480], "fx": 500.0, "fy": 501.0,
                "cx": 315.0, "cy": 237.0,
            }
            config_path = Path(directory) / "scene.yaml"
            config_path.write_text(yaml.safe_dump(settings, sort_keys=False))
            output_path = Path(directory) / "scene.xml"
            scene.build(config_path=config_path, output=output_path, with_hand=False)
            central = next(
                node for node in ET.parse(output_path).iter("camera")
                if node.get("name") == "central"
            )
            self.assertEqual(central.get("focalpixel"), "500.0 501.0")
            self.assertEqual(central.get("principalpixel"), "5.0 3.0")
            self.assertIsNone(central.get("fovy"))

    def test_quad_order_and_rejection(self):
        image=np.zeros((480,640,3),dtype=np.uint8)
        cv2.rectangle(image,(200,150),(350,300),(0,200,0),-1)
        np.testing.assert_allclose(tray_corners(image),[[200,150],[350,150],[350,300],[200,300]])
        with self.assertRaises(ValueError):
            tray_corners(np.zeros_like(image))

    def test_static_joint_and_mimic_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            model=mujoco.MjModel.from_xml_path(str(scene.build(output=Path(directory)/"scene.xml")))
            adapter=SimulationAdapter(model)
            try:
                state=np.r_[model.key_qpos[0,adapter.qpos_indices],hand_control.poses()["closed"]]
                set_pose(adapter,state)
                np.testing.assert_allclose(adapter.data.qpos[adapter.qpos_indices],state[:6])
                np.testing.assert_allclose(adapter.data.qpos[adapter.hand_indices],state[6:])
                for i in range(model.neq):
                    name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_EQUALITY,i)
                    if name.startswith("mimic_"):
                        a,b=model.eq_obj1id[i],model.eq_obj2id[i]
                        expected=np.polynomial.polynomial.polyval(adapter.data.qpos[model.jnt_qposadr[b]],model.eq_data[i,:5])
                        self.assertAlmostEqual(adapter.data.qpos[model.jnt_qposadr[a]],expected)
            finally:
                adapter.close()


if __name__=="__main__":
    unittest.main()
