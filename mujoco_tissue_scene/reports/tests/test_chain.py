import tempfile
import unittest
from pathlib import Path

import numpy as np
import scene
import mujoco
from smoke_chain import SimulationAdapter,run


class ChainTests(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            report=run(Path(directory),ticks=2)
            self.assertEqual(report["status"],"PASS_BINARY_HAND_ARM_CAMERA_CHAIN")
            self.assertFalse(report["model_loaded"])
            self.assertTrue(report["hand_control_ready"])

    def test_reject_invalid_or_unsupported_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            model=mujoco.MjModel.from_xml_path(str(scene.build(output=Path(directory)/"scene.xml")))
            adapter=SimulationAdapter(model)
            try:
                for action in [np.zeros(15),np.full(16,np.nan)]:
                    with self.assertRaises(ValueError):
                        adapter.execute(action)
                action=np.zeros(16)
                action[6]=.1
                with self.assertRaises(ValueError):
                    adapter.execute(action)
                action=np.zeros(32)
                action[20]=1
                with self.assertRaises(ValueError):
                    adapter.execute(action)
            finally:
                adapter.close()

    def test_30hz_scheduler_preserves_elapsed_time(self):
        with tempfile.TemporaryDirectory() as directory:
            model=mujoco.MjModel.from_xml_path(str(scene.build(output=Path(directory)/"scene.xml")))
            adapter=SimulationAdapter(model)
            try:
                action=adapter.observe()["observation.state"]
                # 500 Hz physics cannot represent 1/30 s in one equal-sized step;
                # residual scheduling must nevertheless total exactly one second.
                for _ in range(30):
                    adapter.execute(action,control_dt=1/30)
                self.assertAlmostEqual(adapter.data.time,1.0,places=9)
            finally:
                adapter.close()
