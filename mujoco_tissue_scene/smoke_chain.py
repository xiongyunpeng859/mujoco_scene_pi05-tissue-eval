"""CPU-only observation -> mock action chunk -> simulation feedback smoke test.

No Torch/JAX checkpoint is loaded. Hand commands select real binary gestures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import scene  # Select OSMesa before any graphics imports.
import hand_control
import mujoco
import numpy as np
from PIL import Image

JOINT_NAMES = [f"joint{i}.pos" for i in range(1,7)] + [
    "thumb_cm_roll.pos", "thumb_cm_yaw.pos", "thumb_cm_pitch.pos",
    "index_mp_yaw.pos", "index_mp_pitch.pos", "middle_mp_pitch.pos",
    "ring_mp_yaw.pos", "ring_mp_pitch.pos", "pinky_mp_yaw.pos", "pinky_mp_pitch.pos",
]


class SimulationAdapter:
    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model,self.data,0)
        mujoco.mj_forward(model,self.data)
        self.qpos_indices = [model.joint(f"joint{i}").qposadr[0] for i in range(1,7)]
        self.ctrl_indices = [model.actuator(f"arm_joint{i}_position").id for i in range(1,7)]
        self.hand_indices=[model.joint("hand_"+name).qposadr[0] for name in hand_control.HAND_JOINTS]
        self.supported = np.ones(16,dtype=bool)
        self.options = mujoco.MjvOption()
        self.options.geomgroup[3] = 0
        self.renderer = mujoco.Renderer(model,height=480,width=640)
        self._control_step_remainder = 0.0

    def close(self):
        self.renderer.close()

    def observe(self):
        state = np.zeros(16,dtype=np.float32)
        state[:6] = self.data.qpos[self.qpos_indices]
        state[6:] = self.data.qpos[self.hand_indices]
        observation = {"observation.state":state,
                       "supported_state_mask":self.supported.copy(),
                       "timestamp":float(self.data.time),
                       "task":"pick up the tissue pack and place it on the right side"}
        for key,camera in [("observation.images.top","central"),
                           ("observation.images.left","left_wrist")]:
            self.renderer.update_scene(self.data,camera=camera,scene_option=self.options)
            image = self.renderer.render().copy()
            assert image.shape == (480,640,3) and image.dtype == np.uint8
            observation[key] = image
        return observation

    def execute(self, action, control_dt=.1):
        action=np.asarray(action,dtype=np.float32)
        if action.shape not in {(16,),(32,)} or not np.isfinite(action).all():
            raise ValueError("Expected finite 16D action or 32D zero-padded action")
        if action.shape == (32,) and np.any(action[16:] != 0):
            raise ValueError("Nonzero padded action channels")
        pose_values=hand_control.poses()
        distances={name:np.linalg.norm(action[6:16]-values) for name,values in pose_values.items()}
        hand_state=min(distances,key=distances.get)
        if distances[hand_state] > .15:
            raise ValueError("Hand action must match a real open/closed pose; no free finger control")
        exact_steps = control_dt / self.model.opt.timestep + self._control_step_remainder
        steps = round(exact_steps)
        self._control_step_remainder = exact_steps - steps
        if steps < 1:
            raise ValueError("control_dt must be at least one physics timestep")
        for index,command in enumerate(action[:6],1):
            bounds = self.model.jnt_range[self.model.joint(f"joint{index}").id]
            if not bounds[0] <= command <= bounds[1]:
                raise ValueError(f"joint{index} target outside limits")
        self.data.ctrl[self.ctrl_indices] = action[:6]
        hand_control.command(self.model,self.data,hand_state)
        for _ in range(steps):
            mujoco.mj_step(self.model,self.data)
            if not np.isfinite(self.data.qpos).all():
                raise RuntimeError("Non-finite simulation state")
        mujoco.mj_forward(self.model,self.data)


class MockPolicy:
    """Explicit synthetic policy. Not preprocessing or evaluating Pi0.5."""
    def predict(self, observation):
        state=observation["observation.state"]
        assert state.shape == (16,)
        assert observation["observation.images.top"].shape == (480,640,3)
        assert observation["observation.images.left"].shape == (480,640,3)
        chunk=np.zeros((4,32),dtype=np.float32)
        chunk[:,:6]=state[:6]
        chunk[:,6:16]=hand_control.poses()["closed" if observation["timestamp"] >= .4 else "open"]
        chunk[:,0]+=np.linspace(.015,.06,4)
        return {"actions":chunk}


def run(output_dir, ticks=12):
    if ticks < 1:
        raise ValueError("ticks must be positive")
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    path=scene.build(output=output_dir/"scene.xml",target_on_table=True)
    adapter=SimulationAdapter(mujoco.MjModel.from_xml_path(str(path)))
    policy=MockPolicy()
    trace=[]
    try:
        initial=adapter.observe()
        camera_id=adapter.model.camera("left_wrist").id
        camera_before=adapter.data.cam_xpos[camera_id].copy()
        for tick in range(ticks):
            before=adapter.observe()
            # Replan every tick and execute the first action (receding horizon).
            chunk=policy.predict(before)["actions"]
            assert chunk.shape == (4,32)
            adapter.execute(chunk[0])
            after=adapter.observe()
            assert after["timestamp"] > before["timestamp"]
            trace.append({"tick":tick,"timestamp":after["timestamp"],
                          "arm_target":chunk[0,:6].tolist(),
                          "arm_state":after["observation.state"][:6].tolist()})
        final=adapter.observe()
        arm_change=float(np.linalg.norm(final["observation.state"][:6]-initial["observation.state"][:6]))
        camera_change=float(np.linalg.norm(adapter.data.cam_xpos[camera_id]-camera_before))
        assert arm_change>.005 and camera_change>.001
        image_changed=not np.array_equal(initial["observation.images.left"],final["observation.images.left"])
        assert image_changed
        for suffix,observation in [("initial",initial),("final",final)]:
            for key,name in [("observation.images.top","central"),("observation.images.left","left_wrist")]:
                Image.fromarray(observation[key]).save(output_dir/f"{name}_{suffix}.png")
        report={"status":"PASS_BINARY_HAND_ARM_CAMERA_CHAIN","model_loaded":False,
                "backend":scene.os.environ["MUJOCO_GL"],"ticks":ticks,"control_hz":10,
                "image_shape":[480,640,3],"state_shape":[16],"mock_chunk_shape":[4,32],
                "joint_names":JOINT_NAMES,"supported_state_mask":adapter.supported.tolist(),
                "arm_state_change_rad":arm_change,"wrist_camera_displacement_m":camera_change,
                "wrist_image_changed":image_changed,"hand_control_ready":True,
                "checkpoint_processors_tested":False,"grasp_tested":False,
                "limitations":["Hand API restricted to real open/closed gesture targets",
                               "Robot-environment contact enabled; self-contact disabled",
                               "Motor dynamics and camera calibration incomplete",
                               "Checkpoint normalization/tokenization not tested"]}
        (output_dir/"trace.json").write_text(json.dumps(trace,indent=2))
        (output_dir/"report.json").write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
        return report
    finally:
        adapter.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--ticks",type=int,default=12)
    args=parser.parse_args()
    # Default writable path for whichever account is logged into NoMachine.
    output=args.output_dir or Path(tempfile.mkdtemp(prefix="mujoco_chain_"))
    print("Output directory:",output)
    run(output,args.ticks)
