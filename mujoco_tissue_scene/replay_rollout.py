"""View recorded model rollout in NoMachine without loading a checkpoint/GPU policy."""
import argparse
import json
from pathlib import Path
import time

import scene
import mujoco
import mujoco.viewer
import numpy as np


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir",type=Path,required=True)
    parser.add_argument("--backend",choices=["glfw"],default="glfw")
    args=parser.parse_args()
    model=mujoco.MjModel.from_xml_path(str(args.rollout_dir/"scene.xml"))
    trace=json.loads((args.rollout_dir/"trace.json").read_text())
    if not trace:
        raise ValueError("Rollout contains no recorded steps")
    if any(len(frame["qpos"])!=model.nq for frame in trace):
        raise ValueError("Recorded state/model dimensions differ")
    data=mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model,data,0)
    requested={"restart":False}
    def key_callback(key):
        if key in {ord("R"),ord("r")}:
            requested["restart"]=True
    print("Recorded Torch rollout, NOT live inference. R = replay from start.")
    with mujoco.viewer.launch_passive(model,data,key_callback=key_callback) as viewer:
        viewer.opt.geomgroup[3]=0
        index=0
        while viewer.is_running():
            started=time.monotonic()
            if requested["restart"]:
                index=0
                requested["restart"]=False
            frame=trace[min(index,len(trace)-1)]
            with viewer.lock():
                data.qpos[:]=np.asarray(frame["qpos"])
                data.qvel[:]=0
                data.ctrl[:]=np.asarray(frame["ctrl"])
                data.time=frame["time"]
                mujoco.mj_forward(model,data)
            viewer.sync()
            index+=1
            time.sleep(max(0,.1-(time.monotonic()-started)))
