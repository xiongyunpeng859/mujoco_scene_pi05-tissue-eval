"""Short bounded Pi0.5 rollout, isolated CPU simulation and CUDA policy processes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import scene
from smoke_chain import SimulationAdapter
import hand_control
import mujoco
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
PROJECT=Path("/workspace/users/fmc3-6-workspace/pi0.5_recap")
CHECKPOINT=Path("/home/fmc3-6/workspace/shared/new_program_qiuzhi/output/torch_o10_tissue_normal_recovery_pi05_fullft_bs32_010000/checkpoints/010000/pretrained_model")


def project_action(adapter,raw,max_delta=.04):
    raw=np.asarray(raw,dtype=np.float32)
    if raw.shape!=(32,) or not np.isfinite(raw).all():
        raise ValueError("Expected finite raw 32D model action")
    action=raw[:16].copy()
    current=adapter.data.qpos[adapter.qpos_indices]
    for index in range(6):
        limits=adapter.model.jnt_range[adapter.model.joint(f"joint{index+1}").id]
        action[index]=np.clip(np.clip(raw[index],*limits),current[index]-max_delta,current[index]+max_delta)
    values=hand_control.poses()
    state=min(values,key=lambda name:np.linalg.norm(action[6:16]-values[name]))
    action[6:16]=values[state]
    return action,state


def read_message(process,prefix):
    while True:
        line=process.stdout.readline()
        if not line:
            raise RuntimeError(f"Policy worker exited ({process.poll()}); inspect policy_worker.log")
        if line.startswith(prefix+" "):
            return json.loads(line[len(prefix)+1:])
        print("Policy:",line.rstrip(),flush=True)


def run(checkpoint,output,ticks=20,execute_chunk=5,control_hz=30):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if ticks<1 or execute_chunk<1 or execute_chunk>50 or control_hz<1:
        raise ValueError("Invalid rollout length/chunk execution length")
    output.mkdir(parents=True)
    model=mujoco.MjModel.from_xml_path(str(scene.build(output=output/"scene.xml",target_on_table=True)))
    adapter=SimulationAdapter(model)
    environment=os.environ.copy()
    environment.update({"PYTHONNOUSERSITE":"1","HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1",
        "WANDB_MODE":"disabled","TOKENIZERS_PARALLELISM":"false","CUDA_VISIBLE_DEVICES":"0",
        "PYTHONPATH":str(PROJECT/".local_deps/transformers_lerobot_openpi")+":"+str(PROJECT/"src")})
    log=(output/"policy_worker.log").open("w")
    process=subprocess.Popen(["/opt/miniconda3/envs/lerobot-pi05/bin/python","-u",
                              str(ROOT/"torch_policy_worker.py"),"--checkpoint",str(checkpoint)],
                              stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,env=environment)
    trace=[]
    calls=[]
    images=[]
    target=model.body("target").id
    try:
        print("Loading trained Torch Pi0.5; simulation only",flush=True)
        ready=read_message(process,"READY")
        print("POLICY_READY",ready,flush=True)
        initial=adapter.observe()
        target_initial=adapter.data.xpos[target].copy()
        for key,name in [("observation.images.top","central"),("observation.images.left","left_wrist")]:
            Image.fromarray(initial[key]).save(output/f"{name}_initial.png")
        chunk=None
        cursor=0
        for tick in range(ticks):
            observation=adapter.observe()
            if chunk is None or cursor>=execute_chunk:
                state=np.zeros(32,dtype=np.float32)
                state[:16]=observation["observation.state"]
                packet=output/"observation_packet.npz"
                np.savez(packet,state=state,central=observation["observation.images.top"],
                         left_wrist=observation["observation.images.left"])
                request={"id":tick,"observation_path":str(packet),"task":observation["task"]}
                process.stdin.write(json.dumps(request)+"\n")
                process.stdin.flush()
                result=read_message(process,"RESULT")
                assert result["id"]==tick
                chunk=np.asarray(result.pop("actions"),dtype=np.float32)
                assert chunk.shape==(50,32)
                np.save(output/f"model_actions_tick_{tick:03d}.npy",chunk)
                calls.append(result)
                cursor=0
                print(f"Tick {tick}: action chunk {chunk.shape}, inference {result['inference_seconds']:.2f}s",flush=True)
            raw=chunk[cursor]
            action,hand_state=project_action(adapter,raw)
            adapter.execute(action,control_dt=1/control_hz)
            cursor+=1
            trace.append({"tick":tick,"time":float(adapter.data.time),"raw_action":raw.tolist(),
                          "executed_action":action.tolist(),"hand_state":hand_state,
                          "qpos":adapter.data.qpos.tolist(),"ctrl":adapter.data.ctrl.tolist(),
                          "contacts":int(adapter.data.ncon),"target_position":adapter.data.xpos[target].tolist()})
            images.append(Image.fromarray(observation["observation.images.top"]).resize((480,360)))
        final=adapter.observe()
        for key,name in [("observation.images.top","central"),("observation.images.left","left_wrist")]:
            Image.fromarray(final[key]).save(output/f"{name}_final.png")
        images.append(Image.fromarray(final["observation.images.top"]).resize((480,360)))
        images[0].save(output/"rollout.gif",save_all=True,append_images=images[1:],duration=100,loop=0)
        report={"status":"PASS_TORCH_MODEL_SIMULATION_ROLLOUT","checkpoint":str(checkpoint),
                "strict_checkpoint_load":ready["strict_load"],"ticks":ticks,"control_hz":control_hz,
                "action_time_semantics":"consecutive model actions at dataset fps",
                "simulation_seconds":float(adapter.data.time),
                "policy_calls":calls,"arm_delta_limit_rad_per_tick":.04,
                "hand_projection":"nearest real open/closed pose","robot_environment_collision":True,
                "target_initial":target_initial.tolist(),"target_final":adapter.data.xpos[target].tolist(),
                "arm_state_change_rad":float(np.linalg.norm(final["observation.state"][:6]-initial["observation.state"][:6])),
                "hand_state_change_rad":float(np.linalg.norm(final["observation.state"][6:]-initial["observation.state"][6:])),
                "not_a_success_rate_benchmark":True,
                "limitations":["Approximate camera/robot pose, objects, lighting and motor dynamics",
                               "Joint zero/sign not calibrated against real robot",
                               "Actions rate-limited; fingers projected to binary real gestures",
                               "Single short rollout; no visual classifier or recovery switching"]}
        (output/"report.json").write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2),flush=True)
        return report
    except Exception as error:
        (output/"failure.json").write_text(json.dumps({"error":repr(error),"ticks_completed":len(trace)},indent=2))
        raise
    finally:
        (output/"trace.json").write_text(json.dumps(trace,indent=2))
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        adapter.close()
        log.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,default=CHECKPOINT)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--ticks",type=int,default=20)
    parser.add_argument("--control-hz",type=int,default=30)
    args=parser.parse_args()
    checkpoint=args.checkpoint
    if (checkpoint/"pretrained_model").is_dir():
        checkpoint=checkpoint/"pretrained_model"
    output=args.output_dir or Path(tempfile.gettempdir())/f"torch_mujoco_rollout_{time.time_ns()}"
    print("Output:",output,flush=True)
    run(checkpoint,output,args.ticks,control_hz=args.control_hz)
