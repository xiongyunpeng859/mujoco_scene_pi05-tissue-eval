"""Offline policy probe on recorded real observations; never commands hardware."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pyarrow.parquet as pq
from align_real_dataset import video_frame
import hand_control
from test_torch_rollout import PROJECT, ROOT, CHECKPOINT, read_message


def run(alignment, checkpoint, output):
    if output.exists():
        raise FileExistsError(output)
    source=json.loads((alignment/"report.json").read_text())
    root=Path(source["data_root"])
    rows=[]
    for path in sorted((root/"data").rglob("*.parquet")):
        rows.extend(pq.read_table(path,columns=["episode_index","frame_index","observation.state","action"]).to_pylist())
    lookup={(r["episode_index"],r["frame_index"]):r for r in rows}
    task=pq.read_table(root/"meta/tasks.parquet").to_pylist()[0]
    task=task.get("task",task.get("tasks"))
    if not isinstance(task,str):
        raise ValueError("Cannot determine original task text")
    output.mkdir(parents=True)
    environment=os.environ.copy()
    environment.update({"PYTHONNOUSERSITE":"1","HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1",
                        "PYTHONPATH":str(PROJECT/".local_deps/transformers_lerobot_openpi")+":"+str(PROJECT/"src"),
                        "TOKENIZERS_PARALLELISM":"false"})
    results=[]
    with (output/"worker.log").open("w") as log:
        process=subprocess.Popen(["/opt/miniconda3/envs/lerobot-pi05/bin/python","-u",str(ROOT/"torch_policy_worker.py"),
                                  "--checkpoint",str(checkpoint)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,env=environment)
        try:
            ready=read_message(process,"READY")
            print("Strict load:",ready["strict_load"],flush=True)
            for index,entry in enumerate(source["comparison_frames"]):
                row=lookup[(entry["episode"],entry["frame"])]
                state=np.zeros(32,dtype=np.float32)
                state[:16]=row["observation.state"]
                images=[]
                for key in ["observation.images.top","observation.images.left"]:
                    record=entry[key]
                    image,_,_=video_frame(Path(record["video"]),record["seconds"])
                    images.append(np.asarray(image))
                packet=output/"packet.npz"
                np.savez(packet,state=state,central=images[0],left_wrist=images[1])
                process.stdin.write(json.dumps({"id":index,"observation_path":str(packet),"task":task})+"\n")
                process.stdin.flush()
                result=read_message(process,"RESULT")
                actions=np.asarray(result.pop("actions"))
                np.save(output/f"actions_{index:03d}.npy",actions)
                poses=hand_control.poses()
                distances=np.stack([np.linalg.norm(actions[:,6:16]-poses[k],axis=1) for k in ["open","closed"]],axis=1)
                truth=min(poses,key=lambda k:np.linalg.norm(np.asarray(row["action"])[6:]-poses[k]))
                result.update({"episode":entry["episode"],"frame":entry["frame"],"recorded_hand_command":truth,
                               "first_predicted_hand":"closed" if distances[0].argmin()==1 else "open",
                               "predicted_closed_frames_in_chunk":int((distances.argmin(1)==1).sum()),
                               "first_action_arm_error_rad":float(np.linalg.norm(actions[0,:6]-np.asarray(row["action"])[:6]))})
                results.append(result)
                print(f"ep{entry['episode']} frame{entry['frame']}: GT {truth}, predicted {result['first_predicted_hand']}, closed/chunk {result['predicted_closed_frames_in_chunk']}",flush=True)
            report={"strict_load":ready["strict_load"],"task":task,"checkpoint":str(checkpoint),"results":results,
                    "not_a_success_rate_test":True,"limitations":["Sparse offline teacher-forced observations, not closed-loop robot control",
                    "Hand decoding uses nearest gesture; transition frames may be ambiguous",
                    "Original success-only source may not equal the exact mixed training dataset"]}
            (output/"report.json").write_text(json.dumps(report,indent=2))
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-dir",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path,default=CHECKPOINT)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    if (args.checkpoint/"pretrained_model").is_dir():
        args.checkpoint=args.checkpoint/"pretrained_model"
    run(args.alignment_dir,args.checkpoint,args.output_dir)
