"""Bounded independent full-physics grasp trials, preserving failed results."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from grasp_candidate_selection import summarize_selection

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--objects',type=float,nargs='+',default=[0,45,90])
    parser.add_argument('--grasps',type=float,nargs='+',default=[0,*range(30,181,15),210,240,270,300,330])
    parser.add_argument('--seeds',type=int,nargs='+',default=[7])
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    trials=[(o,g,s) for o in args.objects for g in args.grasps for s in args.seeds]
    def run(trial):
        angle,grasp,seed=trial
        folder=args.output/f'object{angle:g}_grasp{grasp:g}_seed{seed}'
        folder.mkdir()
        cmd=[sys.executable,'-u',str(ROOT/'reports/tools/scripted_pick_place.py'),
             '--episodes','1','--seed',str(seed),'--speed','1.5','--object-yaw-deg',str(angle),
             '--grasp-yaw-offset',str(grasp-angle-90),'--no-render','--output-dir',str(folder)]
        env=os.environ.copy();env.update(MUJOCO_GL='osmesa',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        with (folder/'run.log').open('w') as log:
            result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        row={'object_deg':angle,'grasp_deg':grasp,'seed':seed,'exit_code':result.returncode,'directory':str(folder)}
        if (folder/'result.json').exists(): row.update(json.loads((folder/'result.json').read_text())[0])
        print(json.dumps({k:row.get(k) for k in ('object_deg','grasp_deg','seed','success','continuous_carry','settled_in_tray','actual_joint_travel_total_rad')}),flush=True)
        return row
    rows=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(run,trials):
            rows.append(row)
            (args.output/'summary.json').write_text(json.dumps(rows,indent=2))
    (args.output/'selection.json').write_text(json.dumps(summarize_selection(rows),indent=2))


if __name__=='__main__': main()
