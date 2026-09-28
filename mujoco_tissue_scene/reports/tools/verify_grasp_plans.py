"""Replay selected candidates with explicit obstacle-contact rejection."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

from grasp_candidate_selection import summarize_selection,select_candidate
ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--searches',nargs='+',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    rows=sum([json.loads((p/'summary.json').read_text()) for p in args.searches],[])
    groups=summarize_selection(rows)
    def verify(group):
        candidates=[r for r in rows if (r['object_deg'],r['seed'])==(group['object_deg'],group['seed']) and select_candidate([r])]
        candidates.sort(key=lambda r:r['actual_joint_travel_total_rad'])
        for old in candidates:
            angle,grasp,seed=old['object_deg'],old['grasp_deg'],old['seed']
            folder=args.output/f'object{angle:g}_grasp{grasp:g}_seed{seed}'
            folder.mkdir()
            env=os.environ.copy();env.update(MUJOCO_GL='osmesa',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
            with (folder/'run.log').open('w') as log:
                result=subprocess.run([sys.executable,'-u',str(ROOT/'reports/tools/scripted_pick_place.py'),
                    '--episodes','1','--seed',str(seed),'--speed','1.5','--object-yaw-deg',str(angle),
                    '--grasp-yaw-offset',str(grasp-angle-90),'--no-render','--output-dir',str(folder)],
                    cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            if not (folder/'result.json').exists():continue
            row=json.loads((folder/'result.json').read_text())[0]
            row.update(object_deg=angle,grasp_deg=grasp,seed=seed,exit_code=result.returncode,directory=str(folder))
            if select_candidate([row]) and row['obstacle_contact_frames']==0:
                print('VERIFIED',angle,seed,grasp,flush=True)
                return row
        print('NO_VERIFIED_PLAN',group['object_deg'],group['seed'],flush=True)
        return dict(object_deg=group['object_deg'],seed=group['seed'],success=False)
    with ThreadPoolExecutor(max_workers=3) as pool:
        verified=list(pool.map(verify,groups))
    (args.output/'summary.json').write_text(json.dumps(verified,indent=2))


if __name__=='__main__':main()
