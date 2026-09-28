"""Combine completed searches, select stable low-motion plans, replay videos."""
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
    parser.add_argument('--searches',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=7)
    args=parser.parse_args()
    rows=[]
    for directory in args.searches: rows.extend(json.loads((directory/'summary.json').read_text()))
    selections=summarize_selection(rows)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'selection.json').write_text(json.dumps(selections,indent=2))
    choices=[g['selected'] for g in selections if g['seed']==args.seed and g['selected'] is not None]
    def render(choice):
        angle,grasp,seed=choice['object_deg'],choice['grasp_deg'],choice['seed']
        folder=args.output/f'object{angle:g}_grasp{grasp:g}'
        folder.mkdir()
        env=os.environ.copy();env.update(MUJOCO_GL='egl',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        with (folder/'run.log').open('w') as log:
            subprocess.run([sys.executable,'-u',str(ROOT/'reports/tools/scripted_pick_place.py'),
                '--episodes','1','--seed',str(seed),'--speed','1.5','--object-yaw-deg',str(angle),
                '--grasp-yaw-offset',str(grasp-angle-90),'--save','--save-every','1','--output-dir',str(folder)],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads((folder/'result.json').read_text())[0]
        if not result['success']: raise RuntimeError(f'Replay did not reproduce success: {folder}')
        for camera in ('top','wrist'):
            subprocess.run(['/usr/bin/ffmpeg','-v','error','-framerate','30','-pattern_type','glob',
                '-i',str(folder/f'ep00_{camera}_*.png'),'-c:v','libx264','-pix_fmt','yuv420p',str(folder/f'{camera}.mp4')],check=True)
        label=f'Object {angle:g}deg / grasp {grasp:g}deg - TOP / WRIST'
        subprocess.run(['/usr/bin/ffmpeg','-v','error','-i',str(folder/'top.mp4'),'-i',str(folder/'wrist.mp4'),
            '-filter_complex',f"hstack,drawtext=text='{label}':x=10:y=10:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.6",
            '-c:v','libx264','-pix_fmt','yuv420p',str(folder/'dual.mp4')],check=True)
        print('REPLAY_OK',folder,flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(render,choices))


if __name__=='__main__': main()
