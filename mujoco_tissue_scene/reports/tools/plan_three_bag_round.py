"""Greedy stable-motion teacher: preserve the physical three-bag round state.

Each candidate replays the selected prefix from the same seed before testing
the next grasp. No object teleport or reset occurs between picks beyond the
existing round lifecycle. Not a global optimization over all 17**3 plans.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
import math
import os
import random
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import yaml

from grasp_candidate_selection import (GRASP_FAMILIES, grasp_family,
                                       relative_grasp_angle_deg, select_candidate)
from grasp_plan_contract import source_hashes, digest
ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--yaw-range',type=float,default=None,help='optional symmetric spawn yaw range in degrees')
    parser.add_argument('--grasps',type=float,nargs='+',default=list(range(0,360,15)))
    parser.add_argument('--max-pregrasp-palm-deg',type=float,default=90.,
                        help='maximum actual palm orientation change from each pick start; no automatic relaxation')
    parser.add_argument('--compact',action='store_true',help='discard only temporary rebuilt simulation assets; retain logs and result JSON')
    parser.add_argument('--max-wrist-deg',type=float,default=60.)
    parser.add_argument('--family',choices=GRASP_FAMILIES,default=None,
                        help='use one object-relative grasp family for all three picks')
    args=parser.parse_args()
    if not math.isfinite(args.max_pregrasp_palm_deg) or not 0 < args.max_pregrasp_palm_deg <= 180:
        parser.error('--max-pregrasp-palm-deg must be in (0,180]')
    if not math.isfinite(args.max_wrist_deg) or not 1 < args.max_wrist_deg <= 180:
        parser.error('--max-wrist-deg must be in (1,180]')
    hashes = source_hashes()
    args.output.mkdir(parents=True,exist_ok=False)
    if args.yaw_range is not None:
        config=yaml.safe_load(args.config.read_text())
        config['box_randomization']['yaw_jitter_deg']=args.yaw_range
        config.setdefault('sampled_domain',{})['yaw_jitter_deg']=args.yaw_range
        args.config=args.output/'config.yaml'
        args.config.write_text(yaml.safe_dump(config,sort_keys=False))
    chosen=[]
    chosen_relative_angles=[]
    family_order = [args.family] * 3
    chosen_families=[]
    for pick in range(3):
        desired_family = family_order[pick]
        def trial(angle):
            folder=args.output/f'pick{pick}_yaw{angle:g}';folder.mkdir()
            plan=chosen+[angle]
            workspace = tempfile.TemporaryDirectory(prefix='candidate-',dir=folder) if args.compact else nullcontext(str(folder))
            with workspace as scratch:
                cmd=[sys.executable,'-u',str(ROOT/'reports/tools/scripted_pick_place.py'),
                     '--config',str(args.config.resolve()),'--seed',str(args.seed),'--speed','1.5',
                     '--bags-per-round','3','--randomize-appearance','--episodes',str(pick+1),
                     '--grasp-yaw-plan',*map(str,plan),'--max-wrist-deg',str(args.max_wrist_deg),
                     '--no-render','--output-dir',str(scratch)]
                env=os.environ.copy();env.update(MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
                with (folder/'run.log').open('w') as log:
                    result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                if args.compact:
                    for record in Path(scratch).glob('*.json'):
                        shutil.copy2(record,folder/record.name)
            path=folder/'result.json'
            if result.returncode or not path.exists(): return None
            rows=json.loads(path.read_text())
            if len(rows)!=pick+1 or not all(r['success'] for r in rows): return None
            for prior_index, (prior, yaw) in enumerate(zip(rows, plan)):
                candidate = prior | dict(exit_code=0,grasp_deg=yaw)
                if select_candidate([candidate], args.max_pregrasp_palm_deg,
                                    args.max_wrist_deg, family_order[prior_index]) is None:
                    return None
            row=rows[-1]|dict(exit_code=result.returncode,grasp_deg=angle,directory=str(folder))
            return row if select_candidate([row]) and row.get('obstacle_contact_frames')==0 else None
        # Evaluate three at a time; do not spend simulations proving a global
        # minimum after a fully validated grasp has already been found.
        # Order is only a search heuristic; actual wrist limits remain mandatory.
        preferred = [90,75,105,60,120,45,135,30,150,165,180,0,210,240,270,300,330]
        angles = sorted(dict.fromkeys(args.grasps), key=lambda a:
                        preferred.index(a) if a in preferred else len(preferred))
        if args.family is None:
            # Vary search order without requiring any family to be feasible.
            # Every winner still passes the same physical and motion checks.
            random.Random(args.seed * 3 + pick).shuffle(angles)
        rows=[]
        remaining=list(angles)
        with ThreadPoolExecutor(max_workers=3) as pool:
            while remaining:
                batch=remaining[:3]; del remaining[:3]
                rows.extend(pool.map(trial,batch))
                if select_candidate([r for r in rows if r], args.max_pregrasp_palm_deg,
                                    args.max_wrist_deg, desired_family) is not None:
                    break
                observed=next((r for r in rows if r),None)
                if observed is not None and desired_family is not None:
                    target={'short_edges':0.,'diagonal':45.,'long_edges':90.}[desired_family]
                    remaining.sort(key=lambda angle:abs(relative_grasp_angle_deg(angle,observed['yaw_deg'])-target))
        (args.output/f'pick{pick}_candidates.json').write_text(json.dumps(rows,indent=2))
        winner=select_candidate([r for r in rows if r], args.max_pregrasp_palm_deg,
                                args.max_wrist_deg, desired_family)
        if winner is None:
            raise RuntimeError(f'No stable candidate at pick {pick}; prefix={chosen}')
        if source_hashes() != hashes:
            raise RuntimeError('Planner source changed during search; refusing mixed plan')
        chosen.append(winner['grasp_deg'])
        chosen_families.append(grasp_family(winner['grasp_deg'],winner['yaw_deg']))
        chosen_relative_angles.append(relative_grasp_angle_deg(winner['grasp_deg'],winner['yaw_deg']))
        (args.output/'plan.json').write_text(json.dumps(dict(seed=args.seed,config=str(args.config.resolve()),
            complete=pick==2,yaws=chosen,families=chosen_families,
            relative_angles_deg=chosen_relative_angles,
            max_pregrasp_palm_deg=args.max_pregrasp_palm_deg,
            max_wrist_deg=args.max_wrist_deg,
            speed=1.5, config_sha256=digest(args.config), source_sha256=hashes,
            selection='minimum joint travel in first feasible candidate batch; seeded diverse search; optional explicit family filter; greedy prefix',
            tested_candidates=len(rows)),indent=2))
        print('SELECTED',pick,chosen,'cost',winner['actual_joint_travel_total_rad'],flush=True)


if __name__=='__main__': main()
