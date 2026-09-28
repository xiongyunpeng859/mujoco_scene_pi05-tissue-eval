#!/usr/bin/env python3
"""Durable multi-process collector: accept complete 3-pick rounds, merge LeRobot v3."""
from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'reports/tools'))
from lerobot_writer import LerobotWriter, CAMERAS, atomic_json, statistics
from scripted_pick_place import episode_frames

GRASP_BLOCK_ORDER=('long_edges','diagonal','short_edges')


def grasp_block_quotas(episodes):
    rounds=math.ceil(episodes/3)
    base,remainder=divmod(rounds,len(GRASP_BLOCK_ORDER))
    return {family:base+(index<remainder) for index,family in enumerate(GRASP_BLOCK_ORDER)}


def completed_grasp_family(directory):
    try:
        families=json.loads((directory/'planning/plan.json').read_text())['families']
        return families[0] if len(families)==3 and len(set(families))==1 else None
    except (OSError,ValueError,KeyError,IndexError):
        return None


def complete_round(directory, planned=False):
    try:
        info=json.loads((directory/'dataset/meta/info.json').read_text())
        rows=json.loads((directory/'run/result.json').read_text())
        good=[r for r in rows if r['success']]
        # Any failed pick can move the object off its sampled table pose. Reject
        # that entire round rather than training on a recovery grasp from the bin.
        complete = info['total_episodes']==3 and len(rows)==len(good)==3 and [r['pick_in_round'] for r in good]==[0,1,2]
        if complete and planned:
            from replay_three_bag_plan import verify_replay
            plan=json.loads((directory/'planning/plan.json').read_text())
            if plan.get('complete') is not True or plan.get('max_pregrasp_palm_deg') != 90:
                return False
            if plan.get('max_wrist_deg') != 60:return False
            verify_replay(rows, plan['yaws'], 90, 60, plan.get('families'))
            if any(not r.get('grasp_plan') for r in rows):return False
            if [r.get('grasp_family') for r in rows] != plan.get('families'):return False
        return complete
    except (OSError,ValueError,KeyError,RuntimeError):return False


def merge_rounds(directories, destination, count, task=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    writer=LerobotWriter(destination, **({'task':task} if task is not None else {}))
    vectors={'action':[],'observation.state':[]}
    for directory in directories:
        source=directory/'dataset'
        episodes=pq.read_table(source/'meta/episodes/chunk-000/file-000.parquet').to_pylist()
        for old in episodes:
            eid=len(writer.episodes)
            if eid==count:break
            chunk,file=divmod(eid,1000)
            oldsuffix=f"chunk-{old['data/chunk_index']:03d}/file-{old['data/file_index']:03d}.parquet"
            table=pq.read_table(source/'data'/oldsuffix);n=len(table)
            for key,array in [('episode_index',np.full(n,eid,dtype=np.int64)),('index',np.arange(writer.total_frames,writer.total_frames+n,dtype=np.int64))]:
                table=table.set_column(table.schema.get_field_index(key),key,pa.array(array))
            path=writer.root/'data'/f'chunk-{chunk:03d}'/f'file-{file:03d}.parquet'
            path.parent.mkdir(parents=True,exist_ok=True);pq.write_table(table,path)
            row=copy.deepcopy(old)
            row.update({'episode_index':eid,'dataset_from_index':writer.total_frames,'dataset_to_index':writer.total_frames+n,
                        'data/chunk_index':chunk,'data/file_index':file})
            stats={}
            for k,v in old.items():
                if k.startswith('stats/'):
                    _,key,stat=k.split('/',2);stats.setdefault(key,{})[stat]=v
            for key in ['episode_index','index']:
                stats[key]=statistics(table[key].to_pylist())
                row.update({f'stats/{key}/{k}':v for k,v in stats[key].items()})
            for cam in CAMERAS:
                original=source/'videos'/cam/f"chunk-{old[f'videos/{cam}/chunk_index']:03d}"/f"file-{old[f'videos/{cam}/file_index']:03d}.mp4"
                target=writer.root/'videos'/cam/f'chunk-{chunk:03d}'/f'file-{file:03d}.mp4'
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():os.link(original,target)
                row[f'videos/{cam}/chunk_index']=chunk;row[f'videos/{cam}/file_index']=file
            meta=json.loads((source/f"meta/collection/episode-{old['episode_index']:06d}.json").read_text())
            meta['source_round']=directory.name
            atomic_json(writer.root/f'meta/collection/episode-{eid:06d}.json',meta)
            writer.episodes.append(row);writer.episode_stats.append(stats);writer.total_frames+=n
            for key in vectors:vectors[key].append(np.array(table[key].to_pylist(),dtype=np.float32))
        if len(writer.episodes)==count:break
    if len(writer.episodes)!=count:raise ValueError('insufficient accepted episodes')
    writer.write()
    stats=json.loads((writer.root/'meta/stats.json').read_text())
    for key,parts in vectors.items():stats[key]=statistics(np.concatenate(parts))
    atomic_json(writer.root/'meta/stats.json',stats)
    boundary=(int(count*.9)//3)*3
    atomic_json(writer.root/'meta/recommended_split.json',{'train':[0,boundary],'validation':[boundary,count],
        'note':'episode ranges, grouped by three-bag round to avoid leaking appearance/placement across splits'})
    return writer.root


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=1000);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=620260920);p.add_argument('--max-round-attempts',type=int,default=800)
    p.add_argument('--speed',type=float,default=1.5)
    p.add_argument('--full-domain',action='store_true')
    p.add_argument('--planned-grasps',action='store_true',help='offline multi-pose screening plus palm limit before collection')
    p.add_argument('--balanced-grasp-blocks',action='store_true',
                   help='collect contiguous equal-size long-edge, diagonal, and short-edge blocks')
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if not np.isfinite(args.speed) or not 0 < args.speed <= 1.5:p.error('--speed must be positive and at most 1.5')
    if args.planned_grasps and (not args.full_domain or args.speed != 1.5):
        p.error('--planned-grasps requires --full-domain and --speed 1.5')
    if args.balanced_grasp_blocks and not args.planned_grasps:
        p.error('--balanced-grasp-blocks requires --planned-grasps')
    if args.workers < 1 or args.episodes < 1:p.error('workers and episodes must be positive')
    lock=(out/'collector.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.nice(10)
    code=['scene.py','sim_env.py','domain_randomization.py','three_bag_round.py','action_layout.py',
          'reports/tools/scripted_pick_place.py','reports/tools/lerobot_writer.py',
          'reports/tools/collect_randomized_dataset.py','reports/tools/validate_training_dataset.py',
          'configs/scene.yaml','outputs/closure_library/library.json']
    if args.full_domain:
        code += ['reports/tools/full_domain_config.py','reports/tools/preview_wrapper_contact.py']
    if args.planned_grasps:
        code += ['reports/tools/collect_planned_round.py','reports/tools/plan_three_bag_round.py',
                 'reports/tools/grasp_candidate_selection.py','reports/tools/grasp_plan_contract.py',
                 'reports/tools/replay_three_bag_plan.py']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in code}
    if (out/'run_config.json').exists():
        previous=json.loads((out/'run_config.json').read_text())
        if any(previous.get(k,False if k in ('full_domain','planned_grasps','balanced_grasp_blocks') else None)!=getattr(args,k)
               for k in ['episodes','seed','speed','full_domain','planned_grasps','balanced_grasp_blocks']):
            raise RuntimeError('resume configuration differs from original collection')
        if any(hashes.get(f)!=h for f,h in previous['sha256'].items()):
            raise RuntimeError('resume source differs from original collection')
    snapshot=out/'source_snapshot'
    for f in code:
        dest=snapshot/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,dest)
    atomic_json(out/'run_config.json',{'episodes':args.episodes,'workers':args.workers,'seed':args.seed,'sha256':hashes,
        'speed':args.speed,'full_domain':args.full_domain,'planned_grasps':args.planned_grasps,
        'balanced_grasp_blocks':args.balanced_grasp_blocks,
        'grasp_block_quotas':grasp_block_quotas(args.episodes) if args.balanced_grasp_blocks else None,
        'yaw_range_deg':90 if args.planned_grasps else 25,
        'robot_dynamics_randomization':False,'contact_randomization':not args.planned_grasps,
        'max_pregrasp_palm_deg':90 if args.planned_grasps else None,
        'max_wrist_deg':60 if args.planned_grasps else None,
        'return_home':True,'episode_frames':episode_frames(args.speed),
        'bags_per_round':3,'appearance_changes':'between_rounds','clear_bin':'between_episodes','release':'direct_above_tray'})
    rounds=out/'rounds';rounds.mkdir(exist_ok=True)
    all_dirs=sorted(rounds.glob('round-*'))
    completed=[d for d in all_dirs if complete_round(d,args.planned_grasps)]
    next_id=max([int(d.name.split('-')[-1]) for d in all_dirs],default=-1)+1
    target_rounds=math.ceil(args.episodes/3);active={};started=time.time();last_print=0
    family_quotas=grasp_block_quotas(args.episodes) if args.balanced_grasp_blocks else None
    env=os.environ.copy();env.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',LP_NUM_THREADS='1')
    try:
        while len(completed)<target_rounds or active:
            for rid,(process,directory,log,family) in list(active.items()):
                if process.poll() is None:continue
                log.close();del active[rid]
                if process.returncode==0 and complete_round(directory,args.planned_grasps):completed.append(directory)
                else:print(f'rejected round {rid}: exit {process.returncode}',flush=True)
            while len(active)<args.workers and len(completed)+len(active)<target_rounds:
                if next_id>=args.max_round_attempts:raise RuntimeError('maximum round attempts reached')
                if shutil.disk_usage(out).free < 20*1024**3:raise RuntimeError('less than 20 GiB free disk space')
                if any(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()!=h for f,h in hashes.items()):
                    raise RuntimeError('collector source changed; stopping to avoid mixed implementations')
                rid=next_id;next_id+=1;directory=rounds/f'round-{rid:05d}';directory.mkdir()
                seed=(args.seed+rid*9973)%(2**31)
                family=None
                if family_quotas:
                    completed_counts={name:sum(completed_grasp_family(d)==name for d in completed)
                                      for name in GRASP_BLOCK_ORDER}
                    active_counts={name:sum(item[3]==name for item in active.values())
                                   for name in GRASP_BLOCK_ORDER}
                    family=next(name for name in GRASP_BLOCK_ORDER
                                if completed_counts[name]+active_counts[name] < family_quotas[name])
                config_path=snapshot/'configs/scene.yaml'
                if args.full_domain:
                    from full_domain_config import sample_config
                    config_path=sample_config(config_path,directory,seed,
                                              yaw_range=90 if args.planned_grasps else 25,
                                              randomize_contact=not args.planned_grasps)
                cmd=[sys.executable,'-u',str(ROOT/'reports/tools/scripted_pick_place.py'),'--episodes','3','--successes','3',
                     '--seed',str(seed),'--bags-per-round','3','--randomize-appearance',
                     '--speed',str(args.speed),
                     '--config',str(config_path),
                     '--collect',str(directory/'dataset'),'--output-dir',str(directory/'run')]
                if args.planned_grasps:
                    cmd=[sys.executable,'-u',str(ROOT/'reports/tools/collect_planned_round.py'),
                         '--config',str(config_path),'--seed',str(seed),'--directory',str(directory)]
                    if family:
                        cmd += ['--grasp-family',family]
                log=(directory/'run.log').open('w')
                active[rid]=(subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                                             start_new_session=True),directory,log,family)
            status={'status':'collecting','target_episodes':args.episodes,'complete_rounds':len(completed),
                    'accepted_episodes_available':3*len(completed),'active_rounds':list(active),
                    'attempted_rounds':next_id,'elapsed_seconds':round(time.time()-started),'pid':os.getpid()}
            atomic_json(out/'progress.json',status)
            if time.time()-last_print>=30:print(json.dumps(status),flush=True);last_print=time.time()
            if active:time.sleep(2)
        atomic_json(out/'progress.json',{'status':'merging','complete_rounds':len(completed),'target_episodes':args.episodes})
        destination=out/'dataset'
        if not (destination/'MERGE_COMPLETE.json').exists():
            if destination.exists():
                destination.rename(out/f'incomplete-merge-{time.time_ns()}')
            merge_rounds(sorted(completed),destination,args.episodes)
            atomic_json(destination/'MERGE_COMPLETE.json',{'episodes':args.episodes})
        check_env=os.environ.copy();check_env['PYTHONNOUSERSITE']='1'
        subprocess.run(['/opt/miniconda3/envs/lerobot-pi05/bin/python',str(ROOT/'reports/tools/validate_training_dataset.py'),str(destination),str(args.episodes),str(episode_frames(args.speed))],env=check_env,check=True)
        atomic_json(out/'progress.json',{'status':'complete','episodes':args.episodes,'frames':args.episodes*episode_frames(args.speed),
            'complete_rounds':len(completed),'dataset':str(destination),'elapsed_seconds':round(time.time()-started)})
    except BaseException as exc:
        for process,_,log,_ in active.values():
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
        for process,_,log,_ in active.values():process.wait();log.close()
        atomic_json(out/'progress.json',{'status':'interrupted','error':str(exc),'complete_rounds':len(completed),'next_round':next_id})
        raise

if __name__=='__main__':main()
