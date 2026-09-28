"""Closed-loop JAX Pi0.5 rollout in MuJoCo; no robot hardware."""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import subprocess
import yaml

_REPO_ROOT=Path(__file__).resolve().parents[1]
ROOT=_REPO_ROOT/'mujoco_tissue_scene' if (_REPO_ROOT/'mujoco_tissue_scene').is_dir() else Path(__file__).resolve().parent
PROJECT=Path(__import__('os').environ.get('PI05_PROJECT_ROOT', str(Path(__file__).resolve().parents[1])))
CHECKPOINT=Path('/home/fmc3-6/workspace/shared/new_program_qiuzhi/output/sim_to_real_bs8_chunked_20260923/sim_model/20000')

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,default=CHECKPOINT)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--ticks',type=int,default=375)
    p.add_argument('--rigid', action='store_true')
    p.add_argument('--known-round',type=Path)
    p.add_argument('--replay',action='store_true')
    p.add_argument('--execute-chunk',type=int,default=10)
    p.add_argument('--paired-audit',action='store_true')
    p.add_argument('--nominal-cameras',action='store_true')
    p.add_argument('--teacher-prefix',type=int,default=0,help='diagnostic only: execute recorded actions for first N frames')
    p.add_argument('--seed',type=int,default=20260923);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    os.environ.update(MUJOCO_GL='egl',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled')
    import mujoco
    from sim_env import TissueSceneEnv
    sys.path.insert(0,str(ROOT/'reports/tools'))
    from scripted_pick_place import SUCCESS
    from three_bag_round import ThreeBagRound
    from domain_randomization import AppearanceRandomizer
    from full_domain_config import sample_config
    cfg=sample_config(ROOT/'configs/scene.yaml',a.output_dir,a.seed,yaw_range=90.,randomize_contact=False)
    cfg_data=yaml.safe_load(Path(cfg).read_text())
    if a.known_round:
        cfg_data=yaml.safe_load((a.known_round/'run/scene.yaml').read_text())
        for box in cfg_data['boxes']:
            if box.get('wrapper_visual'):
                box['wrapper_visual']['texture']=str((a.known_round/Path(box['wrapper_visual']['texture']).name).resolve())
    for box in cfg_data['boxes']:
        box['soft_body']=not a.rigid
    if a.nominal_cameras:
        nominal=yaml.safe_load((ROOT/'configs/scene.yaml').read_text())
        cfg_data['cameras']=nominal['cameras']
        cfg_data['wrist_camera']=nominal['wrist_camera']
    cfg_data['domain_randomization']={'enabled':True}
    rigid_cfg=a.output_dir/'rigid_eval.yaml'
    rigid_cfg.write_text(yaml.safe_dump(cfg_data,sort_keys=False));cfg=rigid_cfg
    env=TissueSceneEnv(config_path=cfg,dataset=SUCCESS,render=True,seed=a.seed,output_dir=a.output_dir)
    if a.known_round:
        env.settle_seconds=0.0
    home=env.episode_start_states()[0]
    region=env.config['box_randomization']['region_xy_cm_from_left_bottom']
    manager=ThreeBagRound(env,np.random.default_rng(a.seed),region,90.,home,AppearanceRandomizer(env))
    if not a.known_round:
        _,_,_,info=manager.begin_pick();obs=env.observation()
    if a.known_round:
        import pyarrow.parquet as pq
        meta=json.loads((a.known_round/'dataset/meta/collection/episode-000000.json').read_text())
        recorded=pq.read_table(a.known_round/'dataset/data/chunk-000/file-000.parquet').to_pydict()
        expert=np.array(recorded['action'])
        env.reset(options={'state':np.asarray(meta['home_state']), 'randomize_objects':False})
        env.data.qpos[env.joint_adr]=np.asarray(meta['home_state'])
        env.data.qvel[:]=0; mujoco.mj_forward(env.model,env.data)
        for spawn in meta['spawn']:
            box=next(b for b in env.config['boxes'] if b['name']==spawn['name'])
            env.move_object(spawn['name'],[spawn['xy_cm'][0]/100-env.config['table']['size'][0]/2,
                spawn['xy_cm'][1]/100-env.config['table']['size'][1]/2,
                env.config['table']['surface_z']+box['size'][2]/2+.002],spawn['yaw'])
        AppearanceRandomizer(env).apply(meta['appearance']['seed'])
        for _ in range(round(.25/env.model.opt.timestep)):mujoco.mj_step(env.model,env.data)
        env._time_target=env.data.time
        env.data.qpos[env.joint_adr]=np.asarray(recorded['observation.state'][0])
        env.data.qvel[:]=0
        env.data.ctrl[env.ctrl_ids]=np.asarray(recorded['observation.state'][0])
        mujoco.mj_forward(env.model,env.data)
        env.set_target(meta['target']);obs=env.observation();info=meta
        print('initial state max error',float(np.abs(obs['observation.state']-recorded['observation.state'][0]).max()),flush=True)
    (a.output_dir/'initial.json').write_text(json.dumps(info,indent=2))
    np.savez(a.output_dir/'initial_observation.npz',state=obs['observation.state'],
             top=obs['observation.images.top'],left=obs['observation.images.left'])
    worker_env=dict(os.environ,PYTHONPATH=os.environ.get('OPENPI_ROOT', '/workspace/shared/openpi_jax')+'/src:'+str(PROJECT/'pi05_scripts'),
                    XLA_PYTHON_CLIENT_PREALLOCATE='false',JAX_PLATFORMS='cuda',POLICY_TEST_SEED=str(a.seed))
    worker=subprocess.Popen(['/opt/miniconda3/envs/openpi-jax-o10/bin/python','-u',str(Path(__file__).resolve()),
          '--policy-worker',str(a.checkpoint)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
          stderr=(a.output_dir/'policy.log').open('w'),text=True,env=worker_env)
    def receive():
        while True:
            line=worker.stdout.readline()
            if not line:raise RuntimeError('Policy worker exited; see policy.log')
            if line.startswith('PACKET '):return json.loads(line[7:])
    receive()
    # Same canonical keys as the deployment server; no scripted action is used.
    def infer(o):
        packet=a.output_dir/'observation.npz'
        np.savez(packet,state=o['observation.state'],top=o['observation.images.top'],left=o['observation.images.left'])
        worker.stdin.write(str(packet.resolve())+'\n');worker.stdin.flush()
        return np.asarray(receive()['actions'])
    chunk=np.zeros((0,16),np.float32);cursor=0;trace=[];frames=[]
    initial_centers={b['name']:env.object_points(b['name']).mean(0).copy() for b in env.config['boxes']}
    pairs=[]
    if a.paired_audit:
        import cv2
        videos={}
        for cam in ('top','left'):
            cap=cv2.VideoCapture(str(a.known_round/f'dataset/videos/observation.images.{cam}/chunk-000/file-000.mp4'))
            frames_in=[]
            while True:
                ok,frame=cap.read()
                if not ok:break
                frames_in.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            videos[cam]=frames_in;cap.release()
    for tick in range(a.ticks):
        if a.paired_audit and tick in (0,50,100,150,250,350):
            raw={'observation.state':np.asarray(recorded['observation.state'][tick],np.float32),
                 'observation.images.top':videos['top'][tick], 'observation.images.left':videos['left'][tick]}
            same_state=dict(obs);same_state['observation.state']=raw['observation.state']
            pred_raw=infer(raw);pred_sim=infer(same_state)
            target=expert[np.minimum(np.arange(tick,tick+50),len(expert)-1)]
            pairs.append(dict(frame=tick,raw_action_mae=float(np.abs(pred_raw-target).mean()),
                sim_image_action_mae=float(np.abs(pred_sim-target).mean()),
                state_max_error=float(np.abs(obs['observation.state']-raw['observation.state']).max()),
                image_mae={c:float(np.abs(raw['observation.images.'+c].astype(float)-obs['observation.images.'+c]).mean()) for c in ('top','left')}))
            for c in ('top','left'):
                cv2.imwrite(str(a.output_dir/f'compare_{tick}_{c}.png'),cv2.cvtColor(np.concatenate([raw['observation.images.'+c],obs['observation.images.'+c]],axis=1),cv2.COLOR_RGB2BGR))
        if tick==a.teacher_prefix or cursor>=min(a.execute_chunk,len(chunk)):
            chunk=infer(obs);cursor=0
            print('inference tick',tick,flush=True)
            if chunk.shape!=(50,16) or not np.isfinite(chunk).all():raise RuntimeError(f'bad action {chunk.shape}')
        action=np.asarray(chunk[cursor],np.float64);cursor+=1
        if a.replay or tick<a.teacher_prefix:action=expert[tick]
        obs,_,_,_,step=env.step(action)
        trace.append({'tick':tick,'time':step['time'],'success':step['success'],'tracking_error':step['tracking_error'],
                      'action':action.tolist(), 'centers':{b['name']:env.object_points(b['name']).mean(0).tolist() for b in env.config['boxes']},
                      'target_contacts':env.target_hand_contacts(),'qpos':obs['observation.state'].tolist()})
        if tick%3==0:frames.append(env.render('central').copy())
    final=trace[-1]
    import cv2
    if frames:
        h,w=frames[0].shape[:2]; vw=cv2.VideoWriter(str(a.output_dir/'rollout.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(w,h))
        for f in frames:vw.write(cv2.cvtColor(f,cv2.COLOR_RGB2BGR))
        vw.release()
    result={'checkpoint':str(a.checkpoint),'ticks':a.ticks,'seed':a.seed,'policy':'jax_pi05_sim494',
      'known_round':str(a.known_round),'initial_joint_state_exact':bool(a.known_round),
      'teacher_prefix':a.teacher_prefix,
      'nominal_cameras':a.nominal_cameras,
      'peak_lift_m':{name:float(max(r['centers'][name][2]-center[2] for r in trace)) for name,center in initial_centers.items()},
      'object_model':'rigid' if a.rigid else 'soft','execute_chunk':a.execute_chunk,'control_hz':30,
      'model_generated_actions':not a.replay and a.teacher_prefix==0,
      'control_mode':'replay' if a.replay else 'teacher_prefix_then_policy' if a.teacher_prefix else 'policy_only',
      'final_success':bool(final['success']),'max_target_contacts':max(x['target_contacts'] for x in trace),
      'max_tracking_error_rad':float(max(np.max(np.abs(x['tracking_error'])) for x in trace)),
      'simulation_seconds':float(final['time']),'note':'single closed-loop rollout; not a statistical success rate'}
    (a.output_dir/'trace.json').write_text(json.dumps(trace));(a.output_dir/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    (a.output_dir/'paired.json').write_text(json.dumps(pairs,indent=2))
    env.close()
    worker.stdin.close();worker.wait(timeout=30)

def policy_worker(checkpoint):
    sys.path.insert(0,str(PROJECT/'scripts'))
    import train_o10_sim494_chunked as pipeline
    from openpi.policies.policy_config import create_trained_policy
    policy=create_trained_policy(pipeline.make_config(),checkpoint)
    rng=np.random.default_rng(int(os.environ.get('POLICY_TEST_SEED','0')))
    print('PACKET '+json.dumps({'ready':True}),flush=True)
    for line in sys.stdin:
        data=np.load(line.strip())
        result=policy.infer({'observation/state':data['state'],'observation/image':data['top'],
             'observation/wrist_image':data['left'],'prompt':'拿起纸巾包，放进右侧的绿色盒子里。'},noise=rng.normal(size=(50,32)).astype(np.float32))
        print('PACKET '+json.dumps({'actions':np.asarray(result['actions']).tolist()}),flush=True)

if __name__=='__main__':
    if '--policy-worker' in sys.argv:policy_worker(sys.argv[-1])
    else:main()
