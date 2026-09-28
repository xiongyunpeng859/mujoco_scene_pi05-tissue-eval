"""Replay recorded real actions without fitting gains or overwriting simulated state.

Image-derived object poses are provisional, not ground-truth contact validation.
Flange references use the same URDF and cannot certify absolute robot geometry.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import dataset_io
from sim_env import TissueSceneEnv
from align_with_dataset import dataset_frame, episode_metadata, measure_bags
from scripted_pick_place import SUCCESS, rotation_distance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=SUCCESS)
    parser.add_argument('--episodes', type=int, nargs='+', default=[0, 53, 107])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    episodes, _ = dataset_io.load(args.dataset)
    info = json.loads((args.dataset/'meta/info.json').read_text())
    report = {'dataset': str(args.dataset), 'full_real_replay_verified': False,
              'limitations': ['Initial object poses estimated from top image, not ground truth',
                              'Same-URDF FK is not independent absolute end-effector measurement',
                              'These recorded episodes are not claimed to be held-out calibration data'],
              'episodes': []}
    for eid in args.episodes:
        folder = args.output / f'episode-{eid:03d}'
        env = TissueSceneEnv(render=True, dataset=args.dataset, output_dir=folder)
        mj, model, data = env.mujoco, env.model, env.data
        episode = episodes[eid]
        state, action, timestamps = (episode[k] for k in ('observation.state','action','timestamp'))
        if state.shape != action.shape or state.shape[1] != 16 or not np.all(np.diff(timestamps)>0):
            raise ValueError('Invalid real episode layout or timestamps')
        env.settle_seconds = 0
        env.reset(options={'state':state[0], 'randomize_objects':False})
        bags = measure_bags(dataset_frame(args.dataset, eid, 0), env.config)
        for index, box in enumerate(env.config['boxes']):
            if index < len(bags):
                cm = bags[index]['box_centre_cm']
                position = [cm[0]/100-env.config['table']['size'][0]/2,
                            cm[1]/100-env.config['table']['size'][1]/2,
                            env.config['table']['surface_z']+box['size'][2]/2+.001]
                env.move_object(box['name'], position, np.radians(bags[index]['yaw_deg']))
            else:
                env.move_object(box['name'], [-2-index,-2,.8])
        present = [b['name'] for b in env.config['boxes'][:len(bags)]]
        initial_z = {name:float(env.object_points(name).mean(0)[2]) for name in present}
        peaks = {name:0. for name in present}
        contact_frames = {name:0 for name in present}
        ref = mj.MjData(model)
        ref_positions, ref_rotations = [], []
        for q in state:
            ref.qpos[env.joint_adr] = q
            mj.mj_forward(model, ref)
            ref_positions.append(ref.xpos[env.flange_body].copy())
            ref_rotations.append(ref.xmat[env.flange_body].reshape(3,3).copy())
        captures = []
        meta = episode_metadata(args.dataset,eid)
        for key in ('observation.images.top','observation.images.left'):
            path=args.dataset/'videos'/key/f"chunk-{meta[f'videos/{key}/chunk_index']:03d}"/f"file-{meta[f'videos/{key}/file_index']:03d}.mp4"
            capture=cv2.VideoCapture(str(path))
            capture.set(cv2.CAP_PROP_POS_FRAMES,round(meta[f'videos/{key}/from_timestamp']*info['fps']))
            capture.read()  # The first output compares state[t+1] after action[t].
            captures.append(capture)
        encoder=subprocess.Popen(['/usr/bin/ffmpeg','-nostdin','-n','-v','error','-f','rawvideo',
            '-pix_fmt','bgr24','-s','1280x960','-r','10','-i','-','-c:v','libx264','-threads','2',
            '-pix_fmt','yuv420p','-movflags','+faststart',str(folder/'real_vs_sim.mp4')],stdin=subprocess.PIPE)
        achieved=[]; flange_errors=[]; orientation_errors=[]; clipped_frames=0; clock_errors=[]
        start_time=data.time
        try:
            for i in range(len(action)-1):
                target=np.clip(action[i],env.limits_lo,env.limits_hi)
                clipped_frames+=int(np.any(np.abs(target-action[i])>1e-9))
                data.ctrl[env.ctrl_ids]=target
                deadline=start_time+float(timestamps[i+1]-timestamps[0])
                while data.time < deadline-1e-8:
                    before=data.time; mj.mj_step(model,data)
                    if data.time <= before or not np.isfinite(data.qpos).all():
                        raise RuntimeError('Replay physics instability')
                mj.mj_forward(model,data)
                clock_errors.append(data.time-deadline)
                achieved.append(data.qpos[env.joint_adr].copy())
                flange_errors.append(float(np.linalg.norm(data.xpos[env.flange_body]-ref_positions[i+1])))
                orientation_errors.append(float(np.degrees(rotation_distance(
                    data.xmat[env.flange_body].reshape(3,3),ref_rotations[i+1]))))
                for name in present:
                    env.set_target(name)
                    peaks[name]=max(peaks[name],float(env.object_points(name).mean(0)[2])-initial_z[name])
                    contact_frames[name]+=int(env.target_hand_contacts()>0)
                real=[]
                for capture in captures:
                    ok,frame=capture.read()
                    if not ok:raise RuntimeError(f'Real video decode failed ep={eid} frame={i+1}')
                    real.append(frame)
                if i%3==0:
                    top=cv2.cvtColor(env.render('central'),cv2.COLOR_RGB2BGR)
                    wrist=cv2.cvtColor(env.render('left_wrist'),cv2.COLOR_RGB2BGR)
                    panel=np.vstack([np.hstack([real[0],top]),np.hstack([real[1],wrist])])
                    cv2.putText(panel,f'REAL | SIM - episode {eid} frame {i+1}; provisional object poses',
                                (8,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),2)
                    encoder.stdin.write(panel.tobytes())
                    if i==0:cv2.imwrite(str(folder/'initial_comparison.jpg'),panel)
                    last_panel=panel
            encoder.stdin.close()
            if encoder.wait()!=0:raise RuntimeError('Preview encoder failed')
            cv2.imwrite(str(folder/'final_comparison.jpg'),last_panel)
            error=np.asarray(achieved)-state[1:]
            placements={}
            for name in present:
                env.set_target(name)
                placements[name]=dict(peak_lift_m=peaks[name],hand_contact_frames=contact_frames[name],
                                      final_in_tray=env.success(),final_center=env.object_points(name).mean(0).tolist())
            row=dict(episode=eid,frames=len(action),replayed_intervals=len(achieved),
                     recorded_duration_s=float(timestamps[-1]-timestamps[0]),
                     simulated_duration_s=float(data.time-start_time),
                     maximum_clock_error_s=float(max(abs(np.array(clock_errors)))),
                     action_clipped_frames=clipped_frames,
                     joint_rms_rad=np.sqrt(np.mean(error**2,axis=0)).tolist(),
                     arm_rms_rad=float(np.sqrt(np.mean(error[:,:6]**2))),
                     hand_rms_rad=float(np.sqrt(np.mean(error[:,6:]**2))),
                     flange_same_urdf_rms_m=float(np.sqrt(np.mean(np.array(flange_errors)**2))),
                     flange_same_urdf_max_m=float(max(flange_errors)),
                     flange_orientation_rms_deg=float(np.sqrt(np.mean(np.array(orientation_errors)**2))),
                     initial_pose_estimates=bags,provisional_object_outcomes=placements)
            report['episodes'].append(row)
            (args.output/'report.json').write_text(json.dumps(report,indent=2))
            print('REAL_ACTION_REPLAY',json.dumps(row),flush=True)
        finally:
            for capture in captures:capture.release()
            if encoder.poll() is None:encoder.kill();encoder.wait()
            env.close()


if __name__=='__main__':main()
