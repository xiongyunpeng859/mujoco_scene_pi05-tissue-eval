"""Isolated camera/illumination previews at fixed grasp physics and seed."""
import copy
import json
from pathlib import Path
import argparse
import subprocess
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from preview_wrapper_contact import ROOT, make_wrapper, summarize
import scene


def perturb_camera(cfg):
    top = cfg['cameras']['central']
    top['position'] = (np.array(top['position']) + [.015,-.01,.01]).tolist()
    axes = np.array(top['xyaxes']).reshape(2,3)
    top['xyaxes'] = (Rotation.from_euler('xyz',[1.5,-2,1],degrees=True).apply(axes)).ravel().tolist()
    wrist = cfg['wrist_camera']
    wrist['render_perturbation'] = {
        'position':[.003,-.002,.002],
        'xyaxes':Rotation.from_euler('xyz',[1,-1,0.5],degrees=True).apply(np.eye(3)[:2]).ravel().tolist(),
    }
    for camera, scale in ((top,1.03),(wrist,.98)):
        intr = camera['intrinsics']
        intr['fx'] *= scale
        intr['fy'] *= scale
        intr['cx'] += 4
        intr['cy'] -= 3


def perturb_light(cfg, cool=False):
    tint = np.array([.90,.96,1.07] if cool else [1.10,1.0,.86])
    strength = 1.12 if cool else .75
    for light in cfg['lighting']['lights']:
        for key in ('diffuse','ambient','specular'):
            light[key] = np.clip(np.array(light[key])*tint*strength,0,1).tolist()
        light['position'] = (np.array(light['position'])+[.15,-.12,0]).tolist()
        light['direction'] = [.10,.15,-1.0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    base = yaml.safe_load((ROOT/'configs/scene.yaml').read_text())
    manifest=[]
    for name in ('baseline','camera_only','warm_dim','combined_cool'):
        folder=out/name
        folder.mkdir()
        cfg=copy.deepcopy(base)
        if name in ('camera_only','combined_cool'):
            perturb_camera(cfg)
        if name in ('warm_dim','combined_cool'):
            perturb_light(cfg,cool=name=='combined_cool')
        if name=='combined_cool':
            for i,box in enumerate(cfg['boxes']):
                texture=folder/f'wrapper_{i}.png'
                make_wrapper(texture,20260920+i,[(160,105,35),(110,150,35),(145,65,130)][i])
                box['wrapper_visual']={'texture':str(texture),'specular':.45,'shininess':.45}
        config=folder/'config.yaml'
        config.write_text(yaml.safe_dump(cfg,sort_keys=False))
        xml=scene.build(config,folder/'compile_check.xml',with_hand=True)
        model=scene.mujoco.MjModel.from_xml_path(str(xml))
        if name=='baseline':
            baseline=model
        else:
            for field in ('body_mass','body_inertia','qpos0','flex_friction','flex_stiffness','actuator_gainprm'):
                np.testing.assert_array_equal(getattr(model,field),getattr(baseline,field))
        manifest.append({'name':name,'config':str(config)})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for profile in manifest:
        name=profile['name']; folder=out/name
        print('RUN',name,flush=True)
        with (folder/'run.log').open('w') as log:
            subprocess.run([sys.executable,'-u',str(ROOT/'reports/tools/scripted_pick_place.py'),
                            '--config',profile['config'],'--episodes','1','--seed','7','--speed','1.5',
                            '--save','--save-every','3','--output-dir',str(folder/'run')],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        for camera in ('top','wrist'):
            subprocess.run(['/usr/bin/ffmpeg','-v','error','-framerate','10',
                            '-pattern_type','glob','-i',str(folder/'run'/f'ep00_{camera}_*.png'),
                            '-c:v','libx264','-pix_fmt','yuv420p',str(folder/f'{camera}.mp4')],check=True)
        subprocess.run(['/usr/bin/ffmpeg','-v','error','-i',str(folder/'top.mp4'),'-i',str(folder/'wrist.mp4'),
                        '-filter_complex',f"hstack,drawtext=text='{name} - top / wrist':x=12:y=12:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.6",
                        '-c:v','libx264','-pix_fmt','yuv420p',str(folder/'dual.mp4')],check=True)
        result=json.loads((folder/'run/result.json').read_text())[0]
        print('DONE',name,'success=',result['success'],flush=True)
    summarize(out)


if __name__=='__main__':
    main()
