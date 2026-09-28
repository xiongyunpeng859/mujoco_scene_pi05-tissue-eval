"""Deterministic per-round visual and contact sampling around calibrated scene."""
import copy
import colorsys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from preview_wrapper_contact import make_wrapper


def sample_config(source, directory, seed, yaw_range=25., randomize_contact=True):
    rng = np.random.default_rng(seed)
    config = copy.deepcopy(yaml.safe_load(Path(source).read_text()))
    # Centres within demonstrated reach, with a full circumscribed footprint
    # left of the tray (including yaw) and 1 cm clearance.
    region = config['box_randomization']['region_xy_cm_from_left_bottom']
    radius_cm = float(max(np.linalg.norm(b['size'][:2])*50 for b in config['boxes']))
    tray_edge = config['box_randomization']['tray_left_edge_cm']
    region['x'] = [max(region['x'][0],radius_cm+1), min(region['x'][1],tray_edge-radius_cm-1)]
    region['y'] = [max(region['y'][0],45.),min(region['y'][1],64.)]
    config['box_randomization']['yaw_jitter_deg'] = float(yaw_range)
    directory = Path(directory)
    near_real = bool(rng.random() < .3)
    scale = .3 if near_real else 1.
    camera_changes = {}
    for name, camera, distance, angle in (
        ('top', config['cameras']['central'], .015, 2.),
        ('left', config['wrist_camera'], .003, 1.5),
    ):
        delta = rng.uniform(-distance,distance,3)*scale
        degrees = rng.uniform(-angle,angle,3)*scale
        rotation = Rotation.from_euler('xyz',degrees,degrees=True)
        if name == 'top':
            camera['position'] = (np.array(camera['position'])+delta).tolist()
            camera['xyaxes'] = rotation.apply(np.array(camera['xyaxes']).reshape(2,3)).ravel().tolist()
        else:
            camera['render_perturbation'] = {'position':delta.tolist(),
                'xyaxes':rotation.apply(np.eye(3)[:2]).ravel().tolist()}
        focal = 1+rng.uniform(-.03,.03)*scale
        center = rng.uniform(-4,4,2)*scale
        intr = camera['intrinsics']
        intr['fx'] *= focal; intr['fy'] *= focal
        intr['cx'] += float(center[0]); intr['cy'] += float(center[1])
        camera_changes[name] = dict(translation_m=delta.tolist(),rotation_deg=degrees.tolist(),
                                    focal_scale=focal,principal_offset_px=center.tolist())
    strength = 1+rng.uniform(-.25,.2)*scale
    tint = rng.uniform(.9,1.1,3)
    for light in config['lighting']['lights']:
        for key in ('ambient','diffuse','specular'):
            light[key] = np.clip(np.array(light[key])*strength*tint,0,1).tolist()
        light['position'] = (np.array(light['position'])+rng.uniform(-.12,.12,3)*scale).tolist()
        light['direction'] = (np.array(light['direction'])+rng.uniform(-.08,.08,3)*scale).tolist()
    contacts=[]
    for i,box in enumerate(config['boxes']):
        # Moderate range first: the preview's mu=1.2/80g stress cases failed.
        sampled_mass = float(rng.uniform(.045,.060))
        sampled_young = float(rng.uniform(80000,120000))
        sampled_friction = [float(rng.uniform(3.2,4.5)),.1,.002]
        if randomize_contact:
            box['mass'] = sampled_mass
            box['young'] = sampled_young
            box['friction'] = sampled_friction
        color = tuple(int(x*255) for x in colorsys.hsv_to_rgb(rng.random(),rng.uniform(.35,.75),rng.uniform(.4,.8)))
        texture = directory/f'wrapper_{i}.png'
        make_wrapper(texture,int(rng.integers(2**31)),color)
        box['wrapper_visual'] = dict(texture=str(texture.resolve()),
            specular=float(rng.uniform(.15,.5)),shininess=float(rng.uniform(.15,.5)))
        contacts.append({k:box[k] for k in ('name','mass','young','friction','wrapper_visual')})
    config['sampled_domain'] = dict(seed=int(seed),near_real=near_real,cameras=camera_changes,
                                   lighting=config['lighting'],contacts=contacts,
                                   position_region_cm=region,yaw_jitter_deg=float(yaw_range),
                                   scope='constant within a three-pick round')
    config['sampled_domain']['contact_randomization'] = bool(randomize_contact)
    config['sampled_domain']['robot_dynamics_randomization'] = False
    path = directory/'randomized_config.yaml'
    path.write_text(yaml.safe_dump(config,sort_keys=False))
    return path
