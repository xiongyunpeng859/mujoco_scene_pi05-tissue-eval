"""Small isolated visual/contact ablation; never writes training datasets."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import scene


def make_wrapper(path: Path, seed: int, color: tuple):
    rng = np.random.default_rng(seed)
    size = 512
    y, x = np.mgrid[:size, :size]
    # Printed film: branding panel, sealed edges and visual crease shading.
    base = np.full((size, size, 3), color, dtype=np.float32)
    shade = 6*np.sin(x*.12 + 2*np.sin(y*.028)) + rng.normal(0, 1.5, (size,size))
    base = np.uint8(np.clip(base + shade[...,None], 0, 255))
    cv2.rectangle(base, (65,95), (450,375), (240,240,229), -1)
    cv2.ellipse(base, (256,130), (95,18), 0, 0, 360, (160,170,163), 3)
    cv2.putText(base, 'SOFT', (115,235), cv2.FONT_HERSHEY_SIMPLEX, 2.2, color, 6)
    cv2.putText(base, 'TISSUE', (137,295), cv2.FONT_HERSHEY_SIMPLEX, 1.25, color, 3)
    cv2.putText(base, '3 PLY  /  100', (151,340), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
    for edge in (12, 21, 491, 500):
        cv2.line(base, (edge,0), (edge,511), (200,210,200), 2)
    for i in range(60):
        xx = 80+i*6
        cv2.line(base, (xx,420), (xx,463), (35,45,40), 1 + i%3)
    cv2.imwrite(str(path), base)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    if args.summarize_only:
        summarize(out)
        return
    out.mkdir(parents=True, exist_ok=False)
    original = yaml.safe_load((ROOT/'configs/scene.yaml').read_text())
    profiles = [
        ('baseline', False, 4.0, 100000., .05),
        ('printed', True, 4.0, 100000., .05),
        ('slippery', True, 1.2, 100000., .05),
        ('soft_heavy', True, 4.0, 50000., .08),
    ]
    manifest = []
    for name, printed, friction, young, mass in profiles:
        folder = out/name
        folder.mkdir()
        cfg = copy.deepcopy(original)
        for i, box in enumerate(cfg['boxes']):
            box['mass'], box['young'] = mass, young
            box['friction'] = [friction,.1,.002]
            if printed:
                texture = folder/f'wrapper_{i}.png'
                make_wrapper(texture, 20260920+i, [(160,105,35),(110,150,35),(145,65,130)][i])
                box['wrapper_visual'] = dict(texture=str(texture), specular=.45, shininess=.45)
        path = folder/'config.yaml'
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        # Compile upfront: reject unsupported flex/material settings before runs.
        xml = scene.build(path, folder/'compile_check.xml', with_hand=True)
        model = scene.mujoco.MjModel.from_xml_path(str(xml))
        assert not model.flex_rigid.any()
        if printed:
            assert (model.flex_matid >= 0).all()
        manifest.append(dict(name=name, printed=printed, friction=friction,
                             young_pa=young, mass_kg=mass, config=str(path)))
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    if args.prepare_only:
        return
    for profile in manifest:
        folder = out/profile['name']
        command = [sys.executable, '-u', str(ROOT/'reports/tools/scripted_pick_place.py'),
                   '--config', profile['config'], '--episodes','1', '--seed','7',
                   '--speed','1.5','--save','--output-dir',str(folder/'run')]
        print('RUN', profile['name'], flush=True)
        with (folder/'run.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Preview failed: {folder/'run.log'}")
        for camera in ('top','wrist'):
            subprocess.run(['/usr/bin/ffmpeg','-v','error','-framerate','2.5',
                            '-pattern_type','glob','-i',str(folder/'run'/f'ep00_{camera}_*.png'),
                            '-c:v','libx264','-pix_fmt','yuv420p',str(folder/f'{camera}.mp4')],check=True)
        subprocess.run(['/usr/bin/ffmpeg','-v','error','-i',str(folder/'top.mp4'),
                        '-i',str(folder/'wrist.mp4'),'-filter_complex','hstack',
                        '-c:v','libx264','-pix_fmt','yuv420p',str(folder/'dual.mp4')], check=True)
        print('DONE', profile['name'], (folder/'run/result.json').read_text(), flush=True)
    summarize(out)


def summarize(out):
    manifest = json.loads((out/'manifest.json').read_text())
    rows, results = [], []
    for profile in manifest:
        name = profile['name']
        result = json.loads((out/name/'run/result.json').read_text())[0]
        results.append({**profile, **{k: result[k] for k in
                       ('success','peak_lift','continuous_carry','settled_in_tray','home_ok')}})
        panels = []
        for frame in (0,108,180,300):
            img = cv2.imread(str(out/name/'run'/f'ep00_top_{frame:03d}.png'))
            img = cv2.resize(img,(320,240))
            cv2.putText(img, f'{name}  t={frame/30:.1f}s', (8,20),
                        cv2.FONT_HERSHEY_SIMPLEX,.48,(0,0,0),3)
            cv2.putText(img, f'{name}  t={frame/30:.1f}s', (8,20),
                        cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
            panels.append(img)
        rows.append(np.concatenate(panels,axis=1))
    cv2.imwrite(str(out/'comparison.jpg'), np.concatenate(rows,axis=0))
    (out/'summary.json').write_text(json.dumps(results,indent=2))
    inputs, filters = [], []
    for i,p in enumerate(manifest):
        inputs += ['-i',str(out/p['name']/'top.mp4')]
        filters.append(f"[{i}:v]drawtext=text='{p['name']}':x=12:y=12:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.65[v{i}]")
    filters.append('[v0][v1]hstack[top];[v2][v3]hstack[bottom];[top][bottom]vstack[out]')
    subprocess.run(['/usr/bin/ffmpeg','-v','error',*inputs,'-filter_complex',';'.join(filters),
                    '-map','[out]','-c:v','libx264','-pix_fmt','yuv420p',str(out/'comparison.mp4')],check=True)


if __name__ == '__main__':
    main()
