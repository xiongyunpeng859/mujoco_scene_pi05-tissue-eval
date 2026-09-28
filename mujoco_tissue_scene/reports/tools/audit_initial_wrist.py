"""Read-only pose audit: render recorded initial state in an existing scene."""
import json
from pathlib import Path
import sys

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import action_layout
import dataset_io
from align_with_dataset import dataset_frame
from scripted_pick_place import SUCCESS


def main():
    out = ROOT / 'outputs/initial_wrist_audit_20260921'
    out.mkdir(exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(ROOT / 'outputs/grasp_direction_v7_wrist60_preview/scene.xml'))
    data = mujoco.MjData(model)
    episodes, _ = dataset_io.load(SUCCESS, fields=('observation.state',))
    state = episodes[0]['observation.state'][0]
    data.qpos[action_layout.joint_ids(model, mujoco)] = state
    mujoco.mj_forward(model, data)
    evidence = {'source': str(SUCCESS), 'episode': 0, 'frame': 0,
                'arm_degrees': np.degrees(state[:6]).tolist(), 'bodies': {}}
    for name in ['link3', 'link4', 'link5', 'link6', 'hand_L_palm', 'wrist_camera_mount']:
        bid = model.body(name).id
        evidence['bodies'][name] = {'position': data.xpos[bid].tolist(),
                                   'rotation': data.xmat[bid].reshape(3, 3).tolist()}
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    renderer = mujoco.Renderer(model, height=480, width=640)
    frames = []
    try:
        for label, azimuth, elevation in [('SIDE', 0, -15), ('OTHER SIDE', 180, -15),
                                           ('ABOVE', 90, -65), ('FRONT', 90, -15)]:
            camera = mujoco.MjvCamera()
            camera.lookat[:] = data.xpos[model.body('link6').id] + [0, .06, 0]
            camera.distance = .55
            camera.azimuth = azimuth
            camera.elevation = elevation
            renderer.update_scene(data, camera=camera, scene_option=opt)
            frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            cv2.putText(frame, label, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
            cv2.imwrite(str(out / (label.lower().replace(' ', '_') + '.jpg')), frame)
            frames.append(frame)
        cv2.imwrite(str(out / 'wrist_views.jpg'), np.vstack([np.hstack(frames[:2]), np.hstack(frames[2:])]))
        renderer.update_scene(data, camera='overview', scene_option=opt)
        cv2.imwrite(str(out / 'overview.jpg'), cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
        pilot = ROOT / 'outputs/visual_dr_planned_pilot_20260921/dataset'
        pilot_episodes, _ = dataset_io.load(pilot, fields=('observation.state',))
        pilot_start = pilot_episodes[3]['observation.state'][0]
        evidence['pilot_episode_003_initial_arm_degrees'] = np.degrees(pilot_start[:6]).tolist()
        data.qpos[action_layout.joint_ids(model, mujoco)] = pilot_start
        mujoco.mj_forward(model, data)
        camera.azimuth, camera.elevation = 90, -65
        renderer.update_scene(data, camera=camera, scene_option=opt)
        cv2.imwrite(str(out / 'pilot_003_initial_wrist.jpg'), cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
        comparison = np.hstack([dataset_frame(SUCCESS, 0, 0), dataset_frame(pilot, 3, 0)])
        cv2.putText(comparison, 'REAL ep0 frame0 | SIM DATA ep3 frame0 (different cameras)',
                    (5, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 2)
        cv2.imwrite(str(out / 'recorded_first_frames.jpg'), comparison)
    finally:
        renderer.close()
    (out / 'evidence.json').write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
