#!/usr/bin/env python3
"""Validate actual training-loader reads, episode boundaries, and encoded videos."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

root=Path(sys.argv[1]);expected=int(sys.argv[2])
data=LeRobotDataset('local/tissue_pick_place',root=root,video_backend='pyav',delta_timestamps={'action':[0.,1/30,2/30]})
frames=int(sys.argv[3]) if len(sys.argv)>3 else 450
assert data.num_episodes==expected and len(data)==expected*frames
info=json.loads((root/'meta/info.json').read_text());assert info['codebase_version']=='v3.0'
for eid in range(expected):
    # Every image stream's frame count/codec is checked, not just file existence.
    for cam in ['observation.images.top','observation.images.left']:
        path=root/'videos'/cam/f'chunk-{eid//1000:03d}'/f'file-{eid%1000:03d}.mp4'
        stream=json.loads(subprocess.check_output(['/usr/bin/ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=codec_name,nb_frames,r_frame_rate,width,height','-of','json',str(path)]))['streams'][0]
        assert int(stream['nb_frames'])==frames and stream['codec_name']=='h264' and stream['r_frame_rate']=='30/1'
        assert (stream['width'],stream['height'])==(640,480)
    meta=json.loads((root/f'meta/collection/episode-{eid:06d}.json').read_text())
    for index in [eid*frames,eid*frames+frames-1]:
        f=data[index]
        assert f['episode_index'].item()==eid and f['action'].shape==(3,16)
        assert f['observation.images.top'].shape==(3,480,640) and f['observation.images.left'].shape==(3,480,640)
        assert f['action_is_pad'].tolist()==([False,True,True] if index%frames==frames-1 else [False]*3)
        assert np.isfinite(f['action'].numpy()).all()
        if meta.get('return_home') and index%frames==frames-1:
            np.testing.assert_allclose(f['action'][0].numpy(),meta['home_state'],atol=1e-6)
            np.testing.assert_allclose(f['observation.state'].numpy(),meta['home_state'],atol=.03)
(root/'VALIDATION.json').write_text(json.dumps({'passed':True,'episodes':expected,'frames':len(data),
    'reader':'LeRobotDataset(video_backend=pyav)','checks':'all video headers; first/last frames, action padding and requested return-home pose of every episode'},indent=2))
print('VALIDATED',expected,'episodes',len(data),'frames',flush=True)
