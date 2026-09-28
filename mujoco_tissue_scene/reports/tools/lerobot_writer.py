#!/usr/bin/env python3
"""Incremental LeRobot v3 writer; completed episodes never retain image frames.

One independently decodable video and Parquet file per episode. Metadata is
checkpointed after each completed episode. Existing datasets are never overwritten.
"""
from __future__ import annotations
import json
import subprocess
from pathlib import Path
import numpy as np

CAMERAS = ('observation.images.top', 'observation.images.left')
TASK = 'pick up the tissue pack and place it on the right side'
FPS = 30
FFMPEG = '/usr/bin/ffmpeg'


def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj, indent=2)); tmp.replace(path)


def statistics(array, image=False, count=None):
    a = np.asarray(array, dtype=np.float64)
    stats = {'min':a.min(0), 'max':a.max(0), 'mean':a.mean(0), 'std':a.std(0)}
    for q in (1,10,50,90,99):
        stats[f'q{q:02d}'] = np.percentile(a,q,axis=0)
    return {k:(v.reshape(3,1,1) if image else np.atleast_1d(v)).tolist() for k,v in stats.items()} | {'count':[int(count or len(a))]}


def aggregate_statistics(stats_list):
    result = {}
    for key in stats_list[0]:
        entries = [s[key] for s in stats_list]
        weights = np.array([s['count'][0] for s in entries],float)
        weights /= weights.sum()
        means = np.array([s['mean'] for s in entries]); stds=np.array([s['std'] for s in entries])
        w = weights.reshape((-1,)+(1,)*(means.ndim-1))
        mean = (w*means).sum(0)
        variance = (w*(stds**2+(means-mean)**2)).sum(0)
        result[key]={'min':np.min([s['min'] for s in entries],axis=0).tolist(),
                     'max':np.max([s['max'] for s in entries],axis=0).tolist(),
                     'mean':mean.tolist(),'std':np.sqrt(variance).tolist(),
                     'count':[sum(s['count'][0] for s in entries)]}
    return result


class LerobotWriter:
    def __init__(self, root, fps=FPS, task=TASK, width=640, height=480):
        self.root=Path(root); self.fps=int(fps); self.task=task
        self.width,self.height=width,height
        if (self.root/'meta/info.json').exists():
            raise FileExistsError(f'dataset already exists: {self.root}')
        self.episodes=[]; self.episode_stats=[]; self.total_frames=0

    def add_episode(self, frames, states, actions, meta=None):
        import cv2
        import pyarrow as pa
        import pyarrow.parquet as pq
        states=np.asarray(states,dtype=np.float32); actions=np.asarray(actions,dtype=np.float32)
        n=len(frames); eid=len(self.episodes); chunk,file=eid//1000,eid%1000
        if n<2 or states.shape!=(n,16) or actions.shape!=(n,16):
            raise ValueError('episode frames/state/action lengths or shapes disagree')
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError('non-finite state/action')
        suffix=f'chunk-{chunk:03d}/file-{file:03d}'
        sinks={}; temporary={}; samples={cam:[] for cam in CAMERAS}
        try:
            for cam in CAMERAS:
                path=self.root/'videos'/cam/(suffix+'.mp4');path.parent.mkdir(parents=True,exist_ok=True)
                temp=path.with_name(path.stem+'.partial.mp4');temporary[cam]=(temp,path)
                sinks[cam]=subprocess.Popen([FFMPEG,'-v','error','-nostdin','-y','-f','rawvideo','-pix_fmt','rgb24',
                    '-s',f'{self.width}x{self.height}','-r',str(self.fps),'-i','-',
                    '-an','-c:v','libx264','-threads','2','-pix_fmt','yuv420p','-preset','fast',
                    '-crf','23','-movflags','+faststart',str(temp)],stdin=subprocess.PIPE)
            for t,frame in enumerate(frames):
                for cam in CAMERAS:
                    im=np.asarray(frame[cam])
                    if im.dtype!=np.uint8 or im.shape!=(self.height,self.width,3):
                        raise ValueError(f'invalid image {cam}: {im.shape}, {im.dtype}')
                    sinks[cam].stdin.write(im.tobytes())
                    if t%15==0:
                        samples[cam].append(cv2.resize(im,(32,32),interpolation=cv2.INTER_AREA).reshape(-1,3)/255.)
            for cam,sink in sinks.items():
                sink.stdin.close()
                if sink.wait()!=0:raise RuntimeError(f'ffmpeg failed for {cam}')
                temporary[cam][0].replace(temporary[cam][1])
        finally:
            for sink in sinks.values():
                if sink.poll() is None:
                    sink.kill();sink.wait()
        cols={'action':pa.array(actions.tolist(),pa.list_(pa.float32())),
              'observation.state':pa.array(states.tolist(),pa.list_(pa.float32())),
              'timestamp':pa.array(np.arange(n,dtype=np.float32)/self.fps),
              'frame_index':pa.array(np.arange(n,dtype=np.int64)),
              'episode_index':pa.array(np.full(n,eid,dtype=np.int64)),
              'index':pa.array(np.arange(self.total_frames,self.total_frames+n,dtype=np.int64)),
              'task_index':pa.array(np.zeros(n,dtype=np.int64))}
        path=self.root/'data'/(suffix+'.parquet');path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix('.partial');pq.write_table(pa.table(cols),tmp);tmp.replace(path)
        stats={key:statistics(value.to_pylist()) for key,value in cols.items()}
        stats.update({cam:statistics(np.concatenate(samples[cam]),image=True,count=n) for cam in CAMERAS})
        row={'episode_index':eid,'tasks':[self.task],'length':n,'data/chunk_index':chunk,'data/file_index':file,
             'dataset_from_index':self.total_frames,'dataset_to_index':self.total_frames+n,
             'meta/episodes/chunk_index':0,'meta/episodes/file_index':0}
        for cam in CAMERAS:
            row.update({f'videos/{cam}/chunk_index':chunk,f'videos/{cam}/file_index':file,
                        f'videos/{cam}/from_timestamp':0.,f'videos/{cam}/to_timestamp':n/self.fps})
        for key,stat in stats.items():
            row.update({f'stats/{key}/{k}':v for k,v in stat.items()})
        self.episodes.append(row);self.episode_stats.append(stats);self.total_frames+=n
        atomic_json(self.root/f'meta/collection/episode-{eid:06d}.json',meta or {})
        self.write()
        return True

    def write(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        import action_layout
        if not self.episodes:return {'episodes':0,'frames':0,'root':str(self.root)}
        features={key:{'dtype':'float32','shape':[16],'names':action_layout.DATASET_NAMES}
                  for key in ('action','observation.state')}
        for key in ('timestamp','frame_index','episode_index','index','task_index'):
            features[key]={'dtype':'float32' if key=='timestamp' else 'int64','shape':[1],'names':None}
        for cam in CAMERAS:
            features[cam]={'dtype':'video','shape':[self.height,self.width,3],'names':['height','width','channels'],
                          'info':{'video.height':self.height,'video.width':self.width,'video.codec':'h264',
                                  'video.pix_fmt':'yuv420p','video.is_depth_map':False,'video.fps':self.fps,
                                  'video.channels':3,'has_audio':False}}
        info={'codebase_version':'v3.0','robot_type':'play_e2_omnihand','fps':self.fps,'features':features,
              'total_episodes':len(self.episodes),'total_frames':self.total_frames,'total_tasks':1,
              'chunks_size':1000,'data_files_size_in_mb':100,'video_files_size_in_mb':200,
              'splits':{'train':f'0:{len(self.episodes)}'},
              'data_path':'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
              'video_path':'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4'}
        path=self.root/'meta/episodes/chunk-000/file-000.parquet';path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix('.partial');pq.write_table(pa.Table.from_pylist(self.episodes),tmp);tmp.replace(path)
        tasks = pa.table({'task_index':pa.array([0], pa.int64()), 'task':pa.array([self.task])})
        pandas_meta = {'index_columns':['task'], 'column_indexes':[],
                       'columns':[{'name':'task_index','field_name':'task_index','pandas_type':'int64','numpy_type':'int64','metadata':None},
                                  {'name':'task','field_name':'task','pandas_type':'unicode','numpy_type':'object','metadata':None}],
                       'creator':{'library':'pyarrow','version':pa.__version__},'pandas_version':'2.2.0'}
        tasks = tasks.replace_schema_metadata({b'pandas':json.dumps(pandas_meta).encode()})
        pq.write_table(tasks,self.root/'meta/tasks.parquet')
        atomic_json(self.root/'meta/stats.json',aggregate_statistics(self.episode_stats))
        atomic_json(self.root/'meta/info.json',info)
        return {'episodes':len(self.episodes),'frames':self.total_frames,'root':str(self.root)}
