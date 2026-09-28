"""Compare JAX policy actions with original simulation training samples."""
import argparse,json,sys
from pathlib import Path
import cv2,numpy as np,pyarrow.parquet as pq
PROJECT=Path('/workspace/users/fmc3-6-workspace/pi0.5_recap');ROOT=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--episodes',type=int,default=3);p.add_argument('--frames',type=int,nargs='+',default=[0,50,100,150,250,350]);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 sys.path.insert(0,str(PROJECT/'scripts'));sys.path.insert(0,'/workspace/shared/openpi_jax/src')
 import train_o10_sim494_chunked as pipeline
 from openpi.policies.policy_config import create_trained_policy
 policy=create_trained_policy(pipeline.make_config(),str(a.checkpoint))
 rows=[]
 for eid in range(a.episodes):
  tab=pq.read_table(a.dataset/f'data/chunk-{eid//1000:03d}/file-{eid%1000:03d}.parquet').to_pydict(); n=len(tab['frame_index'])
  vids=[]
  for cam in ('top','left'):
   path=a.dataset/f'videos/observation.images.{cam}/chunk-{eid//1000:03d}/file-{eid%1000:03d}.mp4'; cap=cv2.VideoCapture(str(path)); arr=[]
   for _ in range(n):
    ok,fr=cap.read()
    if not ok: raise RuntimeError(f'video read failed {path} frame {len(arr)}')
    arr.append(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB))
   cap.release();vids.append(arr)
  for f in a.frames:
   if f>=n:continue
   obs={'observation/state':np.asarray(tab['observation.state'][f],np.float32), 'observation/image':vids[0][f], 'observation/wrist_image':vids[1][f], 'prompt':'拿起纸巾包，放进右侧的绿色盒子里。'}
   pred=np.asarray(policy.infer(obs)['actions']); expert=np.asarray(tab['action'])[np.minimum(np.arange(f,f+50),n-1)]
   d=np.abs(pred-expert); rows.append({'episode':eid,'frame':f,'phase':('approach' if f<100 else 'close/lift' if f<180 else 'transfer' if f<280 else 'release/home'),'pred_first_action':pred[0].tolist(),'expert_first_action':expert[0].tolist(),'mae_first':float(d[0].mean()),'mae_chunk':float(d.mean()),'arm_mae_first':float(d[0,:6].mean()),'max_abs_first':float(d[0].max())})
 print(json.dumps({'checkpoint':str(a.checkpoint),'rows':rows},indent=2));(a.out/'report.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
