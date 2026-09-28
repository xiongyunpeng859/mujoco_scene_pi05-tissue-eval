"""Diagnostic image ablation on recorded observations; no robot or training writes."""
import os,sys,json
from pathlib import Path
import cv2,numpy as np,pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'/workspace/users/fmc3-6-workspace/pi0.5_recap/scripts')
from train_o10_sim494_chunked import make_config
from openpi.policies.policy_config import create_trained_policy
checkpoint='/workspace/shared/new_program_qiuzhi/output/sim_to_real_bs8_chunked_20260923/sim_model/20000'
out=ROOT/'outputs/visual_state_conditioning_audit_20260924';out.mkdir(exist_ok=True)
rounds=sorted((ROOT/'outputs/tissue_pick_place_adaptive_grasps_494_20260921/rounds').glob('round-*'))[:10]
samples=[]
for rd in rounds:
 tab=pq.read_table(rd/'dataset/data/chunk-000/file-000.parquet').to_pydict()
 images={}
 for cam in ('top','left'):
  cap=cv2.VideoCapture(str(rd/f'dataset/videos/observation.images.{cam}/chunk-000/file-000.mp4'))
  for t in range(31):
   ok,im=cap.read()
   if not ok:raise RuntimeError('video decode failed')
   if t in (0,10,30):images[cam,t]=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
  cap.release()
 samples.append((tab,images))
policy=create_trained_policy(make_config(),checkpoint)
from openpi.training import data_loader
config=make_config();dc=config.data.create(config.assets_dirs,config.model)
ds=data_loader.create_torch_dataset(dc,50,config.model)
transformed=data_loader.transform_dataset(ds,dc)
batch=transformed[0]
audit={'repo_id':dc.repo_id,'discrete_state_input':config.model.discrete_state_input,
 'freeze_filter':str(config.freeze_filter),'image_masks':{k:bool(v) for k,v in batch['image_mask'].items()},
 'images':{k:{'shape':list(v.shape),'min':float(np.min(v)),'max':float(np.max(v)),
 'std':float(np.std(v))} for k,v in batch['image'].items()},
 'state_shape':list(batch['state'].shape),'state_range':[float(np.min(batch['state'])),float(np.max(batch['state']))],
 'prompt':ds[0]['prompt']}
(out/'training_inputs.json').write_text(json.dumps(audit,indent=2))
rows=[]
noise=np.random.default_rng(123).normal(size=(50,32)).astype(np.float32)
for i,(tab,images) in enumerate(samples):
 for t in (0,10,30):
  expert=np.array(tab['action'][t:t+50]);state=np.array(tab['observation.state'][t],np.float32)
  preds={}
  for variant in ('original','other_scene','other_state'):
   view=samples[(i+1)%len(samples)][1] if variant=='other_scene' else images
   input_state=np.array(samples[(i+1)%len(samples)][0]['observation.state'][t],np.float32) if variant=='other_state' else state
   preds[variant]=np.asarray(policy.infer({'observation/state':input_state,'observation/image':view['top',t],
    'observation/wrist_image':view['left',t],'prompt':'拿起纸巾包，放进右侧的绿色盒子里。'},noise=noise)['actions'])
  rows.append(dict(episode=i,frame=t,original_arm_mae=float(np.abs(preds['original'][:,:6]-expert[:,:6]).mean()),
    state_swap_action_change=float(np.abs(preds['original'][:,:6]-preds['other_state'][:,:6]).mean()),
    state_swap_input_change=float(np.abs(state[:6]-np.array(samples[(i+1)%len(samples)][0]['observation.state'][t])[:6]).mean()),
    other_image_arm_mae=float(np.abs(preds['other_scene'][:,:6]-expert[:,:6]).mean()),
    image_swap_action_change=float(np.abs(preds['original'][:,:6]-preds['other_scene'][:,:6]).mean()),
    original_first_arm_mae=float(np.abs(preds['original'][0,:6]-expert[0,:6]).mean())))
  (out/'report.json').write_text(json.dumps(rows,indent=2))
  print(rows[-1],flush=True)
