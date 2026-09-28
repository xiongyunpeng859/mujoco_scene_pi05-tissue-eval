"""Paired prefix and camera tests. Two independent simulation/policy workers."""
import json,os,sys,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
root=Path(__file__).resolve().parent
out=root/'outputs/prefix_camera_audit_20260924';out.mkdir(exist_ok=True,parents=True)
rounds=sorted((root/'outputs/tissue_pick_place_adaptive_grasps_494_20260921/rounds').glob('round-*'))[:10]
jobs=[(f'prefix{n}',n,False,i,r) for n in (10,30,50,70) for i,r in enumerate(rounds)]
jobs += [('nominal',0,True,i,r) for i,r in enumerate(rounds)]
def run(job):
 label,n,nominal,i,rd=job;folder=out/label/f'ep-{i:02d}';folder.mkdir(exist_ok=True,parents=True)
 if (folder/'result.json').exists():return json.loads((folder/'result.json').read_text())
 cmd=[sys.executable,'-u',str(root/'test_jax_policy_rollout.py'),'--known-round',str(rd),
      '--output-dir',str(folder),'--execute-chunk','50','--teacher-prefix',str(n),'--seed',str(2026092400+i)]
 if nominal:cmd+=['--nominal-cameras']
 with (folder/'run.log').open('w') as log:
  result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,
     MUJOCO_GL='egl',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
 return json.loads((folder/'result.json').read_text()) if result.returncode==0 else {'error':result.returncode,'folder':str(folder)}
rows=[]
with ThreadPoolExecutor(max_workers=2) as pool:
 futures={pool.submit(run,j):j for j in jobs}
 for f in as_completed(futures):
  j=futures[f];r=f.result();r['group']=j[0];r['episode']=j[3];rows.append(r)
  (out/'summary.json').write_text(json.dumps(rows,indent=2))
  print(len(rows),'/50',j[0],j[3],r.get('final_success'),flush=True)
