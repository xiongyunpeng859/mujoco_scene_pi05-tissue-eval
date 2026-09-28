"""Ten distinct recorded scenes; isolated model-generated closed-loop evaluation."""
import os,json,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
out=root/'outputs/jax_ten_teacher100_20260924';out.mkdir(parents=True,exist_ok=True)
rounds=sorted((root/'outputs/tissue_pick_place_adaptive_grasps_494_20260921/rounds').glob('round-*'))[:10]
rows=[]
for i,rd in enumerate(rounds):
 folder=out/f'ep-{i:02d}';folder.mkdir(exist_ok=True)
 with (folder/'run.log').open('w') as log:
  r=subprocess.run([sys.executable,'-u',str(root/'test_jax_policy_rollout.py'),'--known-round',str(rd),
   '--output-dir',str(folder),'--execute-chunk','50','--teacher-prefix','100','--seed',str(2026092400+i)],
   env=dict(os.environ,MUJOCO_GL='egl',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT)
 result=json.loads((folder/'result.json').read_text()) if r.returncode==0 else {'error':r.returncode}
 rows.append(result);(out/'summary.json').write_text(json.dumps(rows,indent=2))
 print(i,rd.name,result.get('final_success'),result.get('peak_lift_m'),flush=True)
