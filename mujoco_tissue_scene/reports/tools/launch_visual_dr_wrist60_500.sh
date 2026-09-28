#!/usr/bin/env bash
# 用法: tmux new-session -d -s o10_visual_dr_wrist60_500 'bash reports/tools/launch_visual_dr_wrist60_500.sh'
# 作用: 新规则 6 条试采和原生回读通过后，自动开始独立的 500 条采集。
# 默认配置: 腕部相对起点 ≤60°，相邻帧 ≤3°；视觉随机、物理固定；2 轮并行。
set -euo pipefail
cd /workspace/shared/mujoco_tissue_scene
task_pilot=outputs/visual_dr_wrist60_pilot_20260921
task_output=outputs/tissue_pick_place_visual_dr_wrist60_500_20260921
task_pipeline=outputs/visual_dr_wrist60_pipeline_20260921
for task_directory in "$task_pilot" "$task_output" "$task_pipeline"; do
    if [[ -e "$task_directory" ]]; then
        echo "Refusing existing output: $task_directory" >&2
        exit 1
    fi
done
mkdir -p "$task_pipeline"
exec > >(tee -a "$task_pipeline/pipeline.log") 2>&1
trap 'task_exit=$?; echo "PIPELINE_EXIT code=$task_exit time=$(date -Is)"' EXIT
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
task_python=/opt/miniconda3/envs/turbovla-libero/bin/python
echo "PILOT_START $(date -Is)"
"$task_python" -u reports/tools/collect_randomized_dataset.py \
    --output "$task_pilot" --episodes 6 --workers 2 --seed 1020260921 \
    --speed 1.5 --max-round-attempts 30 --full-domain --planned-grasps
"$task_python" - "$task_pilot" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path.cwd()
sys.path.insert(0, str(root / 'reports/tools'))
from collect_randomized_dataset import complete_round

pilot = Path(sys.argv[1])
status = json.loads((pilot / 'progress.json').read_text())
validation = json.loads((pilot / 'dataset/VALIDATION.json').read_text())
config = json.loads((pilot / 'run_config.json').read_text())
assert status['status'] == 'complete'
assert validation['passed'] and validation['episodes'] == 6
assert config['planned_grasps'] and config['full_domain']
assert not config['contact_randomization'] and not config['robot_dynamics_randomization']
assert config['max_wrist_deg'] == 60 and config['max_pregrasp_palm_deg'] == 90
assert config['yaw_range_deg'] == 90 and config['return_home']
assert config['speed'] == 1.5 and config['episode_frames'] == 375
assert sum(complete_round(p, planned=True) for p in (pilot / 'rounds').glob('round-*')) >= 2
assert all(hashlib.sha256((root / name).read_bytes()).hexdigest() == sha
           for name, sha in config['sha256'].items()), 'source changed since pilot'
print('PILOT_GATE_PASSED: native read + motion checks + fixed physics + source hashes', flush=True)
PY
echo "BATCH_500_START $(date -Is)"
"$task_python" -u reports/tools/collect_randomized_dataset.py \
    --output "$task_output" --episodes 500 --workers 2 --seed 1120260921 \
    --speed 1.5 --max-round-attempts 1000 --full-domain --planned-grasps
echo "BATCH_500_VALIDATED $(date -Is)"
