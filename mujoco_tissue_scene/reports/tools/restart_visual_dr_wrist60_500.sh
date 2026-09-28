#!/usr/bin/env bash
# 用法: bash reports/tools/restart_visual_dr_wrist60_500.sh（在 tmux 中运行）
# 作用: 复用已验收的新规则试采，从独立目录重启 500 条，不改变选姿规则。
# 默认配置: 2 轮并行，腕部 60°，视觉随机、物理固定。
set -euo pipefail
cd /workspace/shared/mujoco_tissue_scene
task_output=outputs/tissue_pick_place_visual_dr_wrist60_500_restart_20260921
if [[ -e "$task_output" ]]; then
    echo "Refusing existing output: $task_output" >&2
    exit 1
fi
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
task_python=/opt/miniconda3/envs/turbovla-libero/bin/python
"$task_python" - <<'PY'
import hashlib, json
from pathlib import Path
import sys
root = Path.cwd()
sys.path.insert(0, str(root / 'reports/tools'))
from collect_randomized_dataset import complete_round
pilot = root / 'outputs/visual_dr_wrist60_pilot_20260921'
config = json.loads((pilot / 'run_config.json').read_text())
validation = json.loads((pilot / 'dataset/VALIDATION.json').read_text())
assert json.loads((pilot / 'progress.json').read_text())['status'] == 'complete'
assert validation['passed'] and validation['episodes'] == 6
assert config['planned_grasps'] and config['full_domain']
assert config['max_wrist_deg'] == 60 and config['max_pregrasp_palm_deg'] == 90
assert not config['contact_randomization'] and not config['robot_dynamics_randomization']
assert config['return_home'] and config['yaw_range_deg'] == 90
assert sum(complete_round(p, True) for p in (pilot / 'rounds').glob('round-*')) >= 2
assert all(hashlib.sha256((root / p).read_bytes()).hexdigest() == h for p,h in config['sha256'].items()), 'source changed since validated pilot'
print('VALIDATED_PILOT_AND_UNCHANGED_SOURCE_OK', flush=True)
PY
mkdir -p "$task_output"
exec > >(tee -a "$task_output/collection.log") 2>&1
trap 'task_exit=$?; echo "COLLECTION_EXIT code=$task_exit time=$(date -Is)"' EXIT
"$task_python" -u reports/tools/collect_randomized_dataset.py \
    --output "$task_output" --episodes 500 --workers 2 --seed 1120260921 \
    --speed 1.5 --max-round-attempts 1000 --full-domain --planned-grasps
