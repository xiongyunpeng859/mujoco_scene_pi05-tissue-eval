#!/usr/bin/env bash
# 用法: 在 tmux 中运行 bash reports/tools/launch_full_dr_1000.sh
# 作用: 小批 LeRobot 验收通过后采集全参数随机化 1000 条。
# 默认配置: 4 workers，30 Hz，1.5 倍动作速度，每集复位；只接受完整成功轮次。
set -euo pipefail
cd /workspace/shared/mujoco_tissue_scene
task_output=outputs/tissue_pick_place_full_dr_1000_20260920
if [[ -e "$task_output" ]]; then
    echo "Refusing existing output: $task_output" >&2
    exit 1
fi
/opt/miniconda3/envs/turbovla-libero/bin/python -c 'import json; p=json.load(open("outputs/full_dr_pilot_v2_20260920/dataset/VALIDATION.json")); assert p["passed"] and p["episodes"] == 6'
mkdir -p "$task_output"
exec > >(tee -a "$task_output/collection.log") 2>&1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONNOUSERSITE=1
exec /opt/miniconda3/envs/turbovla-libero/bin/python -u reports/tools/collect_randomized_dataset.py \
    --output "$task_output" --episodes 1000 --workers 4 --seed 820260920 \
    --speed 1.5 --max-round-attempts 1600 --full-domain
