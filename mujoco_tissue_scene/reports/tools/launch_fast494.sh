#!/usr/bin/env bash
# 用法: 在 tmux 中执行 bash reports/tools/launch_fast494.sh
# 作用: 分批候选搜索采 494 条，再与已验收 6 条合并。
# 默认配置: 腕部 60°，物理固定，2 轮并行；任何验收失败都不放宽约束。
set -euo pipefail
cd /workspace/shared/mujoco_tissue_scene
task_output=outputs/tissue_pick_place_visual_dr_wrist60_fast494_20260921
if [[ -e "$task_output" ]]; then exit 1; fi
mkdir -p "$task_output"
exec > >(tee -a "$task_output/collection.log") 2>&1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/opt/miniconda3/envs/turbovla-libero/bin/python -u reports/tools/collect_randomized_dataset.py \
 --output "$task_output" --episodes 494 --workers 2 --seed 1120260921 \
 --speed 1.5 --max-round-attempts 1000 --full-domain --planned-grasps
/opt/miniconda3/envs/turbovla-libero/bin/python -u reports/tools/finalize_500_with_pilot.py
