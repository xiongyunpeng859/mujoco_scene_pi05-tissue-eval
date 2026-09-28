#!/usr/bin/env bash
# 用法: 在工程目录用 tmux 启动 bash reports/tools/launch_visual_dr_planned_500.sh
# 作用: 用户确认先合成 500 条；视觉/相机/摆放随机，物理参数固定。
# 默认配置: 2 轮并行，各轮 3 个候选并行；90 度翻掌限制；30 Hz、每集复位。
set -euo pipefail
cd /workspace/shared/mujoco_tissue_scene
task_output=outputs/tissue_pick_place_visual_dr_planned_500_20260921
if [[ -e "$task_output" ]]; then
    echo "Refusing existing output: $task_output" >&2
    exit 1
fi
/opt/miniconda3/envs/turbovla-libero/bin/python -c '
import hashlib,json,pathlib
root=pathlib.Path(".")
pilot=root/"outputs/visual_dr_planned_pilot_20260921"
status=json.loads((pilot/"progress.json").read_text())
validation=json.loads((pilot/"dataset/VALIDATION.json").read_text())
config=json.loads((pilot/"run_config.json").read_text())
assert status["status"]=="complete" and validation["passed"] and validation["episodes"]==6
assert config["planned_grasps"] and not config["contact_randomization"] and not config["robot_dynamics_randomization"]
assert config["max_pregrasp_palm_deg"]==90 and config["yaw_range_deg"]==90
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in config["sha256"].items()), "source changed since pilot"
'
mkdir -p "$task_output"
exec > >(tee -a "$task_output/collection.log") 2>&1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONNOUSERSITE=1
exec /opt/miniconda3/envs/turbovla-libero/bin/python -u reports/tools/collect_randomized_dataset.py \
    --output "$task_output" --episodes 500 --workers 2 --seed 920260921 \
    --speed 1.5 --max-round-attempts 1000 --full-domain --planned-grasps
