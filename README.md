# mujoco_scene_pi05-tissue-eval

Pi0.5 JAX MuJoCo evaluation and training workflow for the 494-trajectory tissue pick/place simulation.

## What is in Git

This repository contains source code, configs, normalization statistics, and cloud launch scripts. Large artifacts are intentionally excluded: Pi0.5 checkpoints, base weights, robot meshes, videos, and datasets.

## Cloud layout

Place the artifacts as follows (paths can be overridden in `cloud_env.sh`):

```text
$HOME/openpi_jax/                         OpenPI JAX source checkout
$HOME/artifacts/pi05_base/params          JAX pi05_base/params checkpoint
$HOME/artifacts/uniform_step15000/10000   cumulative 15000-step Orbax checkpoint
$HOME/artifacts/tissue_dataset_494/dataset 494-trajectory dataset (training only)
$HOME/artifacts/round-00001              one recorded MuJoCo test round
```

Clone OpenPI separately:

```bash
git clone https://github.com/Fmc3-Robotics-Devs/openpi-jax.git "$HOME/openpi_jax"
```

Install the matching CUDA/JAX environment before running. Then set `PI05_BASE_PARAMS`, `CHECKPOINT`, `TEST_ROUND`, and `OPENPI_ROOT` if the artifact locations differ.

## Single-episode smoke evaluation

```bash
source cloud_env.sh
bash run_eval_cloud.sh "$HOME/eval_ep01"
```

A successful run writes `result.json`, `trace.json`, and `rollout.mp4`.

## Training

The training entrypoint and its helper modules are in `pi05_scripts/`. Use a 80–96GB GPU for the original batch-size-8 training configuration. A 32GB GPU is intended only for inference and small smoke tests.

## Multiple evaluation scenes

`round-00001` is only one recorded initial scene. For repeated evaluation, upload a whole `rounds/` directory containing many `round-*` directories, then run:

```bash
source cloud_env.sh
bash run_many_eval_cloud.sh "$HOME/artifacts/rounds" "$HOME/eval_many" 20
```

The third argument is an optional maximum number of scenes; use `0` to evaluate all scenes. Each scene gets its own output directory and `summary.tsv` records completed runs and final success.

## Evaluation protocol

Evaluation uses `--known-round`: each recorded `round-*` contains a complete, reproducible initial scene, including object placement, robot state, appearance, and target. This is the canonical path for comparing checkpoints. Use multiple rounds for repeated evaluation; do not report an unrecorded random scene as the standard result.
