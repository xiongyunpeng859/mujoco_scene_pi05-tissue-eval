#!/usr/bin/env bash
set -euo pipefail
# Set these paths before sourcing this file.
export PI05_PROJECT_ROOT="${PI05_PROJECT_ROOT:-$PWD}"
export OPENPI_ROOT="${OPENPI_ROOT:-$HOME/openpi_jax}"
export PI05_BASE_PARAMS="${PI05_BASE_PARAMS:-$HOME/artifacts/pi05_base/params}"
export CHECKPOINT="${CHECKPOINT:-$HOME/artifacts/uniform_step15000/10000}"
export SIM494_DATASET="${SIM494_DATASET:-$HOME/artifacts/tissue_dataset_494/dataset}"
export TEST_ROUND="${TEST_ROUND:-$HOME/artifacts/round-00001}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
export MUJOCO_GL=egl
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Robot assets: point these at the extracted URDF and mesh directories.
export O10_ARM_URDF="${O10_ARM_URDF:-$HOME/artifacts/o10_robot_assets/play_e2/urdf/play_e2.urdf}"
export O10_ARM_MESH_DIR="${O10_ARM_MESH_DIR:-$HOME/artifacts/o10_robot_assets/play_e2/meshes}"
export O10_HAND_URDF="${O10_HAND_URDF:-$HOME/artifacts/o10_robot_assets/omnihand_left.urdf}"
export O10_HAND_MESH_DIR="${O10_HAND_MESH_DIR:-$HOME/artifacts/o10_robot_assets/meshes}"
export PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
