"""Standalone OpenPI/JAX configuration for the 108 successful O10 episodes."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
import runpy
import sys
import types

import numpy as np
import orbax.checkpoint as ocp
import pandas as pd

from openpi import transforms
from openpi.models import pi0_config
from openpi.policies.libero_policy import LiberoInputs
from openpi.shared import array_typing as at
from openpi.shared import normalize
from openpi.training import config as training_config
from openpi.training import optimizer, weight_loaders

# This OpenPI checkout uses LeRobot's former common.datasets import path.
# Alias only that namespace; keep the environment and shared repositories unchanged.
import lerobot
import lerobot.datasets
import lerobot.datasets.lerobot_dataset

common = types.ModuleType("lerobot.common")
common.__path__ = []
common.datasets = lerobot.datasets
lerobot.common = common
sys.modules.setdefault("lerobot.common", common)
sys.modules.setdefault("lerobot.common.datasets", lerobot.datasets)
sys.modules.setdefault(
    "lerobot.common.datasets.lerobot_dataset", lerobot.datasets.lerobot_dataset
)
from openpi.training import checkpoints
from openpi.training import data_loader


PROJECT = Path(__file__).resolve().parents[1]
OPENPI_ROOT = Path(__import__("os").environ.get("OPENPI_ROOT", "/workspace/shared/openpi_jax"))
SOURCE = Path(
    "/home/fmc3-6/workspace/shared/new_program_qiuzhi/without_tactile/"
    "success_episode/pick_up_the_tissue_pack_and_place_it_on_the_right_side_"
    "20260908_merged_all_108eps"
)
OUTPUT = Path(
    "/home/fmc3-6/workspace/shared/new_program_qiuzhi/output/"
    "total_suceesful_trajectory_model_jax_bs8_noema"
)
ASSET_ID = "o10_success_only_108eps"


@dataclasses.dataclass(frozen=True)
class O10Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :16]}


@dataclasses.dataclass(frozen=True)
class SuccessOnlyTrainConfig(training_config.TrainConfig):
    @property
    def checkpoint_dir(self) -> Path:
        return OUTPUT


def make_config() -> SuccessOnlyTrainConfig:
    repack = transforms.Group(inputs=[transforms.RepackTransform({
        "observation/image": "observation.images.top",
        "observation/wrist_image": "observation.images.left",
        "observation/state": "observation.state",
        "actions": "action",
        "prompt": "prompt",
    })])
    return SuccessOnlyTrainConfig(
        name="pi05_o10_success_only_jax",
        exp_name="success_only_fullft_bs8_noema_10k",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
        data=training_config.SimpleDataConfig(
            repo_id=str(SOURCE),
            assets=training_config.AssetsConfig(asset_id=ASSET_ID),
            base_config=training_config.DataConfig(
                repack_transforms=repack,
                action_sequence_keys=("action",),
                prompt_from_task=True,
            ),
            data_transforms=lambda model: transforms.Group(
                inputs=[LiberoInputs(model_type=model.model_type)],
                outputs=[O10Outputs()],
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            str(Path(__import__("os").environ.get("PI05_BASE_PARAMS", "/workspace/models/openpi-assets/checkpoints/pi05_base/params")))
        ),
        assets_base_dir=str(PROJECT / "assets_jax_success_only"),
        seed=1000,
        batch_size=8,
        num_workers=4,
        num_train_steps=10000,
        log_interval=50,
        save_interval=5000,
        keep_period=5000,
        optimizer=optimizer.AdamW(weight_decay=0.01),
        ema_decay=None,
        wandb_enabled=False,
        policy_metadata={
            "task": "pick up the tissue pack and place it inside the green box on the right",
            "action_dim": 16,
            "camera_rotation": "NO_ROTATION",
            "framework": "OpenPI JAX",
            "ema": False,
        },
    )


def prepare_stats(config: SuccessOnlyTrainConfig) -> None:
    """Compute stats over all frames and endpoint-padded 50-step chunks, without decoding videos."""
    frames = pd.concat([
        pd.read_parquet(path, columns=["episode_index", "frame_index", "observation.state", "action"])
        for path in sorted((SOURCE / "data").rglob("*.parquet"))
    ], ignore_index=True)
    state_stats, action_stats = normalize.RunningStats(), normalize.RunningStats()
    for _, episode in frames.groupby("episode_index", sort=True):
        episode = episode.sort_values("frame_index")
        states = np.stack(episode["observation.state"])
        actions = np.stack(episode["action"])
        if states.shape[1:] != (16,) or actions.shape[1:] != (16,):
            raise ValueError("Expected 16D O10 state/action")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError("Non-finite state/action values")
        state_stats.update(states)
        indices = np.minimum(
            np.arange(len(actions))[:, None] + np.arange(config.model.action_horizon)[None],
            len(actions) - 1,
        )
        action_stats.update(actions[indices])
    destination = config.assets_dirs / ASSET_ID
    normalize.save(destination, {
        "state": state_stats.get_statistics(),
        "actions": action_stats.get_statistics(),
    })
    print(f"NORM_STATS_OK: {len(frames)} frames, {frames.episode_index.nunique()} episodes; {destination}")


def check(config: SuccessOnlyTrainConfig) -> None:
    loader = data_loader.create_data_loader(config, shuffle=False, num_batches=1)
    observation, actions = next(iter(loader))
    print("BATCH_OK state=", observation.state.shape, "actions=", actions.shape)
    print("image masks=", {key: np.asarray(value).tolist() for key, value in observation.image_masks.items()})
    assert observation.state.shape == (8, 32)
    assert actions.shape == (8, 50, 32)
    assert not np.asarray(observation.image_masks["right_wrist_0_rgb"]).any()


def install_deployable_checkpoint_saver(
    *, save_concurrent_gb: int = 2, stage_on_host: bool = False,
    synchronous: bool = False, max_to_keep: int | None = 2,
) -> None:
    """Save policy params/assets only; full Adam state was too large to commit reliably.

    These checkpoints are directly deployable through OpenPI. They intentionally
    do not contain optimizer state and therefore cannot resume training.
    """
    def initialize(directory, *, keep_period, overwrite, resume):
        directory = Path(directory).resolve()
        if resume:
            raise ValueError("Deployable-only checkpoints do not support resume")
        if directory.exists():
            if not overwrite:
                if not directory.is_dir() or any(directory.iterdir()):
                    raise FileExistsError(directory)
            else:
                raise ValueError("Refusing destructive overwrite for this training run")
        else:
            directory.mkdir(parents=True)
        manager = ocp.CheckpointManager(
            directory,
            item_handlers={
                "assets": checkpoints.CallbackHandler(),
                "params": ocp.PyTreeCheckpointHandler(save_concurrent_gb=save_concurrent_gb),
            },
            options=ocp.CheckpointManagerOptions(
                max_to_keep=max_to_keep,
                keep_period=keep_period,
                create=False,
                enable_async_checkpointing=not synchronous,
                async_options=ocp.AsyncOptions(timeout_secs=7200),
            ),
        )
        return manager, False

    def save(manager, state, loader, step):
        def save_assets(directory):
            data_config = loader.data_config()
            if data_config.norm_stats is not None and data_config.asset_id is not None:
                normalize.save(directory / data_config.asset_id, data_config.norm_stats)

        with at.disable_typechecking():
            _, params = checkpoints._split_params(state)
        if stage_on_host:
            import jax

            # Transfer leaves sequentially; avoid a burst of simultaneous device
            # copies during Orbax's asynchronous GPU-array serialization.
            leaves, structure = jax.tree.flatten(params)
            host_leaves = []
            for index, leaf in enumerate(leaves):
                host_leaves.append(jax.device_get(leaf))
                if index % 10 == 0 or index == len(leaves) - 1:
                    print(f"CHECKPOINT_HOST_COPY {index + 1}/{len(leaves)}", flush=True)
            params = jax.tree.unflatten(structure, host_leaves)
        manager.save(step, {
            "assets": save_assets,
            "params": {"params": params},
        })

    checkpoints.initialize_checkpoint_dir = initialize
    checkpoints.save_state = save
    print("CHECKPOINT_MODE: deployable policy params + assets only; optimizer resume unavailable", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "check", "train"])
    args = parser.parse_args()
    config = make_config()
    if args.mode == "prepare":
        prepare_stats(config)
    elif args.mode == "check":
        check(config)
    else:
        install_deployable_checkpoint_saver()
        runpy.run_path(str(OPENPI_ROOT / "scripts/train.py"))["main"](config)
