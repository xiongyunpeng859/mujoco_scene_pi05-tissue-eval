"""Pi0.5 simulation fine-tune with extra sampling of early approach frames.

The underlying trajectories are unchanged.  Samples whose frame_index is in the
first EARLY_FRAMES frames of an episode are repeated EARLY_WEIGHT times in the
training index.  Normalization statistics remain computed over every frame.
"""
from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path
import numpy as np
import torch

import train_o10_sim494_chunked as sim
import train_o10_left_success_jax_bs4 as runner

EARLY_FRAMES = 60
EARLY_WEIGHT = int(os.environ.get("O10_EARLY_WEIGHT", "3"))
STATE_AUGMENT = os.environ.get("O10_STATE_AUGMENT", "0") == "1"
SAVE_EVERY = int(os.environ.get("O10_SAVE_EVERY", "10000"))
WRIST_DROPOUT_PROB = float(os.environ.get("O10_WRIST_DROPOUT", "0"))
STATE_NOISE_STD = 0.003  # radians; about 0.17 degrees
STATE_MASK_PROB = 0.10  # per sample: at most one of six arm joints
CHECKPOINT = Path(os.environ.get(
    "O10_PILOT_INIT",
    "/workspace/shared/new_program_qiuzhi/output/sim_to_real_bs8_chunked_20260923/sim_model/20000",
))

if not 0.0 <= WRIST_DROPOUT_PROB <= 1.0:
    raise ValueError("O10_WRIST_DROPOUT must be between 0 and 1")


@dataclasses.dataclass(frozen=True)
class WristCameraDropout:
    """Mask and zero the left wrist view for a random subset of samples."""

    probability: float

    def __call__(self, data: dict) -> dict:
        if self.probability <= 0.0 or np.random.random() >= self.probability:
            return data
        out = dict(data)
        images = dict(data["image"])
        image_masks = dict(data["image_mask"])
        images["left_wrist_0_rgb"] = np.zeros_like(images["left_wrist_0_rgb"])
        image_masks["left_wrist_0_rgb"] = np.False_
        # The central camera remains present and is the fallback visual input.
        image_masks["base_0_rgb"] = np.True_
        out["image"] = images
        out["image_mask"] = image_masks
        return out


class RepeatedIndexDataset:
    def __init__(self, dataset, early_frames=EARLY_FRAMES, early_weight=EARLY_WEIGHT):
        if early_weight < 1 or early_frames < 1:
            raise ValueError('Positive integer early weight and frame count required')
        self.dataset = dataset
        # Walk through Prompt/transform wrappers to the local LeRobot dataset.
        raw = dataset
        while hasattr(raw, "_dataset"):
            raw = raw._dataset
        table = getattr(raw, "_data", None)
        if table is None or "frame_index" not in table:
            raise RuntimeError("early weighting requires LocalLeRobotV3Dataset with frame_index")
        indices = []
        for i, frame in enumerate(table["frame_index"].to_numpy()):
            indices.append(i)
            if int(frame) < early_frames:
                indices.extend([i] * (early_weight - 1))
        self.indices = indices
        self.early_count = sum(int(x) < early_frames for x in table["frame_index"].to_numpy())
        self.state_mean = torch.as_tensor(
            np.stack(table['observation.state'].to_numpy())[:, :6].mean(axis=0),
            dtype=torch.float32,
        ) if STATE_AUGMENT else None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        item = self.dataset[self.indices[index]]
        if not STATE_AUGMENT:
            return item
        out = dict(item)
        state = item['observation.state'].clone()
        # Training input only, before normalization/tokenization. Never alter labels.
        state[:6] += torch.randn(6, dtype=state.dtype).clamp(-3, 3) * STATE_NOISE_STD
        if torch.rand(()).item() < STATE_MASK_PROB:
            joint = int(torch.randint(0, 6, ()).item())
            state[joint] = self.state_mean[joint]
        out['observation.state'] = state
        return out


_original_create = sim.pipeline.data_loader.create_torch_dataset


def create_weighted_dataset(data_config, action_horizon, model_config):
    base = _original_create(data_config, action_horizon, model_config)
    weighted = RepeatedIndexDataset(base)
    print(
        f"EARLY_FRAME_WEIGHTING: first {EARLY_FRAMES} frames x{EARLY_WEIGHT}; "
        f"base={len(base)} weighted={len(weighted)} early_rows={weighted.early_count}; "
        f"state_augment={STATE_AUGMENT} noise_std={STATE_NOISE_STD} mask_prob={STATE_MASK_PROB}",
        flush=True,
    )
    return weighted


def make_config():
    from openpi.training import optimizer, weight_loaders
    base = sim.make_config()
    if WRIST_DROPOUT_PROB:
        original_data_transforms = base.data.data_transforms

        def data_transforms(model):
            return original_data_transforms(model).push(
                inputs=[WristCameraDropout(WRIST_DROPOUT_PROB)]
            )

        data = dataclasses.replace(base.data, data_transforms=data_transforms)
    else:
        data = base.data
    return dataclasses.replace(
        base,
        data=data,
        weight_loader=weight_loaders.CheckpointWeightLoader(str(CHECKPOINT / 'params')),
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=1000, peak_lr=1e-5, decay_steps=20000, decay_lr=1e-6),
        batch_size=8,
        num_workers=2,
    )


if __name__ == "__main__":
    # The reused runner otherwise defaults to batch 4; this experiment must use 8.
    if '--batch-size' not in sys.argv and not any(x.startswith('--batch-size=') for x in sys.argv):
        sys.argv.extend(['--batch-size', '8'])
    # New launches use the requested 10k checkpoint interval even when an older
    # queue script still carries the previous CLI value.
    for index, argument in enumerate(sys.argv):
        if argument == '--save-every' and index + 1 < len(sys.argv):
            sys.argv[index + 1] = str(SAVE_EVERY)
        elif argument.startswith('--save-every='):
            sys.argv[index] = f'--save-every={SAVE_EVERY}'
    sim.pipeline.data_loader.create_torch_dataset = create_weighted_dataset
    sim.pipeline.make_config = make_config
    print(
        f'PILOT_INIT={CHECKPOINT}; optimizer reset; early weight={EARLY_WEIGHT}; '
        f'save_every={SAVE_EVERY}; wrist_dropout={WRIST_DROPOUT_PROB}',
        flush=True,
    )
    # Keep the existing runner and chunked checkpoint saver unchanged.
    runner.main()
