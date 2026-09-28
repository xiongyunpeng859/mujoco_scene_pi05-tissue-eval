"""Simulation-only Pi0.5 base fine-tuning with the tested streaming saver."""
import dataclasses
from pathlib import Path
import sys

import train_o10_pi05_success_only_jax as pipeline
import train_o10_chunked as streaming
import train_o10_left_success_jax_bs4 as runner

pipeline.SOURCE = Path(__import__('os').environ.get('SIM494_DATASET', '/workspace/shared/mujoco_tissue_scene/outputs/tissue_pick_place_adaptive_grasps_494_20260921/dataset'))
pipeline.ASSET_ID = 'o10_sim_adaptive_494eps'
original_factory = pipeline.make_config


def make_config():
    config = original_factory()
    return dataclasses.replace(
        config, name='pi05_o10_sim494_jax',
        assets_base_dir=str(pipeline.PROJECT / 'assets_jax_sim494'),
        policy_metadata={**config.policy_metadata,
                         'task': '拿起纸巾包，放进右侧的绿色盒子里。',
                         'dataset': 'simulation_494'},
    )


pipeline.make_config = make_config
pipeline.install_deployable_checkpoint_saver = streaming.install

if __name__ == '__main__':
    if sys.argv[1:] == ['prepare']:
        pipeline.prepare_stats(make_config())
    else:
        runner.main()
