"""Opt-in streaming saver; leaves the running trainer and shared source unchanged."""
from pathlib import Path
import runpy
import train_o10_pi05_success_only_jax as pipeline
from chunked_jax_checkpoint import make_handler

original_install = pipeline.install_deployable_checkpoint_saver

def install(**kwargs):
    kwargs['stage_on_host'] = False
    kwargs['synchronous'] = True
    original_install(**kwargs)
    def initialize(directory, *, keep_period, overwrite, resume):
        directory = Path(directory).resolve()
        if overwrite or resume:
            raise ValueError('Streaming training refuses overwrite/resume')
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(directory)
        directory.mkdir(parents=True, exist_ok=True)
        ocp = pipeline.ocp
        manager = ocp.CheckpointManager(directory, item_handlers={
            'assets': pipeline.checkpoints.CallbackHandler(), 'params': make_handler()},
            options=ocp.CheckpointManagerOptions(max_to_keep=None,keep_period=keep_period,
                create=False,enable_async_checkpointing=False))
        return manager, False
    pipeline.checkpoints.initialize_checkpoint_dir = initialize
    print('STREAMING_SAVE: 64 MiB slices; exact block readback; stock Orbax commit',flush=True)

if __name__ == '__main__':
    pipeline.install_deployable_checkpoint_saver = install
    runpy.run_path(str(Path(__file__).with_name('train_o10_left_success_jax_bs4.py')),run_name='__main__')
