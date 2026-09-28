"""Left O10 success-only Pi0.5: isolated batch-4 runs, without EMA."""

import argparse
import dataclasses
from pathlib import Path
import runpy
import traceback

import numpy as np

import train_o10_pi05_success_only_jax as pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["check", "smoke", "train"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=["success", "recovery"], default="success")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--best-effort-save", action='store_true')
    args = parser.parse_args()
    if args.steps < 2 or args.steps % 2:
        parser.error('--steps must be positive and even (two checkpoints)')
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if args.save_every is not None and args.save_every < 1:
        parser.error('--save-every must be positive')
    factory = pipeline.make_config
    if args.dataset == "recovery":
        import train_o10_pi05_normal_recovery_jax as recovery

        factory = recovery.make_config
    pipeline.OUTPUT = args.output.resolve()
    config = dataclasses.replace(
        factory(),
        exp_name=f"left_{args.dataset}_fullft_bs{args.batch_size}_noema_{args.steps}steps",
        batch_size=args.batch_size,
        num_train_steps=2 if args.mode == "smoke" else args.steps,
        log_interval=1 if args.mode == "smoke" else 50,
        # OpenPI calls save_state with a zero-based loop index. Filter below
        # using completed steps, rather than saving at steps 5001 and 10000.
        save_interval=1,
    )
    print(f"LEFT_ARM dataset={args.dataset} batch={config.batch_size} steps={config.num_train_steps} "
          f"source={pipeline.SOURCE} output={config.checkpoint_dir}", flush=True)
    if args.mode == "check":
        loader = pipeline.data_loader.create_data_loader(config, shuffle=False, num_batches=1)
        obs, actions = next(iter(loader))
        assert obs.state.shape == (args.batch_size, 32)
        assert actions.shape == (args.batch_size, 50, 32)
        assert np.isfinite(np.asarray(obs.state)).all()
        assert np.isfinite(np.asarray(actions)).all()
        assert not np.asarray(obs.image_masks["right_wrist_0_rgb"]).any()
        for key in ("base_0_rgb", "left_wrist_0_rgb"):
            assert np.asarray(obs.image_masks[key]).all(), key
        print("BATCH_OK: left arm, 16 real dimensions padded to 32; top + left cameras", flush=True)
    else:
        # Largest FP32 parameter leaf exceeds the old 2 GB host-write budget.
        pipeline.install_deployable_checkpoint_saver(
            save_concurrent_gb=None if args.best_effort_save else 6,
            stage_on_host=not args.best_effort_save, synchronous=True,
            max_to_keep=None if args.save_every else 2,
        )
        save_state = pipeline.checkpoints.save_state
        save_steps = {2} if args.mode == "smoke" else {args.steps // 2, args.steps}
        if args.save_every and args.mode != 'smoke':
            save_steps = set(range(args.save_every, args.steps + 1, args.save_every))
            # Always retain the completed stage, even when a resumed tail is
            # shorter than the regular checkpoint interval.
            save_steps.add(args.steps)

        def save_selected_steps(manager, state, loader, step):
            completed_steps = step + 1
            if completed_steps in save_steps:
                try:
                    return save_state(manager, state, loader, completed_steps)
                except Exception:
                    if not args.best_effort_save:
                        raise
                    print(f'CHECKPOINT_FAILED step={completed_steps}; continuing training', flush=True)
                    traceback.print_exc()

        pipeline.checkpoints.save_state = save_selected_steps
        print(f"CHECKPOINT_STEPS: {sorted(save_steps)}", flush=True)
        print("EMA is OFF. Checkpoints contain model + normalization, NOT optimizer state.", flush=True)
        runpy.run_path(str(pipeline.OPENPI_ROOT / "scripts/train.py"))["main"](config)


if __name__ == "__main__":
    main()
