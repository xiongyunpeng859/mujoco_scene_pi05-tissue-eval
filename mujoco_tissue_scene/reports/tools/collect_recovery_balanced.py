"""Durable screened recovery collection. Never count failed candidates."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'reports/tools'))
from full_domain_config import sample_config
from lerobot_writer import atomic_json

MODES = ('closed_empty', 'execution_error', 'slip', 'taken_away',
         'target_moves', 'corner_grasp')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=1000)
    p.add_argument('--max-attempts-per-mode', type=int, default=1000)
    p.add_argument('--workers', type=int, default=1)
    args = p.parse_args()
    if args.episodes < len(MODES):
        p.error('--episodes must cover all six modes')
    if not 1 <= args.workers <= len(MODES):
        p.error('--workers must be between 1 and 6')
    quotas = {mode:args.episodes//len(MODES)+(index<args.episodes%len(MODES))
              for index,mode in enumerate(MODES)}
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'collector.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = dict(os.environ, MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    accepted = {mode: [] for mode in MODES}
    attempts = {mode: 0 for mode in MODES}
    blocked = {}
    progress = out/'progress.json'
    if progress.exists():
        old = json.loads(progress.read_text())
        accepted, attempts, blocked = old['accepted'], old['attempts'], old['blocked']

    def save():
        atomic_json(progress, dict(target=args.episodes, quotas=quotas,
            workers=args.workers,
            accepted=accepted, attempts=attempts, blocked=blocked,
            accepted_total=sum(map(len,accepted.values())), updated=time.time(),
            status='complete' if all(len(v)==quotas[m] for m,v in accepted.items()) else 'incomplete'))

    def run_candidate(mode):
            attempt = attempts[mode]
            folder = out/mode/f'attempt-{attempt:05d}'
            # A previous coordinator may have stopped just after the writer
            # committed. Adopt that completed candidate without overwriting it.
            info = folder/'dataset/meta/info.json'
            result_path = folder/'render/result.json'
            if info.exists() and result_path.exists():
                if (json.loads(info.read_text())['total_episodes']==1
                        and json.loads(result_path.read_text())[0]['success']):
                    return str(folder)
            if folder.exists():
                # Keep partial artifacts recoverable, and avoid existing-writer
                # collisions when retrying an interrupted candidate.
                backup = folder.with_name(folder.name+f'.interrupted-{time.time_ns()}')
                folder.rename(backup)
            folder.mkdir(parents=True, exist_ok=True)
            seed = 23092600 + MODES.index(mode)*100000 + attempt
            cfg = sample_config(ROOT/'configs/scene.yaml', folder, seed,
                                yaw_range=25., randomize_contact=False)
            command = [sys.executable, '-u', str(ROOT/'reports/tools/scripted_pick_place.py'),
                '--config', str(cfg), '--episodes', '1', '--seed', str(seed),
                '--randomize-appearance', '--speed','1.5','--max-wrist-deg','60',
                '--failure-mode',mode,'--diversify-failure',
                '--miss-offset',str(-(.10 + (attempt%7)*.012)),
                str(((-1)**attempt) * (.015 + (attempt%5)*.010)),
                '--miss-drift',str(.018+(attempt%6)*.010),
                '--recovery-delay-frames',str((attempt*5)%19)]
            with (folder/'screen.log').open('w') as log:
                result = subprocess.run(command+['--output-dir',str(folder/'screen'),'--no-render'],
                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            rowfile = folder/'screen/result.json'
            good = result.returncode==0 and rowfile.exists() and json.loads(rowfile.read_text())[0]['success']
            if good:
                with (folder/'render.log').open('w') as log:
                    result = subprocess.run(command+['--output-dir',str(folder/'render'),
                        '--collect',str(folder/'dataset')], cwd=ROOT, env=env,
                        stdout=log, stderr=subprocess.STDOUT)
                info = folder/'dataset/meta/info.json'
                if result.returncode==0 and info.exists() and json.loads(info.read_text())['total_episodes']==1:
                    return str(folder)
            return None

    # Threads only coordinate independent Python simulator subprocesses; a
    # single coordinator alone writes progress and final merged metadata.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
      while True:
        active = [m for m in MODES if len(accepted[m])<quotas[m] and m not in blocked]
        if not active:
            break
        for mode, candidate in zip(active, pool.map(run_candidate, active)):
            if candidate is not None:
                accepted[mode].append(candidate)
            attempts[mode] += 1
            # Explicit stall guard: do not burn thousands of attempts when a
            # new failure mechanism or recovery controller is not yet working.
            if attempts[mode]>=20 and not accepted[mode]:
                blocked[mode] = 'No accepted pilot in 20 attempts; controller review required'
            elif attempts[mode]>=args.max_attempts_per_mode:
                blocked[mode] = 'Candidate budget reached'
            save()
            print(mode, len(accepted[mode]), '/', quotas[mode], 'attempt', attempts[mode], flush=True)
    save()
    if blocked:
        print('INCOMPLETE: classes require review',blocked,flush=True)
        return 2
    # Keep the authoritative per-class manifest; merge only a balanced complete
    # set and ensure the prompt is recovery, not the normal-task merge default.
    from collect_randomized_dataset import merge_rounds
    recovery_task = 'Previous grasp failed. Regrasp the tissue pack and place it inside the green box on the right.'
    directories = [Path(folder) for mode in MODES for folder in accepted[mode]]
    destination = merge_rounds(directories, out/'dataset', args.episodes, task=recovery_task)
    print('COMPLETE', destination, flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
