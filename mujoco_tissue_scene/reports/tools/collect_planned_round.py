"""Plan in independent simulations, then replay once into a LeRobot round."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from replay_three_bag_plan import verify_replay
from grasp_candidate_selection import GRASP_FAMILIES

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--grasp-family', choices=GRASP_FAMILIES, default=None)
    args = parser.parse_args()
    folder = args.directory.resolve()
    planning = folder / 'planning'
    plan_command=[sys.executable, '-u', str(ROOT / 'reports/tools/plan_three_bag_round.py'),
                    '--config', str(args.config.resolve()), '--seed', str(args.seed),
                    '--output', str(planning), '--max-pregrasp-palm-deg', '90', '--compact']
    if args.grasp_family:
        plan_command += ['--family',args.grasp_family]
    subprocess.run(plan_command,cwd=ROOT,check=True)
    plan = json.loads((planning / 'plan.json').read_text())
    env = os.environ.copy()
    env.update(MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    subprocess.run([sys.executable, '-u', str(ROOT / 'reports/tools/scripted_pick_place.py'),
                    '--config', plan['config'], '--seed', str(args.seed), '--episodes', '3',
                    '--bags-per-round', '3', '--randomize-appearance', '--speed', '1.5',
                    '--validated-grasp-plan', str(planning / 'plan.json'),
                    '--collect', str(folder / 'dataset'), '--output-dir', str(folder / 'run')],
                   cwd=ROOT, env=env, check=True)
    rows = json.loads((folder / 'run/result.json').read_text())
    verify_replay(rows, plan['yaws'], plan['max_pregrasp_palm_deg'],
                  plan['max_wrist_deg'], plan['families'])
    print('PLANNED_ROUND_ACCEPTED', folder, flush=True)


if __name__ == '__main__':
    main()
