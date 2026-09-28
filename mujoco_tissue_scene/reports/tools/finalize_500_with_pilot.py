"""Wait for ongoing collection; combine validated pilot and batch into 500 episodes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from collect_randomized_dataset import complete_round, merge_rounds

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / 'outputs/visual_dr_wrist60_pilot_20260921'
BATCH = ROOT / 'outputs/tissue_pick_place_visual_dr_wrist60_fast494_20260921'
DEST = ROOT / 'outputs/tissue_pick_place_visual_dr_wrist60_final500_20260921/dataset'


def main():
    if DEST.exists():
        raise RuntimeError('Refusing existing final dataset')
    while True:
        status = json.loads((BATCH / 'progress.json').read_text())
        if status['status'] == 'complete':
            break
        if status['status'] == 'interrupted':
            raise RuntimeError(f'Collection interrupted: {status}')
        time.sleep(30)
    pilot_config = json.loads((PILOT / 'run_config.json').read_text())
    batch_config = json.loads((BATCH / 'run_config.json').read_text())
    # Explicitly authorized search-only change; require identical physics,
    # controller, acceptance checks and data writer across the two batches.
    differences = {k for k in set(pilot_config['sha256']) | set(batch_config['sha256'])
                   if pilot_config['sha256'].get(k) != batch_config['sha256'].get(k)}
    assert differences == {'reports/tools/plan_three_bag_round.py'}, differences
    for source in (PILOT, BATCH):
        assert json.loads((source / 'dataset/VALIDATION.json').read_text())['passed']
    pilot_rounds = sorted(p for p in (PILOT / 'rounds').glob('round-*') if complete_round(p, True))
    batch_rounds = sorted(p for p in (BATCH / 'rounds').glob('round-*') if complete_round(p, True))
    assert len(pilot_rounds) == 2 and len(batch_rounds) >= 165
    merge_rounds(pilot_rounds + batch_rounds, DEST, 500)
    env = os.environ.copy()
    env['PYTHONNOUSERSITE'] = '1'
    subprocess.run(['/opt/miniconda3/envs/lerobot-pi05/bin/python',
                    str(ROOT / 'reports/tools/validate_training_dataset.py'),
                    str(DEST), '500', '375'], env=env, check=True)
    (DEST / 'COMPOSITION.json').write_text(json.dumps({
        'episodes': 500, 'pilot_episodes': 6, 'batch_episodes': 494,
        'pilot_source': str(PILOT), 'batch_source': str(BATCH),
        'note': 'First 6 episodes are pilot; subsequent 494 are batch, in sorted accepted-round order.'
    }, indent=2))
    print('FINAL_500_VALIDATED', DEST, flush=True)


if __name__ == '__main__':
    main()
