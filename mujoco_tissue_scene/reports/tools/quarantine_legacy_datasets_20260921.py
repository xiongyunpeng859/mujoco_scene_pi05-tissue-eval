"""Quarantine explicitly scoped pre-wrist60 synthetic training data; keep diagnostics."""
import argparse
import json
from pathlib import Path

ROOT = Path('/workspace/shared/mujoco_tissue_scene/outputs')
TRASH = Path('/workspace/shared/.dataset_trash/legacy_pre_wrist60_20260921')
# Inspected historical batches only; never select current outputs by exclusion.
SCOPES = [
    'tissue_pick_place_full_dr_1000_20260920',
    'tissue_pick_place_dr_1000_20260920',
    'full_dr_pilot_20260920', 'full_dr_pilot_v2_20260920',
    'visual_dr_planned_pilot_20260921', 'speed_1p5_home',
    'domain_randomization', 'lerobot_sim', 'lerobot_verified_20260920',
    'lerobot_video_continuous_20260920', 'lerobot_direct_release_preview',
    'lerobot_three_bag_preview', 'lerobot_three_bag_preview_v3',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    targets = set()
    for name in SCOPES:
        scope = ROOT / name
        assert scope.is_dir() and not scope.is_symlink(), scope
        for directory in scope.rglob('*'):
            if directory.name in ('data', 'videos') and directory.is_dir():
                parent = directory.parent
                if 'source_snapshot' in parent.parts:
                    continue
                assert parent.resolve() == parent and parent.is_relative_to(scope), parent
                assert (parent / 'data').is_dir() and (parent / 'videos').is_dir(), parent
                assert not any(p.is_symlink() for p in parent.rglob('*')), parent
                targets.add(parent)
    rows = []
    for target in sorted(targets):
        assert not any(other != target and target.is_relative_to(other) for other in targets)
        dest = TRASH / target.relative_to(ROOT)
        assert not dest.exists(), dest
        info = target / 'meta/info.json'
        rows.append({'source': str(target), 'destination': str(dest),
                     'episodes': json.loads(info.read_text()).get('total_episodes') if info.exists() else None})
    print(json.dumps({'dataset_directories': len(rows), 'targets': rows}, indent=2))
    if not args.apply:
        return
    TRASH.mkdir(parents=True, exist_ok=False)
    manifest = {'reason': 'Pre-wrist60 synthetic data: not accepted under current collection rules',
                'recoverable': True, 'targets': rows, 'moved': []}
    manifest_path = TRASH / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    for row in rows:
        source, destination = Path(row['source']), Path(row['destination'])
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
        assert not source.exists() and destination.is_dir()
        manifest['moved'].append(row['source'])
        manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f'QUARANTINED {len(rows)} dataset directories; manifest={manifest_path}')


if __name__ == '__main__':
    main()
