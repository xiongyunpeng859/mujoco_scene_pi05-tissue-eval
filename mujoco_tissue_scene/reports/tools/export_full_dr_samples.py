"""Export previews from completed successful rounds without touching collection."""
import argparse
import json
from pathlib import Path
import subprocess

from collect_randomized_dataset import complete_round


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    out = args.root/'sample_previews'
    out.mkdir(exist_ok=False)
    rounds = [p for p in sorted((args.root/'rounds').glob('round-*')) if complete_round(p)][:3]
    manifest=[]
    for index, directory in enumerate(rounds, 1):
        meta=json.loads((directory/'dataset/meta/collection/episode-000000.json').read_text())
        inputs=[]
        for camera in ('top','left'):
            inputs += ['-i',str(directory/f'dataset/videos/observation.images.{camera}/chunk-000/file-000.mp4')]
        video=out/f'sample_{index:02d}_{directory.name}.mp4'
        subprocess.run(['/usr/bin/ffmpeg','-v','error',*inputs,'-filter_complex',
                        f"hstack,drawtext=text='{directory.name} - TOP / WRIST':x=10:y=10:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.6",
                        '-c:v','libx264','-crf','19','-pix_fmt','yuv420p',str(video)],check=True)
        subprocess.run(['/usr/bin/ffmpeg','-v','error','-ss','3.5','-i',str(video),
                        '-frames:v','1',str(out/f'sample_{index:02d}.jpg')],check=True)
        manifest.append(dict(video=str(video),source_round=directory.name,episode=0,
                             appearance=meta['appearance'],domain=meta['sampled_domain'],spawn=meta['spawn']))
    (out/'samples.json').write_text(json.dumps(manifest,indent=2))
    for sample in manifest:
        print(sample['video'], 'arm=',sample['appearance']['arm_texture'],
              'mu=',[round(b['friction'][0],2) for b in sample['domain']['contacts']],flush=True)


if __name__=='__main__':
    main()
