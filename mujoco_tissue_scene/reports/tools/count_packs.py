import sys
from pathlib import Path
import yaml
ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import align_with_dataset as align
SUCCESS = Path("/workspace/shared/new_program_qiuzhi/without_tactile/success_episode/"
               "pick_up_the_tissue_pack_and_place_it_on_the_right_side_"
               "20260908_merged_all_108eps")
config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
for e in range(12):
    bags = align.measure_bags(align.dataset_frame(SUCCESS, e, 0), config)
    print("  ep%-3d %d packs: %s" % (e, len(bags),
          " ".join("(%.1f,%.1f)" % tuple(b["box_centre_cm"]) for b in bags)))
