import sys
from pathlib import Path
import numpy as np, yaml, cv2, mujoco
ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import align_with_dataset as align
import dataset_io, sim_env
SUCCESS = Path("/workspace/shared/new_program_qiuzhi/without_tactile/success_episode/"
               "pick_up_the_tissue_pack_and_place_it_on_the_right_side_"
               "20260908_merged_all_108eps")
OUT = ROOT / "outputs/flex_visible"
OUT.mkdir(parents=True, exist_ok=True)
config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
p = OUT / "scene.yaml"
p.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
data, order = dataset_io.load(SUCCESS, fields=("observation.state",))
env = sim_env.TissueSceneEnv(config_path=p, dataset=SUCCESS, render=True, output_dir=OUT)
obs, _ = env.reset(options={"state": data[0]["observation.state"][0],
                            "randomize_objects": False})
img = np.asarray(obs["observation.images.top"])
cv2.imwrite(str(OUT / "with_packs.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
# the flex rgba 0.5,0.64,0.72 renders as a light blue-grey, far from the black table
rgb = img.astype(float) / 255.0
target = np.array([0.5, 0.64, 0.72])
dist = np.linalg.norm(rgb - target[None, None, :], axis=2)
n = int((dist < 0.18).sum())
print("pixels close to the pack colour: %d of %d" % (n, img.shape[0] * img.shape[1]))
print("VERDICT:", "PACKS VISIBLE" if n > 500 else "still invisible")
env.close()
