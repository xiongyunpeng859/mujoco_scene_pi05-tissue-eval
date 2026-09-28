#!/usr/bin/env python3
"""What does lerobot 0.4.4 need to create a dataset offline?"""
import inspect
import traceback

print("=== import ===")
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    print("   LeRobotDataset OK")
except Exception:
    traceback.print_exc()
    raise SystemExit(1)

print()
print("=== create signature ===")
try:
    print(inspect.signature(LeRobotDataset.create))
except Exception as exc:
    print("   ", exc)
print()
doc = (LeRobotDataset.create.__doc__ or "").strip().splitlines()
for line in doc[:40]:
    print("   ", line)
print()
print("=== save_episode / add_frame ===")
for name in ("add_frame", "save_episode", "clear_episode_buffer", "create",
             "start_image_writer", "stop_image_writer"):
    fn = getattr(LeRobotDataset, name, None)
    if fn is None:
        print("   %-24s MISSING" % name)
    else:
        try:
            print("   %-24s %s" % (name, inspect.signature(fn)))
        except Exception as exc:
            print("   %-24s %s" % (name, exc))

print()
print("=== does the real dataset load with this library? ===")
D = ("/workspace/shared/new_program_qiuzhi/without_tactile/success_episode/"
     "pick_up_the_tissue_pack_and_place_it_on_the_right_side_"
     "20260908_merged_all_108eps")
try:
    ds = LeRobotDataset(repo_id="local/success_108", root=D)
    print("   loaded: %d episodes, %d frames" % (ds.num_episodes, ds.num_frames))
    print("   features:", list(ds.features))
    item = ds[0]
    for k, v in item.items():
        print("      %-32s %s" % (k, getattr(v, "shape", type(v).__name__)))
except Exception:
    print("   load failed:")
    traceback.print_exc()
