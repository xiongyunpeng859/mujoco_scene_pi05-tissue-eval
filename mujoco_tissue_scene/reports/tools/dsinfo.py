#!/usr/bin/env python3
"""Print the real dataset's exact structure so a writer can match it."""
import glob
import json
import os

D = ("/workspace/shared/new_program_qiuzhi/without_tactile/success_episode/"
     "pick_up_the_tissue_pack_and_place_it_on_the_right_side_"
     "20260908_merged_all_108eps")
print("=== top level ===")
print(sorted(os.listdir(D)))
print()
print("=== meta/ ===")
print(sorted(os.listdir(D + "/meta")))
print()
print("=== info.json ===")
print(open(D + "/meta/info.json").read()[:3000])
print()
for sub in ("data", "videos"):
    print("=== %s tree ===" % sub)
    for p in sorted(glob.glob(D + "/" + sub + "/**", recursive=True))[:14]:
        print("   ", p.replace(D + "/", ""))
    print()
print("=== one episode metadata file ===")
for cand in ("episodes.jsonl", "episodes/chunk-000/file-000.jsonl"):
    path = D + "/meta/" + cand
    if os.path.exists(path):
        with open(path) as fh:
            print(cand, "->", fh.readline()[:800])
        break
else:
    print("no jsonl found; meta contains:", sorted(os.listdir(D + "/meta")))
