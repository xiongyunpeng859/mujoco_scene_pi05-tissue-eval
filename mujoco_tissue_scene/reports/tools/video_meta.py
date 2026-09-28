#!/usr/bin/env python3
"""Is dataset_frame() reading the right video frames at all?"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import align_with_dataset as align
import dataset_io

SUCCESS = Path("/workspace/shared/new_program_qiuzhi/without_tactile/success_episode/"
               "pick_up_the_tissue_pack_and_place_it_on_the_right_side_"
"20260908_merged_all_108eps")


def main() -> int:
    info = json.loads((SUCCESS / "meta/info.json").read_text())
    fps = float(info["fps"])
    print("fps %.3f   total_frames %s   total_episodes %s"
          % (fps, info.get("total_frames"), info.get("total_episodes")))
    data, order = dataset_io.load(SUCCESS)
    print()
    print("  %-3s %-6s %-4s %-4s %12s %10s %9s %14s"
          % ("ep", "chunk", "file", "len", "from_ts(s)", "start_frm", "end_frm", "file_frames"))
    files = {}
    for episode in range(10):
        meta = align.episode_metadata(SUCCESS, episode)
        key = "videos/observation.images.top"
        chunk, fidx = meta[key + "/chunk_index"], meta[key + "/file_index"]
        ts = meta[key + "/from_timestamp"]
        n = len(data[episode]["action"])
        start = int(round(ts * fps))
        path = SUCCESS / ("videos/observation.images.top/chunk-%03d/file-%03d.mp4"
                          % (chunk, fidx))
        if path not in files:
            cap = cv2.VideoCapture(str(path))
            files[path] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        print("  %-3d %-6d %-4d %-4d %12.3f %10d %9d %14d"
              % (episode, chunk, fidx, n, ts, start, start + n - 1, files[path]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
