#!/usr/bin/env python3
"""A strip of frames across an episode: where does the pack start, where does it end?"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import cv2

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import align_with_dataset as align
import dataset_io

SUCCESS = Path("/workspace/shared/new_program_qiuzhi/without_tactile/success_episode/"
               "pick_up_the_tissue_pack_and_place_it_on_the_right_side_"
               "20260908_merged_all_108eps")
OUT = ROOT / "outputs/detector_debug"


def strip(episode):
    data, order = dataset_io.load(SUCCESS)
    n = len(data[episode]["action"])
    idx = [0, n // 5, 2 * n // 5, 3 * n // 5, 4 * n // 5, n - 1]
    tiles = []
    for i in idx:
        img = align.dataset_frame(SUCCESS, episode, int(i))
        tile = cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, "f%d/%d" % (i, n), (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)
        tiles.append(tile)
    top = np.hstack(tiles[:3])
    bot = np.hstack(tiles[3:])
    grid = np.vstack([top, bot])
    cv2.imwrite(str(OUT / ("strip_ep%02d.png" % episode)), grid)
    print("ep%d: %d frames, strip at %s" % (episode, n, idx))


for e in (2, 0):
    strip(e)
