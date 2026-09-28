#!/usr/bin/env python3
"""Read the collected dataset back with the project's own reader.

If `dataset_io.load()` returns 16-dim states/actions and `align.dataset_frame()` decodes a
frame from each camera, the format is compatible with the tooling the real dataset uses.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import align_with_dataset as align
import dataset_io

D = ROOT / "outputs/lerobot_sim"


def main() -> int:
    print("tree:")
    for p in sorted(D.rglob("*")):
        if p.is_file():
            print("   %-58s %8d bytes" % (p.relative_to(D), p.stat().st_size))
    episodes, order = dataset_io.load(D)
    print()
    print("dataset_io.load(): %d episodes %s" % (len(order), order))
    for e in order:
        st = episodes[e]["observation.state"]
        ac = episodes[e]["action"]
        print("   ep%d  state %s action %s timestamp %d frames"
              % (e, st.shape, ac.shape, len(episodes[e]["timestamp"])))
    st = episodes[order[0]]["observation.state"]
    ac = episodes[order[0]]["action"]
    assert st.shape[1] == 16 and ac.shape[1] == 16, "expected 16-dim state and action"
    print("   state/action widths: 16 and 16  OK")
    print()
    for key in ("observation.images.top", "observation.images.left"):
        img = align.dataset_frame(D, order[0], 0, key=key)
        print("   %-30s decoded %s  mean %.1f" % (key, img.shape, img.mean()))
    print()
    print("PASS: the collected dataset reads back with the project's own reader.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
