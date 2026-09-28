#!/usr/bin/env python3
"""Revert the in-tray filter: packs DO legitimately sit in the tray at episode
start, because the 108 episodes are one continuous recording and packs left in
the tray by earlier episodes persist into later ones."""
from pathlib import Path
import shutil

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
SRC = ROOT / "align_with_dataset.py"
BAK = ROOT / "align_with_dataset.py.bak_before_bagfilter"
if BAK.exists():
    shutil.copy(BAK, SRC)
    print("reverted align_with_dataset.py from %s" % BAK.name)
else:
    print("no backup found; leaving as is")
print("in-tray filter present now:", "rejected_inside_tray" in SRC.read_text())
