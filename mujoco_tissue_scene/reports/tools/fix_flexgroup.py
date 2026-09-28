#!/usr/bin/env python3
"""The flexible packs were built with group='3', which MuJoCo hides by default, so the
pack never appeared in any render -- the camera images showed an empty table.  Put them
in a visible group.  This is why every render so far (sim.png, overlay.png, the scripted
controller frames) looked as if the table had no packs on it."""
from pathlib import Path
import shutil
import re

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
SRC = ROOT / "scene.py"
text = SRC.read_text()
hits = re.findall(r"group=.?3.?\s*\)?", text)
print("occurrences of group 3 in scene.py:", len(hits))
for h in hits[:6]:
    print("   ", h)
shutil.copy(SRC, SRC.with_suffix(".py.bak_before_flexgroup"))
new = text.replace('group="3"', 'group="1"').replace("group='3'", "group='1'")
if new == text:
    new = re.sub(r'group\s*=\s*[\'"]?3[\'"]?', 'group="1"', text)
changed = sum(1 for a, b in zip(text.splitlines(), new.splitlines()) if a != b)
SRC.write_text(new)
print("lines changed:", changed)
import ast
ast.parse(new)
print("scene.py patched; syntax ok")
