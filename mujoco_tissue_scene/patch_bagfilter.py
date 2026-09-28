#!/usr/bin/env python3
"""A tissue bag cannot start inside the green box: those detections are the box's
own shadow, and keeping them makes 'bag in tray' trivially true before the arm moves."""
from pathlib import Path
import shutil

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
SRC = ROOT / "align_with_dataset.py"
text = SRC.read_text()
if "rejected_inside_tray" in text:
    print("filter already present")
    raise SystemExit(0)
old = """            result["rejected_off_table"] = True
            continue
        found.append(result)"""
new = """            result["rejected_off_table"] = True
            continue
        tray = config["tray"]
        tcx = (tray["center"][0] + config["table"]["size"][0] / 2.0) * 100.0
        tcy = (tray["center"][1] + config["table"]["size"][1] / 2.0) * 100.0
        tyaw = float(tray.get("yaw", 0.0))
        ddx, ddy = cx - tcx, cy - tcy
        lx = np.cos(-tyaw) * ddx - np.sin(-tyaw) * ddy
        ly = np.sin(-tyaw) * ddx + np.cos(-tyaw) * ddy
        if (abs(lx) < tray["size"][0] * 50.0 + 2.0
                and abs(ly) < tray["size"][1] * 50.0 + 2.0):
            # inside the green box already -> the box's own shadow, not a bag
            result["rejected_inside_tray"] = True
            continue
        found.append(result)"""
assert text.count(old) == 1, "anchor not unique: %d" % text.count(old)
shutil.copy(SRC, SRC.with_suffix(".py.bak_before_bagfilter"))
SRC.write_text(text.replace(old, new))
print("bag filter installed in align_with_dataset.py")
