#!/usr/bin/env python3
"""Record the fourth pack position, inferred as the parallelogram's missing corner.

The user says the four packs roughly form a rectangle and asked me to work the fourth
out.  With A(5.53, 70.2), B(44.76, 71.33), C(51.12, 43.11) measured, the fourth corner is
D = A + C - B = (11.89, 41.98) cm, and it is geometrically exact: C - D == B - A.
Marked as INFERRED, not measured.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
CFG = ROOT / "configs/scene.yaml"
config = yaml.safe_load(CFG.read_text())
shutil.copy(CFG, CFG.with_suffix(".yaml.bak_before_fourth"))
br = config["box_randomization"]
A = [5.53, 70.2]
B = [44.76, 71.33]
C = [51.12, 43.11]
D = [round(A[0] + C[0] - B[0], 2), round(A[1] + C[1] - B[1], 2)]
samples = br["measured_samples"]
samples.append({"name": "inferred_fourth_corner",
                "centre_cm_from_left_bottom": D,
                "yaw_deg": None,
                "source": "INFERRED, not measured: the fourth corner of the parallelogram "
                          "formed by the three measured packs (D = A + C - B). The user "
                          "confirmed the four packs roughly form a rectangle (2026)."})
xs = [s["centre_cm_from_left_bottom"][0] for s in samples]
ys = [s["centre_cm_from_left_bottom"][1] for s in samples]
box = {"x": [round(min(xs), 2), round(max(xs), 2)],
       "y": [round(min(ys), 2), round(max(ys), 2)]}
old_region = dict(br["region_xy_cm_from_left_bottom"])
br["measured_bounding_box_cm"] = box
br["note"] = ((br.get("note") or "") +
              " | Fourth pack added 2026 as the inferred parallelogram corner %s; the "
              "bounding box of all four is x %s y %s, so the sampling region %s is kept "
              "unchanged (it is a superset)." % (D, box["x"], box["y"], old_region))
CFG.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
print("fourth pack recorded at", D)
print("bounding box of all four:", box)
print("sampling region kept   :", old_region)
