#!/usr/bin/env python3
"""Why is the arm still rendered bright?  Inspect the compiled geoms."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import scene                            # noqa: E402


def main() -> int:
    import mujoco
    out = ROOT / "outputs/armcolor_diag"
    out.mkdir(parents=True, exist_ok=True)
    xml = scene.build(ROOT / "configs/scene.yaml", out / "scene.xml", with_hand=True)
    model = mujoco.MjModel.from_xml_path(str(xml))

    print("materials in the model: %d" % model.nmat)
    for i in range(model.nmat):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, i)
        print("   mat %-3d %-24s rgba %s" % (i, name, np.round(model.mat_rgba[i], 3).tolist()))
    print()
    print("  %-5s %-22s %-7s %4s %5s %-22s %6s" %
          ("geom", "body", "type", "con", "group", "geom_rgba", "matid"))
    arm_visual = 0
    for g in range(model.ngeom):
        body = model.body(int(model.geom_bodyid[g])).name or ""
        if not (body.startswith("link") or body == "qiuzhi_arm_mount"):
            continue
        gtype = int(model.geom_type[g])
        contype = int(model.geom_contype[g])
        group = int(model.geom_group[g])
        matid = int(model.geom_matid[g])
        if contype == 0:
            arm_visual += 1
        if arm_visual > 0 and contype == 0 and arm_visual <= 12:
            print("  %-5d %-22s %-7d %4d %5d %-22s %6d"
                  % (g, body, gtype, contype, group,
                     np.round(model.geom_rgba[g], 3).tolist(), matid))
    print()
    print("arm visual geoms (contype==0): %d" % arm_visual)
    bright = [g for g in range(model.ngeom)
              if int(model.geom_contype[g]) == 0
              and (model.body(int(model.geom_bodyid[g])).name or "").startswith("link")
              and float(model.geom_rgba[g][:3].mean()) > 0.5]
    print("of those, still bright (>0.5): %d" % len(bright))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
