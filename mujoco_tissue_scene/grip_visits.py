#!/usr/bin/env python3
"""Does the grip point actually visit the box, given the settled base?

Necessary condition: in every successful episode the real robot put the pack in the
box, so its grip point must have reached the box.  This uses the grip point (where the
object is held) rather than the palm, and imposes no height band -- the earlier
palm-based version was sensitive to exactly those choices.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import align_with_dataset as align      # noqa: E402
import dataset_io                       # noqa: E402
import scene                            # noqa: E402

SUCCESS = Path("/workspace/shared/new_program_qiuzhi/without_tactile/success_episode/"
               "pick_up_the_tissue_pack_and_place_it_on_the_right_side_"
               "20260908_merged_all_108eps")
OUT = ROOT / "outputs/grip_visits"
THUMB = "hand_L_thumb_tip"
FINGERS = ["hand_L_index_tip", "hand_L_middle_tip", "hand_L_ring_tip", "hand_L_pinky_tip"]


def main() -> int:
    import mujoco
    OUT.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    surface = config["table"]["surface_z"]
    table = config["table"]["size"]
    tray = config["tray"]
    tray_cm = [(tray["center"][0] + table[0] / 2) * 100.0,
               (tray["center"][1] + table[1] / 2) * 100.0]
    half = tray["size"][0] / 2.0

    xml = scene.build(ROOT / "configs/scene.yaml", OUT / "scene.xml", with_hand=True)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mount = model.body("qiuzhi_arm_mount").id
    thumb = model.body(THUMB).id
    fingers = [model.body(n).id for n in FINGERS]
    tray_body = model.body("tray").id
    adr = [model.joint("joint%d" % i).qposadr[0] for i in range(1, 7)]
    mujoco.mj_kinematics(model, data)
    tray_world = data.xpos[tray_body][:2].copy()

    episodes, order = dataset_io.load(SUCCESS, fields=("observation.state",))
    print("box centre (table cm) = (%.1f, %.1f), half %.3f m ; base (18,10) yaw 90deg"
          % (tray_cm[0], tray_cm[1], half))
    print()
    print("  %-4s %12s %12s %12s  %s" %
          ("ep", "min d(grip,box)", "at frame", "grip z there", "verdict"))
    rows = []
    for episode in order[:20]:
        state = episodes[episode]["observation.state"]
        best, best_frame, best_z = 9.9, -1, 0.0
        for frame, row in enumerate(state):
            if frame < 40:
                continue
            data.qpos[adr] = row[:6]
            mujoco.mj_kinematics(model, data)
            grip = 0.5 * (data.xpos[thumb]
                          + np.mean([data.xpos[i] for i in fingers], axis=0))
            d = float(np.linalg.norm(grip[:2] - tray_world))
            if d < best:
                best, best_frame, best_z = d, frame, float(grip[2])
        verdict = "visits the box" if best < 0.12 else ("near miss" if best < 0.20
                                                        else "never gets there")
        rows.append({"episode": episode, "min_distance_m": best, "frame": best_frame,
                     "grip_z": best_z, "verdict": verdict})
        print("  %-4d %12.4f %12d %12.3f  %s" % (episode, best, best_frame, best_z, verdict))
    print()
    visits = sum(1 for r in rows if r["min_distance_m"] < 0.12)
    print("  episodes whose grip point reaches within 12 cm of the box centre: %d / %d"
          % (visits, len(rows)))
    print("  mean min distance %.4f m ; worst %.4f m"
          % (np.mean([r["min_distance_m"] for r in rows]),
             max(r["min_distance_m"] for r in rows)))
    if visits < len(rows) * 0.8:
        print()
        print("  => the recorded trajectories do NOT bring the grip point to the box under")
        print("     this base, yet the real robot placed the pack in the box every time.")
        print("     So either the base is still off, or something else is inconsistent.")
    (OUT / "result.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
