#!/usr/bin/env python3
"""Ground truth for the bag positions, straight from the robot's own motion.

The robot closes its fingers on the pack, and that closure is recorded in the state
(finger pitch dimensions jump from ~0 to ~0.7).  The frame where it happens is the
grasp frame, and the grip point -- halfway between the thumb tip and the four
fingertips -- is then sitting on the pack.  Forward kinematics with the settled base
therefore gives the pack's true position without any image detection, without the
tray, and without any search.

Comparing that against what the image detector reports for the same episode measures
the detector's systematic error directly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import action_layout as layout          # noqa: E402
import align_with_dataset as align      # noqa: E402
import dataset_io                       # noqa: E402
import scene                            # noqa: E402

SUCCESS = Path("/workspace/shared/new_program_qiuzhi/without_tactile/success_episode/"
               "pick_up_the_tissue_pack_and_place_it_on_the_right_side_"
               "20260908_merged_all_108eps")
OUT = ROOT / "outputs/bag_truth"
THUMB = "hand_L_thumb_tip"
FINGERS = ["hand_L_index_tip", "hand_L_middle_tip", "hand_L_ring_tip", "hand_L_pinky_tip"]
# the five hand dimensions the policy actually drives, and the value meaning "closed"
PITCH_DIMS = [8, 10, 11, 13, 15]
CLOSED_LEVEL = 0.35


def main() -> int:
    import mujoco
    OUT.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    surface = config["table"]["surface_z"]
    size = config["table"]["size"]

    xml = scene.build(ROOT / "configs/scene.yaml", OUT / "scene.xml", with_hand=True)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    thumb = model.body(THUMB).id
    fingers = [model.body(n).id for n in FINGERS]
    adr = np.array(layout.joint_ids(model, mujoco))

    episodes, order = dataset_io.load(SUCCESS)
    rows = []
    print("  %-4s %-8s %-28s %-28s %s" %
          ("ep", "grasp f", "grip point, table cm", "detected bag, table cm", "delta cm"))
    for episode in order[:20]:
        state = episodes[episode]["observation.state"]
        action = episodes[episode]["action"]
        # the grasp happens when the driven finger dimensions first pass the level
        closed = np.abs(action[:, PITCH_DIMS]).mean(1)
        frames = np.where(closed > CLOSED_LEVEL)[0]
        if len(frames) == 0:
            print("  %-4d  (fingers never close)" % episode)
            continue
        grasp_frame = int(frames[0])
        data.qpos[adr] = state[grasp_frame]
        mujoco.mj_kinematics(model, data)
        grip = 0.5 * (data.xpos[thumb] + np.mean([data.xpos[i] for i in fingers], axis=0))
        truth_cm = np.array([(grip[0] + size[0] / 2) * 100.0,
                             (grip[1] + size[1] / 2) * 100.0])

        bags = [b for b in align.measure_bags(
            align.dataset_frame(SUCCESS, episode, 0), config) if "box_centre_cm" in b]
        if bags:
            centres = np.array([b["box_centre_cm"] for b in bags])
            nearest = centres[int(np.argmin(np.linalg.norm(centres - truth_cm, axis=1)))]
            delta = nearest - truth_cm
            print("  %-4d %-8d (%.1f, %.1f)%s(%s) (%.1f, %.1f)%s%s %s"
                  % (episode, grasp_frame, truth_cm[0], truth_cm[1], " " * 8,
                     "", nearest[0], nearest[1], " " * 8, "",
                     np.round(delta, 1).tolist()))
            rows.append({"episode": episode, "grasp_frame": grasp_frame,
                         "truth_cm": truth_cm.tolist(), "nearest_detected_cm": nearest.tolist(),
                         "delta_cm": delta.tolist(),
                         "distance_cm": float(np.linalg.norm(delta)),
                         "n_detected": len(bags)})
        else:
            print("  %-4d %-8d (%.1f, %.1f)          (no bag detected)"
                  % (episode, grasp_frame, truth_cm[0], truth_cm[1]))
            rows.append({"episode": episode, "grasp_frame": grasp_frame,
                         "truth_cm": truth_cm.tolist(), "nearest_detected_cm": None,
                         "n_detected": 0})
    print()
    found = [r for r in rows if r["nearest_detected_cm"]]
    if found:
        deltas = np.array([r["delta_cm"] for r in found])
        distances = np.array([r["distance_cm"] for r in found])
        print("  episodes with both truth and a detection: %d / %d" % (len(found), len(rows)))
        print("  detector error: mean %s cm, median %s cm, max %.1f cm"
              % (np.round(deltas.mean(0), 2).tolist(), np.round(np.median(deltas, 0), 2).tolist(),
                 distances.max()))
        print("  -> a systematic bias would show as a consistent sign in the mean")
    print()
    print("  bag truth spread across episodes (are they really randomised?):")
    truths = np.array([r["truth_cm"] for r in rows])
    print("     x %.1f..%.1f   y %.1f..%.1f cm" %
          (truths[:, 0].min(), truths[:, 0].max(), truths[:, 1].min(), truths[:, 1].max()))
    (OUT / "result.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
