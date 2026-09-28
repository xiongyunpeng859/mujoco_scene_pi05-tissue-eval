#!/usr/bin/env python3
"""Pin the base from two image features: the hand at rest, and the held bag over the box.

The hand blob at the start pose pins two degrees of freedom (translating and rotating
the base can hold one point fixed, leaving a one-parameter family).  The held bag at
the moment of placing sits at a much larger radius from the base, so it breaks that
family.  Both comparisons are done in image space, which avoids back-projecting an
unknown height.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import align_with_dataset as align      # noqa: E402
import dataset_io                       # noqa: E402
import measure_layout                   # noqa: E402
import scene                            # noqa: E402

SUCCESS = Path("/workspace/shared/new_program_qiuzhi/without_tactile/success_episode/"
               "pick_up_the_tissue_pack_and_place_it_on_the_right_side_"
               "20260908_merged_all_108eps")
OUT = ROOT / "outputs/base_twofeature"
HAND_POINTS = ["static_omnihand_reference", "hand_L_thumb_tip", "hand_L_index_tip",
               "hand_L_middle_tip", "hand_L_ring_tip", "hand_L_pinky_tip"]
THUMB = "hand_L_thumb_tip"
FINGERS = ["hand_L_index_tip", "hand_L_middle_tip", "hand_L_ring_tip", "hand_L_pinky_tip"]
PLACE_FRAME = 150


def hand_blob_centre(image, region):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(grey, bool)
    y0, y1, x0, x1 = region
    mask[y0:y1, x0:x1] = True
    blob = (grey > 140) & mask
    blob = cv2.morphologyEx(blob.astype(np.uint8), cv2.MORPH_OPEN,
                            np.ones((3, 3), np.uint8)) > 0
    ys, xs = np.where(blob)
    if len(xs) < 40:
        return None, None
    return np.array([xs.mean(), ys.mean()]), (xs, ys)


def held_bag_centre(image, config):
    """The pack in the hand at the placing moment: the blue blob nearest the box."""
    table = measure_layout.TableFrame(config)
    blue = measure_layout.blue_dominance(image)
    mask = (blue > 12).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    tray = config["tray"]
    size = config["table"]["size"]
    centre_cm = [(tray["center"][0] + size[0] / 2) * 100.0,
                 (tray["center"][1] + size[1] / 2) * 100.0]
    tray_px = table.project(table.from_cm(centre_cm[0], centre_cm[1]))
    best = None
    for i in range(1, count):
        if stats[i, 4] < 150:
            continue
        d = float(np.linalg.norm(centroids[i] - tray_px))
        if best is None or d < best[0]:
            best = (d, centroids[i], stats[i, 4])
    return (None, None) if best is None else (best[1], best)


def main() -> int:
    import mujoco
    OUT.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    surface = config["table"]["surface_z"]
    table = config["table"]["size"]
    ref = measure_layout.TableFrame(config)
    episodes, order = dataset_io.load(SUCCESS)
    episode = order[0]
    states = episodes[episode]["observation.state"]

    # The hand at rest is the SAME in every episode, so the 40-frame median is the
    # right source for it; a single frame carries that episode's bags and lighting.
    first = cv2.imread(str(ROOT / "outputs/base_arm_mask/consensus.png"))
    if first is None:
        first = align.dataset_frame(SUCCESS, episode, 0)
    placing = align.dataset_frame(SUCCESS, episode, PLACE_FRAME)
    cv2.imwrite(str(OUT / "placing_frame.png"), placing)
    hand_c, hand_px = hand_blob_centre(first, (230, 400, 0, 170))
    bag_c, bag_info = held_bag_centre(placing, config)
    print("hand blob at t=0    : %s (%d px)"
          % (None if hand_c is None else np.round(hand_c, 1).tolist(),
             0 if hand_px is None else len(hand_px[0])))
    print("held bag at t=%d   : %s (%s px)"
          % (PLACE_FRAME, None if bag_c is None else np.round(bag_c, 1).tolist(),
             0 if bag_info is None else bag_info[2]))
    if hand_c is None or bag_c is None:
        raise SystemExit("a feature was not found")

    xml = scene.build(ROOT / "configs/scene.yaml", OUT / "scene.xml", with_hand=True)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mount = model.body("qiuzhi_arm_mount").id
    hand_ids = [model.body(n).id for n in HAND_POINTS]
    thumb = model.body(THUMB).id
    fingers = [model.body(n).id for n in FINGERS]
    adr = [model.joint("joint%d" % i).qposadr[0] for i in range(1, 7)]

    def errors(x_cm, y_cm, yaw):
        model.body_pos[mount] = [x_cm / 100.0 - table[0] / 2.0,
                                 y_cm / 100.0 - table[1] / 2.0, surface]
        model.body_quat[mount] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        data.qpos[adr] = states[0][:6]
        mujoco.mj_kinematics(model, data)
        hand = np.mean([ref.project(data.xpos[i]) for i in hand_ids], axis=0)
        data.qpos[adr] = states[PLACE_FRAME][:6]
        mujoco.mj_kinematics(model, data)
        grip = 0.5 * (data.xpos[thumb]
                      + np.mean([data.xpos[i] for i in fingers], axis=0))
        grip_px = ref.project(grip)
        e1 = float(np.linalg.norm(hand - hand_c))
        e2 = float(np.linalg.norm(grip_px - bag_c))
        return e1, e2

    rows = []
    for x_cm in np.arange(6.0, 46.1, 1.0):
        for y_cm in np.arange(-6.0, 30.1, 1.0):
            for yaw in np.radians(np.arange(40.0, 141.0, 2.5)):
                e1, e2 = errors(float(x_cm), float(y_cm), float(yaw))
                rows.append({"x": float(x_cm), "y": float(y_cm), "yaw": float(yaw),
                             "hand_px": e1, "bag_px": e2, "total": e1 + e2})
    rows.sort(key=lambda r: r["total"])
    print()
    print("  %-30s %9s %9s %9s" % ("base / yaw", "hand px", "bag px", "total"))
    for r in rows[:14]:
        print("  x=%5.1f y=%5.1f yaw=%6.1f (%5.1f deg) %9.1f %9.1f %9.1f"
              % (r["x"], r["y"], r["yaw"], np.degrees(r["yaw"]), r["hand_px"],
                 r["bag_px"], r["total"]))
    best = rows[0]
    print()
    print("  BEST x=%.1f y=%.1f yaw=%.1f deg -> hand %.1f px, bag %.1f px"
          % (best["x"], best["y"], np.degrees(best["yaw"]), best["hand_px"],
             best["bag_px"]))
    for label, x, y, deg in (("grasp fit (18,10,90)", 18.0, 10.0, 90.0),
                             ("tape (25,10,90)", 25.0, 10.0, 90.0),
                             ("kinematic (34,24,60)", 34.0, 24.0, 60.0)):
        e1, e2 = errors(x, y, np.radians(deg))
        print("  %-24s hand %6.1f  bag %6.1f" % (label, e1, e2))
    (OUT / "result.json").write_text(json.dumps(rows[:500], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
