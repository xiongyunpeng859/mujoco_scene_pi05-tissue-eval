#!/usr/bin/env python3
"""Is the sim's bag-start-to-box distance the same as the real image says?

If they agree, the geometry is right and a failed placement means the carry is
short.  If the sim's distance is larger, the base pose is wrong and moving it
toward the box fixes both the start and the release point.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import action_layout as layout
import align_with_dataset as align
import dataset_io
import measure_layout
import scene

SUCCESS = Path("/workspace/shared/new_program_qiuzhi/without_tactile/success_episode/"
               "pick_up_the_tissue_pack_and_place_it_on_the_right_side_"
               "20260908_merged_all_108eps")
OUT = ROOT / "outputs/geom_audit"
THUMB = "hand_L_thumb_tip"
FINGERS = ["hand_L_index_tip", "hand_L_middle_tip", "hand_L_ring_tip", "hand_L_pinky_tip"]
PITCH = [8, 10, 11, 13, 15]


def main() -> int:
    import cv2
    import mujoco
    OUT.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    table = config["table"]["size"]
    tray = config["tray"]
    box_cm = [(tray["center"][0] + table[0] / 2.0) * 100.0,
              (tray["center"][1] + table[1] / 2.0) * 100.0]
    print("box centre (config)                 : (%.2f, %.2f) cm" % tuple(box_cm))

    episodes, order = dataset_io.load(SUCCESS)
    chosen = order[:6]

    # --- the box position, measured again from an image, independent of config
    frame = align.dataset_frame(SUCCESS, chosen[0], 0)
    fitted = measure_layout.fit_tray(measure_layout.TableFrame(config), frame, tray)
    if fitted:
        fc = fitted["centre_cm_from_left_bottom"]
        print("box centre (re-fit from image ep%d)  : (%.2f, %.2f) cm  -> delta (%.2f, %.2f)"
              % (chosen[0], fc[0], fc[1], fc[0] - box_cm[0], fc[1] - box_cm[1]))

    # --- FK truth: the bag's centre if the sim base pose is right
    ref = mujoco.MjModel.from_xml_path(str(scene.build(
        ROOT / "configs/scene.yaml", OUT / "ref.xml", with_hand=True)))
    rd = mujoco.MjData(ref)
    ra = np.array(layout.joint_ids(ref, mujoco))
    thumb, fingers = ref.body(THUMB).id, [ref.body(n).id for n in FINGERS]
    print()
    print("  %-3s %-26s %-26s %9s | %-26s %9s" %
          ("ep", "image bag cm", "sim truth bag cm", "d truth-img",
           "box ctr cm", "d(sim)->box"))
    rows = []
    for e in chosen:
        st, ac = episodes[e]["observation.state"], episodes[e]["action"]
        rd.qpos[ra] = st[0]
        mujoco.mj_kinematics(ref, rd)
        g = 0.5 * (rd.xpos[thumb] + np.mean([rd.xpos[i] for i in fingers], axis=0))
        truth = [(g[0] + table[0] / 2) * 100.0, (g[1] + table[1] / 2) * 100.0]

        img = align.measure_bags(align.dataset_frame(SUCCESS, e, 0), config)
        if img:
            ic = img[0]["box_centre_cm"]
        else:
            ic = [float("nan")] * 2
        d_truth_img = float(np.hypot(truth[0] - ic[0], truth[1] - ic[1]))
        d_truth_box = float(np.hypot(truth[0] - box_cm[0], truth[1] - box_cm[1]))
        d_img_box = float(np.hypot(ic[0] - box_cm[0], ic[1] - box_cm[1]))
        print("  %-3d (%6.1f,%6.1f)%s (%6.1f,%6.1f)      %9.2f | (%6.1f,%6.1f) %9.2f"
              % (e, ic[0], ic[1],
                 " " if img else "*",
                 truth[0], truth[1], d_truth_img,
                 box_cm[0], box_cm[1], d_truth_box))
        rows.append({"episode": e, "image_cm": ic, "truth_cm": truth,
                     "d_truth_image": d_truth_img, "d_truth_box": d_truth_box,
                     "d_image_box": d_img_box, "n_image_bags": len(img)})
    print()
    good = [r for r in rows if r["d_truth_image"] < 0.08]
    print("  episodes where the image and the FK agree (< 8 cm): %d/%d" % (len(good), len(rows)))
    if good:
        print("  mean |truth - image|      : %.2f cm" %
              np.mean([r["d_truth_image"] for r in good]))
        print("  mean sim  bag->box        : %.2f cm" %
              np.mean([r["d_truth_box"] for r in good]))
        print("  mean real bag->box        : %.2f cm" %
              np.mean([r["d_image_box"] for r in good]))
        print("  mean signed offset (truth-image) : (%.2f, %.2f) cm"
              % tuple(np.mean([[r["truth_cm"][0] - r["image_cm"][0],
                                r["truth_cm"][1] - r["image_cm"][1]] for r in good], axis=0)))
    print()
    print("  bag half-length %.2f cm, box inner half-extent (%.2f, %.2f) cm" %
          (config["boxes"][0]["size"][0] * 50.0,
           tray["size"][0] * 50.0 - tray["wall_thickness"] * 100.0,
           tray["size"][1] * 50.0 - tray["wall_thickness"] * 100.0))
    (OUT / "result.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
