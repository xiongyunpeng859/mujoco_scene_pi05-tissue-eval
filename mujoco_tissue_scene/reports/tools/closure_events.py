#!/usr/bin/env python3
"""Find every open->closed event in the hand signal and check which one lands on
a real pack.  The old heuristic ("first frame past a threshold") fires on the
wrong event in ~5/23 episodes; a grasp is a transition, not a level."""
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
import scene

SUCCESS = Path("/workspace/shared/new_program_qiuzhi/without_tactile/success_episode/"
               "pick_up_the_tissue_pack_and_place_it_on_the_right_side_"
               "20260908_merged_all_108eps")
OUT = ROOT / "outputs/closure_events"
THUMB = "hand_L_thumb_tip"
FINGERS = ["hand_L_index_tip", "hand_L_middle_tip", "hand_L_ring_tip", "hand_L_pinky_tip"]
PITCH = [8, 10, 11, 13, 15]
CFG = ROOT / "configs/scene.yaml"


def main() -> int:
    import mujoco
    OUT.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(CFG.read_text())
    table = config["table"]["size"]
    model = mujoco.MjModel.from_xml_path(str(scene.build(
        CFG, OUT / "ref.xml", with_hand=True)))
    data = mujoco.MjData(model)
    ra = np.array(layout.joint_ids(model, mujoco))
    thumb, fingers = model.body(THUMB).id, [model.body(n).id for n in FINGERS]
    dataset, order = dataset_io.load(SUCCESS)

    def grip(q):
        data.qpos[ra] = q
        mujoco.mj_kinematics(model, data)
        g = 0.5 * (data.xpos[thumb]
                   + np.mean([data.xpos[i] for i in fingers], axis=0))
        return np.array([(g[0] + table[0] / 2) * 100.0, (g[1] + table[1] / 2) * 100.0])

    episodes = [0, 1, 2, 3, 5, 8, 10, 11]
    report = {}
    for e in episodes:
        state, action = dataset[e]["observation.state"], dataset[e]["action"]
        pitch = state[:, PITCH]
        closed = (np.abs(pitch) > 0.35).all(axis=1)
        # every open->closed transition, plus the level-change count
        if e in (2, 0):
            print("    pitch signal, every 6th frame (dims %s):" % PITCH)
            for t2 in range(0, len(pitch), 6):
                if np.abs(pitch[t2]).max() > 0.01 or 60 <= t2 <= 130:
                    print("      t=%-4d %s" % (t2, np.array2string(pitch[t2], precision=2)))
        starts = [t for t in range(1, len(closed)) if closed[t] and not closed[t - 1]]
        if closed[0]:
            starts = [0] + starts
        packs = align.measure_bags(align.dataset_frame(SUCCESS, e, 0), config)
        pack_cm = np.array([p["box_centre_cm"] for p in packs]) if packs else np.zeros((0, 2))
        print()
        print("=== ep%-3d  %d frames, hand closed in %d frames, %d open->closed event(s), "
              "%d packs detected" % (e, len(state), int(closed.sum()), len(starts), len(packs)))
        events = []
        for t in starts:
            g = grip(state[t])
            if len(pack_cm):
                d = np.linalg.norm(pack_cm - g, axis=1)
                j = int(np.argmin(d))
                near = "pack@(%.1f,%.1f) d=%.1f cm" % (pack_cm[j][0], pack_cm[j][1], d[j])
                dist = float(d[j])
            else:
                near, dist = "no packs detected", float("nan")
            events.append({"frame": t, "grip_cm": [round(float(g[0]), 2),
                                                   round(float(g[1]), 2)],
                           "nearest_cm": dist})
            print("    t=%-4d grip ( %7.2f, %7.2f ) cm   nearest %s"
                  % (t, g[0], g[1], near))
        old = np.where(np.abs(action[:, PITCH]).mean(1) > 0.35)[0]
        old_f = int(old[0]) if len(old) else -1
        if old_f >= 0:
            g = grip(state[old_f])
            d = (np.linalg.norm(pack_cm - g, axis=1).min() if len(pack_cm)
                 else float("nan"))
            picked = min(events, key=lambda r: r["nearest_cm"]) if events else None
            print("    OLD heuristic picked t=%d -> grip (%.2f, %.2f), nearest pack %.1f cm"
                  % (old_f, g[0], g[1], d))
            if picked:
                print("    BEST event (nearest pack) t=%d -> %.1f cm"
                      % (picked["frame"], picked["nearest_cm"]))
        report[e] = {"events": events, "old_frame": old_f}
    (OUT / "result.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
