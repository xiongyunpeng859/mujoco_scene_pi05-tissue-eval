#!/usr/bin/env python3
"""Pin the base by how well the simulated arm overlaps the real arm.

Colour difference has a large irreducible floor (the vendor mesh does not look like
the real arm), so this uses overlap instead: over the tabletop, in the pixels the
simulated arm occupies, is the real image showing an object?  And conversely.  The
score is the F1 of those two masks, with the tray and the detected bags excluded.

The scene is rebuilt per candidate because the base lives in the XML, but only a few
frames are rendered, so a 27-candidate grid is affordable.
"""
from __future__ import annotations

import argparse
import itertools
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
import sim_env                          # noqa: E402

SUCCESS = Path("/workspace/shared/new_program_qiuzhi/without_tactile/success_episode/"
               "pick_up_the_tissue_pack_and_place_it_on_the_right_side_"
               "20260908_merged_all_108eps")
OUT = ROOT / "outputs/base_overlap"
FRAMES = (45, 60, 75, 90, 105, 120)
BRIGHT = 50          # the table renders ~20, the real arm ~79


def table_polygon(config, shape):
    table = measure_layout.TableFrame(config)
    size = table.size_cm
    points = []
    for x_cm, y_cm in ((2, 2), (size[0] - 2, 2), (size[0] - 2, size[1] - 2), (2, size[1] - 2)):
        p = table.project(table.from_cm(x_cm, y_cm))
        points.append([int(p[0]), int(p[1])])
    mask = np.zeros(shape.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.array(points), 255)
    return mask


def main() -> int:
    import mujoco
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["coarse", "fine"], default="coarse")
    parser.add_argument("--best", type=str, default=None)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
    episodes, order = dataset_io.load(SUCCESS)
    episode = order[0]
    states = episodes[episode]["observation.state"]
    reals = [align.dataset_frame(SUCCESS, episode, f) for f in FRAMES]
    for f, img in zip(FRAMES, reals):
        cv2.imwrite(str(OUT / ("real_%03d.png" % f)), img)
    real_bright = [(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > BRIGHT) for img in reals]

    # exclude the tray and the detected bags: they are bright but are not the arm
    exclude = np.zeros(reals[0].shape[:2], np.uint8)
    tray = config["tray"]
    size = config["table"]["size"]
    tf = measure_layout.TableFrame(config)
    centre = [(tray["center"][0] + size[0] / 2) * 100.0, (tray["center"][1] + size[1] / 2) * 100.0]
    quad = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        quad.append(tf.project(tf.from_cm(centre[0] + sx * 13.0, centre[1] + sy * 12.5)))
    cv2.fillConvexPoly(exclude, np.array(quad, np.int32), 255)
    for f, img in zip(FRAMES, reals):
        for bag in align.measure_bags(img, config):
            bx, by = bag["box_centre_cm"]
            p = tf.project(tf.from_cm(bx, by, 0.035))
            cv2.circle(exclude, (int(p[0]), int(p[1])), 42, 255, -1)
    allowed = (table_polygon(config, reals[0]) > 0) & (exclude == 0)
    print("frames %s ; allowed pixels over the table (tray and bags excluded): %d"
          % (FRAMES, allowed.sum()))

    def score(bx, by, byaw, tag):
        cfg = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
        cfg["measured_layout"]["arm_mount_xy_cm_from_left_bottom"] = [float(bx), float(by)]
        cfg["arm"]["euler"][2] = float(byaw)
        cdir = OUT / tag
        cdir.mkdir(parents=True, exist_ok=True)
        path = cdir / "scene.yaml"
        path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
        xml = scene.build(path, cdir / "noarm.xml", with_hand=False)
        model = mujoco.MjModel.from_xml_path(str(xml))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        with mujoco.Renderer(model, height=480, width=640) as renderer:
            renderer.update_scene(data, camera="central")
            no_arm = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        f1s = []
        for f, bright in zip(FRAMES, real_bright):
            env = sim_env.TissueSceneEnv(config_path=path, dataset=SUCCESS, render=True,
                                         output_dir=cdir)
            env.reset(options={"state": states[f], "randomize_objects": False})
            with_arm = cv2.cvtColor(env.render("central"), cv2.COLOR_RGB2BGR)
            env.close()
            difference = np.abs(with_arm.astype(np.float32)
                                - no_arm.astype(np.float32)).max(2)
            sim_mask = (difference > 20) & allowed
            real_mask = bright & allowed
            tp = float((sim_mask & real_mask).sum())
            precision = tp / max(sim_mask.sum(), 1)
            recall = tp / max(real_mask.sum(), 1)
            f1s.append(0.0 if precision + recall == 0 else
                       2 * precision * recall / (precision + recall))
        return float(np.mean(f1s))

    if args.stage == "coarse":
        candidates = [(x, y, np.radians(deg)) for x, y, deg in
                      itertools.product((26.0, 30.0, 34.0, 38.0, 42.0),
                                        (16.0, 20.0, 24.0, 28.0),
                                        (40.0, 50.0, 60.0, 70.0, 80.0))]
    else:
        bx, by, byaw = (float(v) for v in args.best.split(","))
        candidates = [(bx + dx, by + dy, byaw + np.radians(dd)) for dx, dy, dd in
                      itertools.product((-3.0, -1.5, 0.0, 1.5, 3.0),
                                        (-3.0, -1.5, 0.0, 1.5, 3.0),
                                        (-8.0, -4.0, 0.0, 4.0, 8.0))]

    print("evaluating %d candidates" % len(candidates))
    rows = []
    for x_cm, y_cm, yaw in candidates:
        tag = "x%05.1f_y%05.1f_a%06.4f" % (x_cm, y_cm, yaw)
        f1 = score(x_cm, y_cm, yaw, tag)
        rows.append({"x": float(x_cm), "y": float(y_cm), "yaw": float(yaw), "f1": f1})
    rows.sort(key=lambda r: -r["f1"])
    print()
    print("  %-30s %8s" % ("base / yaw", "F1"))
    for r in rows[:14]:
        print("  x=%5.1f y=%5.1f yaw=%6.1f (%5.1f deg) %8.4f"
              % (r["x"], r["y"], r["yaw"], np.degrees(r["yaw"]), r["f1"]))
    best = rows[0]
    print()
    print("  BEST x=%.1f y=%.1f yaw=%.1f deg (%.4f rad) -> F1 %.4f"
          % (best["x"], best["y"], np.degrees(best["yaw"]), best["yaw"], best["f1"]))
    (OUT / ("result_%s.json" % args.stage)).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
