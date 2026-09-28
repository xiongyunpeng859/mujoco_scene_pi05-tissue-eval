#!/usr/bin/env python3
"""Pin the arm base using a consensus image from many real episodes.

Every episode starts from the same arm pose, so the median of many first frames
erases the randomly placed bags and leaves a clean picture of the static scene
including the arm.  Matching candidate bases against that consensus is far more
sensitive than matching one frame, because the only thing that changes with the
base is the arm silhouette.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import dataset_io                       # noqa: E402
import sim_env                          # noqa: E402

SUCCESS_DATASET = Path("/workspace/shared/new_program_qiuzhi/without_tactile/success_episode/"
                       "pick_up_the_tissue_pack_and_place_it_on_the_right_side_"
                       "20260908_merged_all_108eps")
OUT = ROOT / "outputs/base_consensus"
# the idle second arm sits here in every frame; blank it so it cannot flatter a wrong base
IDLE = (400, 640, 240, 480)


def consensus_first_frames(dataset, limit):
    """Median of the first frame of many episodes, plus their mean start state."""
    import pyarrow.parquet as pq
    info = json.loads((dataset / "meta/info.json").read_text())
    fps = float(info["fps"])
    key = "observation.images.top"
    meta_files = sorted(glob.glob(str(dataset / "meta/episodes/**/*.parquet"), recursive=True))
    table = pq.read_table(meta_files[0])
    rows = {name: table[name].to_pylist() for name in table.column_names}
    index_of = {value: i for i, value in enumerate(rows["episode_index"])}

    episodes, order = dataset_io.load(dataset, fields=("observation.state",))
    chosen = order[:limit]
    frames, states = [], []
    by_file = {}
    for episode in chosen:
        i = index_of[episode]
        chunk = rows[f"videos/{key}/chunk_index"][i]
        file_index = rows[f"videos/{key}/file_index"][i]
        when = rows[f"videos/{key}/from_timestamp"][i]
        by_file.setdefault((chunk, file_index), []).append((episode, when))
        states.append(episodes[episode]["observation.state"][0])

    for (chunk, file_index), items in by_file.items():
        path = dataset / f"videos/{key}/chunk-{chunk:03d}/file-{file_index:03d}.mp4"
        capture = cv2.VideoCapture(str(path))
        for episode, when in sorted(items, key=lambda item: item[1]):
            capture.set(cv2.VideoCapture_PROP_POS_FRAMES if False else cv2.CAP_PROP_POS_FRAMES,
                        int(round(when * fps)))
            ok, image = capture.read()
            if ok:
                frames.append(image.astype(np.float32))
        capture.release()
    return np.median(np.stack(frames), axis=0), np.array(states), len(frames)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=SUCCESS_DATASET)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--stage", choices=["coarse", "fine"], default="coarse")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    median, states, count = consensus_first_frames(args.dataset, args.limit)
    median = median.astype(np.uint8)
    cv2.imwrite(str(OUT / "consensus_first_frame.png"), median)
    start_state = np.median(states, axis=0)
    spread = np.abs(states - start_state).max(0)
    print("built consensus from %d first frames" % count)
    print("start-state spread across episodes (max |dev| per dim): %s"
          % np.round(spread, 4).tolist())
    print("-> arm pose is essentially identical every episode: %s"
          % bool(spread[:6].max() < 0.05))
    np.save(OUT / "start_state.npy", start_state)

    grey = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY).astype(np.float32)
    x0, x1, y0, y1 = IDLE
    grey[y0:y1, x0:x1] = 0.0

    if args.stage == "coarse":
        candidates = [(x, y, yaw) for x in (14.0, 18.0, 22.0, 26.0)
                      for y in (2.0, 6.0, 10.0, 14.0)
                      for yaw in (1.45, 1.5708, 1.70)]
    else:
        candidates = [(x, y, 1.5708) for x in np.arange(17.0, 23.1, 1.0)
                      for y in np.arange(3.0, 9.1, 1.0)]

    print()
    print("  %-24s %10s %10s" % ("candidate", "mean|diff|", "arm-region"))
    rows = []
    for x_cm, y_cm, yaw in candidates:
        config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())
        config["measured_layout"]["arm_mount_xy_cm_from_left_bottom"] = [x_cm, y_cm]
        config["arm"]["euler"][2] = yaw
        tag = "x%05.1f_y%05.1f_a%06.4f" % (x_cm, y_cm, yaw)
        path = OUT / tag / "scene.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
        env = sim_env.TissueSceneEnv(config_path=path, dataset=args.dataset, render=True,
                                     output_dir=path.parent)
        env.reset(options={"state": start_state, "randomize_objects": False})
        sim = cv2.cvtColor(env.render("central"), cv2.COLOR_RGB2BGR)
        env.close()
        sim_grey = cv2.cvtColor(sim, cv2.COLOR_BGR2GRAY).astype(np.float32)
        sim_grey[y0:y1, x0:x1] = 0.0
        diff = np.abs(grey - sim_grey)
        overall = float(diff.mean())
        # arm region: where either image differs from the table's flat black
        rows.append((x_cm, y_cm, yaw, overall))
        print("  x=%5.1f y=%5.1f a=%.4f %10.2f" % (x_cm, y_cm, yaw, overall))
    rows.sort(key=lambda r: r[3])
    best = rows[0]
    print()
    print("  best: x=%.1f y=%.1f yaw=%.4f -> mean|diff| %.2f" % best[:4])
    (OUT / "result.json").write_text(json.dumps(
        [{"x_cm": r[0], "y_cm": r[1], "yaw": r[2], "mean_abs_diff": r[3]} for r in rows],
        indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
