#!/usr/bin/env python3
"""Align the simulation against a real dataset episode frame.

    python align_with_dataset.py --dataset <path> --episode 0 --frame 0 \
        --out outputs/alignment_dataset --place-bags-from-image

Unlike comparing against a hand-held photo, this compares the render with a frame
whose exact robot state is recorded, so the arm pose is identical by construction.
With --place-bags-from-image the tissue bags are measured in the real frame with
measure_layout's detector and written into the simulation, closing the loop:
real image -> measured object poses -> simulation -> rendered image.

Writes real.png, sim.png, blend.png, overlay.png and alignment.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure_layout                     # noqa: E402
import sim_env                            # noqa: E402

ROOT = Path(__file__).resolve().parent


def episode_metadata(dataset, episode):
    import pyarrow.parquet as pq
    files = sorted(glob.glob(str(Path(dataset) / "meta/episodes/**/*.parquet"), recursive=True))
    table = pq.read_table(files[0])
    rows = {name: table[name].to_pylist() for name in table.column_names}
    position = rows["episode_index"].index(episode)
    return {name: values[position] for name, values in rows.items()}


def dataset_frame(dataset, episode, frame_index, key="observation.images.top"):
    """Decode one frame of a real episode, honouring the video time offsets."""
    dataset = Path(dataset)
    info = json.loads((dataset / "meta/info.json").read_text())
    fps = float(info["fps"])
    meta = episode_metadata(dataset, episode)
    chunk = meta[f"videos/{key}/chunk_index"]
    file_index = meta[f"videos/{key}/file_index"]
    from_time = meta[f"videos/{key}/from_timestamp"]
    path = dataset / f"videos/{key}/chunk-{chunk:03d}/file-{file_index:03d}.mp4"
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(round(from_time * fps)) + int(frame_index))
    ok, image = capture.read()
    capture.release()
    if not ok:
        raise SystemExit("could not decode %s frame %d" % (path, frame_index))
    return image


def measure_bags(frame, config):
    """Bag centres in table cm, using the same detector as measure_layout."""
    table = measure_layout.TableFrame(config)
    blue = measure_layout.blue_dominance(frame)
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bag_size = config["boxes"][0]["size"]
    found = []
    table_cm = [config["table"]["size"][0] * 100.0, config["table"]["size"][1] * 100.0]
    # half of the bag's footprint, so a placement never overhangs the table edge
    half = [bag_size[0] * 50.0, bag_size[1] * 50.0]
    for seed in measure_layout.find_bag_seeds(frame):
        points = measure_layout.contact_points(grey, blue, seed)
        if len(points) < 4:
            continue
        result = measure_layout.bag_from_contact(table, points, bag_size)
        if not result:
            continue
        cx, cy = result["box_centre_cm"]
        if not (cx - half[0] > 1.0 and cx + half[0] < table_cm[0] - 1.0
                and cy - half[1] > 1.0 and cy + half[1] < table_cm[1] - 1.0):
            # measured off the tabletop: placing it would drop the bag out of the world
            result["rejected_off_table"] = True
            continue
        found.append(result)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/alignment_dataset")
    parser.add_argument("--camera", default="central")
    parser.add_argument("--place-bags-from-image", action="store_true")
    parser.add_argument("--no-distortion", action="store_true")
    args = parser.parse_args()

    import dataset_io
    import yaml
    args.out.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "configs/scene.yaml").read_text())

    real = dataset_frame(args.dataset, args.episode, args.frame)
    episodes, _ = dataset_io.load(args.dataset)
    state = episodes[args.episode]["observation.state"][args.frame]
    print("episode %d frame %d: state[:6] = %s" %
          (args.episode, args.frame, np.round(state[:6], 4).tolist()))

    env = sim_env.TissueSceneEnv(dataset=args.dataset, render=True,
                                 apply_distortion=not args.no_distortion,
                                 output_dir=args.out)
    obs, info = env.reset(options={"state": state, "randomize_objects": True})

    placement = None
    if args.place_bags_from_image:
        found = measure_bags(real, config)
        print("bags measured in the real frame: %d" % len(found))
        table_size_cm = [config["table"]["size"][0] * 100.0, config["table"]["size"][1] * 100.0]
        surface = config["table"]["surface_z"]
        for box, bag in zip(config["boxes"], found):
            x_cm, y_cm = bag["box_centre_cm"]
            yaw = np.radians(bag["yaw_deg"])
            qpos_adr, dof_adr = env.box_free[box["name"]]
            env.data.qpos[qpos_adr + 0] = x_cm / 100.0 - table_size_cm[0] / 200.0
            env.data.qpos[qpos_adr + 1] = y_cm / 100.0 - table_size_cm[1] / 200.0
            env.data.qpos[qpos_adr + 2] = surface + box["size"][2] / 2.0 + 0.001
            env.data.qpos[qpos_adr + 3] = np.cos(yaw / 2.0)
            env.data.qpos[qpos_adr + 6] = np.sin(yaw / 2.0)
            env.data.qvel[dof_adr:dof_adr + 6] = 0.0
            print("   %-16s -> (%.1f, %.1f) cm  yaw %.0f deg" %
                  (box["name"], x_cm, y_cm, bag["yaw_deg"]))
        env.mujoco.mj_forward(env.model, env.data)
        placement = [dict(box=config["boxes"][index]["name"], centre_cm=bag["box_centre_cm"],
                          yaw_deg=bag["yaw_deg"])
                     for index, bag in enumerate(found)]

    sim = env.render(args.camera)

    cv2.imwrite(str(args.out / "real.png"), real)
    cv2.imwrite(str(args.out / "sim.png"), cv2.cvtColor(sim, cv2.COLOR_RGB2BGR))
    sim_bgr = cv2.cvtColor(sim, cv2.COLOR_RGB2BGR)

    real_grey = cv2.cvtColor(real, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sim_grey = cv2.cvtColor(sim_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    diff = np.abs(real_grey - sim_grey)

    def edges(image):
        return cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0),
                         40, 110)
    kernel = np.ones((5, 5), np.uint8)
    real_edges, sim_edges = edges(real), edges(sim_bgr)
    sim_near = cv2.dilate(sim_edges, kernel) > 0
    real_near = cv2.dilate(real_edges, kernel) > 0
    matched_real = int(((real_edges > 0) & sim_near).sum())
    matched_sim = int(((sim_edges > 0) & real_near).sum())
    total_real = int((real_edges > 0).sum())
    total_sim = int((sim_edges > 0).sum())

    cv2.imwrite(str(args.out / "blend.png"),
                cv2.addWeighted(real, 0.5, sim_bgr, 0.5, 0.0))
    overlay = real.copy()
    overlay[sim_near] = (0.45 * overlay[sim_near] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
    overlay[real_edges > 0] = (0, 255, 0)
    cv2.imwrite(str(args.out / "overlay.png"), overlay)

    report = {
        "dataset": str(args.dataset), "episode": args.episode, "frame": args.frame,
        "camera": args.camera, "distortion_applied": not args.no_distortion,
        "bags_placed_from_image": placement,
        "mean_abs_grey_diff": round(float(diff.mean()), 2),
        "rmse_grey": round(float(np.sqrt((diff ** 2).mean())), 2),
        "real_edges_matching_sim_within_2px_percent":
            round(100.0 * matched_real / max(total_real, 1), 2),
        "sim_edges_matching_real_within_2px_percent":
            round(100.0 * matched_sim / max(total_sim, 1), 2),
    }
    (args.out / "alignment.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
