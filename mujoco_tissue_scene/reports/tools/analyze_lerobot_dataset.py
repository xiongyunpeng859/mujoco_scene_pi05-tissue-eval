#!/usr/bin/env python3
"""Analyse a LeRobot v3 dataset and print every number the simulator must match.

    python analyze_lerobot_dataset.py --dataset <path> [--json out.json]

Reports the shared 16-D `action` / `observation.state` layout dimension by
dimension (range, spread, how many distinct values, whether it is constant),
the distribution of episode start states (used to randomise the reset pose),
per-step action deltas, episode lengths, tasks and camera streams.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def read_data(dataset: Path):
    import pyarrow.parquet as pq
    files = sorted((dataset / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise SystemExit(f"no parquet under {dataset}/data")
    tables = [pq.read_table(path) for path in files]
    import pyarrow as pa
    table = pa.concat_tables(tables)
    columns = {}
    for name in ("action", "observation.state", "episode_index", "frame_index",
                 "timestamp", "task_index"):
        if name in table.column_names:
            column = table[name].to_pylist()
            columns[name] = np.array(column, dtype=np.float64 if name in
                                     ("action", "observation.state", "timestamp") else np.int64)
    return columns, len(files)


def describe(matrix, names):
    rows = []
    for index, name in enumerate(names):
        column = matrix[:, index]
        unique = np.unique(np.round(column, 6))
        rows.append({
            "index": index,
            "name": name,
            "min": float(column.min()),
            "max": float(column.max()),
            "mean": float(column.mean()),
            "std": float(column.std()),
            "p01": float(np.percentile(column, 1)),
            "p99": float(np.percentile(column, 99)),
            "n_unique": int(unique.size),
            "constant": bool(unique.size == 1),
            "constant_value": float(unique[0]) if unique.size == 1 else None,
            "step": float(np.median(np.diff(unique))) if unique.size > 1 else None,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    dataset = args.dataset
    info = json.loads((dataset / "meta/info.json").read_text())
    columns, n_files = read_data(dataset)

    names = info["features"]["observation.state"]["names"]
    state, action = columns["observation.state"], columns["action"]
    episode = columns["episode_index"]

    report = {"dataset": str(dataset), "parquet_files": n_files,
              "codebase_version": info.get("codebase_version"),
              "robot_type": info.get("robot_type"),
              "total_episodes": info.get("total_episodes"),
              "total_frames": int(state.shape[0]), "fps": info.get("fps"),
              "cameras": {key: spec["shape"] for key, spec in info["features"].items()
                          if spec.get("dtype") == "video"},
              "action_names": info["features"]["action"]["names"]}

    print("dataset      : %s" % dataset)
    print("lerobot      : %s   robot_type=%s" % (info.get("codebase_version"), info.get("robot_type")))
    print("episodes     : %s   frames=%d   fps=%s" % (info.get("total_episodes"), state.shape[0], info.get("fps")))
    print("cameras      : %s" % report["cameras"])
    print()

    # action and state share one layout; verify that before trusting either.
    same_names = info["features"]["action"]["names"] == names
    report["action_and_state_same_layout"] = bool(same_names)
    print("action names == state names : %s" % same_names)
    tasks = json.loads((dataset / "meta/tasks.json").read_text()) if (dataset / "meta/tasks.json").is_file() else None
    if tasks is None:
        import pyarrow.parquet as pq
        tasks = json.loads((dataset / "meta/info.json").read_text()).get("tasks")
    if tasks:
        print("tasks        : %s" % tasks)
    print()

    state_rows = describe(state, names)
    action_rows = describe(action, names)
    report["state_per_dim"] = state_rows
    report["action_per_dim"] = action_rows

    print("--- observation.state / action per dimension ---")
    print("  %-3s %-18s %9s %9s %9s %9s %8s %s" %
          ("#", "name", "min", "max", "std", "p01..p99", "n_uniq", "note"))
    for s, a in zip(state_rows, action_rows):
        note = "CONSTANT %g" % s["constant_value"] if s["constant"] else ""
        if not s["constant"] and s["step"]:
            note += " step~%.5f" % s["step"]
        if abs(s["min"]) < 3 and abs(s["max"]) < 3 and s["n_unique"] > 50:
            pass
        print("  %-3d %-18s %9.4f %9.4f %9.4f %4.2f..%4.2f %8d %s" %
              (s["index"], s["name"], s["min"], s["max"], s["std"],
               s["p01"], s["p99"], s["n_unique"], note))
    print()

    constant = [row["name"] for row in state_rows if row["constant"]]
    varying = [row["name"] for row in state_rows if not row["constant"]]
    report["constant_dims"] = constant
    report["varying_dims"] = varying
    print("constant dims (%d): %s" % (len(constant), constant))
    print("varying  dims (%d): %s" % (len(varying), varying))
    print()

    # Episode start states drive reset randomisation.
    starts, lengths = [], []
    for index in np.unique(episode):
        rows = np.where(episode == index)[0]
        starts.append(state[rows[0]])
        lengths.append(rows.size)
    starts = np.array(starts)
    lengths = np.array(lengths)
    report["episodes"] = len(lengths)
    report["episode_length"] = {"min": int(lengths.min()), "median": float(np.median(lengths)),
                                "max": int(lengths.max()), "mean": float(lengths.mean())}
    print("episode length: min=%d median=%.0f max=%d" %
          (lengths.min(), np.median(lengths), lengths.max()))

    start_rows = []
    for index, name in enumerate(names):
        column = starts[:, index]
        start_rows.append({"name": name, "min": float(column.min()), "max": float(column.max()),
                           "mean": float(column.mean()), "std": float(column.std()),
                           "p05": float(np.percentile(column, 5)),
                           "p95": float(np.percentile(column, 95))})
    report["episode_start_state"] = start_rows
    print("--- episode start state (reset randomisation range) ---")
    print("  %-18s %9s %9s %9s %9s %9s" % ("name", "min", "max", "mean", "std", "p05..p95"))
    for row in start_rows:
        print("  %-18s %9.4f %9.4f %9.4f %9.4f %5.3f..%.3f" %
              (row["name"], row["min"], row["max"], row["mean"], row["std"], row["p05"], row["p95"]))
    print()

    # Action smoothness: how far the commanded target moves per control step.
    delta_rows = []
    for index, name in enumerate(names):
        inside = np.diff(action[:, index])[episode[1:] == episode[:-1]]
        delta_rows.append({"name": name, "mean_abs": float(np.abs(inside).mean()),
                           "max_abs": float(np.abs(inside).max())})
    report["action_step_delta"] = delta_rows
    print("--- |action[t]-action[t-1]| within episodes ---")
    for row in delta_rows:
        print("  %-18s mean=%.5f  max=%.5f" % (row["name"], row["mean_abs"], row["max_abs"]))
    print()

    task_hist = Counter(columns["task_index"].tolist())
    report["task_frame_counts"] = {str(k): v for k, v in sorted(task_hist.items())}
    print("frames per task_index: %s" % report["task_frame_counts"])

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
