#!/usr/bin/env python3
"""Read a LeRobot v3 dataset into per-episode numpy arrays."""
from __future__ import annotations

from pathlib import Path

import numpy as np

FIELDS = ("action", "observation.state", "episode_index", "timestamp",
          "frame_index", "task_index")


def load(dataset, fields=FIELDS):
    """Return {episode_index: {field: ndarray}} plus the episode order."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    dataset = Path(dataset)
    files = sorted((dataset / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise SystemExit(f"no parquet under {dataset}/data")
    table = pa.concat_tables([pq.read_table(path) for path in files])
    present = [name for name in fields if name in table.column_names]
    # Grouping needs episode_index even when the caller did not ask for it.
    if "episode_index" not in present and "episode_index" in table.column_names:
        present.append("episode_index")
    columns = {name: table[name].to_pylist() for name in present}
    episode_index = np.asarray(columns["episode_index"], dtype=np.int64)
    float_fields = {"action", "observation.state", "timestamp"}
    episodes = {}
    for index in np.unique(episode_index):
        rows = np.where(episode_index == index)[0]
        entry = {}
        for name in present:
            if name == "episode_index":
                continue
            values = [columns[name][row] for row in rows]
            entry[name] = np.asarray(values, dtype=np.float64 if name in float_fields else np.int64)
        episodes[int(index)] = entry
    return episodes, sorted(episodes)


def frame_times(fps, count):
    return np.arange(count) / float(fps)
