#!/usr/bin/env python3
"""Feature schema, episode metadata schema, and whether lerobot is importable."""
import glob
import json
import os

D = ("/workspace/shared/new_program_qiuzhi/without_tactile/success_episode/"
     "pick_up_the_tissue_pack_and_place_it_on_the_right_side_"
     "20260908_merged_all_108eps")
info = json.load(open(D + "/meta/info.json"))
print("=== info.json top-level scalars ===")
for k, v in info.items():
    if k != "features":
        print("   %-24s %s" % (k, v))
print()
print("=== features ===")
for name, spec in info["features"].items():
    print("   %-30s dtype=%-8s shape=%s" % (name, spec.get("dtype"), spec.get("shape")))
print()
print("=== meta/episodes/ ===")
for p in sorted(glob.glob(D + "/meta/episodes/**", recursive=True))[:8]:
    print("   ", p.replace(D + "/", ""))
try:
    import pyarrow.parquet as pq
    ep = sorted(glob.glob(D + "/meta/episodes/**/*.parquet", recursive=True))
    if ep:
        t = pq.read_table(ep[0])
        print("   rows:", t.num_rows)
        for nm in t.schema.names:
            print("      %-28s %s" % (nm, t.schema.field(nm).type))
        print("   first episode keys sample:")
        row = t.slice(0, 1).to_pylist()[0]
        for k in list(row)[:40]:
            v = row[k]
            s = str(v)
            print("      %-40s %s" % (k, s[:70]))
except Exception as exc:
    print("   parquet read failed:", exc)
print()
print("=== data parquet schema ===")
f = sorted(glob.glob(D + "/data/**/*.parquet", recursive=True))[0]
import pyarrow.parquet as pq
t = pq.read_table(f)
print("   rows:", t.num_rows, " files:", len(glob.glob(D + "/data/**/*.parquet", recursive=True)))
for nm in t.schema.names:
    print("      %-28s %s" % (nm, t.schema.field(nm).type))
print()
print("=== tasks.parquet ===")
tp = pq.read_table(D + "/meta/tasks.parquet")
print("   rows:", tp.num_rows)
print("   ", tp.to_pylist()[:3])
print()
print("=== lerobot availability ===")
for mod in ("lerobot", "datasets", "pyarrow", "av", "torch"):
    try:
        m = __import__(mod)
        print("   %-10s OK  %s" % (mod, getattr(m, "__version__", "")))
    except Exception as exc:
        print("   %-10s MISSING (%s)" % (mod, type(exc).__name__))
