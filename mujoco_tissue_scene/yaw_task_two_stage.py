#!/usr/bin/env python3
"""Two-stage task-level search: yaw first, then the base position at that yaw."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import arm_yaw_task_search as search     # noqa: E402


def report(title, rows):
    print("  %-24s %10s %9s %9s  %s" % ("candidate", "closest(m)", "contacts", "in tray",
                                        "per-bag closest"))
    scored = []
    for x_cm, y_cm, yaw, result in rows:
        print("  x=%5.1f y=%5.1f a=%.4f %10.4f %9d %9s  %s"
              % (x_cm, y_cm, yaw, result["min_closest"], result["contacts"],
                 any(result["contained"].values()),
                 {k[:9]: round(v, 3) for k, v in result["closest"].items()}))
        # rank by: task success, then contacts, then closest approach
        scored.append((any(result["contained"].values()), result["contacts"],
                       -result["min_closest"], (x_cm, y_cm, yaw), result))
    scored.sort(reverse=True)
    best = scored[0]
    print("  --> best: x=%.1f y=%.1f yaw=%.4f  (in tray %s, contacts %d, closest %.4f)"
          % (*best[3], best[0], best[1], -best[2]))
    return best[3]


def main() -> int:
    print("=== stage 1: base yaw, wide range ===")
    stage1 = [(25.0, 10.0, yaw) for yaw in
              (1.5708, 1.70, 1.80, 1.90, 2.00, 2.10, 2.25, 2.40, 2.60)]
    rows = [(x, y, a, search.run_candidate(x, y, a)) for x, y, a in stage1]
    best_yaw = report("yaw", rows)[2]

    print()
    print("=== stage 2: base x, y at yaw %.4f ===" % best_yaw)
    stage2 = [(x, y, best_yaw) for x in (10.0, 18.0, 25.0, 32.0, 40.0)
              for y in (0.0, 10.0, 20.0)]
    rows = [(x, y, a, search.run_candidate(x, y, a)) for x, y, a in stage2]
    best = report("base", rows)

    print()
    print("=== summary ===")
    print("  chosen: base x=%.1f cm  y=%.1f cm  yaw=%.4f rad (%.1f deg)"
          % (best[0], best[1], best[2], __import__("math").degrees(best[2])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
