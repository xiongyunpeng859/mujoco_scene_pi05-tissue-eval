#!/usr/bin/env python3
"""Use grasp postures for the pack waypoints, release postures for the box waypoints,
and report IK residual + tracking error PER WAYPOINT."""
from pathlib import Path

SRC = Path("/workspace/shared/mujoco_tissue_scene/reports/tools/scripted_pick_place.py")
t = SRC.read_text()
if "release_lib" in t:
    print("already patched")
    raise SystemExit(0)

old = '''    import json
    library = json.loads((ROOT / "outputs/closure_library/library.json").read_text())
    lib_x = np.array([p["x_cm"] for p in library])
    lib_y = np.array([p["y_cm"] for p in library])
    print("closure library: %d real grasp postures, x %.1f..%.1f, y %.1f..%.1f"
          % (len(library), lib_x.min(), lib_x.max(), lib_y.min(), lib_y.max()))'''
new = '''    import json
    library = json.loads((ROOT / "outputs/closure_library/library.json").read_text())
    grasp_lib, release_lib = library["grasp"], library["release"]
    lib_x = np.array([p["x_cm"] for p in grasp_lib])
    lib_y = np.array([p["y_cm"] for p in grasp_lib])
    rel_x = np.array([p["x_cm"] for p in release_lib])
    rel_y = np.array([p["y_cm"] for p in release_lib])
    print("closure library: %d grasp postures (x %.1f..%.1f, y %.1f..%.1f) and "
          "%d release postures" % (len(grasp_lib), lib_x.min(), lib_x.max(),
                                   lib_y.min(), lib_y.max(), len(release_lib)))'''
assert t.count(old) == 1, "lib anchor %d" % t.count(old)
t = t.replace(old, new)

old = '''        want_yaw = pyaw + np.pi / 2.0
        near = int(np.argmin(np.hypot(lib_x - px_cm, lib_y - py_cm)))
        entry = library[near]
        seed = np.array(entry["q"], dtype=float)
        q_ref_ep = seed.copy()
        ref_z = float(entry["z_m"])
        solutions, worst, worst_move = [], 0.0, 0.0
        for name, dz, hand, steps in PLAN:
            xy = (px, py) if name in ("approach", "descend", "close", "lift") \\
                else (bx_world, by_world)
            q, err, moved = arm.solve(xy, ref_z + dz, want_yaw, seed, q_ref_ep)
            seed = q.copy()
            worst = max(worst, err)
            worst_move = max(worst_move, moved)
            solutions.append((q, hand, steps))
        print("      nearest real posture ep%d at (%.1f,%.1f) %.1f cm away; "
              "z %.3f" % (entry["episode"], entry["x_cm"], entry["y_cm"],
              float(np.hypot(entry["x_cm"] - px_cm, entry["y_cm"] - py_cm)),
              entry["z_m"]))
        print("      worst joint move between waypoints: %.3f rad" % worst_move)'''
new = '''        want_yaw = pyaw + np.pi / 2.0
        near = int(np.argmin(np.hypot(lib_x - px_cm, lib_y - py_cm)))
        entry = grasp_lib[near]
        bx_cm = (bx_world + ts[0] / 2.0) * 100.0
        by_cm = (by_world + ts[1] / 2.0) * 100.0
        near_r = int(np.argmin(np.hypot(rel_x - bx_cm, rel_y - by_cm)))
        rel_entry = release_lib[near_r]
        solutions, worst, worst_move = [], 0.0, 0.0
        seed = np.array(entry["q"], dtype=float)
        ref_z = float(entry["z_m"])
        seed_r = np.array(rel_entry["q"], dtype=float)
        ref_z_r = float(rel_entry["z_m"])
        per_way = []
        for name, dz, hand, steps in PLAN:
            over_pack = name in ("approach", "descend", "close", "lift")
            xy = (px, py) if over_pack else (bx_world, by_world)
            s0 = seed if over_pack else seed_r
            # the pack-aligned yaw is meaningless once the hand is over the box
            yaw_here = want_yaw if over_pack else float(tray["yaw"]) + np.pi / 2.0
            z_here = (ref_z if over_pack else ref_z_r) + dz
            q, err, moved = arm.solve(xy, z_here, yaw_here, s0, q_ref)
            seed = q.copy()
            worst = max(worst, err)
            worst_move = max(worst_move, moved)
            per_way.append((name, err, moved))
            solutions.append((q, hand, steps))
        print("      pack seed ep%d (%.1f,%.1f) %.1f cm away | box seed ep%d "
              "(%.1f,%.1f) %.1f cm from box"
              % (entry["episode"], entry["x_cm"], entry["y_cm"],
                 float(np.hypot(entry["x_cm"] - px_cm, entry["y_cm"] - py_cm)),
                 rel_entry["episode"], rel_entry["x_cm"], rel_entry["y_cm"],
                 float(np.hypot(rel_entry["x_cm"] - bx_cm, rel_entry["y_cm"] - by_cm))))
        print("      per-waypoint: " + "  ".join("%s ik%.3f mv%.2f" % w for w in per_way))'''
assert t.count(old) == 1, "waypoint anchor %d" % t.count(old)
t = t.replace(old, new)
SRC.write_text(t)
import ast
ast.parse(t)
print("controller uses grasp+release postures with per-waypoint diagnostics")
