#!/usr/bin/env python3
"""Seed the IK from the nearest REAL closure posture, and sample where the robot
demonstrably reaches."""
from pathlib import Path

SRC = Path("/workspace/shared/mujoco_tissue_scene/reports/tools/scripted_pick_place.py")
t = SRC.read_text()
if "closure_library" in t:
    print("already patched")
    raise SystemExit(0)

# load the library and clip the sampling region to the demonstrably reachable area
old = '    region = config["box_randomization"]["region_xy_cm_from_left_bottom"]'
new = '''    import json
    library = json.loads((ROOT / "outputs/closure_library/library.json").read_text())
    lib_x = np.array([p["x_cm"] for p in library])
    lib_y = np.array([p["y_cm"] for p in library])
    print("closure library: %d real grasp postures, x %.1f..%.1f, y %.1f..%.1f"
          % (len(library), lib_x.min(), lib_x.max(), lib_y.min(), lib_y.max()))
    region = config["box_randomization"]["region_xy_cm_from_left_bottom"]
    # the real demos only ever grasped inside the library's span; sampling outside it
    # asks the arm for poses nobody has shown it can reach
    region = {"x": [max(region["x"][0], float(lib_x.min()) - 2.0),
                    min(region["x"][1], float(lib_x.max()) + 2.0)],
              "y": [max(region["y"][0], float(lib_y.min()) - 2.0),
                    min(region["y"][1], float(lib_y.max()) + 2.0)]}'''
assert t.count(old) == 1, "region anchor %d" % t.count(old)
t = t.replace(old, new)

# per-episode: pick the nearest real posture and use its height
old = """        want_yaw = pyaw + np.pi / 2.0
        solutions, worst, worst_move = [], 0.0, 0.0
        seed = q_ref.copy()"""
new = """        want_yaw = pyaw + np.pi / 2.0
        near = int(np.argmin(np.hypot(lib_x - px_cm, lib_y - py_cm)))
        entry = library[near]
        seed = np.array(entry["q"], dtype=float)
        q_ref_ep = seed.copy()
        ref_z = float(entry["z_m"])
        solutions, worst, worst_move = [], 0.0, 0.0"""
assert t.count(old) == 1, "seed anchor %d" % t.count(old)
t = t.replace(old, new)

old = "            q, err, moved = arm.solve(xy, ref[2] + dz, want_yaw, seed, q_ref)"
new = "            q, err, moved = arm.solve(xy, ref_z + dz, want_yaw, seed, q_ref_ep)"
assert t.count(old) == 1, "solve anchor %d" % t.count(old)
t = t.replace(old, new)

old = '        print("      worst joint move between waypoints: %.3f rad" % worst_move)'
new = ('        print("      nearest real posture ep%d at (%.1f,%.1f) %.1f cm away; '
       'z %.3f" % (entry["episode"], entry["x_cm"], entry["y_cm"],\n'
       '              float(np.hypot(entry["x_cm"] - px_cm, entry["y_cm"] - py_cm)),\n'
       '              entry["z_m"]))\n'
       '        print("      worst joint move between waypoints: %.3f rad" % worst_move)')
if t.count(old) == 1:
    t = t.replace(old, new)
SRC.write_text(t)
import ast
ast.parse(t)
print("controller now seeds from the real closure library")
