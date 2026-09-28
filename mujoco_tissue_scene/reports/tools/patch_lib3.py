#!/usr/bin/env python3
"""Release-posture seeding for the box waypoints + PER-WAYPOINT diagnostics.

Each replacement is applied independently and reported, so one bad anchor cannot void
the whole patch (that is what happened twice before).
"""
from pathlib import Path

SRC = Path("/workspace/shared/mujoco_tissue_scene/reports/tools/scripted_pick_place.py")
t = SRC.read_text()
done = []


def sub(tag, old, new):
    global t
    n = t.count(old)
    if n == 1:
        t = t.replace(old, new)
        done.append(tag)
    else:
        done.append("%s SKIPPED(count=%d)" % (tag, n))


sub("release-arrays",
    '    release_lib = _lib["release"]\n    library = _lib["grasp"]',
    '    release_lib = _lib["release"]\n    library = _lib["grasp"]\n'
    '    rel_x = np.array([p["x_cm"] for p in release_lib])\n'
    '    rel_y = np.array([p["y_cm"] for p in release_lib])')

sub("waypoints",
    '''        want_yaw = pyaw + np.pi / 2.0
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
            solutions.append((q, hand, steps))''',
    '''        want_yaw = pyaw + np.pi / 2.0
        near = int(np.argmin(np.hypot(lib_x - px_cm, lib_y - py_cm)))
        entry = library[near]
        bx_cm = (bx_world + ts[0] / 2.0) * 100.0
        by_cm = (by_world + ts[1] / 2.0) * 100.0
        near_r = int(np.argmin(np.hypot(rel_x - bx_cm, rel_y - by_cm)))
        rel = release_lib[near_r]
        box_yaw = float(tray["yaw"]) + np.pi / 2.0
        seed_pack = np.array(entry["q"], dtype=float)
        seed_box = np.array(rel["q"], dtype=float)
        ref_z_pack = float(entry["z_m"])
        ref_z_box = float(rel["z_m"])
        solutions, worst, per_way = [], 0.0, []
        for name, dz, hand, steps in PLAN:
            over_pack = name in ("approach", "descend", "close", "lift")
            xy = (px, py) if over_pack else (bx_world, by_world)
            s0 = seed_pack if over_pack else seed_box
            # the pack-aligned yaw means nothing once the hand is over the box
            yaw_here = want_yaw if over_pack else box_yaw
            z_here = (ref_z_pack if over_pack else ref_z_box) + dz
            q, err, moved = arm.solve(xy, z_here, yaw_here, s0, q_ref)
            if over_pack:
                seed_pack = q.copy()
            else:
                seed_box = q.copy()
            worst = max(worst, err)
            per_way.append((name, err, moved))
            solutions.append((q, hand, steps, name))''')

sub("waypoint-print",
    '''        print("      worst joint move between waypoints: %.3f rad" % worst_move)''',
    '''        print("      pack seed ep%d (%.1f,%.1f) %.1f cm away | box seed ep%d "
              "(%.1f,%.1f) %.1f cm from the box"
              % (entry["episode"], entry["x_cm"], entry["y_cm"],
                 float(np.hypot(entry["x_cm"] - px_cm, entry["y_cm"] - py_cm)),
                 rel["episode"], rel["x_cm"], rel["y_cm"],
                 float(np.hypot(rel["x_cm"] - bx_cm, rel["y_cm"] - by_cm))))
        print("      per-waypoint IK: " + "  ".join("%s %.3f" % (w[0], w[1])
                                                    for w in per_way))''')

sub("action-segments",
    '''        actions, prev = [], np.concatenate([q_ref, hand_open])
        for q, hand, steps in solutions:
            target16 = np.concatenate([q, hand_vec[hand]])
            for row in smooth(prev, target16, steps):
                actions.append(row)
            prev = target16''',
    '''        actions, prev, seg_bounds = [], np.concatenate([q_ref, hand_open]), []
        for q, hand, steps, wname in solutions:
            target16 = np.concatenate([q, hand_vec[hand]])
            seg_start = len(actions)
            for row in smooth(prev, target16, steps):
                actions.append(row)
            seg_bounds.append((wname, seg_start, len(actions)))
            prev = target16''')

sub("per-waypoint-tracking",
    '''        for value in actions:
            obs, _, _, _, info = env.step(value)
            err = np.abs(np.asarray(info["tracking_error"]))''',
    '''        seg_track = {}
        for step_i, value in enumerate(actions):
            obs, _, _, _, info = env.step(value)
            err = np.abs(np.asarray(info["tracking_error"]))
            for wname, a0, a1 in seg_bounds:
                if a0 <= step_i < a1:
                    seg_track[wname] = max(seg_track.get(wname, 0.0), float(err.max()))
                    break''')

sub("tracking-print",
    '''        print("  %-4d (%.0f) (%6.1f,%6.1f) y%6.1f %8.4f %8s %9s %8.2f %8s  %s"''',
    '''        print("      tracking err per waypoint: " + "  ".join(
            "%s %.2f" % (k, v) for k, v in seg_track.items()))
        print("  %-4d (%.0f) (%6.1f,%6.1f) y%6.1f %8.4f %8s %9s %8.2f %8s  %s"''')

SRC.write_text(t)
import ast
ast.parse(t)
print("patch results:")
for d in done:
    print("   ", d)
