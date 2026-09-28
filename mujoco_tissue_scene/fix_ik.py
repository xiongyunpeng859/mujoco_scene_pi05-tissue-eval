#!/usr/bin/env python3
"""Stop the IK from choosing unreachable postures.

The old residual weighted the fingertip position at 10.0 but the pull toward a real
grasp posture at only 0.02 -- a 500:1 ratio, so the solver bought position accuracy with
any joint configuration it liked.  That is why the arm ended at the far left of the table
with its hand pointing up (section 31).

Fixes: regularise toward the SEED (the previous waypoint's solution, starting from a real
closure posture) at weight 1.0, keep a light anchor to q_ref, solve sequentially, and
report each waypoint's largest joint move so an unreachable waypoint is visible.
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path("/workspace/shared/mujoco_tissue_scene/reports/tools/scripted_pick_place.py")
text = SRC.read_text()

old_solve = '''    def solve(self, target_xy, target_z, want_yaw, q_ref):
        """Damped least squares on the six arm joints."""
        from scipy.optimize import least_squares
        tgt = np.array([target_xy[0], target_xy[1], target_z])
        want = np.array([np.cos(want_yaw), np.sin(want_yaw)])

        def residual(q):
            self.set_arm(q)
            p = self.tip_mid()
            c = self.closing_dir()
            horiz = c[:2] / (np.linalg.norm(c[:2]) + 1e-9)
            ang = np.arctan2(horiz[0] * want[1] - horiz[1] * want[0],
                             horiz @ want)
            return np.concatenate([(p - tgt) * 10.0, [ang * 1.5],
                                   (q - q_ref) * 0.02])

        res = least_squares(residual, q_ref, bounds=(self.lo, self.hi),
                            xtol=1e-10, ftol=1e-10, max_nfev=300)
        self.set_arm(res.x)
        err = float(np.linalg.norm(self.tip_mid() - tgt))
        return res.x, err'''

new_solve = '''    def solve(self, target_xy, target_z, want_yaw, seed, q_ref):
        """Damped least squares on the six arm joints.

        Regularised toward `seed` (the previous waypoint's solution) at a weight
        comparable to the position term, so the arm keeps a sane, reachable posture
        instead of buying fingertip accuracy with an arbitrary configuration.
        """
        from scipy.optimize import least_squares
        tgt = np.array([target_xy[0], target_xy[1], target_z])
        want = np.array([np.cos(want_yaw), np.sin(want_yaw)])

        def residual(q):
            self.set_arm(q)
            p = self.tip_mid()
            c = self.closing_dir()
            horiz = c[:2] / (np.linalg.norm(c[:2]) + 1e-9)
            ang = np.arctan2(horiz[0] * want[1] - horiz[1] * want[0],
                             horiz @ want)
            return np.concatenate([(p - tgt) * 10.0, [ang * 1.5],
                                   (q - seed) * 1.0, (q - q_ref) * 0.05])

        res = least_squares(residual, seed, bounds=(self.lo, self.hi),
                            xtol=1e-10, ftol=1e-10, max_nfev=400)
        self.set_arm(res.x)
        err = float(np.linalg.norm(self.tip_mid() - tgt))
        moved = float(np.max(np.abs(res.x - seed)))
        return res.x, err, moved'''

assert text.count(old_solve) == 1, "solve anchor %d" % text.count(old_solve)
text = text.replace(old_solve, new_solve)

old_loop = """        want_yaw = pyaw + np.pi / 2.0
        solutions, worst = [], 0.0
        for name, dz, hand, steps in PLAN:
            xy = (px, py) if name in ("approach", "descend", "close", "lift") \\
                else (bx_world, by_world)
            q, err = arm.solve(xy, ref[2] + dz, want_yaw, q_ref)
            worst = max(worst, err)
            solutions.append((q, hand, steps))"""
new_loop = """        want_yaw = pyaw + np.pi / 2.0
        solutions, worst, worst_move = [], 0.0, 0.0
        seed = q_ref.copy()
        moves = []
        for name, dz, hand, steps in PLAN:
            xy = (px, py) if name in ("approach", "descend", "close", "lift") \\
                else (bx_world, by_world)
            q, err, moved = arm.solve(xy, ref[2] + dz, want_yaw, seed, q_ref)
            seed = q.copy()                     # sequential: next waypoint starts here
            worst = max(worst, err)
            worst_move = max(worst_move, moved)
            moves.append("%s %.2f" % (name, moved))
            solutions.append((q, hand, steps))"""
assert text.count(old_loop) == 1, "loop anchor %d" % text.count(old_loop)
text = text.replace(old_loop, new_loop)

old_print = """        print("  %-4d (%.0f) (%6.1f,%6.1f) y%6.1f %8.4f %8s %9s %8.2f %8s  %s\""""
new_print = """        print("      waypoint joint moves (rad): %s" % "  ".join(moves))
        print("      worst joint move %.3f rad" % worst_move)
        print("  %-4d (%.0f) (%6.1f,%6.1f) y%6.1f %8.4f %8s %9s %8.2f %8s  %s\""""
assert text.count(old_print) == 1, "print anchor %d" % text.count(old_print)
text = text.replace(old_print, new_print)

SRC.write_text(text)
import ast
ast.parse(text)
print("IK regularisation fixed; syntax ok")
