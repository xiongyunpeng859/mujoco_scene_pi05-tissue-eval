#!/usr/bin/env python3
"""Derive the target pack's yaw from the hand instead of the tray.

Fingers pinch across a flat pack, so the pack's long axis runs PERPENDICULAR to the
thumb->fingertip closing direction.  Setting every pack's yaw to the tray's yaw was
arbitrary and changes which side of the pack is pinched.
"""
from pathlib import Path
import shutil

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
SRC = ROOT / "validate_kinematic.py"
text = SRC.read_text()
if "target_yaw" in text:
    print("already patched")
    raise SystemExit(0)

old_fk = """        g = 0.5 * (rd.xpos[thumb] + np.mean([rd.xpos[i] for i in fingers], axis=0))
        target_cm[e] = np.array([(g[0] + ts[0] / 2) * 100.0,
                                 (g[1] + ts[1] / 2) * 100.0])"""
new_fk = """        finger_mid = np.mean([rd.xpos[i] for i in fingers], axis=0)
        g = 0.5 * (rd.xpos[thumb] + finger_mid)
        target_cm[e] = np.array([(g[0] + ts[0] / 2) * 100.0,
                                 (g[1] + ts[1] / 2) * 100.0])
        # pinch closes across the pack, so the long axis is perpendicular to it
        closing = finger_mid[:2] - rd.xpos[thumb][:2]
        norm = float(np.linalg.norm(closing))
        if norm > 1e-6:
            perp = np.array([-closing[1], closing[0]]) / norm
            target_yaw[e] = float(np.arctan2(perp[1], perp[0]))
        else:
            target_yaw[e] = float(tyaw)"""
assert text.count(old_fk) == 1
text = text.replace(old_fk, new_fk)

old_init = "    target_cm = {}\n"
assert text.count(old_init) == 1
text = text.replace(old_init, "    target_cm = {}\n    target_yaw = {}\n")

old_q = """        data.qpos[adr + 3] = np.cos(tyaw / 2.0)
        data.qpos[adr + 4:adr + 7] = 0.0
        data.qvel[dof:dof + 6] = 0.0
        # other packs at their detected positions"""
new_q = """        data.qpos[adr + 3] = np.cos(target_yaw[e] / 2.0)
        data.qpos[adr + 4:adr + 6] = 0.0
        data.qpos[adr + 6] = np.sin(target_yaw[e] / 2.0)
        data.qvel[dof:dof + 6] = 0.0
        # other packs at their detected positions"""
assert text.count(old_q) == 1
text = text.replace(old_q, new_q)

old_print = ("        print(\"  %-4d (%6.1f,%6.1f) %8.4f %8s %9s %8.2f %8s\"\n"
             "              % (e, cx, cy, peak[best],")
new_print = ("        print(\"  %-4d (%6.1f,%6.1f) y%6.1f %8.4f %8s %9s %8.2f %8s\"\n"
             "              % (e, cx, cy, np.degrees(target_yaw[e]), peak[best],")
assert text.count(old_print) == 1
text = text.replace(old_print, new_print)
text = text.replace('("ep", "target cm", "peak", "nearest", "grasped", "in-tray", "placed")',
                    '("ep", "target cm", "yaw", "peak", "nearest", "grasped", "in-tray", "placed")')
shutil.copy(SRC, SRC.with_suffix(".py.bak_before_yaw"))
SRC.write_text(text)
print("validate_kinematic.py: target yaw now from the closing direction")
