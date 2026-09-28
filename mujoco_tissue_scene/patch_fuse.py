#!/usr/bin/env python3
"""Fuse the two target estimates.

The FK closure point is reliable but assumes the fingers closed on the pack's CENTRE;
the user says the real grasp position varies per episode, so the pack's centre can be up
to half its length (6 cm) away from that point.  The detector measures the centre
directly but cannot always say WHICH detection is the target.  So: use the FK closure to
pick which detection is the target, then take that detection's centre as the pose.
"""
from pathlib import Path
import shutil

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
SRC = ROOT / "validate_kinematic.py"
text = SRC.read_text()
if "target_source" in text:
    print("already patched")
    raise SystemExit(0)

old = "    target_cm = {}\n    target_yaw = {}\n"
new = ("    target_cm = {}\n    target_yaw = {}\n    target_source = {}\n"
       "    all_packs = {}\n")
assert text.count(old) == 1
text = text.replace(old, new)

old = """        target_cm[e] = np.array([(g[0] + ts[0] / 2) * 100.0,
                                 (g[1] + ts[1] / 2) * 100.0])"""
new = """        target_cm[e] = np.array([(g[0] + ts[0] / 2) * 100.0,
                                 (g[1] + ts[1] / 2) * 100.0])
        # the image gives the pack's real CENTRE, capturing an off-centre grasp
        packs = align.measure_bags(align.dataset_frame(SUCCESS, e, 0), config)
        all_packs[e] = packs
        if args.target_source == "fused" and packs:
            dist = [np.hypot(p["box_centre_cm"][0] - target_cm[e][0],
                             p["box_centre_cm"][1] - target_cm[e][1]) for p in packs]
            j = int(np.argmin(dist))
            if dist[j] < 15.0:
                target_cm[e] = np.array(packs[j]["box_centre_cm"], dtype=float)
                target_source[e] = "image (+%.1f cm)" % dist[j]
            else:
                target_source[e] = "FK closure"
        else:
            target_source[e] = "FK closure\""""
assert text.count(old) == 1
text = text.replace(old, new)

old = '    parser.add_argument("--target-yaw", choices=["tray", "closing"],\n                        default="tray")\n'
new = (old + '    parser.add_argument("--target-source", choices=["fk", "fused"],\n'
       '                        default="fused")\n')
assert text.count(old) == 1
text = text.replace(old, new)

old = """        bags = align.measure_bags(align.dataset_frame(SUCCESS, e, 0), config)
        # the FK closure point and the detector agree to ~2.5 cm, so a detection"""
new = """        bags = all_packs.get(e) or []
        # the FK closure point and the detector agree to ~2.5 cm, so a detection"""
assert text.count(old) == 1
text = text.replace(old, new)

old = '        print("  %-4d (%6.1f,%6.1f) y%6.1f %8.4f %8s %9s %8.2f %8s"\n              % (e, cx, cy, np.degrees(target_yaw[e]), peak[best],'
new = '        print("  %-4d (%6.1f,%6.1f) y%6.1f %8.4f %8s %9s %8.2f %8s  %s"\n              % (e, cx, cy, np.degrees(target_yaw[e]), peak[best],'
assert text.count(old) == 1
text = text.replace(old, new)
old = '                 grasped, frac, placed), flush=True)'
new = '                 grasped, frac, placed, target_source.get(e, "")), flush=True)'
assert text.count(old) == 1
text = text.replace(old, new)
text = text.replace('("ep", "target cm", "yaw", "peak", "nearest", "grasped", "in-tray", "placed")',
                    '("ep", "target cm", "yaw", "peak", "nearest", "grasped", "in-tray", "placed", "source")')

shutil.copy(SRC, SRC.with_suffix(".py.bak_before_fuse"))
SRC.write_text(text)
import ast
ast.parse(text)
print("fused target source installed; syntax ok")
