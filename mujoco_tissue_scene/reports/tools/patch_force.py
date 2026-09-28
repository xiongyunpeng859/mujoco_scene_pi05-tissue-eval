#!/usr/bin/env python3
"""Contact FORCE on the target pack, not contact count.

Section 49: at descend -12 cm the fingers straddle the pack with 45 contacts and the pack
still rises only 7 mm.  A large count with near-zero normal force means the fingers straddle
it without pinching.  Sum the normal force over contacts involving the target flex.
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
        done.append("%s SKIP(%d)" % (tag, n))


sub("force-init",
    "        skin = {}\n",
    "        skin = {}\n        fsum = {}\n")

sub("force-sum",
    "            if _n:\n                hp[_seg] = max(hp.get(_seg, 0), _n)",
    "            _f = 0.0\n"
    "            for _k in range(mdata.ncon):\n"
    "                _ct = mdata.contact[_k]\n"
    "                if int(_ct.flex[0]) == fid or int(_ct.flex[1]) == fid:\n"
    "                    _f += float(np.linalg.norm(_ct.force[:3]))\n"
    "            fsum[_seg] = max(fsum.get(_seg, 0.0), _f)\n"
    "            if _n:\n                hp[_seg] = max(hp.get(_seg, 0), _n)")

sub("force-print",
    "        print(\"      hand<->target contacts per segment:",
    "        print(\"      peak contact force on the pack per segment (N): \"\n"
    "              + \"  \".join(\"%s %.2f\" % kv for kv in fsum.items()))\n"
    "        print(\"      hand<->target contacts per segment:")

SRC.write_text(t)
import ast
ast.parse(t)
print("patch:", done)
