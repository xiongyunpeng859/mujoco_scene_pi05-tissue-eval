#!/usr/bin/env python3
"""Constrain the hand's ORIENTATION explicitly.

Section 44: the IK constrains 3 position DOF + 1 yaw and leaves the wrist orientation free,
so the fingers do not point down at the pack and the skin never reaches it.  Section 27's
attempt to fix that by weight alone failed both ways (weight 1 -> reaches xy but wrong
orientation; weight 8 -> keeps orientation but misses xy by 4.7 cm).

So add the missing constraints.  The vector from the arm's base to the fingertip midpoint
encodes the reach DIRECTION; holding its direction to the seed's holds the arm -- and hence
the hand -- in the posture the real demonstration used, while still allowing the small
translation needed to meet the target.  That gives 3 position + 1 yaw + 2 direction = 6
constraints for 6 joints, i.e. a well-posed problem.
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
        done.append("%s SKIPPED(%d)" % (tag, n))


# back to the un-over-constrained posture weight
sub("weight-back",
    "(q - seed) * 8.0, (q - q_ref) * 0.05])",
    "(q - seed) * 1.0, (q - q_ref) * 0.05])")

sub("base-id",
    """        self.thumb = self.model.body(THUMB).id""",
    """        self.thumb = self.model.body(THUMB).id
        self.base = self.model.body("base_link").id""")

sub("ref-direction",
    """        def residual(q):
            self.set_arm(q)
            if hand is not None:""",
    """        # the reach direction of the seed posture: holding this holds the hand's
        # orientation, which is what the free wrist was losing
        self.set_arm(seed)
        if hand is not None:
            self.set_hand(hand)
            self.mujoco.mj_kinematics(self.model, self.data)
        _v0 = self.tip_mid() - self.data.xpos[self.base]
        ref_dir = _v0 / (np.linalg.norm(_v0) + 1e-9)

        def residual(q):
            self.set_arm(q)
            if hand is not None:""")

sub("direction-term",
    """            return np.concatenate([(p - tgt) * 10.0, [ang * 1.5],
                                   (q - seed) * 1.0, (q - q_ref) * 0.05])""",
    """            _v = p - self.data.xpos[self.base]
            _u = _v / (np.linalg.norm(_v) + 1e-9)
            return np.concatenate([(p - tgt) * 10.0, [ang * 1.5],
                                   (_u - ref_dir) * 3.0,
                                   (q - seed) * 1.0, (q - q_ref) * 0.05])""")

SRC.write_text(t)
import ast
ast.parse(t)
print("patch results:")
for d in done:
    print("   ", d)
