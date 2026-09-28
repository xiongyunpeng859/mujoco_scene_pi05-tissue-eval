#!/usr/bin/env python3
"""The IK was aiming with the WRONG HAND POSE.

`Arm.set_arm()` sets only the six arm joints, so the IK model's ten hand joints stay at
their defaults while the environment's hand is driven to the episode's real hand values
(the thumb yaw alone spans 1.36-1.64 rad).  The fingertip midpoint therefore differs by
several centimetres between the two, which is exactly the 4-6 cm offset that survived every
kp/force/seed change.  Set the hand in the IK model to the same values that will be
commanded.
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


sub("hand-qadr",
    """        self.arm_qadr = np.array([self.model.joint(n).qposadr[0]
                                  for n in arm_joint_names])""",
    """        self.arm_qadr = np.array([self.model.joint(n).qposadr[0]
                                  for n in arm_joint_names])
        self.hand_qadr = np.array([self.model.joint(n).qposadr[0]
                                   for n in layout.SIM_HAND_JOINTS])""")

sub("set-hand",
    """    def set_arm(self, q):
        self.data.qpos[self.arm_qadr] = q
        self.mujoco.mj_kinematics(self.model, self.data)""",
    """    def set_arm(self, q):
        self.data.qpos[self.arm_qadr] = q
        self.mujoco.mj_kinematics(self.model, self.data)

    def set_hand(self, h):
        \"\"\"The hand pose moves the fingertips by centimetres, so the IK must use the
        same hand pose that will actually be commanded.\"\"\"
        self.data.qpos[self.hand_qadr] = np.asarray(h, dtype=float)""")

sub("solve-signature",
    "    def solve(self, target_xy, target_z, want_yaw, seed, q_ref):",
    "    def solve(self, target_xy, target_z, want_yaw, seed, q_ref, hand=None):")

sub("solve-set-hand",
    """        def residual(q):
            self.set_arm(q)
            p = self.tip_mid()""",
    """        def residual(q):
            self.set_arm(q)
            if hand is not None:
                self.set_hand(hand)
                self.mujoco.mj_kinematics(self.model, self.data)
            p = self.tip_mid()""")

sub("solve-initial-hand",
    """        res = least_squares(residual, seed, bounds=(self.lo, self.hi),
                            xtol=1e-10, ftol=1e-10, max_nfev=400)
        self.set_arm(res.x)""",
    """        if hand is not None:
            self.set_hand(hand)
            self.mujoco.mj_kinematics(self.model, self.data)
        res = least_squares(residual, seed, bounds=(self.lo, self.hi),
                            xtol=1e-10, ftol=1e-10, max_nfev=400)
        self.set_arm(res.x)
        if hand is not None:
            self.set_hand(hand)
            self.mujoco.mj_kinematics(self.model, self.data)""")

sub("call-site",
    "            q, err, moved = arm.solve(xy, z_here, yaw_here, s0, q_ref)",
    "            q, err, moved = arm.solve(\n"
    "                xy, z_here, yaw_here, s0, q_ref,\n"
    "                hand=hand_open if hand == \"open\" else hand_closed)")

SRC.write_text(t)
import ast
ast.parse(t)
print("patch results:")
for d in done:
    print("   ", d)
