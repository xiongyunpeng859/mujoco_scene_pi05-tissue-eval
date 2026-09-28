#!/usr/bin/env python3
"""The one place that maps the dataset's 16-D vector onto MuJoCo joints.

Dataset (`meta/info.json`) uses these names, in this order, for BOTH
`observation.state` and `action`:

    0..5    joint1.pos .. joint6.pos                     arm
    6..15   thumb_cm_roll, thumb_cm_yaw, thumb_cm_pitch,
            index_mp_yaw, index_mp_pitch, middle_mp_pitch,
            ring_mp_yaw, ring_mp_pitch, pinky_mp_yaw, pinky_mp_pitch

The simulation exposes the same information through six arm joints plus the ten
OmniHand joints listed below.  Keeping the correspondence in one module means the
environment, the checker and any analysis script cannot drift apart.
"""
from __future__ import annotations

DATASET_ARM_NAMES = ["joint%d.pos" % index for index in range(1, 7)]

# Dataset hand names, in dataset order.
DATASET_HAND_NAMES = [
    "thumb_cm_roll.pos",
    "thumb_cm_yaw.pos",
    "thumb_cm_pitch.pos",
    "index_mp_yaw.pos",
    "index_mp_pitch.pos",
    "middle_mp_pitch.pos",
    "ring_mp_yaw.pos",
    "ring_mp_pitch.pos",
    "pinky_mp_yaw.pos",
    "pinky_mp_pitch.pos",
]

DATASET_NAMES = DATASET_ARM_NAMES + DATASET_HAND_NAMES

# MuJoCo joint names, positionally aligned with DATASET_NAMES.
SIM_ARM_JOINTS = ["joint%d" % index for index in range(1, 7)]

# scene.py prefixes imported hand bodies/joints with "hand_".  The remaining
# OmniHand joints (thumb_pip/dip, index/middle/ring/pinky dip) are mimic joints
# and are not actuated, so they never appear in the 16-D vector.
HAND_JOINT_BASE = [
    "L_thumb_roll_joint",     # thumb_cm_roll
    "L_thumb_abad_joint",     # thumb_cm_yaw
    "L_thumb_mcp_joint",      # thumb_cm_pitch
    "L_index_abad_joint",     # index_mp_yaw
    "L_index_pip_joint",      # index_mp_pitch
    "L_middle_pip_joint",     # middle_mp_pitch
    "L_ring_abad_joint",      # ring_mp_yaw
    "L_ring_pip_joint",       # ring_mp_pitch
    "L_pinky_abad_joint",     # pinky_mp_yaw
    "L_pinky_pip_joint",      # pinky_mp_pitch
]

SIM_HAND_JOINTS = ["hand_" + name for name in HAND_JOINT_BASE]

SIM_JOINTS = SIM_ARM_JOINTS + SIM_HAND_JOINTS

# Actuator names produced by scene.py.
ARM_ACTUATORS = ["arm_joint%d_position" % index for index in range(1, 7)]
HAND_ACTUATORS = ["hand_" + name + "_position" for name in HAND_JOINT_BASE]
ACTUATORS = ARM_ACTUATORS + HAND_ACTUATORS

# Dataset indices of the hand joints the policy actually modulates.  Every other
# hand dimension is written once per episode and then held, matching the data
# (their within-episode |delta| is exactly zero).
MODULATED_HAND_INDICES = [8, 10, 11, 13, 15]
HELD_HAND_INDICES = [6, 7, 9, 12, 14]

# Open/closed endpoints, read from the real reset gesture file by hand_control.
OPEN_PITCH = 0.0
CLOSED_PITCH = {"thumb_cm_pitch.pos": -0.7}
CLOSED_PITCH_DEFAULT = 0.7


def split(vector):
    """Split a 16-D dataset vector into (arm, hand) parts."""
    values = [float(value) for value in vector]
    if len(values) != len(DATASET_NAMES):
        raise ValueError("expected %d values, got %d" % (len(DATASET_NAMES), len(values)))
    return values[:6], values[6:]


def actuator_ids(model, mujoco):
    """Ctrl indices in dataset order, so ctrl[ids] = action works directly."""
    ids = []
    for name in ACTUATORS:
        actuator = model.actuator(name)
        if actuator is None:
            raise KeyError("actuator %r missing from the model" % name)
        ids.append(actuator.id)
    return ids


def joint_ids(model, mujoco):
    ids = []
    for name in SIM_JOINTS:
        joint = model.joint(name)
        if joint is None:
            raise KeyError("joint %r missing from the model" % name)
        ids.append(joint.qposadr[0])
    return ids
