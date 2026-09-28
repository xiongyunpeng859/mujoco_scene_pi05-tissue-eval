"""Hand commands: the two recorded reset gestures, or any continuous target.

The real robot receives an absolute joint-position target for all ten hand
joints on every control step.  Those targets come from the dataset's 16-D action
vector, dimensions 6..15, in exactly the HAND_JOINTS order below.  The recorded
"open"/"closed" gestures remain available for previews, but they are only two
points inside the reachable space, so the simulator must accept the continuous
values too.
"""
import json
from pathlib import Path

POSE_SOURCE = Path("/workspace/shared/o10-openpi-demo/arm-hand-teleop-o10-openpi-demo-stable/configs/reset_poses/o10_dual_reset.json")
HAND_JOINTS = ["L_thumb_roll_joint","L_thumb_abad_joint","L_thumb_mcp_joint",
               "L_index_abad_joint","L_index_pip_joint","L_middle_pip_joint",
               "L_ring_abad_joint","L_ring_pip_joint","L_pinky_abad_joint","L_pinky_pip_joint"]


def poses():
    values=json.loads(POSE_SOURCE.read_text())["gestures"]["cylindrical_straight"]["left"]
    return {state:[float(value) for value in values[state]] for state in ["open","closed"]}


def command(model, data, target):
    """Write the ten hand actuator targets.

    `target` is either the name of a recorded gesture ("open"/"closed") or a
    sequence of ten positions in HAND_JOINTS order -- the same order as dataset
    dimensions 6..15.
    """
    if isinstance(target, str):
        if target not in {"open", "closed"}:
            raise ValueError("Hand command must be open, closed, or ten joint positions")
        values = poses()[target]
    else:
        values = [float(value) for value in target]
        if len(values) != len(HAND_JOINTS):
            raise ValueError("Expected %d hand joint positions, got %d"
                             % (len(HAND_JOINTS), len(values)))
    for joint, value in zip(HAND_JOINTS, values):
        data.ctrl[model.actuator("hand_"+joint+"_position").id] = value
    return values
