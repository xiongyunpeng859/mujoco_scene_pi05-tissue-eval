"""Select minimal joint travel only among fully validated simulated rollouts.

This is an offline simulation teacher, not a real-robot trial-and-error policy.
Candidates must all begin from the same initial scene. A failed trial is never
used as the starting state of another candidate.
"""
import math


GRASP_FAMILIES = ('short_edges', 'diagonal', 'long_edges')


def relative_grasp_angle_deg(grasp_deg, object_deg):
    """Unsigned pinch-closing angle to the bag long axis, modulo 180 degrees."""
    return abs((float(grasp_deg) - float(object_deg) + 90.) % 180. - 90.)


def grasp_family(grasp_deg, object_deg):
    """Name the contacted edge pair; diagonal occupies the middle 45-degree band."""
    angle = relative_grasp_angle_deg(grasp_deg, object_deg)
    if angle <= 22.5:
        return 'short_edges'  # closing parallel to long axis; fingers touch short ends
    if angle >= 67.5:
        return 'long_edges'   # closing perpendicular; fingers touch the long sides
    return 'diagonal'


def select_candidate(rows, max_pregrasp_palm_deg=None, max_wrist_deg=None,
                     desired_family=None):
    if desired_family is not None and desired_family not in GRASP_FAMILIES:
        raise ValueError(f'unknown grasp family: {desired_family}')
    valid = []
    for row in rows:
        if row.get('exit_code') != 0:
            continue
        if not all(row.get(k) is True for k in
                   ('success','continuous_carry','settled_in_tray','home_ok','motion_ok')):
            continue
        cost = row.get('actual_joint_travel_total_rad',float('inf'))
        if not math.isfinite(cost) or cost < 0:
            continue
        if desired_family is not None:
            if 'yaw_deg' not in row or grasp_family(row['grasp_deg'], row['yaw_deg']) != desired_family:
                continue
        if max_pregrasp_palm_deg is not None:
            palm = row.get('pregrasp_palm_max_from_start_deg', float('inf'))
            if not math.isfinite(palm) or not 0 <= palm <= max_pregrasp_palm_deg:
                continue
        if max_wrist_deg is not None:
            fields = ('wrist_actual_max_from_start_deg', 'wrist_command_max_from_start_deg')
            if any(not isinstance(row.get(k), list) or len(row[k]) != 3 for k in fields):
                continue
            values = row[fields[0]] + row[fields[1]]
            if any(not isinstance(v, (int, float)) or not math.isfinite(v)
                   or not 0 <= v <= max_wrist_deg + 1e-7 for v in values):
                continue
            steps = row.get('actual_joint_max_step_deg', [])
            command_step = row.get('command_joint_max_step_deg', float('inf'))
            if not isinstance(steps, list) or len(steps) != 6:
                continue
            if any(not isinstance(v, (int, float)) or not math.isfinite(v)
                   or not 0 <= v <= 3. + 1e-7 for v in steps + [command_step]):
                continue
        valid.append(row)
    if not valid:
        return None
    return min(valid,key=lambda r:(r['actual_joint_travel_total_rad'],r['grasp_deg']))


def summarize_selection(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row['object_deg'],row['seed']),[]).append(row)
    summary = []
    for (angle,seed),candidates in sorted(groups.items()):
        choice = select_candidate(candidates)
        summary.append(dict(object_deg=angle,seed=seed,tested=len(candidates),
                            successful=sum(bool(r.get('success')) for r in candidates),
                            selected=choice))
    return summary
