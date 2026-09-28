"""Reject stale or mismatched offline plans before writing training episodes."""
import hashlib
import json
import math
from pathlib import Path
from grasp_candidate_selection import GRASP_FAMILIES

ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = ["scene.py", "sim_env.py", "three_bag_round.py", "domain_randomization.py",
                "action_layout.py", "reports/tools/scripted_pick_place.py",
                "reports/tools/grasp_candidate_selection.py", "reports/tools/plan_three_bag_round.py",
                "reports/tools/grasp_plan_contract.py", "outputs/closure_library/library.json"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    return {name: digest(ROOT / name) for name in SOURCE_FILES}


def validate_collection_plan(path, config, seed, speed, episodes, bags, appearance):
    plan = json.loads(Path(path).read_text())
    if plan.get("complete") is not True or episodes != 3 or bags != 3 or not appearance:
        raise ValueError("Planned collection requires one complete three-pick appearance-randomized round")
    if plan.get("seed") != seed or plan.get("speed") != speed:
        raise ValueError("Plan seed/speed mismatch")
    if plan.get("config_sha256") != digest(config) or plan.get("source_sha256") != source_hashes():
        raise ValueError("Plan config/source changed; replan instead of collecting stale actions")
    yaws = plan.get("yaws", [])
    families = plan.get("families", [])
    limit = plan.get("max_pregrasp_palm_deg", float("nan"))
    if len(yaws) != 3 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in yaws):
        raise ValueError("Plan needs three finite grasp angles")
    if len(families) != 3 or any(family not in GRASP_FAMILIES for family in families):
        raise ValueError("Plan needs three recognized object-relative grasp families")
    if not isinstance(limit, (int, float)) or not math.isfinite(limit) or not 0 < limit <= 90:
        raise ValueError("Training collection palm limit must be at most 90 degrees")
    wrist = plan.get('max_wrist_deg', float('nan'))
    if not isinstance(wrist, (int, float)) or not math.isfinite(wrist) or not 1 < wrist <= 60:
        raise ValueError('Training collection requires a wrist excursion limit at most 60 degrees')
    return plan
