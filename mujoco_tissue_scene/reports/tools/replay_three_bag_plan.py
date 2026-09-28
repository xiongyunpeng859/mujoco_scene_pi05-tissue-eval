"""Independently replay a completed simulation-only grasp plan and encode previews."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from grasp_candidate_selection import grasp_family, select_candidate

ROOT = Path(__file__).resolve().parents[2]


def load_plan(path):
    plan = json.loads(path.read_text())
    yaws = plan.get("yaws", [])
    if plan.get("complete") is not True or len(yaws) != 3:
        raise ValueError("Replay requires a completed three-pick plan")
    if not all(isinstance(yaw, (int, float)) and math.isfinite(yaw) for yaw in yaws):
        raise ValueError("Grasp angles must be finite")
    config = Path(plan["config"])
    if not config.is_absolute() or not config.is_file():
        raise ValueError("Plan must reference an existing absolute config path")
    return plan


def verify_replay(rows, yaws, max_pregrasp_palm_deg=None, max_wrist_deg=None, families=None):
    if len(rows) != 3:
        raise RuntimeError("Replay must produce exactly three episodes")
    for index, (row, yaw) in enumerate(zip(rows, yaws)):
        candidate = dict(row, exit_code=0, grasp_deg=yaw)
        desired = families[index] if families is not None else None
        if select_candidate([candidate], max_pregrasp_palm_deg, max_wrist_deg, desired) is None or row.get("obstacle_contact_frames") != 0:
            raise RuntimeError(f"Replay failed physical acceptance at pick {index}")
        if desired is not None and grasp_family(yaw,row['yaw_deg']) != desired:
            raise RuntimeError(f"Replay grasp family mismatch at pick {index}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = load_plan(args.plan)
    args.output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(MUJOCO_GL="egl", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    command = [sys.executable, "-u", str(ROOT / "reports/tools/scripted_pick_place.py"),
               "--config", plan["config"], "--seed", str(plan["seed"]),
               "--speed", "1.5", "--bags-per-round", "3", "--randomize-appearance",
               "--episodes", "3", "--grasp-yaw-plan", *map(str, plan["yaws"]),
               "--save", "--save-every", "3", "--output-dir", str(args.output.resolve())]
    if 'max_wrist_deg' in plan:
        command += ['--max-wrist-deg', str(plan['max_wrist_deg'])]
    with (args.output / "run.log").open("w") as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    rows = json.loads((args.output / "result.json").read_text())
    verify_replay(rows, plan["yaws"], plan.get("max_pregrasp_palm_deg"),
                  plan.get('max_wrist_deg'), plan.get('families'))
    for camera in ("top", "wrist"):
        subprocess.run(["/usr/bin/ffmpeg", "-nostdin", "-n", "-v", "error",
                        "-framerate", "10", "-pattern_type", "glob",
                        "-i", str(args.output / f"ep??_{camera}_*.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        str(args.output / f"{camera}.mp4")], check=True)
    subprocess.run(["/usr/bin/ffmpeg", "-nostdin", "-n", "-v", "error",
                    "-i", str(args.output / "top.mp4"), "-i", str(args.output / "wrist.mp4"),
                    "-filter_complex", "hstack", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(args.output / "dual.mp4")], check=True)
    (args.output / "replay_verified.json").write_text(json.dumps({
        "source_plan": str(args.plan.resolve()), "plan": plan,
        "accepted_episodes": 3, "video": "dual.mp4", "video_fps": 10,
        "note": "Preview samples every third control frame plus the final frame; control remains 30 Hz.",
    }, indent=2))
    print("REPLAY_OK", args.output / "dual.mp4", flush=True)


if __name__ == "__main__":
    main()
