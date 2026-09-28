"""Read-only real trajectory audit and timestamp-aligned static pose comparisons.

No policy or hardware is connected. Does not claim camera/dynamics calibration.
"""
import argparse
import json
from pathlib import Path

import scene
import hand_control
from smoke_chain import SimulationAdapter, JOINT_NAMES
import mujoco
import numpy as np
import pyarrow.parquet as pq
import cv2
import yaml
from PIL import Image, ImageDraw


def video_frame(path, seconds):
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        index = round(seconds * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Cannot decode {path} at {seconds}s")
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), index, fps
    finally:
        cap.release()


def set_pose(adapter, state):
    adapter.data.qpos[adapter.qpos_indices] = state[:6]
    adapter.data.qpos[adapter.hand_indices] = state[6:]
    # Static forward kinematics must explicitly populate URDF mimic coordinates.
    for index in range(adapter.model.neq):
        name = mujoco.mj_id2name(adapter.model, mujoco.mjtObj.mjOBJ_EQUALITY, index)
        if name and name.startswith("mimic_"):
            first, second = adapter.model.eq_obj1id[index], adapter.model.eq_obj2id[index]
            coeff = adapter.model.eq_data[index, :5]
            value = adapter.data.qpos[adapter.model.jnt_qposadr[second]]
            adapter.data.qpos[adapter.model.jnt_qposadr[first]] = np.polynomial.polynomial.polyval(value, coeff)
    adapter.data.qvel[:] = 0
    mujoco.mj_forward(adapter.model, adapter.data)


def run(root, output, episodes, config=None, mount_candidate=False):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    info = json.loads((root / "meta/info.json").read_text())
    for key in ["action", "observation.state"]:
        if info["features"][key]["names"] != JOINT_NAMES:
            raise ValueError(f"Unexpected {key} order")
    metadata = []
    for path in sorted((root / "meta/episodes").rglob("*.parquet")):
        metadata.extend(pq.read_table(path).to_pylist())
    rows = []
    for path in sorted((root / "data").rglob("*.parquet")):
        rows.extend(pq.read_table(path, columns=["episode_index", "frame_index", "timestamp", "observation.state", "action"]).to_pylist())
    state = np.asarray([r["observation.state"] for r in rows])
    action = np.asarray([r["action"] for r in rows])
    if state.shape[1:] != (16,) or not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("Invalid state/action shape or values")
    output.mkdir(parents=True)
    if mount_candidate:
        settings=yaml.safe_load(Path(config or scene.ROOT/"configs/scene.yaml").read_text())
        # Photo-constrained hypothesis: at real rest pose flange +Y is down,
        # flange +Z is distal. Native hand thickness is +X: rotate adapter +90deg.
        settings["arm"]["hand_mount_euler"]=[0,0,float(np.pi/2)]
        settings["wrist_camera"].update(position=[.02,.048,.14],euler=[2.80,0,0],bracket_anchor=[.02,.029,.118])
        config=output/"mount_candidate.yaml"
        config.write_text(yaml.safe_dump(settings,sort_keys=False))
    kwargs = {"config_path": config} if config else {}
    adapter = SimulationAdapter(mujoco.MjModel.from_xml_path(str(scene.build(output=output / "scene.xml", target_on_table=True, **kwargs))))
    poses = hand_control.poses()
    distances = np.stack([np.linalg.norm(action[:, 6:] - poses[k], axis=1) for k in ["open", "closed"]], axis=1)
    report = {"data_root": str(root), "frames": len(rows), "episodes": len(metadata),
              "joint_names": JOINT_NAMES, "state_min": state.min(0).tolist(), "state_max": state.max(0).tolist(),
              "action_min": action.min(0).tolist(), "action_max": action.max(0).tolist(),
              "action_state_arm_rmse": np.sqrt(np.mean((action[:, :6]-state[:, :6])**2, axis=0)).tolist(),
              "nearest_binary_hand_counts": {"open": int((distances.argmin(1)==0).sum()), "closed": int((distances.argmin(1)==1).sum())},
              "hand_actions_farther_than_0_15": int((distances.min(1) > .15).sum()),
              "comparison_frames": [], "calibration_status": "BASELINE_NOT_CALIBRATED",
              "limitations": ["Identity joint mapping is a hypothesis, not visually verified calibration",
                              "No automatic camera fit without matched 2D/3D landmarks",
                              "Objects remain at estimated locations, not tracked from real video",
                              "Static pose comparisons do not measure physical grasp or dynamics"]}
    try:
        for episode in episodes:
            erows = sorted([r for r in rows if r["episode_index"] == episode], key=lambda r: r["frame_index"])
            meta = next(m for m in metadata if m["episode_index"] == episode)
            ea = np.asarray([r["action"] for r in erows])
            d = np.stack([np.linalg.norm(ea[:,6:]-poses[k], axis=1) for k in ["open", "closed"]], axis=1)
            closed = d.argmin(1) == 1
            transitions = np.flatnonzero(closed & ~np.r_[False, closed[:-1]])
            indices = {0, len(erows)//2, len(erows)-1}
            for index in transitions[:1]:
                indices.update([max(0,int(index)-15), int(index), min(len(erows)-1,int(index)+30)])
            for index in sorted(indices):
                row = erows[index]
                set_pose(adapter, np.asarray(row["observation.state"]))
                obs = adapter.observe()
                canvas = Image.new("RGB", (1280, 1000), "white")
                draw = ImageDraw.Draw(canvas)
                record = {"episode": episode, "frame": int(row["frame_index"]), "timestamp": row["timestamp"],
                          "hand_world_position": adapter.data.xpos[adapter.model.body("hand_mount").id].tolist() if "hand_mount" in [mujoco.mj_id2name(adapter.model,mujoco.mjtObj.mjOBJ_BODY,i) for i in range(adapter.model.nbody)] else adapter.data.xpos[adapter.model.body("link6").id].tolist()}
                for camera_index, key in enumerate(["observation.images.top", "observation.images.left"]):
                    prefix = f"videos/{key}"
                    path = root / info["video_path"].format(video_key=key, chunk_index=meta[prefix+"/chunk_index"], file_index=meta[prefix+"/file_index"])
                    seconds = meta[prefix+"/from_timestamp"] + row["timestamp"]
                    real, video_index, fps = video_frame(path, seconds)
                    record[key] = {"video": str(path), "seconds": seconds, "video_frame": video_index, "fps": fps}
                    y = camera_index * 500
                    canvas.paste(real.resize((640,480)), (0,y+20))
                    canvas.paste(Image.fromarray(obs[key]), (640,y+20))
                    draw.text((5,y+3), f"REAL {key} ep={episode} frame={row['frame_index']}", fill="black")
                    draw.text((645,y+3), "SIM: recorded state, estimated camera", fill="black")
                name = f"episode_{episode:03d}_frame_{row['frame_index']:04d}.jpg"
                canvas.save(output/name)
                record["comparison"] = name
                report["comparison_frames"].append(record)
                print(name, flush=True)
        (output/"report.json").write_text(json.dumps(report,indent=2))
        html = "<meta charset='utf-8'><h1>真实 / 仿真姿态对齐基线（未标定）</h1>"
        html += "<p>左列真实，右列仿真；上排中央，下排腕部。物体尚未跟踪；不应据此评估抓取成功率。</p>"
        html += "".join(f"<h3>{r['comparison']}</h3><img width='100%' src='{r['comparison']}'>" for r in report["comparison_frames"])
        (output/"index.html").write_text(html)
        print(json.dumps({k:v for k,v in report.items() if k != "comparison_frames"},indent=2))
    finally:
        adapter.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", default=[0,54,107])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mount-candidate", action="store_true", help="Test photo-constrained mounting hypothesis; does not replace defaults")
    args = parser.parse_args()
    run(args.data_root,args.output_dir,args.episodes,args.config,args.mount_candidate)
