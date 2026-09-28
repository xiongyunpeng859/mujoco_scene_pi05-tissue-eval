"""Interactive checkerboard capture for the O10 left-wrist D405.

With ``--arm-port``, the program records six arm joint angles and T_base_eef
without enabling, disabling, changing control mode, or commanding the arm.
Without it, saving is blocked unless explicitly enabled for camera diagnostics.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import time
from urllib.parse import urlparse

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_SERIAL = "260322276846"


@dataclass(frozen=True)
class DetectionQuality:
    found: bool
    sharpness: float
    coverage: float
    minimum_margin_px: float
    stability_rms_px: float | None
    accepted: bool
    reasons: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default=DEFAULT_SERIAL)
    parser.add_argument(
        "--columns", type=int, default=7,
        help="Checkerboard INNER corners across the image (default: 7).",
    )
    parser.add_argument(
        "--rows", type=int, default=8,
        help="Checkerboard INNER corners down the image (default: 8).",
    )
    parser.add_argument(
        "--square-size-mm", type=float, default=None,
        help="Measured square edge length. Required later for metric pose solving.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--exposure-us", type=float, default=14000.0)
    parser.add_argument("--gain", type=float, default=16.0)
    parser.add_argument("--auto-exposure", action="store_true")
    parser.add_argument(
        "--sharpness-min", type=float, default=40.0,
        help="Minimum Laplacian sharpness (relaxed default: 40).",
    )
    parser.add_argument(
        "--margin-min", type=float, default=3.0,
        help="Minimum inner-corner distance from an image edge in pixels (default: 3).",
    )
    parser.add_argument(
        "--stability-max", type=float, default=3.0,
        help="Maximum corner motion across recent frames in pixels (default: 3).",
    )
    parser.add_argument(
        "--stability-frames", type=int, default=3,
        help="Number of consecutive detections used for stability (default: 3).",
    )
    parser.add_argument(
        "--web", action="store_true",
        help="Use a browser interface instead of a local desktop window.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Web UI listen address.")
    parser.add_argument("--port", type=int, default=8765, help="Web UI listen port.")
    parser.add_argument(
        "--smoke-seconds", type=float, default=0.0,
        help="Run camera/detection without a UI for this many seconds, then exit.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="New session directory. Defaults to a timestamped outputs directory.",
    )
    parser.add_argument(
        "--arm-port", default=None,
        help=(
            "Record the six arm joints and T_base_eef from this CAN interface "
            "without enabling, disabling, changing mode, or commanding the arm."
        ),
    )
    parser.add_argument(
        "--min-joint-change-rad", type=float, default=0.05,
        help=(
            "Reject a hand-eye sample when every joint is within this wrapped "
            "distance of an existing sample (default: 0.05 rad)."
        ),
    )
    parser.add_argument(
        "--allow-camera-only-save", action="store_true",
        help=(
            "Explicitly allow diagnostic images without robot poses. These images "
            "cannot be used as complete hand-eye samples."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.columns < 3 or args.rows < 3:
        raise ValueError("--columns and --rows must both be at least 3")
    if args.square_size_mm is not None and args.square_size_mm <= 0:
        raise ValueError("--square-size-mm must be positive")
    if min(args.width, args.height, args.fps, args.stability_frames) <= 0:
        raise ValueError("resolution, fps and stability frames must be positive")
    if min(
        args.sharpness_min,
        args.margin_min,
        args.stability_max,
        args.min_joint_change_rad,
    ) < 0:
        raise ValueError("quality thresholds cannot be negative")
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be between 1 and 65535")
    if args.smoke_seconds < 0:
        raise ValueError("--smoke-seconds cannot be negative")


def _canonicalize_corners(corners: np.ndarray) -> np.ndarray:
    """Keep detector fallbacks from returning the same grid in reverse order."""
    points = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    if points[0].sum() > points[-1].sum():
        points = points[::-1].copy()
    return points


def detect_checkerboard(gray: np.ndarray, pattern: tuple[int, int]):
    """Detect every inner corner, with robust fallbacks for difficult views."""
    if gray.ndim != 2:
        raise ValueError("detect_checkerboard expects one grayscale image")
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Fast path for the normal case. Keeping this first avoids making the live
    # preview unnecessarily slow when the checkerboard is already clear.
    found, corners = cv2.findChessboardCornersSB(
        gray,
        pattern,
        flags=cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if found:
        return True, _canonicalize_corners(corners)

    # Local contrast helps with D405 auto/manual exposure, shadows and glare.
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    robust_flags = (
        cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_EXHAUSTIVE
        | cv2.CALIB_CB_ACCURACY
    )
    for candidate in (gray, enhanced):
        found, corners = cv2.findChessboardCornersSB(
            candidate,
            pattern,
            flags=robust_flags,
        )
        if found:
            return True, _canonicalize_corners(corners)

    # The classic detector sometimes succeeds on steep perspective views that
    # the sector-based detector rejects. Sub-pixel refinement keeps these
    # fallback corners suitable for pose estimation.
    classic_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    for candidate in (gray, enhanced):
        found, corners = cv2.findChessboardCorners(
            candidate,
            pattern,
            flags=classic_flags,
        )
        if found:
            corners = cv2.cornerSubPix(
                candidate,
                corners,
                (5, 5),
                (-1, -1),
                (
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.01,
                ),
            )
            return True, _canonicalize_corners(corners)
    return False, None


def assess_detection(
    gray: np.ndarray,
    corners: np.ndarray | None,
    corner_history: deque[np.ndarray],
    *,
    sharpness_min: float,
    margin_min: float,
    stability_max: float,
    stability_frames: int,
) -> DetectionQuality:
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if corners is None:
        return DetectionQuality(
            found=False,
            sharpness=sharpness,
            coverage=0.0,
            minimum_margin_px=0.0,
            stability_rms_px=None,
            accepted=False,
            reasons=("complete 7x8 pattern not found; keep all 56 inner corners visible",),
        )

    height, width = gray.shape
    minimum = corners.min(axis=0)
    maximum = corners.max(axis=0)
    coverage = float(
        (maximum[0] - minimum[0]) * (maximum[1] - minimum[1])
        / (width * height)
    )
    margin = float(min(minimum[0], minimum[1], width - 1 - maximum[0], height - 1 - maximum[1]))
    compatible = [past for past in corner_history if past.shape == corners.shape]
    stability = None
    if len(compatible) >= stability_frames - 1:
        recent = compatible[-(stability_frames - 1):]
        errors = [np.sqrt(np.mean(np.square(corners - past))) for past in recent]
        stability = float(max(errors, default=0.0))

    reasons: list[str] = []
    if sharpness < sharpness_min:
        reasons.append(f"blur {sharpness:.1f} < {sharpness_min:.1f}")
    if margin < margin_min:
        reasons.append(f"border margin {margin:.1f}px < {margin_min:.1f}px")
    if stability is None:
        reasons.append("hold still")
    elif stability > stability_max:
        reasons.append(f"motion {stability:.2f}px > {stability_max:.2f}px")
    return DetectionQuality(
        found=True,
        sharpness=sharpness,
        coverage=coverage,
        minimum_margin_px=margin,
        stability_rms_px=stability,
        accepted=not reasons,
        reasons=tuple(reasons),
    )


def configure_sensor(profile, args: argparse.Namespace, rs_module) -> dict[str, float | bool]:
    rs = rs_module
    sensors = list(profile.get_device().query_sensors())
    sensor = next(
        (
            candidate
            for candidate in sensors
            if any(
                stream_profile.stream_type() == rs.stream.color
                for stream_profile in candidate.get_stream_profiles()
            )
        ),
        None,
    )
    if sensor is None:
        names = [
            candidate.get_info(rs.camera_info.name)
            for candidate in sensors
        ]
        raise RuntimeError(
            f"No sensor exposing a color stream; available sensors: {names}"
        )
    settings: dict[str, float | bool] = {}
    if sensor.supports(rs.option.enable_auto_exposure):
        sensor.set_option(rs.option.enable_auto_exposure, 1.0 if args.auto_exposure else 0.0)
        settings["auto_exposure"] = bool(args.auto_exposure)
    if not args.auto_exposure and sensor.supports(rs.option.exposure):
        exposure_range = sensor.get_option_range(rs.option.exposure)
        exposure = float(np.clip(args.exposure_us, exposure_range.min, exposure_range.max))
        sensor.set_option(rs.option.exposure, exposure)
        settings["exposure_us"] = exposure
    if sensor.supports(rs.option.gain):
        gain_range = sensor.get_option_range(rs.option.gain)
        gain = float(np.clip(args.gain, gain_range.min, gain_range.max))
        sensor.set_option(rs.option.gain, gain)
        settings["gain"] = gain
    return settings


def make_session_directory(args: argparse.Namespace) -> Path:
    output = args.output_dir
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = ROOT / "outputs" / "calibration" / f"left_wrist_checkerboard_{stamp}"
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse existing session: {output}")
    output.mkdir(parents=True)
    return output


def intrinsics_payload(intrinsics) -> dict:
    return {
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "cx": float(intrinsics.ppx),
        "cy": float(intrinsics.ppy),
        "distortion_model": str(intrinsics.model).split(".")[-1],
        "distortion_coefficients": [float(value) for value in intrinsics.coeffs],
    }


class ReadOnlyArmState:
    """Read O10 joint feedback and FK without issuing a control command."""

    def __init__(self, port: str) -> None:
        package_root = Path(
            "/workspace/shared/o10-openpi-demo/arm-hand-teleop-o10-openpi-demo-stable/"
            "qiuzhi/lerobot_play_1.0.4/x86/noble"
        )
        for child in (
            package_root / "lerobot_play-1.0.4-py3-none-any",
            package_root / "mmk2_kdl_py-0.1.4-py3-none-any",
        ):
            if not child.is_dir():
                raise RuntimeError(f"Required O10 package directory is missing: {child}")
            if str(child) not in sys.path:
                sys.path.insert(0, str(child))

        import airbot_hardware_py as ah
        from mmk2_kdl_py import ArmKdlNumerical

        self.port = port
        self._arm = ah.Play.create(
            ah.MotorType.OD,
            ah.MotorType.OD,
            ah.MotorType.OD,
            ah.MotorType.DM,
            ah.MotorType.DM,
            ah.MotorType.DM,
            ah.EEFType.NA,
            ah.MotorType.NA,
        )
        self._executor = ah.create_asio_executor(8)
        if not self._arm.init(self._executor.get_io_context(), port, 250):
            raise RuntimeError(f"Failed to initialize read-only arm feedback on {port}")
        self._initialized = True
        # Deliberately do not call enable(), disable(), set_param(), pvt() or mit().
        self._kinematics = ArmKdlNumerical(eef_type="none")
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            state = self._arm.state()
            if bool(state.is_valid) and len(list(state.pos)) >= 6:
                self.snapshot()
                return
            time.sleep(0.02)
        self.close()
        raise RuntimeError(f"No valid six-joint feedback received from {port}")

    def snapshot(self) -> dict:
        state = self._arm.state()
        if not bool(state.is_valid):
            raise RuntimeError("Arm feedback is not valid")
        joints = np.asarray(list(state.pos)[:6], dtype=float)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise RuntimeError("Arm did not return six finite joint angles")
        base_to_eef = np.asarray(
            self._kinematics.forward_kinematics(joints), dtype=float
        )
        if base_to_eef.shape != (4, 4) or not np.isfinite(base_to_eef).all():
            raise RuntimeError("Forward kinematics did not return a finite 4x4 transform")
        return {
            "arm_state_unix_ns": time.time_ns(),
            "arm_joint_positions_rad": joints.tolist(),
            "T_base_eef": base_to_eef.tolist(),
        }

    def close(self) -> None:
        if getattr(self, "_initialized", False):
            self._arm.uninit()
            self._initialized = False


def robot_pose_or_rejection(
    arm_reader: ReadOnlyArmState | None,
    *,
    allow_camera_only_save: bool,
) -> tuple[dict | None, str | None]:
    if arm_reader is None:
        if allow_camera_only_save:
            return {
                "arm_state_unix_ns": None,
                "arm_joint_positions_rad": None,
                "T_base_eef": None,
            }, None
        return None, (
            "robot pose unavailable; restart with --arm-port can0, or use "
            "--allow-camera-only-save only for diagnostics"
        )
    try:
        return arm_reader.snapshot(), None
    except RuntimeError as error:
        return None, f"robot pose unavailable: {error}"


def repeated_arm_pose_rejection(
    robot_pose: dict,
    records: list[dict],
    *,
    min_joint_change_rad: float,
) -> str | None:
    """Reject a pose unless at least one joint differs from every prior pose."""
    joints = robot_pose.get("arm_joint_positions_rad")
    if joints is None or not records or min_joint_change_rad == 0:
        return None
    current = np.asarray(joints, dtype=float)
    nearest = float("inf")
    nearest_index = None
    for record in records:
        previous_values = record.get("arm_joint_positions_rad")
        if previous_values is None:
            continue
        previous = np.asarray(previous_values, dtype=float)
        wrapped_difference = np.abs(
            (current - previous + np.pi) % (2.0 * np.pi) - np.pi
        )
        distance = float(wrapped_difference.max())
        if distance < nearest:
            nearest = distance
            nearest_index = record.get("sample_index")
    if nearest_index is not None and nearest < min_joint_change_rad:
        return (
            "arm pose repeats sample "
            f"{int(nearest_index):03d}: max joint change {nearest:.4f} rad < "
            f"{min_joint_change_rad:.4f} rad; keep the board fixed and move the arm"
        )
    return None


def draw_overlay(
    bgr: np.ndarray,
    pattern: tuple[int, int],
    corners: np.ndarray | None,
    quality: DetectionQuality,
    saved_count: int,
    message: str,
) -> np.ndarray:
    preview = bgr.copy()
    if corners is not None:
        cv2.drawChessboardCorners(
            preview, pattern, corners.reshape(-1, 1, 2), True
        )
    color = (40, 210, 40) if quality.accepted else (30, 80, 240)
    status = "READY - press S to save" if quality.accepted else "NOT READY"
    lines = [
        f"Pattern: {pattern[0]}x{pattern[1]} INNER corners",
        f"Status: {status}",
        f"Sharpness: {quality.sharpness:.1f}  Coverage: {100 * quality.coverage:.1f}%",
        "Reason: " + (", ".join(quality.reasons) if quality.reasons else "quality gates passed"),
        f"Saved: {saved_count}   S/Space save   D delete last   Q/Esc quit",
    ]
    if message:
        lines.append(message)
    overlay_height = 24 * len(lines) + 8
    cv2.rectangle(preview, (0, 0), (preview.shape[1], overlay_height), (0, 0, 0), -1)
    for index, line in enumerate(lines):
        cv2.putText(
            preview, line, (10, 22 + 24 * index), cv2.FONT_HERSHEY_SIMPLEX,
            0.53, color if index == 1 else (235, 235, 235), 1, cv2.LINE_AA,
        )
    return preview


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


WEB_PAGE = b"""<!doctype html>
<html><head><meta charset="utf-8"><title>O10 left wrist calibration</title>
<style>
body{font-family:sans-serif;background:#181818;color:#eee;text-align:center;margin:16px}
img{max-width:96vw;border:2px solid #555;background:#000}
button{font-size:18px;margin:8px;padding:10px 24px} #status{white-space:pre-wrap;margin:10px}
</style></head><body>
<h2>O10 left wrist checkerboard capture</h2>
<img id="view" src="/frame.jpg"><div id="status">Starting...</div>
<button onclick="act('/save')">Save</button><button onclick="act('/delete')">Delete last</button>
<button onclick="act('/shutdown')">Quit</button>
<script>
const view=document.getElementById('view'), status=document.getElementById('status');
setInterval(()=>{view.src='/frame.jpg?t='+Date.now()},100);
setInterval(()=>fetch('/status.json').then(r=>r.json()).then(x=>status.textContent=JSON.stringify(x,null,2)).catch(()=>{}),500);
function act(path){fetch(path,{method:'POST'}).then(r=>r.json()).then(x=>status.textContent=JSON.stringify(x,null,2));}
</script></body></html>"""


def run_smoke_capture(pipeline, pattern, history, args) -> None:
    deadline = time.monotonic() + args.smoke_seconds
    frames_seen = 0
    detections = 0
    accepted = 0
    last_quality = None
    while time.monotonic() < deadline:
        color_frame = pipeline.wait_for_frames(3000).get_color_frame()
        if not color_frame:
            continue
        rgb = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        _, corners = detect_checkerboard(gray, pattern)
        quality = assess_detection(
            gray, corners, history,
            sharpness_min=args.sharpness_min,
            margin_min=args.margin_min,
            stability_max=args.stability_max,
            stability_frames=args.stability_frames,
        )
        frames_seen += 1
        detections += int(corners is not None)
        accepted += int(quality.accepted)
        last_quality = quality
        if corners is None:
            history.clear()
        else:
            history.append(corners.copy())
    if frames_seen == 0:
        raise RuntimeError("Smoke test received no color frames")
    print(json.dumps({
        "status": "PASS_CAMERA_DETECTION_LOOP",
        "frames_seen": frames_seen,
        "complete_pattern_detections": detections,
        "quality_accepted_frames": accepted,
        "last_quality": None if last_quality is None else asdict(last_quality),
    }, indent=2))


def run_web_capture(
    pipeline,
    pattern,
    history,
    args,
    output: Path,
    records: list[dict],
    arm_reader: ReadOnlyArmState | None,
) -> None:
    lock = threading.Lock()
    stop = threading.Event()
    state: dict[str, object | None] = {
        "bgr": None, "corners": None, "quality": None,
        "camera_timestamp_ms": None, "preview_jpeg": None,
        "message": "Starting camera", "message_until": 0.0,
        "error": None,
    }

    def notify(message: str) -> None:
        with lock:
            state["message"] = message
            state["message_until"] = time.monotonic() + 2.0

    def capture_loop() -> None:
        try:
            while not stop.is_set():
                color_frame = pipeline.wait_for_frames(3000).get_color_frame()
                if not color_frame:
                    continue
                rgb = np.asanyarray(color_frame.get_data())
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                _, corners = detect_checkerboard(gray, pattern)
                quality = assess_detection(
                    gray, corners, history,
                    sharpness_min=args.sharpness_min,
                    margin_min=args.margin_min,
                    stability_max=args.stability_max,
                    stability_frames=args.stability_frames,
                )
                if corners is None:
                    history.clear()
                else:
                    history.append(corners.copy())
                with lock:
                    if time.monotonic() > float(state["message_until"] or 0.0):
                        state["message"] = ""
                    message = str(state["message"] or "")
                preview = draw_overlay(bgr, pattern, corners, quality, len(records), message)
                encoded, jpeg = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 88])
                if not encoded:
                    raise RuntimeError("Failed to encode preview JPEG")
                with lock:
                    state.update({
                        "bgr": bgr.copy(),
                        "corners": None if corners is None else corners.copy(),
                        "quality": quality,
                        "camera_timestamp_ms": float(color_frame.get_timestamp()),
                        "preview_jpeg": jpeg.tobytes(),
                    })
        except BaseException as error:
            with lock:
                state["error"] = repr(error)
            stop.set()

    def save_latest() -> tuple[int, dict]:
        with lock:
            bgr = state["bgr"]
            corners = state["corners"]
            quality = state["quality"]
            timestamp_ms = state["camera_timestamp_ms"]
            if isinstance(bgr, np.ndarray):
                bgr = bgr.copy()
            if isinstance(corners, np.ndarray):
                corners = corners.copy()
        if not isinstance(quality, DetectionQuality) or not quality.accepted:
            reasons = quality.reasons if isinstance(quality, DetectionQuality) else ("no frame",)
            notify("REJECTED: " + ", ".join(reasons))
            return 409, {"saved": False, "reasons": reasons}
        if not isinstance(bgr, np.ndarray) or not isinstance(corners, np.ndarray):
            return 409, {"saved": False, "reasons": ["no valid frame"]}
        robot_pose, rejection = robot_pose_or_rejection(
            arm_reader,
            allow_camera_only_save=args.allow_camera_only_save,
        )
        if rejection is not None:
            notify("REJECTED: " + rejection)
            return 409, {"saved": False, "reasons": [rejection]}
        assert robot_pose is not None
        rejection = repeated_arm_pose_rejection(
            robot_pose,
            records,
            min_joint_change_rad=args.min_joint_change_rad,
        )
        if rejection is not None:
            notify("REJECTED: " + rejection)
            return 409, {"saved": False, "reasons": [rejection]}
        index = len(records) + 1
        raw_name = f"sample_{index:03d}_raw.png"
        detected_name = f"sample_{index:03d}_detected.png"
        annotated = draw_overlay(bgr, pattern, corners, quality, index, "ACCEPTED")
        if not cv2.imwrite(str(output / raw_name), bgr):
            raise RuntimeError(f"Failed to save {raw_name}")
        if not cv2.imwrite(str(output / detected_name), annotated):
            raise RuntimeError(f"Failed to save {detected_name}")
        record = {
            "sample_index": index, "captured_unix_ns": time.time_ns(),
            "camera_timestamp_ms": timestamp_ms, "raw_image": raw_name,
            "detected_image": detected_name, "corners_px": corners.tolist(),
            "quality": asdict(quality),
            **robot_pose,
        }
        records.append(record)
        write_json(output / "samples.json", records)
        notify(f"SAVED sample {index:03d}")
        print(f"Saved sample {index:03d}", flush=True)
        return 200, {"saved": True, "sample_index": index, "output": str(output / raw_name)}

    def delete_last() -> tuple[int, dict]:
        if not records:
            notify("Nothing to delete")
            return 409, {"deleted": False, "reason": "no samples"}
        removed = records.pop()
        for field in ("raw_image", "detected_image"):
            path = output / removed[field]
            if path.exists():
                path.unlink()
        write_json(output / "samples.json", records)
        notify(f"Deleted sample {removed['sample_index']:03d}")
        return 200, {"deleted": True, "sample_index": removed["sample_index"]}

    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, status: int, content_type: str, payload: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def send_json(self, status: int, value: dict) -> None:
            self.send_bytes(status, "application/json", json.dumps(value).encode())

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self.send_bytes(200, "text/html; charset=utf-8", WEB_PAGE)
                return
            if path == "/frame.jpg":
                with lock:
                    jpeg = state["preview_jpeg"]
                if not isinstance(jpeg, bytes):
                    self.send_json(503, {"error": "preview not ready"})
                else:
                    self.send_bytes(200, "image/jpeg", jpeg)
                return
            if path == "/status.json":
                with lock:
                    quality = state["quality"]
                    error = state["error"]
                self.send_json(200, {
                    "ready": bool(isinstance(quality, DetectionQuality) and quality.accepted),
                    "saved": len(records),
                    "quality": None if not isinstance(quality, DetectionQuality) else asdict(quality),
                    "error": error,
                })
                return
            self.send_json(404, {"error": "not found"})

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                if path == "/save":
                    status, payload = save_latest()
                elif path == "/delete":
                    status, payload = delete_last()
                elif path == "/shutdown":
                    status, payload = 200, {"stopping": True}
                    stop.set()
                    threading.Thread(target=server.shutdown, daemon=True).start()
                else:
                    status, payload = 404, {"error": "not found"}
            except BaseException as error:
                status, payload = 500, {"error": repr(error)}
            self.send_json(status, payload)

        def log_message(self, format, *values):
            return

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    capture_thread = threading.Thread(target=capture_loop, daemon=True)
    capture_thread.start()
    display_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    print(f"Open http://{display_host}:{args.port} in a browser", flush=True)
    try:
        server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()
        capture_thread.join(timeout=4.0)
    with lock:
        error = state["error"]
    if error:
        raise RuntimeError(f"Capture loop failed: {error}")


def main() -> None:
    try:
        import pyrealsense2 as rs
    except ImportError as error:
        raise RuntimeError(
            "pyrealsense2 is required; run with the arm-hand-teleop Python environment"
        ) from error
    try:
        import tkinter as tk
        from PIL import Image, ImageTk
    except ImportError as error:
        raise RuntimeError("Tkinter and Pillow ImageTk are required for the preview") from error
    args = parse_args()
    validate_args(args)
    output = make_session_directory(args)
    pattern = (args.columns, args.rows)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(args.serial)
    config.enable_stream(
        rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps
    )
    records: list[dict] = []
    history: deque[np.ndarray] = deque(maxlen=args.stability_frames)
    profile = None
    message = ""
    message_until = 0.0
    root = None
    closed = False
    arm_reader = None
    callback_error: list[BaseException] = []
    try:
        if args.arm_port is not None:
            print(
                f"Opening state-only arm feedback on {args.arm_port}; "
                "no arm control commands will be sent."
            )
            arm_reader = ReadOnlyArmState(args.arm_port)
        profile = pipeline.start(config)
        controls = configure_sensor(profile, args, rs)
        frame = None
        for _ in range(max(args.fps, 15)):
            frame = pipeline.wait_for_frames(3000).get_color_frame()
        if not frame:
            raise RuntimeError("No D405 color frame received")
        intrinsics = frame.profile.as_video_stream_profile().get_intrinsics()
        robot_states_recorded = arm_reader is not None
        session = {
            "purpose": (
                "left-wrist eye-in-hand checkerboard collection"
                if robot_states_recorded
                else "camera-only checkerboard validation; not complete hand-eye data"
            ),
            "camera_serial": args.serial,
            "profile": {"width": args.width, "height": args.height, "fps": args.fps},
            "camera_controls": controls,
            "checkerboard": {
                "inner_corners": [args.columns, args.rows],
                "square_size_mm": args.square_size_mm,
            },
            "intrinsics": intrinsics_payload(intrinsics),
            "robot_joint_states_recorded": robot_states_recorded,
            "arm_port": args.arm_port,
            "arm_feedback_mode": (
                "state_only_no_enable_no_control_commands"
                if robot_states_recorded
                else None
            ),
            "warning": (
                None
                if robot_states_recorded
                else "Camera-only samples cannot be used directly for hand-eye solving."
            ),
        }
        write_json(output / "session.json", session)
        write_json(output / "samples.json", records)
        print(f"Session: {output}")
        if robot_states_recorded:
            print("Robot joint feedback is being recorded; this program cannot move the robot.")
        else:
            print(
                "CAMERA-ONLY PREVIEW: saving is blocked unless "
                "--allow-camera-only-save is explicitly provided."
            )
        if args.smoke_seconds > 0:
            run_smoke_capture(pipeline, pattern, history, args)
            return
        if args.web:
            run_web_capture(
                pipeline, pattern, history, args, output, records, arm_reader
            )
            return
        print("Press S/Space to save a valid frame, D to delete last, Q/Esc to quit.")
        try:
            root = tk.Tk()
        except tk.TclError as error:
            raise RuntimeError(
                "Cannot open the Tk preview. Run this command inside the NoMachine "
                "desktop terminal with DISPLAY set."
            ) from error
        root.title("O10 left wrist checkerboard capture")
        image_label = tk.Label(root)
        image_label.pack(padx=8, pady=8)
        button_row = tk.Frame(root)
        button_row.pack(fill=tk.X, padx=8, pady=(0, 8))
        latest: dict[str, object | None] = {
            "bgr": None,
            "corners": None,
            "quality": None,
            "camera_timestamp_ms": None,
        }

        def notify(text: str) -> None:
            nonlocal message, message_until
            message = text
            message_until = time.monotonic() + 2.0

        def close_window(_event=None):
            nonlocal closed
            closed = True
            if root is not None:
                root.destroy()
            return "break"

        def delete_last(_event=None):
            if not records:
                notify("Nothing to delete")
                return "break"
            removed = records.pop()
            for field in ("raw_image", "detected_image"):
                path = output / removed[field]
                if path.exists():
                    path.unlink()
            write_json(output / "samples.json", records)
            notify(f"Deleted sample {removed['sample_index']:03d}")
            return "break"

        def save_latest(_event=None):
            bgr = latest["bgr"]
            corners = latest["corners"]
            quality = latest["quality"]
            if not isinstance(quality, DetectionQuality) or not quality.accepted:
                reasons = quality.reasons if isinstance(quality, DetectionQuality) else ("no frame",)
                notify("REJECTED: " + ", ".join(reasons))
                return "break"
            if not isinstance(bgr, np.ndarray) or not isinstance(corners, np.ndarray):
                notify("REJECTED: no valid frame")
                return "break"
            robot_pose, rejection = robot_pose_or_rejection(
                arm_reader,
                allow_camera_only_save=args.allow_camera_only_save,
            )
            if rejection is not None:
                notify("REJECTED: " + rejection)
                return "break"
            assert robot_pose is not None
            rejection = repeated_arm_pose_rejection(
                robot_pose,
                records,
                min_joint_change_rad=args.min_joint_change_rad,
            )
            if rejection is not None:
                notify("REJECTED: " + rejection)
                print("Rejected: " + rejection)
                return "break"
            index = len(records) + 1
            raw_name = f"sample_{index:03d}_raw.png"
            detected_name = f"sample_{index:03d}_detected.png"
            annotated = draw_overlay(
                bgr, pattern, corners, quality, index, "ACCEPTED"
            )
            if not cv2.imwrite(str(output / raw_name), bgr):
                raise RuntimeError(f"Failed to save {raw_name}")
            if not cv2.imwrite(str(output / detected_name), annotated):
                raise RuntimeError(f"Failed to save {detected_name}")
            record = {
                "sample_index": index,
                "captured_unix_ns": time.time_ns(),
                "camera_timestamp_ms": latest["camera_timestamp_ms"],
                "raw_image": raw_name,
                "detected_image": detected_name,
                "corners_px": corners.tolist(),
                "quality": asdict(quality),
                **robot_pose,
            }
            records.append(record)
            write_json(output / "samples.json", records)
            notify(f"SAVED sample {index:03d}")
            print(
                f"Saved {index:03d}: sharpness={quality.sharpness:.1f}, "
                f"coverage={100 * quality.coverage:.1f}%, "
                f"margin={quality.minimum_margin_px:.1f}px"
            )
            return "break"

        tk.Button(button_row, text="Save (S / Space)", command=save_latest).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=2
        )
        tk.Button(button_row, text="Delete last (D)", command=delete_last).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=2
        )
        tk.Button(button_row, text="Quit (Q / Esc)", command=close_window).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=2
        )
        root.bind("s", save_latest)
        root.bind("<space>", save_latest)
        root.bind("d", delete_last)
        root.bind("q", close_window)
        root.bind("<Escape>", close_window)
        root.protocol("WM_DELETE_WINDOW", close_window)

        def update_preview() -> None:
            nonlocal message
            if closed:
                return
            try:
                frames = pipeline.wait_for_frames(3000)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    root.after(1, update_preview)
                    return
                rgb = np.asanyarray(color_frame.get_data())
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                _, corners = detect_checkerboard(gray, pattern)
                quality = assess_detection(
                    gray, corners, history,
                    sharpness_min=args.sharpness_min,
                    margin_min=args.margin_min,
                    stability_max=args.stability_max,
                    stability_frames=args.stability_frames,
                )
                if corners is not None:
                    history.append(corners.copy())
                else:
                    history.clear()
                if time.monotonic() > message_until:
                    message = ""
                preview = draw_overlay(
                    bgr, pattern, corners, quality, len(records), message
                )
                latest.update({
                    "bgr": bgr.copy(),
                    "corners": None if corners is None else corners.copy(),
                    "quality": quality,
                    "camera_timestamp_ms": float(color_frame.get_timestamp()),
                })
                photo = ImageTk.PhotoImage(
                    Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
                )
                image_label.configure(image=photo)
                image_label.image = photo
                root.after(1, update_preview)
            except BaseException as error:
                callback_error.append(error)
                close_window()

        root.after(0, update_preview)
        root.mainloop()
        if callback_error:
            raise callback_error[0]
    finally:
        if profile is not None:
            pipeline.stop()
        if arm_reader is not None:
            arm_reader.close()
    print(f"Finished. Accepted samples: {len(records)}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
