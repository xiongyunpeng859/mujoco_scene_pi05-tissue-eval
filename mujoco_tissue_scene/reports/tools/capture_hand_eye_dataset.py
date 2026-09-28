#!/usr/bin/env python3
"""左腕 eye-in-hand 标定数据采集 / 单张质量判定 / T_eef_camera 求解。

核心保证
--------
1. 机器人正运动学**不依赖 mmk2_kdl_py**，而是直接解析厂商 `play_e2.urdf`。
   已验证：该 FK 与 MuJoCo 生成模型的 `link6` body 位姿一致（误差 < 1e-6），
   所以这里解出的 T_eef_camera 可以直接放进 MuJoCo 场景，不需要任何换算。
   注意：项目自带的 ArmKdlNumerical(arm_type="play_short") 与本 URDF **不等价**
   （相对 link6 旋转差约 177°、平移随位姿漂移 0.4~0.9 m），不要用它算 T_base_eef。

2. 机械臂默认**只读**关节角，不下发任何运动指令；`--free-drive` 才会进入
   MIT 零力矩自由拖动（无重力补偿，机械臂会掉，必须先托住）。

3. 每一张照片都会给出「能否用于标定」的判定和**具体原因**，不合格的默认不保存。

常用命令（主机上，务必用 arm-hand-teleop 环境）
-----------------------------------------------
  PY=/opt/miniconda3/envs/arm-hand-teleop/bin/python

  # 0) 不接硬件自检：合成棋盘格 + FK 基准 + 合成手眼求解
  $PY capture_hand_eye_dataset.py --self-test

  # 1) 看有哪些相机
  $PY capture_hand_eye_dataset.py --list-devices

  # 2) 只预览标定板，不连机械臂（最安全的第一步）
  $PY capture_hand_eye_dataset.py --preview --board checkerboard --cols 7 --rows 8 --square-mm 15

  # 3) 正式采集：只读关节角 + 手动拍照
  $PY capture_hand_eye_dataset.py --capture --board checkerboard --cols 7 --rows 8 --square-mm 15

  # 4) 零力矩手推采集（危险：先托住臂、确认急停）
  $PY capture_hand_eye_dataset.py --capture --free-drive

  # 5) 只重新求解
  $PY capture_hand_eye_dataset.py --solve --session outputs/calibration/hand_eye_left_XXXX

  # 6) 没有桌面显示时用浏览器界面
  $PY capture_hand_eye_dataset.py --capture --web --port 8765

按键： s=保存  d=删除上一张  c=求解  q/Esc=退出
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 默认值
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
SDK_ROOT = Path(
    "/workspace/shared/o10-openpi-demo/arm-hand-teleop-o10-openpi-demo-stable"
)
DEFAULT_URDF = (
    SDK_ROOT
    / "qiuzhi/lerobot_play_1.0.4/x86/noble/lerobot_play-1.0.4-py3-none-any"
    / "lerobot_play/urdf/play_e2/urdf/play_e2.urdf"
)
DEFAULT_OUT_ROOT = PROJECT_ROOT / "outputs/calibration"
DEFAULT_SERIAL = "260322276846"        # 配置里的左腕序列号（实际以 --list-devices 为准）
DEFAULT_UVC = "/dev/v4l/by-id/usb-LRCP_ZAWK_LRCP_500W_SN0001-video-index0"
ARM_DOF = 6

# 关节限位（来自 play_e2.urdf，仅用于判定关节读数是否可信）
JOINT_LIMITS = [
    (-3.1416, 2.0944),
    (-2.9671, 0.17453),
    (-0.087266, 3.1416),
    (-3.0107, 3.0107),
    (-1.7628, 1.7628),
    (-3.0107, 3.0107),
]

# 单张质量阈值
Q = {
    "min_margin_px": 20.0,       # 标定板角点到图像边界的余量
    "min_sharpness": 60.0,       # Laplacian 方差
    "max_reproj_px": 1.0,        # solvePnP 重投影 RMS
    "max_tilt_deg": 60.0,        # 板法线与相机光轴夹角上限（避免接近正对/退化）
    "min_area_frac": 0.005,      # 板在画面中的最小占比
    "max_area_frac": 0.85,       # 上限，避免贴太近导致截断/畸变主导
    "stable_frames": 5,          # 稳定性判定用的帧数
    "max_pose_trans_std_m": 0.002,
    "max_pose_rot_std_deg": 0.5,
    "max_joint_motion_rad": 0.005,   # 这几帧内关节最大变化
    "min_joint_change_rad": 0.05,    # 与已有样本的最小差异（防止样本退化）
    "min_samples": 15,
}

# 相机内参标定（--calibrate-camera）专用阈值。与手眼不同：
#   · 不需要关节角（相机固定、只动板）
#   · **允许、而且需要**接近正对镜头的视角（手眼那边反而要避免正对）
#   · 手持会抖，所以位置稳定性判据放宽，靠清晰度 + 重投影误差把关
#   · 关键约束变成"板要出现在画面的不同区域"，引入 4x4 网格覆盖检查
Q_CAM = {
    "min_margin_px": 15.0,
    "min_sharpness": 40.0,
    # 起步阶段还没有真内参，用的是标称 FOV 估的 fx，重投影误差会被高估，
    # 所以这里放宽；真正的质量把关是最后 calibrateCamera 的单视图残差。
    "max_reproj_px": 5.0,
    "max_tilt_deg": 75.0,          # 只排除接近退化的极端视角
    "min_area_frac": 0.02,         # 板至少占画面 2%，否则角点精度不够
    "max_area_frac": 0.90,
    "stable_frames": 2,            # 手持拍摄，2 帧取最清晰的一帧就够
    "max_pose_trans_std_m": 0.020,  # 手持，放宽
    "max_pose_rot_std_deg": 3.0,    # 手持，放宽
    # 内参标定最怕"在同一个地方拍十几张"，所以要求板中心之间拉开距离。
    # 但太大会让人在原地一直红，55px 已足够区分不同区域。
    "min_view_center_sep_px": 55.0,
    "grid": 4,                       # 4x4 覆盖率网格
    "min_views": 15,
    "min_grid_cells": 10,            # 至少覆盖 4x4 中的 10 格（按板占到的格子算）
}


# 实时预览只给一行 ASCII 短原因（cv2 的字体画不了中文）
_REASON_ASCII = [
    ("没有检测到标定板", "no board detected"),
    ("离画面边界太近", "keep the WHOLE board inside the frame"),
    ("太小", "board too small - move closer"),
    ("太大", "board too big - move back"),
    ("模糊", "blurry - hold steady"),
    ("重投影误差偏大", "corners unreliable - hold steady / avoid glare"),
    ("位置太接近", "same area as a saved view - MOVE the board"),
    ("不增加画面覆盖", "MOVE SIDEWAYS - this area is already covered"),
    ("覆盖已经够了", "change DISTANCE or TILT - area already covered"),
    ("太斜", "board too oblique"),
    ("太正对", "too frontal - tilt the board more"),
    ("晃动", "moving - hold still"),
    ("几乎没有相对旋转", "not enough rotation"),
    ("关节", "arm is moving"),
    ("变化", "pose changed during capture"),
]


def reason_ascii(reasons: list[str]) -> str:
    for reason in reasons:
        for key, text in _REASON_ASCII:
            if key in reason:
                return text
    return reasons[0][:52] if reasons else ""


# ---------------------------------------------------------------------------
# 1. URDF 正运动学（与 MuJoCo 的 link6 一致）
# ---------------------------------------------------------------------------


def _rpy_to_R(r: float, p: float, y: float) -> np.ndarray:
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Rz @ Ry @ Rx


def _axis_R(axis, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    n = np.linalg.norm(a)
    if n == 0:
        return np.eye(3)
    a = a / n
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


class UrdfKinematics:
    """从 URDF 解析出的单链正运动学。EEF 默认 link6。"""

    def __init__(self, urdf_path: Path, eef_link: str = "link6") -> None:
        if not urdf_path.is_file():
            raise FileNotFoundError(f"URDF 不存在: {urdf_path}")
        root = ET.parse(urdf_path).getroot()
        self.joints = {j.get("name"): j for j in root.findall("joint")}
        self.eef_link = eef_link
        # 预先算出 base_link -> eef_link 的关节链
        parent = {
            j.find("child").get("link"): (name, j.find("parent").get("link"))
            for name, j in self.joints.items()
        }
        chain: list[str] = []
        link = eef_link
        while link in parent:
            name, up = parent[link]
            chain.append(name)
            link = up
        if not chain:
            raise ValueError(f"URDF 里找不到 {eef_link} 的上游关节链")
        chain.reverse()
        self.chain = chain
        revolute = [n for n in chain if self.joints[n].get("type") == "revolute"]
        if len(revolute) != ARM_DOF:
            raise ValueError(f"{eef_link} 链上有 {len(revolute)} 个转动关节，期望 {ARM_DOF}")

    def _joint_transform(self, name: str, q: float) -> np.ndarray:
        j = self.joints[name]
        origin = j.find("origin")
        xyz = (
            [float(v) for v in origin.get("xyz", "0 0 0").split()]
            if origin is not None
            else [0.0, 0.0, 0.0]
        )
        rpy = (
            [float(v) for v in origin.get("rpy", "0 0 0").split()]
            if origin is not None
            else [0.0, 0.0, 0.0]
        )
        T = np.eye(4)
        T[:3, :3] = _rpy_to_R(*rpy)
        T[:3, 3] = xyz
        axis = j.find("axis")
        if j.get("type") == "revolute" and axis is not None:
            R = np.eye(4)
            R[:3, :3] = _axis_R([float(v) for v in axis.get("xyz").split()], q)
            T = T @ R
        return T

    def fk(self, joints) -> np.ndarray:
        """q(6) -> T_base_eef (4x4)，单位米。"""
        if len(joints) != ARM_DOF:
            raise ValueError(f"需要 {ARM_DOF} 个关节角，收到 {len(joints)}")
        qmap = {f"joint{i + 1}": float(joints[i]) for i in range(ARM_DOF)}
        T = np.eye(4)
        for name in self.chain:
            T = T @ self._joint_transform(name, qmap.get(name, 0.0))
        return T


# ---------------------------------------------------------------------------
# 2. 标定板检测（棋盘格 / ArUco-AprilTag 统一接口）
# ---------------------------------------------------------------------------


class BoardDetector:
    name = "base"

    def detect(self, gray: np.ndarray):
        """返回 (corners_px (N,2) float32, object_pts (N,3) float32) 或 None。"""
        raise NotImplementedError

    def describe(self) -> dict:
        raise NotImplementedError


class CheckerboardDetector(BoardDetector):
    """cols x rows = 内角点数（不是方格数）。"""

    name = "checkerboard"

    def __init__(self, cols: int, rows: int, square_m: float) -> None:
        self.cols = int(cols)
        self.rows = int(rows)
        self.square_m = float(square_m)
        grid = np.zeros((self.cols * self.rows, 3), np.float32)
        grid[:, :2] = (
            np.mgrid[0 : self.cols, 0 : self.rows].T.reshape(-1, 2) * self.square_m
        )
        self.object_pts = grid

    def detect_fast(self, gray):
        """只用默认 flags。EXHAUSTIVE|ACCURACY 在 88 个内角点上很慢（可达秒级），
        连续帧稳定性检查用这个快速版，保存时再对最后一帧做精确检测。"""
        try:
            found, corners = cv2.findChessboardCornersSB(gray, (self.cols, self.rows))
        except cv2.error:
            return None
        if not found or corners is None:
            return None
        return corners.reshape(-1, 2).astype(np.float32), self.object_pts

    def detect(self, gray):
        # 先用 EXHAUSTIVE|ACCURACY 求更准的角点；真实图像（略糊/略斜）它偶尔找不到，
        # 而自动探测阶段用的是默认 flags，所以这里必须回退，否则会出现
        # "自动探测命中 6x7" 紧接着 "没有检测到标定板" 的矛盾提示。
        for flags in (cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY, 0):
            found, corners = cv2.findChessboardCornersSB(
                gray, (self.cols, self.rows), flags=flags
            )
            if found and corners is not None:
                return corners.reshape(-1, 2).astype(np.float32), self.object_pts
        return None

    def describe(self) -> dict:
        return {
            "type": "checkerboard",
            "inner_corners": [self.cols, self.rows],
            "square_size_m": self.square_m,
        }


class MarkerDetector(BoardDetector):
    """ArUco / AprilTag 单标记。"""

    name = "aruco"

    def __init__(self, dictionary: str, marker_id: int, marker_m: float) -> None:
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("当前 OpenCV 没有 aruco 模块")
        dict_id = getattr(cv2.aruco, dictionary, None)
        if dict_id is None:
            raise ValueError(f"未知字典 {dictionary}")
        self.dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        self.dictionary_name = dictionary
        self.marker_id = int(marker_id)
        self.marker_m = float(marker_m)
        if hasattr(cv2.aruco, "ArucoDetector"):
            self._detector = cv2.aruco.ArucoDetector(
                self.dictionary, cv2.aruco.DetectorParameters()
            )
        else:  # OpenCV < 4.7
            self._detector = None
            self._params = cv2.aruco.DetectorParameters_create()
        s = self.marker_m / 2.0
        self.object_pts = np.array(
            [[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]], np.float32
        )

    def detect(self, gray):
        if self._detector is not None:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self.dictionary, parameters=self._params
            )
        if ids is None or len(ids) == 0:
            return None
        for index, marker in enumerate(ids.flatten()):
            if int(marker) == self.marker_id:
                return corners[index].reshape(-1, 2).astype(np.float32), self.object_pts
        return None

    def describe(self) -> dict:
        return {
            "type": "aruco",
            "dictionary": self.dictionary_name,
            "marker_id": self.marker_id,
            "marker_size_m": self.marker_m,
        }


def make_detector(args) -> BoardDetector:
    if args.board == "checkerboard":
        return CheckerboardDetector(args.cols, args.rows, args.square_mm / 1000.0)
    return MarkerDetector(args.dict, args.marker_id, args.marker_mm / 1000.0)


# 常见棋盘格内角点规格。规格填错是"检测不到标定板"最常见的原因。
CANDIDATE_PATTERNS = [
    (7, 8), (8, 7), (8, 11), (11, 8), (9, 6), (6, 9), (10, 7), (7, 10),
    (9, 7), (7, 9), (6, 8), (8, 6), (5, 7), (7, 5), (4, 11), (11, 4),
    (6, 7), (7, 6), (5, 5), (3, 11), (11, 3), (12, 9), (9, 12),
]


def autodetect_pattern(gray: np.ndarray) -> list[tuple[int, int]]:
    """在一帧图像上尝试常见内角点规格，返回全部命中的 (cols, rows)。

    这里用默认 flags（快）。EXHAUSTIVE 留给正式检测用，否则一次扫描要几十秒。
    """
    hits: list[tuple[int, int]] = []
    for cols, rows in CANDIDATE_PATTERNS:
        try:
            found, _ = cv2.findChessboardCornersSB(gray, (cols, rows))
        except cv2.error:
            continue
        if found:
            hits.append((cols, rows))
    return hits


def gui_available() -> bool:
    """OpenCV 是否是带 GUI 后端的构建（很多 conda 环境是 headless 版）。"""
    if not hasattr(cv2, "imshow"):
        return False
    try:
        info = cv2.getBuildInformation()
        for line in info.splitlines():
            if line.strip().startswith("GUI:"):
                value = line.split(":", 1)[1].strip().upper()
                if value.startswith("NONE"):
                    return False
                break
    except Exception:
        pass
    try:
        cv2.namedWindow("__probe__", cv2.WINDOW_AUTOSIZE)
        cv2.destroyWindow("__probe__")
        return True
    except cv2.error:
        return False


def _pattern_reprojection(gray, cols, rows, square_m, intr):
    """对某个内角点规格算 (重投影RMS, 画面占比)。失败返回 (inf, 0)。

    这里刻意用**默认 flags**：EXHAUSTIVE 在真实（略糊/略斜）图像上常常直接找不到，
    而自动探测阶段本来就用默认 flags 命中的，两者必须一致，否则打分全是 inf。
    """
    try:
        obj = CheckerboardDetector(cols, rows, square_m).object_pts
        found, corners = cv2.findChessboardCornersSB(gray, (cols, rows))
    except Exception:
        return float("inf"), 0.0
    if not found or corners is None:
        return float("inf"), 0.0
    corners = corners.reshape(-1, 2).astype(np.float32)
    if corners.shape[0] != obj.shape[0]:
        return float("inf"), 0.0
    pixels = intr.undistort(corners)
    try:
        ok, rvec, tvec = cv2.solvePnP(
            obj, pixels.reshape(-1, 1, 2), intr.camera_matrix(), intr.dist_coeffs(),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error:
        return float("inf"), 0.0
    if not ok:
        return float("inf"), 0.0
    projected, _ = cv2.projectPoints(
        obj, rvec, tvec, intr.camera_matrix(), intr.dist_coeffs()
    )
    rms = float(np.sqrt(np.mean(
        np.sum((projected.reshape(-1, 2) - pixels) ** 2, axis=1)
    )))
    area = float(np.ptp(corners[:, 0]) * np.ptp(corners[:, 1]))
    return rms, area


def report_autodetect(gray: np.ndarray, args, intr: Intrinsics) -> None:
    """探测棋盘格规格。"""
    print("（正在自动探测棋盘格规格，最多几秒…）", flush=True)
    hits = autodetect_pattern(gray)
    print("\n=== 棋盘格规格自动探测 ===", flush=True)
    if not hits:
        print("  当前画面里没有识别到任何常见规格的棋盘格。")
        print("  请确认：整块板都在画面内、板面清晰、背景对比足够、"
              "且是黑白棋盘格（不是圆点板/ChArUco）。")
        return

    square_m = args.square_mm / 1000.0
    scored = []
    for cols, rows in hits:
        rms, area = _pattern_reprojection(gray, cols, rows, square_m, intr)
        scored.append((cols, rows, rms, area))

    # 选规格：
    #  1) 只考虑误差接近最小的一批（并列/近似并列都算）
    #  2) 其中优先角点最多的——防止锁在"大板被截断后剩下的子窗口"上
    #  3) 仍然并列时优先用户已经传的参数，不要无缘无故让人改命令
    TIE_PX = 0.35
    finite = [s for s in scored if s[2] != float("inf")]
    if finite:
        best_rms = min(s[2] for s in finite)
        near = [s for s in finite if s[2] <= best_rms + TIE_PX]
        most = max(s[0] * s[1] for s in near)
        near = [s for s in near if s[0] * s[1] == most]
        same = [s for s in near if (s[0], s[1]) == (args.cols, args.rows)]
        chosen = same[0] if same else near[0]
    else:
        chosen = scored[0]
    scored.sort(key=lambda item: (item is not chosen, item[2], -item[3]))

    for cols, rows, rms, area in scored:
        marks = []
        if (cols, rows) == (args.cols, args.rows):
            marks.append("当前参数")
        if (cols, rows) == (chosen[0], chosen[1]):
            marks.append("推荐")
        tail = ("  ← " + " / ".join(marks)) if marks else ""
        rms_txt = "位姿求解失败" if rms == float("inf") else f"重投影 {rms:.2f}px"
        print(f"  命中 内角点 {cols} x {rows}（{rms_txt}，占画面约 {area:.0f}px²）{tail}")

    best_cols, best_rows, best_rms, _ = chosen

    # (c,r) 与 (r,c) 是同一组物理点：只差一个板内 90° 旋转，是固定变换，会被吸收
    swapped = [s for s in scored if (s[0], s[1]) == (best_rows, best_cols)]
    if swapped:
        print(f"  说明：{best_cols}x{best_rows} 与 {best_rows}x{best_cols} 是**同一组物理点**"
              "的两种读法（相差板内 90° 旋转）。")
        print("        两者给出的是同一个物理板坐标系，只差一个固定旋转，会被 "
              "T_eef_camera 吸收，因此**用哪个都可以**——只要整个会话保持一致。")

    # 如果还有更大的规格也被命中，说明画面里的板很可能只露出一部分
    bigger = [s for s in scored
              if s[0] * s[1] > best_cols * best_rows and s[2] != float("inf")]
    if bigger:
        b = bigger[0]
        print(f"  ! 还命中了更大的规格 {b[0]}x{b[1]}（重投影 {b[2]:.2f}px）："
              "画面里的板很可能**只露出一部分**，"
              "\n    检测器锁在了大板里的一个子窗口上。这种照片必须重拍——"
              "请把机械臂挪到能看见整块板的位置。")

    if best_rms == float("inf"):
        print("\n  ! 命中了规格但算不出位姿（板可能被截断/太斜/反光）。"
              "\n    请看画面里绿色框是否完整贴住棋盘格外沿，再决定 --cols/--rows。")
        return

    if (best_cols, best_rows) != (args.cols, args.rows):
        print(
            f"\n  ! 你传的是 --cols {args.cols} --rows {args.rows}，"
            f"但更合适的是 {best_cols} x {best_rows}（{best_rms:.2f}px）。"
            f"\n    建议改用：--cols {best_cols} --rows {best_rows} "
            f"--square-mm <实测方格边长mm>"
        )
    else:
        print(f"\n  ✅ 与 --cols/--rows 一致（重投影 {best_rms:.2f}px）。"
              "若仍判不合格，请检查 --square-mm 是否填了实测值。")


# ---------------------------------------------------------------------------
# 3. 相机
# ---------------------------------------------------------------------------


@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    source: str
    serial: str = ""
    distortion_model: str = ""
    distortion: list[float] = field(default_factory=list)
    undistort_pixels: bool = False   # True 时用 RealSense 反投影把像素变无畸变
    rs_intrinsics: Any = None

    def camera_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

    def dist_coeffs(self) -> np.ndarray:
        if self.undistort_pixels:
            return np.zeros(5)
        return np.array(self.distortion[:5] if self.distortion else [0.0] * 5)

    def undistort(self, pixels: np.ndarray) -> np.ndarray:
        if not self.undistort_pixels or self.rs_intrinsics is None:
            return pixels
        import pyrealsense2 as rs

        out = np.empty_like(pixels, dtype=np.float32)
        for i, (u, v) in enumerate(pixels):
            p = rs.rs2_deproject_pixel_to_point(
                self.rs_intrinsics, [float(u), float(v)], 1.0
            )
            out[i, 0] = self.fx * p[0] / p[2] + self.cx
            out[i, 1] = self.fy * p[1] / p[2] + self.cy
        return out

    def as_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "source": self.source,
            "camera_serial": self.serial,
            "distortion_model": self.distortion_model,
            "distortion_coefficients": self.distortion,
            "pixels_undistorted_before_solvepnp": bool(self.undistort_pixels),
        }


class RealsenseCamera:
    def __init__(self, args) -> None:
        import pyrealsense2 as rs

        self.rs = rs
        self.serial = args.serial
        self.width = args.width
        self.height = args.height
        self.fps = args.fps
        # 推流线程和采集线程会同时读同一个 pipeline，RealSense 不允许并发
        # wait_for_frames —— 之前浏览器画面变黑就是这个原因。全部读操作串行化。
        self.lock = threading.Lock()
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        if self.serial:
            cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.profile = self.pipeline.start(cfg)
        device = self.profile.get_device()
        self.actual_serial = device.get_info(rs.camera_info.serial_number)
        self.model = device.get_info(rs.camera_info.name)
        # 关掉自动曝光，和真机采集配置一致
        try:
            for sensor in device.query_sensors():
                if sensor.supports(rs.option.enable_auto_exposure):
                    sensor.set_option(rs.option.enable_auto_exposure, 0)
                if sensor.supports(rs.option.exposure) and args.exposure_us > 0:
                    sensor.set_option(rs.option.exposure, args.exposure_us)
                if sensor.supports(rs.option.gain) and args.gain > 0:
                    sensor.set_option(rs.option.gain, args.gain)
        except Exception as exc:  # 曝光设置失败不该挡住采集
            print(f"  ! 相机曝光设置失败（继续）：{exc}")
        # 丢掉前几帧让自动曝光/白平衡稳定
        for _ in range(15):
            self.pipeline.wait_for_frames(1000)

    def intrinsics(self) -> Intrinsics:
        rs = self.rs
        video = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        i = video.get_intrinsics()
        model = str(i.model)
        undistort = "inverse_brown" in model
        return Intrinsics(
            width=i.width,
            height=i.height,
            fx=float(i.fx),
            fy=float(i.fy),
            cx=float(i.ppx),
            cy=float(i.ppy),
            source=f"realsense_factory:{self.model}",
            serial=self.actual_serial,
            distortion_model=model,
            distortion=[float(c) for c in i.coeffs],
            undistort_pixels=undistort,
            rs_intrinsics=i,
        )

    def read(self):
        with self.lock:
            frames = self.pipeline.wait_for_frames(1000)
            color = frames.get_color_frame()
            if not color:
                return None
            return np.asanyarray(color.get_data()).copy()

    def close(self) -> None:
        try:
            self.pipeline.stop()
        except Exception:
            pass


class UvcCamera:
    def __init__(self, args) -> None:
        device = args.device if args.device.isdigit() else args.device
        self.lock = threading.Lock()
        self.capture = cv2.VideoCapture(int(device) if str(device).isdigit() else device)
        if not self.capture.isOpened():
            raise RuntimeError(f"打不开 UVC 相机 {args.device}")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        self.capture.set(cv2.CAP_PROP_FPS, args.fps)
        # 这台 UVC 相机（LRCP 500W）要 MJPG 才能跑 640x480@30
        try:
            self.capture.set(cv2.CAP_PROP_FOURCC,
                             cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        self.actual_serial = str(args.device)
        self.model = "uvc"

    def intrinsics(self) -> Intrinsics:
        if not args_intrinsics_path:
            if _ALLOW_PROVISIONAL:
                # 内参标定模式：此时还没有内参，先用一个标称 FOV 让质量判定能跑。
                # solvePnP 只用它算重投影误差，真正的内参由 calibrateCamera 解出来。
                f = _PROVISIONAL_HEIGHT / (2.0 * math.tan(math.radians(
                    _PROVISIONAL_FOV / 2.0)))
                print(f"  （内参标定模式：暂用标称 FOV {_PROVISIONAL_FOV}° 估计 "
                      f"fx≈{f:.1f} 做质量判定；真正内参由 calibrateCamera 解出）")
                return Intrinsics(
                    width=args_width, height=args_height,
                    fx=f, fy=f, cx=args_width / 2.0, cy=args_height / 2.0,
                    source=f"provisional_fov{_PROVISIONAL_FOV}",
                    serial=self.actual_serial,
                )
            raise SystemExit(
                "UVC 相机没有可用内参。请用 --intrinsics <yaml> 指定标定好的 "
                "fx/fy/cx/cy，\n或先用 --calibrate-camera 做一次内参标定。"
                "UVC 设备不提供厂家内参。"
            )
        import yaml

        data = yaml.safe_load(Path(args_intrinsics_path).read_text())
        return Intrinsics(
            width=int(data.get("width", args_width)),
            height=int(data.get("height", args_height)),
            fx=float(data["fx"]),
            fy=float(data["fy"]),
            cx=float(data["cx"]),
            cy=float(data["cy"]),
            source=f"yaml:{args_intrinsics_path}",
            serial=str(data.get("camera_serial", self.actual_serial)),
            distortion_model=str(data.get("distortion_model", "")),
            distortion=[float(v) for v in data.get("distortion_coefficients", [])],
        )

    def read(self):
        with self.lock:
            ok, frame = self.capture.read()
            return frame if ok else None

    def close(self) -> None:
        self.capture.release()


def make_camera(args):
    if args.camera == "uvc":
        return UvcCamera(args)
    return RealsenseCamera(args)


# ---------------------------------------------------------------------------
# 3b. 相机内参标定（只动标定板，相机固定）
# ---------------------------------------------------------------------------


def _coverage_cell(cx: float, cy: float, w: int, h: int, n: int):
    gx = min(n - 1, max(0, int(cx / max(1, w) * n)))
    gy = min(n - 1, max(0, int(cy / max(1, h) * n)))
    return gx, gy


def board_cells(corners, w: int, h: int, n: int) -> set:
    """板占到的网格格子：用角点包围盒算。

    这比"板中心落在哪一格"合理：同一个中心、但**更近**的板，角点会伸到画面更外围，
    覆盖的格子更多 —— 对内参有价值。之前按"中心离已有视图够不够远"判，
    会让用户"放远放近"怎么调都被拒（因为中心没动）。
    """
    x0, x1 = float(corners[:, 0].min()), float(corners[:, 0].max())
    y0, y1 = float(corners[:, 1].min()), float(corners[:, 1].max())
    gx0 = min(n - 1, max(0, int(x0 / max(1, w) * n)))
    gx1 = min(n - 1, max(0, int(x1 / max(1, w) * n)))
    gy0 = min(n - 1, max(0, int(y0 / max(1, h) * n)))
    gy1 = min(n - 1, max(0, int(y1 / max(1, h) * n)))
    return {(gx, gy) for gx in range(gx0, gx1 + 1)
            for gy in range(gy0, gy1 + 1)}


def covered_cells(views: list[dict], n: int) -> set:
    """已有视图覆盖到的格子并集。

    优先用存下来的 coverage_cells；老记录没这个字段时**用存下来的角点反算**，
    否则只按板中心算会严重低估覆盖（每一张都会被算成只占 1 格）。
    """
    out: set = set()
    for v in views:
        m = v.get("quality", {}).get("metrics", {})
        cs = m.get("coverage_cells")
        if cs:
            out |= {tuple(c) for c in cs}
            continue
        corners = v.get("corners_px")
        if corners:
            try:
                out |= board_cells(np.asarray(corners, float),
                                   args_width or 640, args_height or 480, n)
                continue
            except Exception:
                pass
        if m.get("coverage_cell"):
            out.add(tuple(m["coverage_cell"]))
    return out


def coverage_ascii(cells, n: int) -> list[str]:
    lines = []
    for gy in range(n):
        lines.append("".join(" ## " if (gx, gy) in cells else " .. "
                             for gx in range(n)))
    return lines


_CELL_NAME = {(0, 0): "左上", (1, 0): "中上左", (2, 0): "中上右", (3, 0): "右上",
              (0, 3): "左下", (1, 3): "中下左", (2, 3): "中下右", (3, 3): "右下"}


def suggest_cells_ascii(views: list[dict], n: int, limit: int = 3) -> str:
    """ASCII 版建议格（画面横幅用：cv2 的字库画不了中文）。"""
    covered = covered_cells(views, n)
    missing = [(gx, gy) for gy in range(n) for gx in range(n)
               if (gx, gy) not in covered]
    if not missing:
        return "all cells covered"
    corners = [(0, 0), (n - 1, 0), (0, n - 1), (n - 1, n - 1)]
    missing.sort(key=lambda c: (0 if c in corners else 1))
    return " ".join(f"({c[0]},{c[1]})" for c in missing[:limit])


def suggest_cells(views: list[dict], n: int, limit: int = 4) -> str:
    """列出还没拍过的格子，优先四角 —— 给用户一个明确的"往哪挪"。"""
    covered = covered_cells(views, n)
    missing = [(gx, gy) for gy in range(n) for gx in range(n)
               if (gx, gy) not in covered]
    if not missing:
        return "所有格子都拍过了"
    # 四角优先
    corners = [(0, 0), (n - 1, 0), (0, n - 1), (n - 1, n - 1)]
    missing.sort(key=lambda c: (0 if c in corners else 1))
    out = []
    for c in missing[:limit]:
        name = _CELL_NAME.get(c, "")
        out.append(f"{name}({c[0]},{c[1]})" if name else f"({c[0]},{c[1]})")
    return "、".join(out) + f"  共剩 {len(missing)} 格"


def assess_board_view(gray, detector, intr, previous_views, *, grid=None,
                      fast: bool = False):
    """内参标定的单张判定：不需要关节角，重点是"清晰 + 够大 + 位置有变化"。

    fast=True 时用快速检测（预览用，保证视频流畅）；保存时用精确检测。
    """
    reasons: list[str] = []
    metrics: dict[str, Any] = {}
    height, width = gray.shape[:2]

    det = (detector.detect_fast(gray) if fast and hasattr(detector, "detect_fast")
           else detector.detect(gray))
    if det is None:
        return ShotQuality(False, ["没有检测到标定板"], metrics), None
    corners_px, object_pts = det

    margin = float(min(corners_px[:, 0].min(), corners_px[:, 1].min(),
                       width - 1 - corners_px[:, 0].max(),
                       height - 1 - corners_px[:, 1].max()))
    metrics["edge_margin_px"] = margin
    if margin < Q_CAM["min_margin_px"]:
        reasons.append(f"标定板离画面边界太近（余量 {margin:.0f}px）")

    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    metrics["sharpness_laplacian_var"] = sharp
    if sharp < Q_CAM["min_sharpness"]:
        reasons.append(f"图像模糊（清晰度 {sharp:.0f} < {Q_CAM['min_sharpness']:.0f}）")

    hull = cv2.convexHull(corners_px.astype(np.float32))
    area_frac = float(cv2.contourArea(hull)) / float(width * height)
    metrics["board_area_fraction"] = area_frac
    if area_frac < Q_CAM["min_area_frac"]:
        reasons.append(f"标定板太小（占画面 {area_frac * 100:.2f}% < "
                       f"{Q_CAM['min_area_frac'] * 100:.0f}%），请把板拿近一些")
    if area_frac > Q_CAM["max_area_frac"]:
        reasons.append(f"标定板太大（占画面 {area_frac * 100:.0f}%），可能被截断")

    pixels = intr.undistort(corners_px)
    ok, rvec, tvec = cv2.solvePnP(
        object_pts, pixels.reshape(-1, 1, 2), intr.camera_matrix(),
        intr.dist_coeffs(),
        flags=(cv2.SOLVEPNP_IPPE_SQUARE if object_pts.shape[0] == 4
               else cv2.SOLVEPNP_ITERATIVE),
    )
    if not ok:
        return ShotQuality(False, reasons + ["solvePnP 失败"]), None
    proj, _ = cv2.projectPoints(object_pts, rvec, tvec, intr.camera_matrix(),
                                intr.dist_coeffs())
    reproj = float(np.sqrt(np.mean(
        np.sum((proj.reshape(-1, 2) - pixels) ** 2, axis=1))))
    metrics["reprojection_rms_px"] = reproj
    if reproj > Q_CAM["max_reproj_px"]:
        reasons.append(f"重投影误差偏大（{reproj:.2f}px）")

    R = cv2.Rodrigues(rvec)[0]
    normal = R @ np.array([0.0, 0.0, 1.0])
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, abs(float(normal[2]))))))
    metrics["board_tilt_deg"] = tilt
    if tilt > Q_CAM["max_tilt_deg"]:
        reasons.append(f"标定板太斜（{tilt:.0f}°），角点不可靠")
    metrics["distance_m"] = float(np.linalg.norm(tvec))

    cx_all = corners_px[:, 0]
    cy_all = corners_px[:, 1]
    center = (float(cx_all.mean()), float(cy_all.mean()))
    metrics["board_center_px"] = [center[0], center[1]]
    span = float(max(np.ptp(cx_all), np.ptp(cy_all)))
    metrics["board_span_px"] = span

    n = grid or Q_CAM["grid"]
    cell = _coverage_cell(center[0], center[1], width, height, n)
    metrics["coverage_cell"] = list(cell)
    cells = board_cells(corners_px, width, height, n)
    metrics["coverage_cells"] = sorted(list(cells))

    # 多样性判据：这张必须给画面覆盖**新增**至少一格。
    # 用"板占到的格子"而不是"板中心的距离"：同一个中心、更近的板会覆盖更多格子，
    # 那是有价值的；而单纯前后移动、格子没变，才是冗余。
    if previous_views:
        already = covered_cells(previous_views, n)
        new_cells = cells - already
        metrics["new_cells"] = sorted(list(new_cells))
        metrics["new_cell_count"] = len(new_cells)
        span_now = metrics["board_span_px"]
        tilt_now = metrics["board_tilt_deg"]
        scale_diff = min(
            (abs(math.log(span_now / s)) for s in
             (v["quality"]["metrics"].get("board_span_px") for v in previous_views)
             if s),
            default=9.9,
        )
        tilt_diff = min(
            (abs(tilt_now - t) for t in
             (v["quality"]["metrics"].get("board_tilt_deg") for v in previous_views)
             if t is not None),
            default=99.0,
        )
        metrics["nearest_span_log_ratio"] = float(scale_diff)
        metrics["nearest_tilt_diff_deg"] = float(tilt_diff)
        if not new_cells:
            enough = len(already) >= Q_CAM["min_grid_cells"]
            if enough and (scale_diff > 0.15 or tilt_diff > 10.0):
                # 覆盖够了，但这张在"板的大小"或"倾角"上与已有视图明显不同，
                # 对焦距/畸变仍有约束价值 —— 放行
                metrics["accepted_by"] = ("scale" if scale_diff > 0.15 else "tilt")
            elif enough:
                reasons.append(
                    "画面覆盖已经够了，但这张在「板的大小」和「倾角」上也和已有视图差不多。"
                    "请改变**距离**（把板拿近或拿远约四分之一）或**明显改变倾角**")
            else:
                reasons.append(
                    "这张不增加画面覆盖（板只落在已经拍过的格子里）——"
                    "请把板**横向挪到还没拍过的区域**（前/后移动解决不了）")
                reasons.append("建议挪到：" + suggest_cells(previous_views, n))
        dmin = min(
            math.hypot(center[0] - v["quality"]["metrics"]["board_center_px"][0],
                       center[1] - v["quality"]["metrics"]["board_center_px"][1])
            for v in previous_views
            if v.get("quality", {}).get("metrics", {}).get("board_center_px")
        )
        metrics["nearest_view_center_px"] = float(dmin)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return ShotQuality(not reasons, reasons, metrics), {
        "corners_px": corners_px.tolist(),
        "T_camera_board": T.tolist(),
    }


def solve_intrinsics(views: list[dict], size: tuple[int, int],
                     detector: BoardDetector, max_rounds: int = 6,
                     fix_focal: float | None = None,
                     rational: bool = False):
    """用一组标定板视图求解相机内参，按单视图重投影误差迭代剔异常视图。

    fix_focal: 给定则把 fx=fy 固定成该值（用实测距离锚定焦距）。
               **广角镜头 + 低阶畸变模型时 f 与畸变会互相补偿**，锚定是标准做法。
    rational:  用 8 参数有理畸变模型（CALIB_RATIONAL_MODEL），适合大畸变广角。
    """
    obj = np.asarray(detector.object_pts, np.float32)
    base_flags = cv2.CALIB_RATIONAL_MODEL if rational else 0
    guess = None
    if fix_focal:
        guess = np.array([[float(fix_focal), 0.0, size[0] / 2.0],
                          [0.0, float(fix_focal), size[1] / 2.0],
                          [0.0, 0.0, 1.0]], float)
        base_flags |= (cv2.CALIB_USE_INTRINSIC_GUESS
                       | cv2.CALIB_FIX_FOCAL_LENGTH)
    cur = list(views)
    history = []
    while True:
        objs = [obj.copy() for _ in cur]
        imgs = [np.asarray(v["corners_px"], np.float32).reshape(-1, 1, 2) for v in cur]
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            objs, imgs, size, None if guess is None else guess.copy(), None,
            flags=base_flags,
        )
        errs = []
        for o, ip, rv, tv in zip(objs, imgs, rvecs, tvecs):
            proj, _ = cv2.projectPoints(o, rv, tv, K, dist)
            errs.append(float(np.sqrt(np.mean(
                np.sum((proj.reshape(-1, 2) - ip.reshape(-1, 2)) ** 2, axis=1)))))
        history.append({"n": len(cur), "rms_px": float(rms),
                        "mean_view_err_px": float(np.mean(errs)),
                        "max_view_err_px": float(np.max(errs))})
        if len(cur) <= max(8, len(views) - max_rounds):
            break
        worst = int(np.argmax(errs))
        med = float(np.median(errs))
        if errs[worst] < max(2.5 * med, 0.8):
            break
        cur.pop(worst)
    objs = [obj.copy() for _ in cur]
    imgs = [np.asarray(v["corners_px"], np.float32).reshape(-1, 1, 2) for v in cur]
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objs, imgs, size, None if guess is None else guess.copy(), None,
        flags=base_flags,
    )
    per_view = []
    for v, o, ip, rv, tv in zip(cur, objs, imgs, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, rv, tv, K, dist)
        per_view.append({
            "name": v["name"],
            "reproj_rms_px": float(np.sqrt(np.mean(
                np.sum((proj.reshape(-1, 2) - ip.reshape(-1, 2)) ** 2, axis=1)))),
            "tilt_deg": v["quality"]["metrics"].get("board_tilt_deg"),
            "distance_m": v["quality"]["metrics"].get("distance_m"),
            "coverage_cell": v["quality"]["metrics"].get("coverage_cell"),
        })
    dropped = [v["name"] for v in views if v not in cur]
    return {
        "distortion_model": ("rational_8param" if rational
                             else "brown_conrady_opencv"),
        "focal_fixed": fix_focal,
        "image_width": size[0],
        "image_height": size[1],
        "camera_matrix": K.tolist(),
        "fx": float(K[0, 0]), "fy": float(K[1, 1]),
        "cx": float(K[0, 2]), "cy": float(K[1, 2]),
        "distortion_coefficients": [float(x) for x in np.asarray(dist).ravel()],
        "rms_reprojection_px": float(rms),
        "views_used": len(cur),
        "views_dropped": dropped,
        "per_view": per_view,
        "refinement_history": history,
    }


def camera_calib_adequacy(views: list[dict], n_grid: int) -> dict:
    cells = covered_cells(views, n_grid)
    tilts = [v["quality"]["metrics"].get("board_tilt_deg", 0) for v in views]
    spans = [v["quality"]["metrics"].get("board_span_px", 0) for v in views]
    notes = [f"视图 {len(views)} 个（建议 ≥ {Q_CAM['min_views']}，"
             f"但 ≥8 个就能先解一版）",
             f"画面覆盖 {len(cells)}/{n_grid * n_grid} 格"
             f"（每格 = {640 // n_grid}x{480 // n_grid} 像素区域）"]
    if tilts:
        notes.append(f"倾角范围 {min(tilts):.0f}°~{max(tilts):.0f}°")
    if spans:
        notes.append(f"板像素跨度 {min(spans):.0f}~{max(spans):.0f}px")
    ok = len(views) >= 8 and len(cells) >= Q_CAM["min_grid_cells"]
    if len(cells) < Q_CAM["min_grid_cells"]:
        notes.append(f"覆盖不足 {Q_CAM['min_grid_cells']} 格：内参需要板出现在画面各处，"
                     "尤其是四个角")
    if spans and max(spans) / max(1.0, min(spans)) < 1.25:
        notes.append("板的像素尺寸变化太小：请在不同距离各拍几张")
    if tilts and (max(tilts) - min(tilts)) < 15:
        notes.append("倾角变化太小：请让板有时正对镜头、有时明显倾斜")
    return {"sufficient": ok, "view_count": len(views),
            "cells": sorted(list(cells)), "notes": notes}


# ---------------------------------------------------------------------------
# 4. 机械臂关节角（默认只读）
# ---------------------------------------------------------------------------


class ArmReader:
    """只读关节角。不发送任何位置/速度目标。"""

    def __init__(self, port: str, free_drive: bool = False,
                 free_damping: float = 1.0) -> None:
        import airbot_hardware_py as ah

        self.ah = ah
        self.port = port
        self.free_drive = free_drive
        self.free_damping = float(free_damping)
        # 这套 SDK 只有 INVALID/MIT/CSP/CSV/PVT，**没有重力补偿**。
        # 所以"手推"只能靠 MIT 零力矩；纯零力矩臂会因自重下坠，
        # 这里额外给一点关节阻尼（kp=0, kd>0），显著减慢下坠速度、也更稳，
        # 但仍然无法抵消重力 —— 手一定要托着。
        self._keep_alive: threading.Thread | None = None
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self.free_drive_active = bool(free_drive)
        self.executor = ah.create_asio_executor(8)
        self.arm = ah.Play.create(
            ah.MotorType.OD,
            ah.MotorType.OD,
            ah.MotorType.OD,
            ah.MotorType.DM,
            ah.MotorType.DM,
            ah.MotorType.DM,
            ah.EEFType.NA,
            ah.MotorType.NA,
        )
        if not self.arm.init(self.executor.get_io_context(), port, 250):
            raise RuntimeError(
                f"{port} 初始化失败（已知问题：'Motor 1 init failed, expected 38 "
                "params, got 35' 是 AIRBOT SDK 与电机固件不兼容，必须先解决）"
            )
        self.arm.enable()
        if free_drive:
            self.arm.set_param("arm.control_mode", ah.MotorControlMode.MIT)
            self._send_free()
            self._stop.clear()
            self._keep_alive = threading.Thread(target=self._loop, daemon=True)
            self._keep_alive.start()
        else:
            # PVT = 位置保持：电机刚性锁住当前姿态，**手推不动**（这是读关节角的代价）
            self.arm.set_param("arm.control_mode", ah.MotorControlMode.PVT)

    def _send_free(self) -> None:
        """MIT：目标=当前角度、kp=0、kd=阻尼、力矩=0 → 可手推，但下坠被阻尼拖慢。"""
        zeros = [0.0] * ARM_DOF
        gain = [self.free_damping] * ARM_DOF
        try:
            current = [float(v) for v in list(self.arm.state().pos)[:ARM_DOF]]
        except Exception:
            current = zeros
        self.arm.mit(current, zeros, zeros, gain, zeros)

    def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                self._send_free()
                time.sleep(0.004)
        except BaseException as exc:
            self._error = exc
            self._stop.set()

    def check(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"零力矩保持线程停了：{self._error}")

    def joints(self) -> list[float] | None:
        try:
            values = [float(v) for v in list(self.arm.state().pos)[:ARM_DOF]]
        except Exception:
            return None
        if len(values) != ARM_DOF or not np.isfinite(values).all():
            return None
        return values

    def close(self) -> None:
        self._stop.set()
        if self._keep_alive is not None:
            self._keep_alive.join(timeout=1.0)
        for step in (self.arm.disable, self.arm.uninit):
            try:
                step()
            except Exception as exc:
                print(f"  ! {step.__name__} 失败：{exc}")


# ---------------------------------------------------------------------------
# 5. 单张质量判定
# ---------------------------------------------------------------------------


@dataclass
class ShotQuality:
    usable: bool
    reasons: list[str]
    metrics: dict

    def summary(self) -> str:
        if self.usable:
            return "✅ 可用于标定"
        return "❌ 不能用于标定：" + "；".join(self.reasons)


def _rotation_angle_deg(R: np.ndarray) -> float:
    value = (np.trace(R) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, float(value)))))


def assess_shot(
    gray: np.ndarray,
    detector: BoardDetector,
    intr: Intrinsics,
    joints: list[float] | None,
    previous: list[dict],
    *,
    min_joint_change_rad: float,
    require_joints: bool = True,
    verbose: bool = True,
    fast: bool = False,
) -> tuple[ShotQuality, dict | None]:
    """判定这一帧能否用于标定。previous 是已保存样本（含 joints_rad）。

    fast=True 用快速检测（预览用，保证视频流畅）；保存时用精确检测。
    """
    reasons: list[str] = []
    metrics: dict[str, Any] = {}

    detection = (detector.detect_fast(gray)
                 if fast and hasattr(detector, "detect_fast")
                 else detector.detect(gray))
    if detection is None:
        return ShotQuality(False, ["没有检测到标定板"], metrics), None

    corners_px, object_pts = detection
    height, width = gray.shape[:2]

    # --- 余量：板四角必须离图像边界足够远 ---
    margin = float(
        min(
            corners_px[:, 0].min(),
            corners_px[:, 1].min(),
            width - 1 - corners_px[:, 0].max(),
            height - 1 - corners_px[:, 1].max(),
        )
    )
    metrics["edge_margin_px"] = margin
    if margin < Q["min_margin_px"]:
        reasons.append(
            f"标定板离画面边界太近（余量 {margin:.0f}px < {Q['min_margin_px']:.0f}px）"
        )

    # --- 清晰度：Laplacian 方差 ---
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    metrics["sharpness_laplacian_var"] = sharpness
    if sharpness < Q["min_sharpness"]:
        reasons.append(
            f"图像模糊（清晰度 {sharpness:.0f} < {Q['min_sharpness']:.0f}）"
        )

    # --- 板在画面中的占比 ---
    hull = cv2.convexHull(corners_px.astype(np.float32))
    area_frac = float(cv2.contourArea(hull)) / float(width * height)
    metrics["board_area_fraction"] = area_frac
    if area_frac < Q["min_area_frac"]:
        reasons.append(f"标定板太小（占画面 {area_frac * 100:.2f}%）")
    if area_frac > Q["max_area_frac"]:
        reasons.append(f"标定板太大（占画面 {area_frac * 100:.1f}%），容易截断/畸变主导")

    # --- 位姿求解 + 重投影误差 ---
    pixels_for_pnp = intr.undistort(corners_px)
    ok, rvec, tvec = cv2.solvePnP(
        object_pts,
        pixels_for_pnp.reshape(-1, 1, 2),
        intr.camera_matrix(),
        intr.dist_coeffs(),
        flags=(
            cv2.SOLVEPNP_IPPE_SQUARE
            if object_pts.shape[0] == 4
            else cv2.SOLVEPNP_ITERATIVE
        ),
    )
    if not ok:
        return ShotQuality(False, reasons + ["solvePnP 失败"]), None
    projected, _ = cv2.projectPoints(
        object_pts, rvec, tvec, intr.camera_matrix(), intr.dist_coeffs()
    )
    reproj = float(
        np.sqrt(
            np.mean(
                np.sum((projected.reshape(-1, 2) - pixels_for_pnp) ** 2, axis=1)
            )
        )
    )
    metrics["reprojection_rms_px"] = reproj
    if reproj > Q["max_reproj_px"]:
        reasons.append(
            f"重投影误差偏大（{reproj:.2f}px > {Q['max_reproj_px']:.2f}px）"
        )

    R_board2cam, _ = cv2.Rodrigues(rvec)
    distance = float(np.linalg.norm(tvec))
    metrics["distance_m"] = distance

    # --- 倾角：板法线（板局部 +Z）与相机光轴的夹角 ---
    normal_cam = R_board2cam @ np.array([0.0, 0.0, 1.0])
    optical = np.array([0.0, 0.0, 1.0])
    tilt = math.degrees(
        math.acos(max(-1.0, min(1.0, float(abs(normal_cam @ optical)))))
    )
    metrics["board_tilt_deg"] = tilt
    if tilt > Q["max_tilt_deg"]:
        reasons.append(
            f"标定板太正对相机（倾角 {tilt:.0f}° > {Q['max_tilt_deg']:.0f}°），"
            "这种姿态对手眼标定贡献小"
        )

    # --- 关节角有效性 ---
    if joints is None:
        if require_joints:
            reasons.append("读不到机械臂关节角（无法算 T_base_eef）")
        else:
            # 预览/未连机械臂时，关节角本来就没有；不因此把画面质量判定也拖成失败
            metrics["joints_note"] = "预览模式：未连机械臂，关节角不参与本次判定"
    else:
        bad = [
            i + 1
            for i, value in enumerate(joints)
            if not (JOINT_LIMITS[i][0] - 0.05 <= value <= JOINT_LIMITS[i][1] + 0.05)
        ]
        if bad:
            reasons.append(f"关节 {bad} 读数超出 URDF 限位，疑似读错")
        metrics["joints_rad"] = list(joints)

        # --- 与已有样本的差异（防止样本退化） ---
        if previous and joints is not None:
            diffs = [
                max(
                    abs((a - b + math.pi) % (2 * math.pi) - math.pi)
                    for a, b in zip(joints, sample["joints_rad"])
                )
                for sample in previous
                if sample.get("joints_rad")
            ]
            nearest = min(diffs) if diffs else float("inf")
            metrics["nearest_sample_joint_diff_rad"] = nearest
            if nearest < min_joint_change_rad:
                reasons.append(
                    f"与已有样本太接近（最近差异 {nearest:.3f} rad < "
                    f"{min_joint_change_rad:.3f} rad），继续增加这种姿态没有信息量"
                )

    metrics["T_camera_board"] = np.eye(4)
    metrics["T_camera_board"][:3, :3] = R_board2cam
    metrics["T_camera_board"][:3, 3] = tvec.reshape(3)

    return ShotQuality(not reasons, reasons, metrics), {
        "corners_px": corners_px.tolist(),
        "T_camera_board": metrics["T_camera_board"].tolist(),
        "tvec_m": tvec.reshape(3).tolist(),
    }


def grab_stable(
    camera,
    detector: BoardDetector,
    intr: Intrinsics,
    arm: ArmReader | None,
    frames: int,
) -> tuple[np.ndarray | None, list[float] | None, dict]:
    """连读 frames 帧，返回最后一帧、关节角、以及稳定性指标。

    稳定性检查只用快速检测（88 个内角点的 EXHAUSTIVE 检测可达秒级，
    5 帧就能把推流卡住十几秒）；精确检测留给最后那一帧。
    """
    fast = getattr(detector, "detect_fast", detector.detect)
    pose_t, pose_R, joint_track = [], [], []
    last = None
    for _ in range(frames):
        frame = camera.read()
        if frame is None:
            continue
        last = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        det = fast(gray)
        if det is not None:
            corners, obj = det
            ok, rvec, tvec = cv2.solvePnP(
                obj,
                intr.undistort(corners).reshape(-1, 1, 2),
                intr.camera_matrix(),
                intr.dist_coeffs(),
                flags=(
                    cv2.SOLVEPNP_IPPE_SQUARE if obj.shape[0] == 4 else cv2.SOLVEPNP_ITERATIVE
                ),
            )
            if ok:
                pose_t.append(tvec.reshape(3))
                pose_R.append(cv2.Rodrigues(rvec)[0])
        if arm is not None:
            j = arm.joints()
            if j is not None:
                joint_track.append(j)
            arm.check()
        time.sleep(0.02)

    stats: dict[str, Any] = {}
    if len(pose_t) >= 2:
        stats["pose_trans_std_m"] = float(np.max(np.std(np.stack(pose_t), axis=0)))
        base = pose_R[0]
        stats["pose_rot_std_deg"] = float(
            max(_rotation_angle_deg(R @ base.T) for R in pose_R[1:])
        )
    if len(joint_track) >= 2:
        arr = np.stack(joint_track)
        stats["joint_motion_rad"] = float(np.max(arr.max(axis=0) - arr.min(axis=0)))
    return last, (joint_track[-1] if joint_track else None), stats


# ---------------------------------------------------------------------------
# 6. 会话读写
# ---------------------------------------------------------------------------


def make_session_dir(args) -> Path:
    if args.session:
        return Path(args.session)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cam = "realsense" if args.camera == "realsense" else "uvc"
    if getattr(args, "calibrate_camera", False):
        return Path(args.out_root) / f"camera_intrinsics_{cam}_{stamp}"
    return Path(args.out_root) / f"hand_eye_left_{cam}_{stamp}"


def save_samples(session: Path, samples: list[dict], meta: dict) -> None:
    session.mkdir(parents=True, exist_ok=True)
    payload = {
        "purpose": "left-wrist eye-in-hand hand-eye calibration samples",
        "eef_frame": meta.get("eef_frame", "link6"),
        "eef_frame_note": (
            "T_base_eef 由厂商 play_e2.urdf 的正运动学给出；已验证与 MuJoCo 生成"
            "模型的 link6 body 位姿一致（误差<1e-6），可直接用于 MuJoCo 场景。"
        ),
        "camera": meta.get("camera", {}),
        "board": meta.get("board", {}),
        "thresholds": meta.get("thresholds", {}),
        "samples": samples,
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    (session / "samples.yaml").write_text(text, encoding="utf-8")


def load_samples(session: Path) -> tuple[dict, list[dict]]:
    path = session / "samples.yaml"
    if not path.is_file():
        raise SystemExit(f"找不到 {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data, data.get("samples", [])


# ---------------------------------------------------------------------------
# 7. 求解 T_eef_camera
# ---------------------------------------------------------------------------

HAND_EYE_METHODS = {
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "park": cv2.CALIB_HAND_EYE_PARK,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def solve_hand_eye(samples: list[dict], method: str = "tsai") -> dict:
    if len(samples) < 3:
        raise SystemExit(f"样本只有 {len(samples)} 个，至少需要 3 个才能求解")
    R_g2b, t_g2b, R_t2c, t_t2c = [], [], [], []
    for s in samples:
        T_be = np.asarray(s["T_base_eef"], float)
        T_cb = np.asarray(s["T_camera_board"], float)
        R_g2b.append(T_be[:3, :3])
        t_g2b.append(T_be[:3, 3].reshape(3, 1))
        R_t2c.append(T_cb[:3, :3])
        t_t2c.append(T_cb[:3, 3].reshape(3, 1))
    R_cam2eef, t_cam2eef = cv2.calibrateHandEye(
        R_g2b,
        t_g2b,
        R_t2c,
        t_t2c,
        method=HAND_EYE_METHODS.get(method, cv2.CALIB_HAND_EYE_TSAI),
    )
    T_eef_camera = np.eye(4)
    T_eef_camera[:3, :3] = R_cam2eef
    T_eef_camera[:3, 3] = t_cam2eef.reshape(3)

    # 残差：所有样本算出的 T_base_board 应一致
    base_boards = []
    for s in samples:
        T_be = np.asarray(s["T_base_eef"], float)
        T_cb = np.asarray(s["T_camera_board"], float)
        base_boards.append(T_be @ T_eef_camera @ T_cb)
    mean_t = np.mean([T[:3, 3] for T in base_boards], axis=0)
    trans_err = [float(np.linalg.norm(T[:3, 3] - mean_t)) for T in base_boards]
    R0 = base_boards[0][:3, :3]
    rot_err = [
        _rotation_angle_deg(T[:3, :3] @ R0.T) for T in base_boards
    ]
    return {
        "method": method,
        "sample_count": len(samples),
        "T_eef_camera": T_eef_camera.tolist(),
        "validation": {
            "translation_rms_m": float(np.sqrt(np.mean(np.square(trans_err)))),
            "translation_max_m": float(np.max(trans_err)),
            "rotation_rms_deg": float(np.sqrt(np.mean(np.square(rot_err)))),
            "rotation_max_deg": float(np.max(rot_err)),
            "per_sample_translation_errors_m": trans_err,
            "per_sample_rotation_errors_deg": rot_err,
        },
    }


def split_half_check(samples: list[dict], method: str = "tsai") -> dict:
    """对半分割验证：两半独立求解，比较结果。

    这个检查比残差 RMS 更重要：**当数据存在系统不一致（比如板滑过）时，
    残差 RMS 会被"平均"掉而显得很小，但对半求解的差异会如实暴露出来。**
    """
    n = len(samples)
    if n < 6:
        return {}
    out = {}
    for tag, A, B in (("odd_even", samples[0::2], samples[1::2]),
                      ("first_second", samples[: n // 2], samples[n // 2:])):
        if len(A) < 3 or len(B) < 3:
            continue
        try:
            Xa = np.asarray(solve_hand_eye(A, method)["T_eef_camera"], float)
            Xb = np.asarray(solve_hand_eye(B, method)["T_eef_camera"], float)
        except Exception:
            continue
        dR = Xa[:3, :3] @ Xb[:3, :3].T
        out[tag] = {
            "n_a": len(A),
            "n_b": len(B),
            "translation_diff_m": float(np.linalg.norm(Xa[:3, 3] - Xb[:3, 3])),
            "rotation_diff_deg": _rotation_angle_deg(dR),
            "distance_a_m": float(np.linalg.norm(Xa[:3, 3])),
            "distance_b_m": float(np.linalg.norm(Xb[:3, 3])),
        }
    return out


def consistency_outliers(samples: list[dict], tol_deg: float = 3.0):
    """不需要知道 X 的自洽性检验。

    手眼关系 `T_base_eef_i @ X @ T_cam_board_i = 常量` 要求：任意两帧之间，
    EEF 的相对旋转角必须等于标定板的相对旋转角（共轭变换保角）。
    某帧若与其余所有帧都对不上，它就是异常帧 —— 常见原因是板被挪过，
    或那一帧的关节角与图像不是同一时刻。

    返回 (inlier_idx, outlier_idx, {样本下标: 中位不一致度数})。
    """
    good = [i for i, s in enumerate(samples)
            if s.get("joints_rad") and s.get("T_base_eef") and s.get("T_camera_board")]
    if len(good) < 3:
        return list(range(len(samples))), [], {}
    A = [np.asarray(samples[i]["T_base_eef"], float) for i in good]
    B = [np.asarray(samples[i]["T_camera_board"], float) for i in good]
    n = len(good)

    def ang(R):
        return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))

    mism = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            ra = ang(A[i][:3, :3].T @ A[j][:3, :3])
            rb = ang(B[i][:3, :3] @ B[j][:3, :3].T)
            mism[i, j] = mism[j, i] = abs(ra - rb)
    med = np.array([np.median(np.delete(mism[i], i)) for i in range(n)])

    inliers = [good[i] for i in range(n) if med[i] <= tol_deg]
    outliers = [good[i] for i in range(n) if med[i] > tol_deg]
    per_sample = {good[i]: float(med[i]) for i in range(n)}
    return inliers, outliers, per_sample


def solve_robust(samples: list[dict], method: str = "tsai", tol_deg: float = 3.0):
    """先按自洽性剔掉异常帧，再求解。返回 (稳健结果, 报告)。"""
    inliers, outliers, per_sample = consistency_outliers(samples, tol_deg)
    report: dict[str, Any] = {
        "total": len(samples),
        "method": method,
        "consistency_tolerance_deg": tol_deg,
        "used_sample_names": [samples[i]["name"] for i in inliers],
        "dropped_sample_names": [samples[i]["name"] for i in outliers],
        "per_sample_mismatch_deg": {samples[k]["name"]: round(v, 3)
                                    for k, v in per_sample.items()},
    }
    raw = solve_hand_eye(samples, method) if len(samples) >= 3 else None
    report["raw"] = raw["validation"] if raw else None
    keep = [samples[i] for i in inliers]
    if len(keep) < 3:
        report["robust"] = None
        return raw, report
    robust = solve_hand_eye(keep, method)
    report["robust"] = robust["validation"]
    return robust, report


def session_adequacy(samples: list[dict]) -> dict:
    """告诉用户这批数据够不够标定，以及缺什么。"""
    notes: list[str] = []
    n = len(samples)
    ok = n >= Q["min_samples"]
    if not ok:
        notes.append(f"样本数 {n} < {Q['min_samples']}，标准 eye-in-hand 标定不够稳")

    joints = [s["joints_rad"] for s in samples if s.get("joints_rad")]
    if len(joints) < n:
        notes.append(f"有 {n - len(joints)} 个样本没有关节角，无法参与手眼求解")
        ok = False
    if joints:
        arr = np.stack(joints)
        spread = arr.max(axis=0) - arr.min(axis=0)
        moving = int(np.sum(spread > 0.15))
        notes.append(
            "各关节跨度(rad)：" + ", ".join(f"j{i + 1}={v:.3f}" for i, v in enumerate(spread))
        )
        if moving < 4:
            notes.append(
                f"只有 {moving} 个关节有明显的位姿变化，腕部姿态多样性不足；"
                "标定需要改变腕部的左右/上下/前后倾斜**和**绕光轴旋转"
            )
            ok = False

        # 手眼可观测性：相对运动的旋转轴必须张成三维
        axes = []
        for i in range(1, len(joints)):
            dR = np.asarray(samples[i]["T_base_eef"], float)[:3, :3] @ np.asarray(
                samples[0]["T_base_eef"], float
            )[:3, :3].T
            angle = math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2)))
            if angle > math.radians(3):
                rvec = cv2.Rodrigues(dR)[0].reshape(3)
                axes.append(rvec / (np.linalg.norm(rvec) + 1e-12))
        if axes:
            sing = np.linalg.svd(np.stack(axes), compute_uv=False)
            ratios = (sing / (sing[0] + 1e-12)).tolist()
            notes.append("相对旋转轴奇异值比：" + ", ".join(f"{v:.3f}" for v in ratios))
            if len(ratios) < 3 or ratios[2] < 0.02:
                notes.append(
                    "相对旋转几乎共面/共轴，手眼标定会退化；请加入绕不同轴的腕部旋转"
                )
                ok = False
        else:
            notes.append(
                f"{len(joints)} 个样本之间几乎没有相对旋转（全部都小于 3°）。"
                "手眼标定的方程 X 只能从旋转差异中解出，纯平移样本无解。"
                "必须让腕部在不同姿态之间发生明显转动。"
            )
            ok = False
    return {"sufficient": ok, "sample_count": n, "notes": notes}


# ---------------------------------------------------------------------------
# 8. 采集主循环（OpenCV 窗口 或 浏览器）
# ---------------------------------------------------------------------------


class CaptureSession:
    def __init__(self, args) -> None:
        self.args = args
        self.detector = make_detector(args)
        self.kdl = UrdfKinematics(Path(args.urdf), args.eef_link)
        self.samples: list[dict] = []
        self.session = make_session_dir(args)
        # 支持"续用已有会话"：重启用同一个 --session 目录时，把已经存下的样本读回来，
        # 否则重启会把 samples.yaml 覆盖成空、前面的数据白拍。
        self.resumed = 0
        existing = self.session / "samples.yaml"
        if existing.is_file():
            try:
                old = json.loads(existing.read_text(encoding="utf-8"))
                restored = [s for s in old.get("samples", [])
                            if s.get("joints_rad") and s.get("T_camera_board")]
                if restored:
                    self.samples = restored
                    self.resumed = len(restored)
            except Exception:
                pass
        (self.session / "images").mkdir(parents=True, exist_ok=True)
        self.solve_requested = False
        self.stop_requested = False
        self.last_frame: np.ndarray | None = None
        self.last_quality: ShotQuality | None = None
        self.last_metrics: dict = {}
        self.status = "启动中…"
        self.lock = threading.Lock()
        self.meta: dict = {}
        # --preview 强制不连机械臂，因此关节角永远拿不到，存下来的图**不能**做手眼。
        # 实测有人在这里拍了 40 张才发现，所以预览模式直接禁止保存。
        self.preview_only = bool(getattr(self.args, "preview", False))
        # 两种采集目标：手眼（要关节角）与相机内参（只动板、不要关节角）
        self.calib_camera = bool(getattr(self.args, "calibrate_camera", False))
        self.views: list[dict] = []          # 相机内参模式的视图（放内存里方便算覆盖）
        if self.calib_camera:
            existing = self.session / "views.json"
            if existing.is_file():
                try:
                    self.views = json.loads(existing.read_text(encoding="utf-8"))
                    self.resumed = len(self.views)
                except Exception:
                    pass

    # -- 相机内参模式：拍一张标定板视图 --
    def _try_capture_view(self, camera, intr) -> None:
        best = None
        for _ in range(Q_CAM["stable_frames"]):
            frame = camera.read()
            if frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            quality, detection = assess_board_view(
                gray, self.detector, intr, self.views)
            if best is None or quality.metrics.get(
                    "sharpness_laplacian_var", 0) > best[2].metrics.get(
                        "sharpness_laplacian_var", 0):
                best = (frame, quality, quality, detection)
        if best is None:
            self.status = "取不到图像帧"
            return
        frame, quality, _, detection = best

        if not quality.usable and not self.args.save_any:
            self.status = "⛔ 这张不能用（没保存）"
            print("⛔ 这张没保存")
            self.log_rejected(quality, None)
            return

        index = len(self.views) + 1
        name = f"view_{index:03d}"
        raw_path = self.session / "images" / f"{name}_raw.png"
        ann_path = self.session / "images" / f"{name}.png"
        cv2.imwrite(str(raw_path), frame)
        cv2.imwrite(str(ann_path), draw_overlay(frame, detection, "ready", index,
                                                len(self.views) + 1))
        record = {
            "name": name,
            "image": str(ann_path),
            "raw_image": str(raw_path),
            "corners_px": detection["corners_px"],
            "T_camera_board": detection["T_camera_board"],
            "quality": {
                "usable": quality.usable,
                "reasons": quality.reasons,
                "metrics": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                            for k, v in quality.metrics.items()},
            },
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        self.views.append(record)
        (self.session / "views.json").write_text(
            json.dumps(self.views, ensure_ascii=False), encoding="utf-8")
        ad = camera_calib_adequacy(self.views, Q_CAM["grid"])
        added = quality.metrics.get("new_cell_count")
        self.status = (f"✅ 已存 {name}（共 {len(self.views)} 个视图，"
                       f"覆盖 {len(ad['cells'])}/{Q_CAM['grid'] ** 2} 格"
                       + (f"，本张新增 {added} 格" if added is not None else "")
                       + "）")
        print(self.status)
        for line in coverage_ascii({tuple(c) for c in ad["cells"]}, Q_CAM["grid"]):
            print("      " + line)

    # -- 采集一张并判定 --
    def log_rejected(self, quality: ShotQuality, joints) -> None:
        """被拒的照片原因写进 rejected.jsonl，终端不再刷屏。"""
        try:
            entry = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "reasons": quality.reasons,
                "joints_rad": joints,
                "metrics": {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in quality.metrics.items()
                },
            }
            with (self.session / "rejected.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def try_capture(self, camera, intr, arm) -> None:
        if self.preview_only:
            self.status = "预览模式不保存 —— 请用 --capture 正式采集"
            print("⛔ 这是预览模式，不保存照片。正式采集请把 --preview 换成 --capture"
                  "（并确保机械臂连上，否则没有关节角、一样做不了手眼标定）。")
            return
        if self.calib_camera:
            return self._try_capture_view(camera, intr)
        frame, joints, stats = grab_stable(
            camera, self.detector, intr, arm, Q["stable_frames"]
        )
        if frame is None:
            self.status = "取不到图像帧"
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        quality, detection = assess_shot(
            gray,
            self.detector,
            intr,
            joints,
            self.samples,
            min_joint_change_rad=Q["min_joint_change_rad"],
            require_joints=arm is not None,
        )
        # 稳定性并入判定（只有在确实检测到板时才有意义，否则会出现
        # "没有检测到标定板 + 标定板在晃动" 这种自相矛盾的话）
        if detection is not None:
            if stats.get("pose_trans_std_m", 0) > Q["max_pose_trans_std_m"]:
                quality.reasons.append(
                    f"标定板/机械臂在拍摄期间晃动（平移 std {stats['pose_trans_std_m'] * 1000:.1f}mm）"
                )
            rot_std = stats.get("pose_rot_std_deg", 0)
            if rot_std > 20.0:
                quality.reasons.append(
                    f"连续帧之间棋盘格的读法发生了跳变（{rot_std:.0f}°）。"
                    "动的板不会真转 90°，最常见原因是**板只露出一部分**、"
                    "检测器锁到了大板里的子窗口；其次是板太斜/反光。请把整块板放进画面重拍"
                )
            elif rot_std > Q["max_pose_rot_std_deg"]:
                quality.reasons.append(
                    f"标定板姿态在拍摄期间变化（{rot_std:.2f}°）"
                )
            if stats.get("joint_motion_rad", 0) > Q["max_joint_motion_rad"]:
                quality.reasons.append(
                    f"拍摄期间机械臂在动（关节最大变化 {stats['joint_motion_rad']:.4f} rad）"
                )
        quality.metrics.update(stats)
        quality.usable = not quality.reasons

        self.last_quality = quality
        self.last_metrics = quality.metrics
        if joints is None and not self.preview_only:
            # 关节角是手眼标定的必需输入，不是可选字段
            print("⚠ 这张没有关节角 —— **不能用于手眼标定**，只能做相机内参。")
            quality.metrics["joints_missing"] = True
        if not quality.usable and not self.args.save_any:
            # 终端只说结论；具体原因写进 rejected.jsonl，不打扰你
            self.status = "⛔ 现在还不能拍（这张没保存）"
            print("⛔ 现在还不能拍 —— 这张没有保存")
            self.log_rejected(quality, joints)
            return

        index = len(self.samples) + 1
        name = f"sample_{index:03d}"
        raw_path = self.session / "images" / f"{name}_raw.png"
        ann_path = self.session / "images" / f"{name}.png"
        cv2.imwrite(str(raw_path), frame)
        cv2.imwrite(
            str(ann_path),
            draw_overlay(frame, detection, "ready", index, len(self.samples) + 1),
        )

        record: dict[str, Any] = {
            "name": name,
            "image": str(ann_path),
            "raw_image": str(raw_path),
            "joints_rad": joints,
            "T_base_eef": (
                self.kdl.fk(joints).tolist() if joints is not None else None
            ),
            "T_camera_board": (detection or {}).get("T_camera_board"),
            "detected_corners_px": (detection or {}).get("corners_px"),
            "quality": {
                "usable": quality.usable,
                "reasons": quality.reasons,
                "metrics": {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in quality.metrics.items()
                },
            },
            "forced_save": (not quality.usable),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        self.samples.append(record)
        save_samples(self.session, self.samples, self.meta)
        self.status = f"✅ 已保存 {name}（共 {len(self.samples)} 张）"
        print(f"{self.status}")

    def delete_last(self) -> None:
        if not self.samples:
            self.status = "没有可删除的样本"
            return
        removed = self.samples.pop()
        for key in ("image", "raw_image"):
            try:
                Path(removed[key]).unlink(missing_ok=True)
            except Exception:
                pass
        save_samples(self.session, self.samples, self.meta)
        self.status = f"已删除 {removed['name']}，剩 {len(self.samples)} 个样本"


class ReadyGate:
    """连续 N 帧都合格才算"可以拍"，避免横幅来回闪。"""

    def __init__(self, window: int = 3) -> None:
        self.window = max(1, int(window))
        self.history: list[bool] = []

    def push(self, usable: bool) -> None:
        self.history.append(bool(usable))
        if len(self.history) > self.window:
            self.history.pop(0)

    @property
    def ready(self) -> bool:
        return len(self.history) >= self.window and all(self.history)


def draw_overlay(frame, detection, state: str, index: int, total: int, note: str = ""):
    """横幅只给一个明确状态，不列原因。

    state:
      "ready"     绿  可以拍
      "notready"  红  先别拍
      "nojoints"  橙  在读不到关节角 —— 拍了也没用（手眼标定必须要关节角）
      "preview"   蓝  预览模式，不保存
      "free"      青  机械臂处于自由拖动（可手推）
    """
    out = frame.copy()
    if detection is not None:
        corners = np.asarray(detection["corners_px"], np.float32).reshape(-1, 1, 2)
        cv2.polylines(out, [corners.astype(np.int32)], True, (0, 255, 0), 2)

    h, w = out.shape[:2]
    bar = 58
    if state == "preview":
        bg = (150, 90, 0)
        text = "PREVIEW ONLY - NOT SAVED"
    elif state == "nojoints":
        bg = (0, 110, 200)
        text = "NO JOINT DATA - NOT USABLE"
    elif state == "free":
        bg = (150, 130, 0)
        text = "ARM FREE - PUSH BY HAND"
    elif state == "ready":
        bg = (0, 150, 0)
        text = "OK - PRESS S TO SAVE"
    else:
        bg = (0, 0, 200)
        text = "NOT READY - DO NOT SAVE"
    cv2.rectangle(out, (0, 0), (w - 1, bar), bg, -1)

    scale, thick = 0.72, 2
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.putText(out, text, ((w - tw) // 2 - 18, (bar + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick)

    # 语言无关的图标
    cx, cy, r = w - 34, bar // 2, 17
    cv2.circle(out, (cx, cy), r, (255, 255, 255), -1)
    if state == "ready":
        cv2.polylines(
            out,
            [np.array([[cx - 8, cy], [cx - 2, cy + 7], [cx + 9, cy - 7]], np.int32)],
            False, (0, 130, 0), 3,
        )
    elif state == "preview":
        cv2.rectangle(out, (cx - 6, cy - 8), (cx + 6, cy + 8), (150, 90, 0), -1)
    elif state == "nojoints":
        cv2.line(out, (cx, cy - 9), (cx, cy + 9), (0, 110, 200), 5)
    elif state == "free":
        cv2.circle(out, (cx, cy), 7, (150, 130, 0), -1)
    else:
        cv2.line(out, (cx - 7, cy - 7), (cx + 7, cy + 7), (0, 0, 190), 3)
        cv2.line(out, (cx - 7, cy + 7), (cx + 7, cy - 7), (0, 0, 190), 3)

    cv2.putText(out, f"saved: {total}", (8, bar + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    if note:
        cv2.putText(out, note[:70], (8, bar + 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return out


def live_assess(session, gray, intr, joints, arm):
    """实时预览用的判定：按模式分派；一律用快速检测保证视频流畅。"""
    if session.calib_camera:
        return assess_board_view(gray, session.detector, intr, session.views,
                                 fast=True)
    return assess_shot(
        gray, session.detector, intr, joints, session.samples,
        min_joint_change_rad=Q["min_joint_change_rad"],
        require_joints=arm is not None,
        fast=True,
    )


def display_state(session, arm, gate) -> str:
    if session.preview_only:
        return "preview"
    if session.calib_camera:
        # 内参标定不需要关节角，所以不存在 nojoints 状态
        return "ready" if gate.ready else "notready"
    if arm is None:
        # 采集模式下读不到关节角 —— 这种照片存了也不能做手眼标定
        return "nojoints"
    if getattr(arm, "free_drive_active", False):
        return "free"
    return "ready" if gate.ready else "notready"


def run_capture(args) -> int:
    session = CaptureSession(args)
    camera = arm = intr = None
    try:
        mode = ("camera_calib" if session.calib_camera
                else ("preview" if getattr(args, "preview", False) else "capture"))
        print("=" * 72)
        if mode == "camera_calib":
            print("模式：相机内参标定（--calibrate-camera）")
            print("  · **相机固定不动，只移动/旋转标定板**（这是标准做法）")
            print("  · 不需要机械臂、不需要关节角")
            print("  · 手持会抖，所以只看清晰度与重投影误差，不算晃动")
            print(f"  · 目标：≥ {Q_CAM['min_views']} 个视图，"
                  f"覆盖画面 {Q_CAM['grid']}x{Q_CAM['grid']} 网格里的至少 "
                  f"{Q_CAM['min_grid_cells']} 格（**四角也要拍到**）")
        elif mode == "preview":
            print("模式：预览（--preview）")
            print("  · 不连接机械臂，**不记录关节角**")
            print("  · 因此本模式**不保存照片** —— 存下来也做不了手眼标定")
            print("  · 这里只用来看画面、确认棋盘格能被检测到、把姿态调好")
            print("  · 正式采集请把 --preview 换成 --capture")
        else:
            print("模式：采集（--capture）")
            print("  · 会保存「照片 + 当时的 6 个关节角」，这才是手眼标定要的数据")
        print("=" * 72)
        print(f"会话目录：{session.session}")
        if session.resumed:
            unit = "个视图" if session.calib_camera else "个有效样本"
            print(f"↻ 续用已有会话：已恢复 {session.resumed} {unit}，继续往上加")
        print(f"标定板：{session.detector.describe()}")
        if args.board == "checkerboard":
            print(f"  假设方格边长 = {args.square_mm} mm  ← 必须是实测值！"
                  "填错会让整个标定尺度线性缩放，而残差照样很小、看不出来")
        if mode == "camera_calib":
            print("  提示：板要举到镜头前约 25~40cm，让板占画面 15%~50%；"
                  "太远角点只有几像素，标不准")
        print(f"EEF 坐标系：{args.eef_link}（URDF: {args.urdf}）")
        camera = make_camera(args)
        intr = camera.intrinsics()
        session.meta = {
            "mode": mode,
            "camera": intr.as_dict(),
            "board": session.detector.describe(),
            "eef_frame": args.eef_link,
            "urdf": str(args.urdf),
            "thresholds": Q,
        }
        if (not session.calib_camera and intr.serial and args.serial
                and args.serial not in intr.serial):
            print(
                f"  ! 警告：配置里的左腕序列号 {args.serial} 与实机 {intr.serial} 不一致。"
                f"\n    若相机换过，scene.yaml 里的内参也需要重新标定。"
            )
        print(
            "相机内参：fx=%.2f fy=%.2f cx=%.2f cy=%.2f（%s，畸变 %s）"
            % (intr.fx, intr.fy, intr.cx, intr.cy, intr.source,
               intr.distortion_model or "none")
        )
        if intr.undistort_pixels:
            print("  → RealSense 反畸变模型，角点会先转成无畸变像素再送入 solvePnP")

        if args.board == "checkerboard" and (args.autodetect_board or args.preview):
            probe = camera.read()
            if probe is not None:
                report_autodetect(cv2.cvtColor(probe, cv2.COLOR_BGR2GRAY), args, intr)

        if not args.no_arm:
            if args.free_drive:
                print(f"连接机械臂 {args.arm_port}（MIT 自由拖动：可手推）…")
                print(
                    "\n!! 这套 SDK 没有重力补偿，自由拖动时机械臂仍会因自重缓慢下坠。\n"
                    f"!! 已加关节阻尼 {args.free_damping}（--free-damping 可调，0=纯零力矩）。\n"
                    "!! 手一定要托着；确认急停可用、清空周围障碍物再继续。"
                )
                input("托稳后按回车继续…")
            else:
                print(f"连接机械臂 {args.arm_port}（只读；会锁住当前姿态）…")
                print("   注意：只读模式电机上电并保持位置，**手推不动**。")
                print("   想手摆姿态请加 --free-drive。")
            try:
                arm = ArmReader(args.arm_port, free_drive=args.free_drive,
                                free_damping=args.free_damping)
                print("机械臂已连接，当前关节角：",
                      [round(v, 4) for v in (arm.joints() or [])])
                print("  ✅ 本会话会记录关节角，可以用于手眼标定")
                if args.free_drive:
                    print("  ✅ 现在是自由拖动状态：直接用手把臂推到目标姿态")
                else:
                    print("  ⚠ 现在是锁定状态：手推不动；用遥操作移动，或改用 --free-drive")
            except Exception as exc:
                print(f"  ! 机械臂连接失败：{exc}")
                if args.require_arm:
                    raise
                print("  → 继续运行，但关节角无法记录，这批数据只能用于内参/预览。")
                print("  ⚠ 这样保存的照片**不能用于手眼标定**。先解决机械臂连接，"
                      "或用 --check-arm 单独诊断。")
                arm = None
        session.meta["arm_connected"] = arm is not None
        session.meta["arm_port"] = args.arm_port
        if arm is None and not session.preview_only and mode == "capture":
            print("\n" + "!" * 72)
            print("!! 本会话没有关节角：保存的照片不能用于手眼标定（只能做相机内参）")
            print("!" * 72)
        if session.calib_camera:
            if not (session.session / "views.json").exists():
                (session.session / "views.json").write_text(
                    json.dumps(session.views, ensure_ascii=False), encoding="utf-8")
        else:
            save_samples(session.session, session.samples, session.meta)

        if args.web:
            return run_web(session, camera, intr, arm, args)
        if not gui_available():
            print(
                "\n! 当前 OpenCV 是 headless 构建，无法弹出窗口（cv2.imshow 不可用）。"
                "\n  → 自动改用浏览器界面。"
            )
            args.web = True
            return run_web(session, camera, intr, arm, args)
        try:
            return run_window(session, camera, intr, arm, args)
        except cv2.error as exc:
            print(f"\n! 窗口界面不可用（{str(exc)[:120]}）\n  → 自动改用浏览器界面。")
            args.web = True
            return run_web(session, camera, intr, arm, args)
    finally:
        if arm is not None:
            arm.close()
        if camera is not None:
            camera.close()
        if session.calib_camera:
            (session.session / "views.json").write_text(
                json.dumps(session.views, ensure_ascii=False), encoding="utf-8")
            adequacy = camera_calib_adequacy(session.views, Q_CAM["grid"])
            (session.session / "adequacy.json").write_text(
                json.dumps(adequacy, indent=2, ensure_ascii=False), encoding="utf-8")
            print("\n=== 内参标定数据充足性 ===")
            print("结论：" + ("够用 ✅" if adequacy["sufficient"] else "还不够 ❌"))
            for note in adequacy["notes"]:
                print("  - " + note)
            print("  画面覆盖情况（## = 已有视图落到这一格）：")
            for line in coverage_ascii({tuple(c) for c in adequacy["cells"]},
                                       Q_CAM["grid"]):
                print("      " + line)
            if session.solve_requested:
                do_calibrate_camera(session)
        else:
            save_samples(session.session, session.samples, session.meta)
            adequacy = session_adequacy(session.samples)
            (session.session / "adequacy.json").write_text(
                json.dumps(adequacy, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print("\n=== 数据充足性 ===")
            print("结论：" + ("够用 ✅" if adequacy["sufficient"] else "还不够 ❌"))
            for note in adequacy["notes"]:
                print("  - " + note)
            if session.solve_requested:
                do_solve(session.session, args.method, args.consistency_tol_deg)


def run_window(session: CaptureSession, camera, intr, arm, args) -> int:
    window = "hand-eye capture (s=save d=delete c=solve q=quit)"
    gate = ReadyGate(2 if session.calib_camera else 3)
    print("\n按键： s=保存  d=删除上一张  c=求解并退出  q/Esc=退出")
    print("画面顶部绿色横幅 = 可以拍；红色 = 先别拍。")
    while True:
        frame = camera.read()
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        joints = arm.joints() if arm is not None else None
        if arm is not None:
            arm.check()
        quality, detection = live_assess(session, gray, intr, joints, arm)
        gate.push(quality.usable)
        display = display_state(session, arm, gate)
        total = len(session.views) if session.calib_camera else len(session.samples)
        preview = draw_overlay(frame, detection, display, total + 1, total)
        cv2.imshow(window, preview)
        key = cv2.waitKey(1) & 0xFF
        if key in {ord("q"), 27}:
            break
        if key == ord("d"):
            session.delete_last()
        elif key == ord("c"):
            if len(session.samples) < 3:
                session.status = "样本少于 3 个，无法求解"
                continue
            session.solve_requested = True
            break
        elif key == ord("s"):
            session.try_capture(camera, intr, arm)
    cv2.destroyAllWindows()
    return 0


def run_web(session: CaptureSession, camera, intr, arm, args) -> int:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    state = {"frame": None, "jpeg": b"", "quit": False, "pause": False,
             "ready": False, "note": "", "fails": 0, "last_ok": 0.0}
    lock = threading.Lock()
    gate = ReadyGate(2 if session.calib_camera else 3)

    def producer() -> None:
        # 这个线程**绝不能退出**：它一死，浏览器就永远是黑屏（之前就是这样）。
        # 所以整个循环体都包在 try 里，出错只计数、不抛出。
        while not state["quit"]:
            try:
                if state["pause"]:
                    # 采集线程要独占相机，这里让开
                    time.sleep(0.03)
                    continue
                frame = camera.read()
                if frame is None:
                    state["fails"] += 1
                    time.sleep(0.05)
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                try:
                    joints = arm.joints() if arm is not None else None
                    quality, detection = live_assess(session, gray, intr, joints, arm)
                except Exception as exc:
                    quality = ShotQuality(False, [f"判定失败: {exc}"], {})
                    detection = None
                gate.push(quality.usable)
                display = display_state(session, arm, gate)
                with lock:
                    note = state["note"]
                total = (len(session.views) if session.calib_camera
                         else len(session.samples))
                if session.calib_camera and detection is not None:
                    # 实时告诉用户：这张能给画面覆盖新增几格
                    ncell = quality.metrics.get("new_cell_count")
                    already = len(covered_cells(session.views, Q_CAM["grid"]))
                    if ncell is None:
                        cur = quality.metrics.get("coverage_cells") or []
                        ncell = len(cur)
                    note = f"cells +{ncell} new (total {already})"
                    span_now = quality.metrics.get("board_span_px")
                    if span_now:
                        note += f"  span {span_now:.0f}px"
                    if ncell == 0:
                        by = quality.metrics.get("accepted_by")
                        if by:
                            note += f"   ok: {by} diversity"
                        else:
                            note += "   MOVE SIDEWAYS or change DISTANCE/TILT"
                if not gate.ready and quality.reasons:
                    # 横幅只给 GO / NO-GO，这里补一行"为什么不能拍"
                    why = reason_ascii(quality.reasons)
                    if any("不增加画面覆盖" in r for r in quality.reasons):
                        why += ("   -> move to cell "
                                + suggest_cells_ascii(session.views, Q_CAM["grid"]))
                    note = (note + "   |   " if note else "") + "why: " + why
                preview = draw_overlay(frame, detection, display, total + 1,
                                       total, note)
                ok, buf = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    with lock:
                        state["frame"] = frame
                        state["jpeg"] = buf.tobytes()
                        state["ready"] = gate.ready
                        state["fails"] = 0
                        state["last_ok"] = time.monotonic()
            except Exception as exc:
                with lock:
                    state["fails"] += 1
                    state["note"] = f"camera error: {str(exc)[:60]}"
                time.sleep(0.2)
            time.sleep(0.03)

    PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>hand-eye capture</title><style>
body{background:#111;color:#eee;font-family:monospace;margin:0}
img{display:block;max-width:100%}
#b{padding:8px}#s{padding:8px;white-space:pre-wrap}
#go{font-size:40px;font-weight:bold;text-align:center;padding:10px;margin:6px}
.ok{background:#0a0;color:#fff}.no{background:#b00;color:#fff}
button{font-size:16px;margin-right:8px;padding:6px 14px}
</style></head><body>
<div id="go" class="no">还不能拍</div>
<div id="b">
<button id="bsave" onclick="cmd('save')">SAVE (s)</button>
<button onclick="cmd('delete')">DELETE (d)</button>
<button onclick="cmd('solve')">SOLVE (c)</button>
<button onclick="cmd('quit')">QUIT (q)</button>
</div>
<img id="v" src="/stream.mjpg" onerror="setTimeout(()=>{this.src='/stream.mjpg?t='+Date.now()},800)">
<div id="s">…</div>
<script>
function cmd(c){fetch('/cmd?c='+c)}
setInterval(()=>{fetch('/state').then(r=>r.json()).then(d=>{
 var g=document.getElementById('go');
 if(d.preview){ g.textContent='预览模式：不保存'; g.className='no'; }
 else if(d.no_joints){ g.textContent='读不到关节角：拍了也不能标定'; g.className='no'; }
 else { g.textContent = d.ready ? '可以拍（按 S 保存）' : '还不能拍';
        g.className = d.ready ? 'ok' : 'no'; }
 var b=document.getElementById('bsave');
 if(b) b.style.display = (d.preview || d.no_joints) ? 'none' : '';
 document.getElementById('s').textContent='已存 '+d.count+' 张  '+d.status})},400);
document.addEventListener('keydown',e=>{
 if(e.key==='s')cmd('save'); if(e.key==='d')cmd('delete');
 if(e.key==='c')cmd('solve'); if(e.key==='q')cmd('quit');});
</script></body></html>"""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.startswith("/stream.mjpg"):
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=f")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                while not state["quit"]:
                    with lock:
                        data = state["jpeg"]
                    if not data:
                        time.sleep(0.05)
                        continue
                    try:
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n"
                                         b"Content-Length: " + str(len(data)).encode()
                                         + b"\r\n\r\n" + data + b"\r\n")
                    except Exception:
                        # 客户端断了就干净退出这个连接；浏览器会自动重连
                        return
                    time.sleep(0.04)
                return
            if self.path.startswith("/cmd"):
                from urllib.parse import parse_qs, urlparse

                c = parse_qs(urlparse(self.path).query).get("c", [""])[0]
                if c == "save":
                    with lock:
                        ready = state["frame"] is not None
                    if ready:
                        state["pause"] = True
                        state["note"] = "SAVING... (相机暂停，请稍等)"
                        try:
                            time.sleep(0.4)      # 等 producer 彻底退出 wait_for_frames
                            session.try_capture(camera, intr, arm)
                        except Exception as exc:
                            session.status = f"保存出错: {str(exc)[:80]}"
                            print(f"! 保存出错：{exc}")
                        finally:
                            state["pause"] = False
                            state["note"] = ""
                elif c == "delete":
                    session.delete_last()
                elif c == "solve":
                    session.solve_requested = True
                    state["quit"] = True
                elif c == "quit":
                    state["quit"] = True
                self.send_response(204)
                self.end_headers()
                return
            if self.path.startswith("/state"):
                body = json.dumps({
                    "status": session.status,
                    "ready": bool(state.get("ready")),
                    "count": (len(session.views) if session.calib_camera
                              else len(session.samples)),
                    "preview": bool(session.preview_only),
                    "no_joints": bool(arm is None and not session.preview_only
                                      and not session.calib_camera),
                }, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    threading.Thread(target=producer, daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    # 必须真正开始 accept，否则端口只是被 bind 而不会响应请求
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"\n浏览器打开 http://{args.host}:{args.port}")
    print("如果从另一台电脑访问，请用 SSH 端口转发，不要把服务暴露到外网。")
    print("按键： s=保存  d=删除  c=求解  q=退出")
    try:
        while not state["quit"]:
            time.sleep(0.2)
            if arm is not None:
                arm.check()
    except KeyboardInterrupt:
        pass
    finally:
        state["quit"] = True
        server.shutdown()
        server.server_close()
        print("web 界面已关闭")
    return 0


# ---------------------------------------------------------------------------
# 9. 自检（不需要任何硬件）
# ---------------------------------------------------------------------------


def render_synthetic_board(detector: BoardDetector, intr: Intrinsics,
                           T_cam_board: np.ndarray) -> np.ndarray:
    """把标定板按给定位姿画到图像上（仅用于自检）。"""
    img = np.full((intr.height, intr.width), 255, np.uint8)
    to_cam = T_cam_board

    def proj(x: float, y: float):
        cam = to_cam @ np.array([x, y, 0.0, 1.0])
        if cam[2] <= 0:
            raise ValueError("板在相机后面")
        return (intr.fx * cam[0] / cam[2] + intr.cx,
                intr.fy * cam[1] / cam[2] + intr.cy)

    if isinstance(detector, CheckerboardDetector):
        s, cols, rows = detector.square_m, detector.cols, detector.rows
        # 内角点在 (i*s, j*s)，i∈[0,cols) j∈[0,rows)；方格顶点网格从 -1 到 cols/rows
        verts = {
            (gx, gy): proj(gx * s, gy * s)
            for gy in range(-1, rows + 2)
            for gx in range(-1, cols + 2)
        }
        for gy in range(-1, rows):
            for gx in range(-1, cols):
                if (gx + gy) % 2 == 0:
                    poly = np.array(
                        [verts[(gx, gy)], verts[(gx + 1, gy)],
                         verts[(gx + 1, gy + 1)], verts[(gx, gy + 1)]],
                        np.int32,
                    )
                    cv2.fillConvexPoly(img, poly, 0)
        return img

    # ArUco / AprilTag：生成真实图案后做透视变换贴上去
    size_px = 400
    if hasattr(cv2.aruco, "generateImageMarker"):
        marker = cv2.aruco.generateImageMarker(
            detector.dictionary, detector.marker_id, size_px
        )
    else:
        marker = cv2.aruco.drawMarker(
            detector.dictionary, detector.marker_id, size_px
        )
    pixels = np.array([proj(p[0], p[1]) for p in detector.object_pts], np.float32)
    src = np.array(
        [[0, 0], [size_px - 1, 0], [size_px - 1, size_px - 1], [0, size_px - 1]],
        np.float32,
    )
    H = cv2.getPerspectiveTransform(src, pixels.astype(np.float32))
    warped = cv2.warpPerspective(
        marker, H, (intr.width, intr.height), flags=cv2.INTER_LINEAR,
        borderValue=255,
    )
    mask = cv2.warpPerspective(
        np.full_like(marker, 255), H, (intr.width, intr.height),
        flags=cv2.INTER_NEAREST, borderValue=0,
    )
    return np.where(mask > 0, warped, img).astype(np.uint8)


def self_test(args) -> int:
    print("=== 自检（不连接任何硬件）===")
    failures: list[str] = []

    # 1) FK 基准：该值已与 MuJoCo 生成模型的 link6 body 位姿核对过
    kdl = UrdfKinematics(Path(args.urdf), args.eef_link)
    q = [0.3, -0.8, 0.5, 1.1, -0.4, 0.7]
    expected_t = np.array([0.16812, 0.05201, 0.27737])
    got = kdl.fk(q)
    err = float(np.linalg.norm(got[:3, 3] - expected_t))
    print(f"1) URDF FK   q={q}\n   t={np.round(got[:3, 3], 5)}  期望≈{expected_t}"
          f"  误差={err * 1000:.3f} mm")
    if err > 1e-3:
        failures.append("FK 与 MuJoCo 基准不一致")

    # 2) 合成板检测 + 角点配对一致性
    #    棋盘格检测器返回的角点顺序与我们的 object_pts 顺序之间，允许相差一个
    #    "板的固定重标号"（实测是绕板法线 180°）。这个固定变换会被 T_eef_camera
    #    吸收，无害；但如果它随位姿变化，就会毁掉标定。这里验证它是否恒定。
    detector = make_detector(args)
    intr = Intrinsics(width=args.width, height=args.height, fx=394.4, fy=393.4,
                      cx=318.9, cy=250.7, source="synthetic")

    def render_pose(rot_axis, rot_deg, cam_roll_deg=0.0, t=(0.0, 0.0, 0.28)):
        B = np.eye(4)
        B[:3, :3] = _axis_R(list(rot_axis), math.radians(rot_deg))
        B[:3, 3] = t
        C = np.eye(4)
        C[:3, :3] = _axis_R([0, 0, 1], math.radians(cam_roll_deg))
        return C @ B

    offsets = []
    for tag, T_true in [
        ("roll 0°", render_pose([0.3, 0.9, 0.2], 25, 0)),
        ("roll 90°", render_pose([0.3, 0.9, 0.2], 25, 90)),
        ("roll 180°", render_pose([0.3, 0.9, 0.2], 25, 180)),
        ("roll 270°", render_pose([0.3, 0.9, 0.2], 25, 270)),
        ("tilt 10°", render_pose([0.4, 0.8, 0.2], 10, 0)),
        ("tilt 40°", render_pose([0.4, 0.8, 0.2], 40, 0)),
    ]:
        img = render_synthetic_board(detector, intr, T_true)
        found = detector.detect(img)
        if found is None:
            print(f"2) 合成板检测：{tag} 未检测到")
            failures.append(f"合成板在 {tag} 未被检测到")
            continue
        corners, obj = found
        ok, rvec, tvec = cv2.solvePnP(
            obj, intr.undistort(corners).reshape(-1, 1, 2),
            intr.camera_matrix(), intr.dist_coeffs(),
            flags=(cv2.SOLVEPNP_IPPE_SQUARE if obj.shape[0] == 4
                   else cv2.SOLVEPNP_ITERATIVE),
        )
        T_est = np.eye(4)
        T_est[:3, :3] = cv2.Rodrigues(rvec)[0]
        T_est[:3, 3] = tvec.reshape(3)
        offsets.append(np.linalg.inv(T_true) @ T_est)
    if len(offsets) >= 2:
        base = offsets[0]
        dts, drs = [], []
        for E in offsets[1:]:
            Dv = E @ np.linalg.inv(base)
            dts.append(float(np.linalg.norm(Dv[:3, 3])))
            drs.append(_rotation_angle_deg(Dv[:3, :3]))
        print(f"2) 合成板检测 {len(offsets)} 个位姿全部命中；角点配对偏移的一致性："
              f"平移漂移 max={max(dts) * 1000:.3f} mm，旋转漂移 max={max(drs):.3f}°")
        if max(dts) > 3e-3 or max(drs) > 1.5:
            failures.append(
                f"角点配对随位姿变化（{max(dts) * 1000:.2f}mm / {max(drs):.2f}°），"
                "会破坏手眼标定"
            )

    # 3) 合成手眼求解（解析构造，检验求解数学）
    rng = np.random.default_rng(3)
    T_eef_cam_true = np.eye(4)
    T_eef_cam_true[:3, :3] = _axis_R([1.0, 0.2, -0.3], math.radians(115))
    T_eef_cam_true[:3, 3] = [0.045, -0.02, 0.044]
    T_base_board = np.eye(4)
    T_base_board[:3, 3] = [0.25, 0.05, 0.76]
    synthetic = []
    for i in range(15):
        qj = rng.uniform(-1.0, 1.0, 6)
        T_base_eef = kdl.fk(list(qj))
        T_cam_board = (np.linalg.inv(T_eef_cam_true)
                       @ np.linalg.inv(T_base_eef) @ T_base_board)
        synthetic.append({"name": f"s{i}", "joints_rad": list(qj),
                          "T_base_eef": T_base_eef.tolist(),
                          "T_camera_board": T_cam_board.tolist(),
                          "detected_corners_px": [], "quality": {"usable": True,
                                                                 "reasons": [],
                                                                 "metrics": {}}})
    result = solve_hand_eye(synthetic, args.method)
    X = np.asarray(result["T_eef_camera"], float)
    t_err = float(np.linalg.norm(X[:3, 3] - T_eef_cam_true[:3, 3]))
    r_err = _rotation_angle_deg(X[:3, :3] @ T_eef_cam_true[:3, :3].T)
    v = result["validation"]
    print(f"3) 合成手眼求解（{args.method}）："
          f"T_eef_camera 误差 平移 {t_err * 1000:.3f} mm，旋转 {r_err:.4f}°；"
          f"残差 RMS {v['translation_rms_m'] * 1000:.4f} mm / {v['rotation_rms_deg']:.4f}°")
    if t_err > 1e-3 or r_err > 0.1:
        failures.append(f"合成手眼求解不收敛（{t_err * 1000:.2f}mm / {r_err:.3f}°）")

    # 3b) 端到端：随机位姿 -> 渲染 -> 检测 -> solvePnP -> 手眼求解 -> 恢复 X
    rng2 = np.random.default_rng(11)
    T_X_true = np.eye(4)
    T_X_true[:3, :3] = _axis_R([1.0, 0.2, -0.3], math.radians(115))
    T_X_true[:3, 3] = [0.045, -0.02, 0.044]
    T_base_board2 = np.eye(4)
    T_base_board2[:3, 3] = [0.25, 0.05, 0.76]
    e2e: list[dict] = []
    attempts = 0
    while len(e2e) < 15 and attempts < 300:
        attempts += 1
        qj = rng2.uniform(-1.0, 1.0, 6)
        T_base_eef = kdl.fk(list(qj))
        T_cam_board_true = (np.linalg.inv(T_X_true) @ np.linalg.inv(T_base_eef)
                            @ T_base_board2)
        if T_cam_board_true[2, 3] <= 0.05:
            continue
        try:
            img = render_synthetic_board(detector, intr, T_cam_board_true)
        except ValueError:
            continue
        found = detector.detect(img)
        if found is None:
            continue
        corners, obj = found
        good, rvec, tvec = cv2.solvePnP(
            obj, intr.undistort(corners).reshape(-1, 1, 2),
            intr.camera_matrix(), intr.dist_coeffs(),
            flags=(cv2.SOLVEPNP_IPPE_SQUARE if obj.shape[0] == 4
                   else cv2.SOLVEPNP_ITERATIVE),
        )
        if not good:
            continue
        T_cam_board_est = np.eye(4)
        T_cam_board_est[:3, :3] = cv2.Rodrigues(rvec)[0]
        T_cam_board_est[:3, 3] = tvec.reshape(3)
        e2e.append({"joints_rad": list(qj), "T_base_eef": T_base_eef.tolist(),
                    "T_camera_board": T_cam_board_est.tolist()})
    if len(e2e) >= 3:
        res = solve_hand_eye(e2e, args.method)
        X = np.asarray(res["T_eef_camera"], float)
        t_err = float(np.linalg.norm(X[:3, 3] - T_X_true[:3, 3]))
        r_err = _rotation_angle_deg(X[:3, :3] @ T_X_true[:3, :3].T)
        print(f"3b) 端到端（渲染→检测→求解，{len(e2e)} 个样本）：恢复 X 误差 "
              f"平移 {t_err * 1000:.2f} mm，旋转 {r_err:.3f}°；"
              f"残差 RMS {res['validation']['translation_rms_m'] * 1000:.2f} mm")
        if t_err > 6e-3 or r_err > 1.0:
            failures.append(f"端到端恢复 X 误差过大（{t_err * 1000:.2f}mm / {r_err:.3f}°）")
    else:
        print(f"3b) 端到端测试样本不足（{len(e2e)}/<15，尝试 {attempts} 次）")
        failures.append("端到端测试无法构造足够样本")

    # 4) 充足性判定逻辑（故意给退化的样本，应该判为不够）
    degenerate = [dict(s) for s in synthetic]
    for i, s in enumerate(degenerate):
        T = np.asarray(s["T_base_eef"], float).copy()
        T[:3, 3] += [0.001 * i, 0.0, 0.0]          # 只平移，不改变姿态
        s["T_base_eef"] = T.tolist()
        s["joints_rad"] = [0.2] * 6                 # 关节也不变
    verdict = session_adequacy(degenerate)
    print(f"4) 退化样本（纯平移、关节不变）充足性判定："
          f"{'够用 ❌（判定错误）' if verdict['sufficient'] else '不够 ✅（判定正确）'}")
    if verdict["sufficient"]:
        failures.append("退化样本被误判为足够")
    good = session_adequacy(synthetic)
    print(f"   正常样本充足性判定：{'够用 ✅' if good['sufficient'] else '不够 ' + str(good['notes'])}")
    if not good["sufficient"]:
        failures.append("正常样本被误判为不足")

    # 5) 相机内参标定闭环：合成视图 -> calibrateCamera -> 恢复内参
    if args.board == "checkerboard":
        intr_true = Intrinsics(width=args.width, height=args.height,
                               fx=420.0, fy=418.0, cx=322.0, cy=238.0,
                               source="synthetic_intrinsics")
        rng3 = np.random.default_rng(5)
        views: list[dict] = []
        tries = 0
        while len(views) < 18 and tries < 500:
            tries += 1
            d = float(rng3.uniform(0.24, 0.45))
            u = float(rng3.uniform(0.22, 0.78) * args.width)
            v = float(rng3.uniform(0.22, 0.78) * args.height)
            tilt = math.radians(float(rng3.uniform(0, 40)))
            roll = float(rng3.uniform(0, 2 * math.pi))
            yaw = float(rng3.uniform(0, 2 * math.pi))
            R_tilt = _axis_R([math.cos(yaw), math.sin(yaw), 0.0], tilt)
            R_roll = _axis_R([0.0, 0.0, 1.0], roll)
            R = R_tilt @ R_roll
            c = np.array([(u - intr_true.cx) * d / intr_true.fx,
                          (v - intr_true.cy) * d / intr_true.fy, d])
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = c
            try:
                img = render_synthetic_board(detector, intr_true, T)
            except ValueError:
                continue
            found = detector.detect(img)
            if found is None:
                continue
            corners, _ = found
            views.append({"name": f"synth_{len(views):03d}",
                          "corners_px": corners.tolist(),
                          "quality": {"metrics": {}}})
        if len(views) >= 8:
            res = solve_intrinsics(views, (args.width, args.height), detector)
            efx, efy = res["fx"], res["fy"]
            ecx, ecy = res["cx"], res["cy"]
            dfx = abs(efx - intr_true.fx) / intr_true.fx * 100
            dfy = abs(efy - intr_true.fy) / intr_true.fy * 100
            dcx = abs(ecx - intr_true.cx)
            dcy = abs(ecy - intr_true.cy)
            print(f"5) 相机内参闭环（{len(views)} 个合成视图）："
                  f"fx {efx:.1f} vs {intr_true.fx:.1f}（{dfx:.2f}%），"
                  f"fy {efy:.1f} vs {intr_true.fy:.1f}（{dfy:.2f}%）；"
                  f"cx {ecx:.1f} vs {intr_true.cx:.1f}，cy {ecy:.1f} vs {intr_true.cy:.1f}；"
                  f"RMS {res['rms_reprojection_px']:.4f}px")
            if dfx > 1.5 or dfy > 1.5 or dcx > 4 or dcy > 4:
                failures.append(
                    f"内参闭环恢复不达标（fx {dfx:.2f}%, fy {dfy:.2f}%, "
                    f"cx {dcx:.2f}px, cy {dcy:.2f}px）")
        else:
            print(f"5) 内参闭环：合成视图只有 {len(views)} 个，跳过")
            failures.append("内参闭环无法构造足够合成视图")

    print()
    if failures:
        print("自检失败 ❌")
        for f in failures:
            print("  - " + f)
        return 1
    print("自检全部通过 ✅")
    return 0


def check_arm(port: str) -> int:
    """只测机械臂能不能连上、能不能读到 6 个关节角。不做任何运动。

    注意：read-only 连接会 arm.enable()（电机使能、保持当前位置），
    这和项目自带的 set_pose.py --read-only 行为一致，但确实会给电机上电。
    """
    print(f"=== 机械臂连接诊断（端口 {port}，只读）===")
    print("说明：会 arm.enable()（电机使能、保持当前位置），不下发任何运动指令。")
    print("      如果只想看能不能 init，请确保周围无人无障碍、急停可用。")
    try:
        arm = ArmReader(port, free_drive=False)
    except Exception as exc:
        print(f"\n❌ 连接失败：{exc}")
        print("\n如果错误里出现 'Motor 1 init failed, expected 38 params, got 35'，")
        print("那是 AIRBOT SDK 与电机固件参数数量不兼容（项目文档里记录过的硬件阻塞），")
        print("必须先解决它，否则做不了手眼标定（没有关节角）。")
        return 1
    try:
        js = None
        for _ in range(10):
            js = arm.joints()
            if js is not None:
                break
            time.sleep(0.1)
        if js is None:
            print("\n❌ 连上了但读不到关节角（arm.state().pos 无效）")
            return 1
        print("\n✅ 机械臂连接正常")
        print("   当前 6 个关节角(rad) =", [round(v, 6) for v in js])
        bad = [i + 1 for i, v in enumerate(js)
               if not (JOINT_LIMITS[i][0] - 0.05 <= v <= JOINT_LIMITS[i][1] + 0.05)]
        if bad:
            print(f"   ! 关节 {bad} 超出 URDF 限位，可能读错了")
        T = UrdfKinematics(DEFAULT_URDF, "link6").fk(js)
        print("   当前 T_base_link6 平移(m) =", [round(v, 4) for v in T[:3, 3]])
        print("\n→ 可以开始正式采集：把命令里的 --preview 换成 --capture")
        return 0
    finally:
        arm.close()


def list_devices() -> int:
    print("=== RealSense ===")
    try:
        import pyrealsense2 as rs

        ctx = rs.context()
        devices = list(ctx.query_devices())
        if not devices:
            print("  没有检测到 RealSense")
        for dev in devices:
            print(
                "  %s | 序列号 %s | 固件 %s"
                % (
                    dev.get_info(rs.camera_info.name),
                    dev.get_info(rs.camera_info.serial_number),
                    dev.get_info(rs.camera_info.firmware_version),
                )
            )
            for sensor in dev.query_sensors():
                for profile in sensor.get_stream_profiles():
                    if profile.stream_type() == rs.stream.color:
                        v = profile.as_video_stream_profile()
                        i = v.get_intrinsics()
                        print(
                            "      彩色 %dx%d@%d  fx=%.2f fy=%.2f cx=%.2f cy=%.2f model=%s"
                            % (i.width, i.height, v.fps(), i.fx, i.fy, i.ppx, i.ppy,
                               i.model)
                        )
                        break
    except Exception as exc:
        print(f"  pyrealsense2 不可用：{exc}")
    print("\n=== V4L2 设备 ===")
    v4l = Path("/dev/v4l/by-id")
    if v4l.is_dir():
        for path in sorted(v4l.glob("*")):
            print("  " + str(path))
    else:
        print("  没有 /dev/v4l/by-id")
    return 0


def focal_check(args, distance_m: float) -> int:
    """用"已知距离"直接测焦距：fx = 板像素跨度 × 距离 ÷ 板实际宽度。

    要求：板**正对镜头**（偏航/俯仰会让视在宽度按 cos 缩小，把 fx 算大），
    距离从**镜头玻璃**量到**板面**。
    """
    camera = make_camera(args)
    detector = make_detector(args)
    try:
        camera.intrinsics()          # 触发相机初始化（内参在此模式下不重要）
        print("=== 焦距现场测定 ===")
        print(f"你量出的距离 = {distance_m * 100:.1f} cm")
        if isinstance(detector, CheckerboardDetector):
            W = (detector.cols - 1) * detector.square_m
            H = (detector.rows - 1) * detector.square_m
            print("板：内角点 %dx%d，方格 %.1f mm → 内角点水平跨度 %.2f m、"
                  "垂直跨度 %.2f m" % (detector.cols, detector.rows,
                                       detector.square_m * 1000, W, H))
        else:
            W = H = detector.marker_m
        best = None
        for _ in range(60):
            frame = camera.read()
            if frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found = detector.detect(gray)
            if found is None:
                continue
            corners, _ = found
            best = (float(np.ptp(corners[:, 0])), float(np.ptp(corners[:, 1])), frame)
            break
        if best is None:
            print("❌ 没检测到完整的标定板：请把板举到镜头前、正对镜头、整块入画")
            return 1
        span_x, span_y, frame = best
        fx = span_x * distance_m / W
        fy = span_y * distance_m / H
        print()
        print(f"画面里：水平跨度 {span_x:.1f} px，垂直跨度 {span_y:.1f} px")
        print(f"→ 由水平跨度算 fx = {fx:.1f}")
        print(f"→ 由垂直跨度算 fy = {fy:.1f}   （两者应接近；差太多说明板歪了）")
        if abs(fx - fy) / max(fx, fy) > 0.06:
            print(f"  ! fx 与 fy 相差 {abs(fx - fy) / max(fx, fy) * 100:.1f}%：板没正对镜头"
                  "（有偏航或俯仰），这个数不可靠，请摆正重测")
        print()
        print(f"对比：本机标定结果 fx=248.8（HFOV 104°）；你这次实测 fx={fx:.1f}")
        out = Path(args.out_root) / "focal_check.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), frame)
        print(f"（这帧已存到 {out}，可复核）")
        return 0
    finally:
        camera.close()


def do_calibrate_camera(session) -> None:
    """求解相机内参并写出 YAML。"""
    views = session.views
    print(f"\n=== 求解相机内参（{session.session}）===")
    print(f"视图 {len(views)} 个")
    if len(views) < 8:
        print("视图少于 8 个，无法可靠求解。")
        return
    res = solve_intrinsics(views, (args_width, args_height), session.detector)
    K = np.asarray(res["camera_matrix"], float)
    fx, fy, cx, cy = res["fx"], res["fy"], res["cx"], res["cy"]
    fovy = 2.0 * math.degrees(math.atan(args_height / (2.0 * fy)))
    fovx = 2.0 * math.degrees(math.atan(args_width / (2.0 * fx)))
    adeq = camera_calib_adequacy(views, Q_CAM["grid"])
    payload = dict(res)
    payload.update({
        "camera_name": "top",
        "camera_serial": session.meta.get("camera", {}).get("camera_serial", ""),
        "source": "checkerboard_calibration",
        "width": args_width,
        "height": args_height,
        "board": session.detector.describe(),
        "fov_deg": {"vertical": fovy, "horizontal": fovx},
        "adequacy": adeq,
        "note": ("内参由棋盘格标定得到（相机固定、只移动标定板）。"
                 "MuJoCo 里可直接用 focalpixel=[[fx,fy]] + "
                 "principalpixel=[[w/2-cx, h/2-cy]]，或用 fovy。"),
    })
    out = session.session / "camera_intrinsics.yaml"
    import yaml as _yaml

    out.write_text(_yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
                   encoding="utf-8")
    print(f"\n视角数 {len(views)} → 采用 {res['views_used']}，"
          f"整体重投影 RMS {res['rms_reprojection_px']:.4f} px")
    if res["views_dropped"]:
        print(f"  剔除视图（误差过大）：{', '.join(res['views_dropped'])}")
    print(f"  fx = {fx:.3f}   fy = {fy:.3f}   cx = {cx:.3f}   cy = {cy:.3f}")
    print(f"  畸变 [k1 k2 p1 p2 k3] = "
          f"{[round(v, 6) for v in res['distortion_coefficients']]}")
    print(f"  FOV: 垂直 {fovy:.2f}°  水平 {fovx:.2f}°")
    errs = [v["reproj_rms_px"] for v in res["per_view"]]
    print(f"  单视图重投影误差：中位 {np.median(errs):.3f} px，"
          f"最大 {max(errs):.3f} px")
    print(f"\n已写入 {out}")
    if res["rms_reprojection_px"] > 0.5:
        print("! 整体重投影 RMS 偏大（>0.5px）。常见原因：板不够平整（打印纸翘曲）、"
              "图像模糊、板太大超出画面导致部分角点不准、或畸变模型不够。")
    if not adeq["sufficient"]:
        print("! 覆盖/视图数不足，内参可能不准（尤其畸变系数）。")
        for note in adeq["notes"]:
            print("  - " + note)


def do_solve(session: Path, method: str, tol_deg: float = 3.0,
             skip_first: int = 0) -> None:
    meta, samples = load_samples(session)
    usable = [s for s in samples if s.get("joints_rad") and s.get("T_camera_board")]
    print(f"\n=== 求解 T_eef_camera（{session}）===")
    print(f"样本 {len(samples)} 个，其中可用于手眼求解 {len(usable)} 个")
    if skip_first:
        # 板在拍摄过程中滑动过时，开头那几帧会被系统性污染；
        # 用 --skip-first 丢掉它们（配合 --solve 的扫描结果使用）
        usable = usable[skip_first:]
        print(f"（已跳过最前面 {skip_first} 帧，剩 {len(usable)} 帧）")
    if len(usable) < 3:
        print("样本不足，无法求解。")
        return

    result, report = solve_robust(usable, method, tol_deg)
    raw_v = report.get("raw")
    robust_v = report.get("robust")
    if raw_v:
        print(f"\n[全部样本]  平移 RMS {raw_v['translation_rms_m'] * 1000:.2f} mm，"
              f"旋转 RMS {raw_v['rotation_rms_deg']:.2f}°")
    dropped = report.get("dropped_sample_names", [])
    if dropped:
        print(f"[自洽性检验] 剔除 {len(dropped)} 个异常帧：{', '.join(dropped)}")
        worst = sorted(report["per_sample_mismatch_deg"].items(),
                       key=lambda kv: -kv[1])[:5]
        print("             不一致度最高的几帧：" +
              ", ".join(f"{k}={v:.1f}°" for k, v in worst))
        print("             常见原因：那一帧之后标定板被挪过，或关节角与图像不同时刻")
    else:
        print("[自洽性检验] 所有样本自洽，无需剔除")
    if robust_v is None:
        print("！剔除后样本不足，只能用全部样本的结果（不建议使用）")
    else:
        print(f"[稳健解]    用 {len(report['used_sample_names'])} 个样本："
              f"平移 RMS {robust_v['translation_rms_m'] * 1000:.2f} mm，"
              f"旋转 RMS {robust_v['rotation_rms_deg']:.2f}°")
    if result is None:
        print("求解失败。")
        return

    # 对半分割验证：比残差 RMS 更能反映真实不确定度
    used_names = set(report.get("used_sample_names") or [])
    used = [s for s in usable if s["name"] in used_names]
    sh = split_half_check(used, method)
    if sh:
        print("\n[对半分割验证] 把样本分成两半各自独立求解，看结果差多少")
        for tag, d in sh.items():
            label = {"odd_even": "奇偶半", "first_second": "前后半"}.get(tag, tag)
            print(f"    {label}({d['n_a']} vs {d['n_b']} 帧)："
                  f"位置差 {d['translation_diff_m'] * 1000:.2f} mm，"
                  f"姿态差 {d['rotation_diff_deg']:.3f}°"
                  f"（距法兰 {d['distance_a_m'] * 1000:.1f} / "
                  f"{d['distance_b_m'] * 1000:.1f} mm）")
        worst = max(d["translation_diff_m"] for d in sh.values()) * 1000
        if worst > 5.0:
            print(f"    ! 两半结果差 {worst:.1f} mm —— 数据内部有**系统不一致**"
                  "（典型原因：标定板在拍摄过程中滑动过）。")
            print("      此时残差 RMS 会偏乐观，真实的绝对精度应以这个差值为量级。")
            print("      建议：把板固定住（胶带/双面胶）重拍一次。")

    result["camera"] = meta.get("camera", {})
    result["board"] = meta.get("board", {})
    result["eef_frame"] = meta.get("eef_frame", "link6")
    result["robust_selection"] = {
        "method": report.get("method"),
        "consistency_tolerance_deg": report.get("consistency_tolerance_deg"),
        "total": report.get("total"),
        "used_sample_names": report.get("used_sample_names"),
        "dropped_sample_names": report.get("dropped_sample_names"),
        "per_sample_mismatch_deg": report.get("per_sample_mismatch_deg"),
    }
    if sh:
        result["split_half_check"] = sh
    result["note"] = (
        "T_eef_camera 把相机坐标映射到 EEF 坐标：p_eef = T_eef_camera @ p_camera。"
        f"EEF 坐标系 = {result['eef_frame']}，与 MuJoCo 生成模型一致，可直接使用。"
        "\n注意：这里的相机系是 OpenCV 约定（+X右 +Y下 +Z为光轴朝前）；"
        "MuJoCo 相机看向局部 -Z，转换用 R_mujoco = R_cv @ diag(1,-1,-1)。"
    )
    out = session / "T_eef_camera.yaml"
    import yaml as _yaml

    out.write_text(_yaml.safe_dump(result, sort_keys=False, allow_unicode=True),
                   encoding="utf-8")
    v = result["validation"]
    print(f"\n平移残差 RMS {v['translation_rms_m'] * 1000:.2f} mm，"
          f"最大 {v['translation_max_m'] * 1000:.2f} mm")
    print(f"旋转残差 RMS {v['rotation_rms_deg']:.2f}°，最大 {v['rotation_max_deg']:.2f}°")
    print("T_eef_camera =")
    for row in np.asarray(result["T_eef_camera"], float):
        print("   " + "  ".join(f"{x: .6f}" for x in row))

    # 尺度自检：方格边长填错会让整个标定尺度线性缩放，而残差依然很小、看不出来，
    # 所以只能靠"相机到法兰的距离是否落在合理范围"来兜住这一类错误。
    t_vec = np.asarray(result["T_eef_camera"], float)[:3, 3]
    t_norm = float(np.linalg.norm(t_vec))
    print(f"\n相机相对 EEF 的位置 = {np.round(t_vec * 1000, 1).tolist()} mm，"
          f"距法兰 {t_norm * 1000:.1f} mm")
    print(f"（参考：scene.yaml 里估的是 48/-20/44.5 mm，距法兰约 {np.linalg.norm([0.048, -0.02, 0.0445]) * 1000:.0f} mm）")
    if not (0.015 <= t_norm <= 0.250):
        print("! 这个距离明显不合理。腕部相机通常距法兰 20~150 mm。"
              "\n  最可能的原因是 --square-mm 填错了：方格边长填错会让标定尺度整体缩放，"
              "\n  而残差照样很小、完全看不出来。请用卡尺实测一个方格的边长再重跑。")

    print(f"\n已写入 {out}")
    if v["translation_rms_m"] > 0.010:
        print("! 残差偏大（>10mm），不建议直接用于仿真；请检查：标定板是否被移动、"
              "内参是否正确、关节角与图像是否同一时刻、样本姿态是否足够分散。")


# ---------------------------------------------------------------------------
# 10. 参数
# ---------------------------------------------------------------------------

args_intrinsics_path = None
args_width = 640
args_height = 480
_ALLOW_PROVISIONAL = False      # 仅 --calibrate-camera 时为 True
_PROVISIONAL_FOV = 60.0         # 标称垂直 FOV，仅用于质量判定的粗估
_PROVISIONAL_HEIGHT = 480


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preview", action="store_true",
                      help="只看画面（不连机械臂、不保存，只用来调姿态）")
    mode.add_argument("--capture", action="store_true",
                      help="正式采集：照片 + 6 个关节角（默认要求机械臂可用）")
    mode.add_argument("--solve", action="store_true", help="从已有会话重新求解")
    mode.add_argument("--self-test", action="store_true", help="无硬件自检")
    mode.add_argument("--list-devices", action="store_true", help="列出相机设备")
    mode.add_argument("--check-arm", action="store_true",
                      help="只诊断机械臂能否连上并读到 6 个关节角")
    mode.add_argument("--calibrate-camera", action="store_true",
                      help="标定相机内参：相机固定不动，只移动/旋转标定板")
    mode.add_argument("--solve-intrinsics", action="store_true",
                      help="从已有会话重新求相机内参")
    mode.add_argument("--focal-check", type=float, default=None,
                      metavar="DIST_M",
                      help="用实测距离(米)现场测焦距：fx = 跨度×距离÷板宽")

    parser.add_argument("--board", choices=["checkerboard", "aruco"], default="checkerboard")
    parser.add_argument("--cols", type=int, default=7, help="棋盘格内角点列数")
    parser.add_argument("--rows", type=int, default=8, help="棋盘格内角点行数")
    parser.add_argument("--square-mm", type=float, default=15.0, help="棋盘格方格边长 mm")
    parser.add_argument("--dict", default="DICT_APRILTAG_36h11", help="ArUco 字典名")
    parser.add_argument("--marker-id", type=int, default=0)
    parser.add_argument("--marker-mm", type=float, default=40.0, help="标记黑框边长 mm")
    parser.add_argument("--autodetect-board", action="store_true",
                        help="启动时自动探测棋盘格内角点规格（--preview 会默认执行）")

    parser.add_argument("--camera", choices=["realsense", "uvc"], default=None,
                        help="不指定时：手眼=realsense，内参标定=uvc")
    parser.add_argument("--fov-deg", type=float, default=60.0,
                        help="--calibrate-camera 时用于质量判定的标称垂直 FOV")
    parser.add_argument("--serial", default=DEFAULT_SERIAL, help="RealSense 序列号（空=第一个）")
    parser.add_argument("--device", default=DEFAULT_UVC, help="UVC 设备路径或编号")
    parser.add_argument("--intrinsics", type=Path, default=None, help="UVC 用的内参 YAML")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--exposure-us", type=float, default=14000.0)
    parser.add_argument("--gain", type=float, default=16.0)

    parser.add_argument("--arm-port", default="can0", help="左臂默认 can0")
    parser.add_argument("--no-arm", action="store_true",
                        help="完全不连机械臂（只能做相机内参，不能做手眼标定）")
    parser.add_argument("--require-arm", action="store_true",
                        help="连不上就退出（--capture 默认就是这种行为）")
    parser.add_argument("--allow-no-arm", action="store_true",
                        help="允许 --capture 在读不到关节角时继续（数据不能做手眼）")
    parser.add_argument("--free-drive", action="store_true",
                        help="MIT 自由拖动（可手推，但无重力补偿、臂会下坠）")
    parser.add_argument("--free-damping", type=float, default=1.0,
                        help="自由拖动时的关节阻尼（越大越慢越稳，0=纯零力矩）")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--eef-link", default="link6",
                        help="EEF 坐标系（MuJoCo 默认 link6）")

    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--session", type=Path, default=None, help="指定会话目录")
    parser.add_argument("--method", default="tsai", choices=sorted(HAND_EYE_METHODS))
    parser.add_argument("--consistency-tol-deg", type=float, default=3.0,
                        help="自洽性检验的容差（度）；超过的帧会被当作异常剔除")
    parser.add_argument("--skip-first", type=int, default=0,
                        help="求解时跳过最前面 N 个样本（板滑动过时用）")
    parser.add_argument("--save-any", action="store_true",
                        help="即使判定不合格也保存（会记录原因）")
    parser.add_argument("--web", action="store_true", help="用浏览器界面而不是窗口")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args(argv)


def main() -> int:
    global args_intrinsics_path, args_width, args_height
    global _ALLOW_PROVISIONAL, _PROVISIONAL_FOV, _PROVISIONAL_HEIGHT
    try:
        # 输出被重定向到文件时默认是块缓冲，日志会看不到；改成行缓冲
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    args = parse_args()
    args_intrinsics_path = args.intrinsics
    args_width, args_height = args.width, args.height
    _PROVISIONAL_FOV = args.fov_deg
    _PROVISIONAL_HEIGHT = args.height

    if args.self_test:
        return self_test(args)
    if args.list_devices:
        return list_devices()
    if args.solve:
        if not args.session:
            raise SystemExit("--solve 需要 --session <目录>")
        do_solve(Path(args.session), args.method, args.consistency_tol_deg,
                 args.skip_first)
        return 0
    if args.solve_intrinsics:
        if not args.session:
            raise SystemExit("--solve-intrinsics 需要 --session <目录>")
        sess = CaptureSession(args)
        do_calibrate_camera(sess)
        return 0
    if args.focal_check is not None:
        if args.camera is None:
            args.camera = "uvc"
        return focal_check(args, args.focal_check)
    if args.check_arm:
        return check_arm(args.arm_port)
    if args.calibrate_camera:
        # 相机固定、只动板：不需要机械臂，允许还没有内参
        _ALLOW_PROVISIONAL = True
        args.no_arm = True
        if args.camera is None:
            args.camera = "uvc"
        return run_capture(args)
    if args.preview:
        if args.camera is None:
            args.camera = "realsense"
        args.no_arm = True
        return run_capture(args)
    if args.capture:
        if args.camera is None:
            args.camera = "realsense"
        # 手眼标定必须要关节角，所以 --capture 默认把"机械臂可用"当作硬要求：
        # 连不上就退出，而不是"优雅降级"成一批没有关节角、做不了标定的废数据。
        if args.no_arm:
            print("! --capture 与 --no-arm 同时给了：这批数据**不能用于手眼标定**，"
                  "只能做相机内参。")
        elif not args.allow_no_arm:
            args.require_arm = True
            print("说明：机械臂可用是 --capture 的硬要求；连不上会直接退出。"
                  "\n      只想拍相机内参请用 --no-arm，或用 --preview 只调姿态。")
        return run_capture(args)
    print(__doc__)
    print("请指定 --capture / --preview / --solve / --check-arm / --self-test / --list-devices")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
