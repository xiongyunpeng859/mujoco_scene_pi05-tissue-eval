#!/usr/bin/env python3
"""把标定/实测数据写进 configs/scene.yaml（自动生成，不手抄数字）。

数据来源：
  outputs/calibration/camera_intrinsics_uvc_*/camera_intrinsics.yaml   中央相机内参
  outputs/calibration/central_camera_extrinsics.yaml                   中央相机外参
  outputs/calibration/hand_eye_left_realsense_*/T_eef_camera.yaml      左腕手眼
实测（用户提供）：桌 120x75cm；托盘 21x20cm 高 7.5cm；纸巾袋 12x8.5x6.5cm；
                  镜头到桌面 56.5cm（与拟合一致）
"""
import glob
import math
import shutil
from pathlib import Path

import numpy as np
import yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
CFG = ROOT / "configs/scene.yaml"

def find(pattern):
    hits = sorted(glob.glob(str(ROOT / pattern)))
    if not hits:
        raise SystemExit("找不到 " + pattern)
    return Path(hits[-1])

intr_p = find("outputs/calibration/camera_intrinsics_uvc_*/camera_intrinsics.yaml")
extr_p = find("outputs/calibration/central_camera_extrinsics.yaml")
he_p = find("outputs/calibration/hand_eye_left_realsense_*/T_eef_camera.yaml")
intr = yaml.safe_load(intr_p.read_text())
extr = yaml.safe_load(extr_p.read_text())
he = yaml.safe_load(he_p.read_text())
cfg = yaml.safe_load(CFG.read_text())

shutil.copy2(CFG, CFG.with_suffix(".yaml.bak_before_calibration"))
print("已备份 ->", CFG.with_suffix(".yaml.bak_before_calibration"))

# ---------------- 中央相机（顶置 UVC）----------------
pos = [float(v) for v in extr["mujoco"]["pos"].split()]
xy = [float(v) for v in extr["mujoco"]["xyaxes"].split()]
c = cfg["cameras"]["central"]
c.pop("target", None)
c["position"] = [round(v, 6) for v in pos]
c["xyaxes"] = [round(v, 6) for v in xy]
c["fovy"] = round(float(extr["mujoco"]["fovy"]), 4)
c["intrinsics"] = {
    "source": "checkerboard_calibration_20260918",
    "serial": "SN0001",
    "resolution": [640, 480],
    "fps": 30,
    "fx": round(float(intr["fx"]), 6),
    "fy": round(float(intr["fy"]), 6),
    "cx": round(float(intr["cx"]), 6),
    "cy": round(float(intr["cy"]), 6),
    "distortion_model": str(intr.get("distortion_model", "brown_conrady_opencv")),
    "distortion_coefficients": [round(float(v), 8) for v in intr["distortion_coefficients"]],
    # MuJoCo 原生渲染是针孔模型；畸变只记录，未施加（与左腕一致）
}
c["intrinsics_calibrated"] = True
c["extrinsics_calibrated"] = True
c["calibration"] = {
    "intrinsics_views": int(intr.get("views_used", 0)),
    "intrinsics_rms_px": round(float(intr.get("rms_reprojection_px", 0)), 4),
    "extrinsics_rms_px": round(float(extr.get("residual_rms_px", 0)), 4),
    "camera_height_above_table_m": round(float(extr["camera_height_above_table_m"]), 4),
    "independent_focal_check": intr.get("independent_focal_check"),
    "ref": str(extr_p.relative_to(ROOT)),
}
print("中央相机: pos=%s  fovy=%.2f  fx=%.2f" % (c["position"], c["fovy"], c["intrinsics"]["fx"]))

# ---------------- 左腕相机：改挂到 link6，用标定的 T_eef_camera ----------------
X = np.asarray(he["T_eef_camera"], float)
R_cv, t_cv = X[:3, :3], X[:3, 3]
# MuJoCo 相机看向局部 -Z、y 向上；OpenCV 是 y 向下、z 向前 → 右乘 diag(1,-1,-1)
R_mj = R_cv @ np.diag([1.0, -1.0, -1.0])
xy_w = list(R_mj[:, 0]) + list(R_mj[:, 1])
w = cfg["wrist_camera"]
w.pop("euler", None)
w["parent_body"] = "link6"
w["position"] = [round(float(v), 6) for v in t_cv]
w["xyaxes"] = [round(float(v), 6) for v in xy_w]
w["bracket_anchor"] = [round(float(t_cv[0]), 6), round(float(t_cv[1]), 6),
                       round(float(t_cv[2] - 0.020), 6)]
w["hand_eye_calibrated"] = True
w["hand_eye"] = {
    "T_eef_camera": [[round(float(v), 8) for v in row] for row in X],
    "eef_frame": "link6",
    "method": str(he.get("method", "?")),
    "sample_count": int(he.get("sample_count", 0)),
    "translation_rms_m": round(float(he["validation"]["translation_rms_m"]), 6),
    "rotation_rms_deg": round(float(he["validation"]["rotation_rms_deg"]), 4),
    "note": ("T_eef_camera 是 OpenCV 约定（+Z 光轴朝前）；上面 position/xyaxes 已"
             "换算成 MuJoCo 约定（看 -Z、y 向上），直接可用。"),
    "ref": str(he_p.relative_to(ROOT)),
}
w["intrinsics"]["note"] = ("厂家内参，未做棋盘格标定；此相机未参与手眼以外的标定。")
print("左腕相机: parent=link6  pos=%s  (距法兰 %.1f mm)"
      % (w["position"], float(np.linalg.norm(t_cv)) * 1000))

# ---------------- 实测尺寸 ----------------
cfg["table"]["size"] = [1.20, 0.75, 0.04]
cfg["table"]["surface_z_measured"] = False        # 桌高仍是估计
cfg["tray"]["size"] = [0.21, 0.20, 0.075]         # 高 7.5cm 实测
cfg["tray"]["size_measured"] = True
for b in cfg["boxes"]:
    b["size"] = [0.12, 0.085, 0.065]              # 12 x 8.5 x 6.5 cm 实测
cfg["boxes_measured"] = True

# ---------------- 控制/时间基准（写进配置备查）----------------
cfg["control"] = {
    "physics_timestep_s": 0.002,          # scene.py 里 MuJoCo timestep，500 Hz
    "physics_frequency_hz": 500,
    "control_frequency_hz": 30,           # 真机 policy/动作频率
    "camera_frequency_hz": 30,
    "action_chunk_size": 50,              # pi0.5 actions_per_chunk
    "chunk_size_threshold": 0.8,
    "action_space": "absolute_joint_position_16d",
    "note": ("物理步长与真机控制频率解耦：500Hz 物理 + 30Hz 控制（16/17 步交替），"
             "不改变 pi0.5 action 的物理含义。"),
}

# ---------------- 标定来源总览 ----------------
cfg["calibration_summary"] = {
    "date": "2026-09-18",
    "wrist_hand_eye": "T_eef_camera（39 帧，平移 RMS 4.63mm / 旋转 RMS 1.45°）",
    "central_intrinsics": ("10 视图，RMS 0.087px；fx=%.2f fy=%.2f 垂直FOV %.1f°；"
                           "卷尺实测 fx=240（差 3.5%%），板/桌面比例验证差 0.2%%"
                           % (intr["fx"], intr["fy"], extr["mujoco"]["fovy"])),
    "central_extrinsics": ("纸四角+花纹 88 点联合平差，RMS 0.157px；"
                           "相机离桌面 %.1f cm（用户卷尺 60cm，已确认以 56.5 为准）"
                           % (extr["camera_height_above_table_m"] * 100)),
    "still_estimated": ["桌面高度 0.75m", "臂底座朝向 euler.z", "托盘壁厚 0.008m",
                        "物体质量 50g", "摩擦系数", "机器人动力学参数"],
}

CFG.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True, width=100),
               encoding="utf-8")
print("\n已写入", CFG)
print("（原文件备份在同目录 .yaml.bak_before_calibration）")
