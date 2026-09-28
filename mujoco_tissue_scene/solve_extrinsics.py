#!/usr/bin/env python3
"""解中央相机相对桌面的完整位姿。

前提（用户已确认）：标定板**平放在桌面上、长边平行桌长边**（无偏航、无倾斜），
板左下角距桌面左下角 (55.0, 29.5) cm。

做法：
  1. PnP 得到 T_cam_board（板坐标系 = 第一个内角点，x 沿 11 个角点方向，z 为板法线）。
  2. 板平放 + 平行桌子 → T_world_board 只差"板坐标系原点落在哪个物理角"这 4 种可能，
     枚举 4 种，用物理合理性（相机在桌面上方、在桌前、光轴朝下朝前）挑出正确的。
  3. T_world_cam = T_world_board @ inv(T_cam_board)。
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import capture_hand_eye_dataset as M   # noqa: E402

INTR = ROOT / "outputs/calibration/camera_intrinsics_uvc_20260918_065532/camera_intrinsics.yaml"
FRAME = Path("/tmp/ext_frame.png")
BOARD_LL_CM = (55.0, 29.5)      # 板左下角（外沿）距桌面左下角
TABLE_CM = (120.0, 75.0)
SURFACE_Z = 0.75
SQ_CM = 1.5
COLS, ROWS = 11, 8


def to_world(x_cm, y_cm):
    return np.array([x_cm / 100 - TABLE_CM[0] / 200, y_cm / 100 - TABLE_CM[1] / 200, 0.0])


def main() -> int:
    cfg = yaml.safe_load(INTR.read_text())
    K = np.array(cfg["camera_matrix"], float)
    dist = np.array(cfg["distortion_coefficients"], float)

    img = cv2.imread(str(FRAME))
    if img is None:
        print("❌ 读不到", FRAME, "（相机被占用？）"); return 1
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    det = M.CheckerboardDetector(COLS, ROWS, SQ_CM / 100.0)
    found = det.detect(gray)
    if found is None:
        print("❌ 没检测到 %dx%d 板" % (COLS, ROWS)); return 1
    corners, obj = found
    ok, rvec, tvec = cv2.solvePnP(obj, corners.reshape(-1, 1, 2), K, dist,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - corners) ** 2, axis=1))))
    span = float(max(np.ptp(corners[:, 0]), np.ptp(corners[:, 1])))
    print("板: %d 角点，跨度 %.0f px，PnP RMS %.3f px，板距相机 %.3f m"
          % (len(corners), span, rms, float(np.linalg.norm(t))))

    # 内角点网格左下角（板坐标系原点处）在桌面的坐标
    ll = (BOARD_LL_CM[0] + SQ_CM, BOARD_LL_CM[1] + SQ_CM)
    # 板坐标系原点 (i=0,j=0) 可能是 4 个物理角里的任一个
    W_CM = (COLS - 1) * SQ_CM          # 15.0
    H_CM = (ROWS - 1) * SQ_CM          # 10.5
    cands = [
        ("左下", to_world(ll[0], ll[1]),            np.array([1.0, 0, 0]), np.array([0, 1.0, 0])),
        ("右下", to_world(ll[0] + W_CM, ll[1]),     np.array([-1.0, 0, 0]), np.array([0, 1.0, 0])),
        ("左上", to_world(ll[0], ll[1] + H_CM),     np.array([1.0, 0, 0]), np.array([0, -1.0, 0])),
        ("右上", to_world(ll[0] + W_CM, ll[1] + H_CM), np.array([-1.0, 0, 0]), np.array([0, -1.0, 0])),
    ]

    print()
    print("枚举板坐标系原点的 4 种物理对应（相机必须在桌面上方、盘前、光轴朝下）：")
    results = []
    for name, origin, xb, yb in cands:
        zb = np.cross(xb, yb)
        R_wb = np.column_stack([xb, yb, zb])
        T_wb = np.eye(4); T_wb[:3, :3] = R_wb; T_wb[:3, 3] = origin
        T_cb = np.eye(4); T_cb[:3, :3] = R; T_cb[:3, 3] = t
        T_wc = T_wb @ np.linalg.inv(T_cb)
        C = T_wc[:3, 3]
        R_cw = T_wc[:3, :3].T
        fwd = R_cw @ np.array([0, 0, 1.0])
        results.append((name, C, R_cw, fwd))
        print("  %s: 相机 pos=[%.3f, %.3f, %.3f]  光轴=%s  z>0:%s"
              % (name, C[0], C[1], C[2], np.round(fwd, 3), C[2] > 0.05))

    # 物理合理性打分：在桌面上方 0.2~2.5m、盘前(y<0.6)、光轴朝下(fwd_z<0)
    scored = []
    for name, C, R_cw, fwd in results:
        ok_pos = 0.20 < C[2] < 2.5
        ok_y = -2.5 < C[1] < 0.6
        ok_dir = fwd[2] < -0.2
        scored.append((int(ok_pos) + int(ok_y) + int(ok_dir), name, C, R_cw, fwd, ok_pos, ok_y, ok_dir))
    scored.sort(key=lambda s: -s[0])
    best = scored[0]
    print()
    print("最合理的解：%s（得分 %d/3：z>0=%s 盘前=%s 朝下=%s）"
          % (best[1], best[0], best[5], best[6], best[7]))
    if best[0] < 3:
        print("⚠ 得分不足 3，说明假设可能有问题，结果仅供参考")
    _, name, C, R_cw, fwd, *_ = best

    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -fwd[2]))))
    R_mj = R_cw @ np.diag([1.0, -1.0, -1.0])
    xyaxes = np.concatenate([R_mj[:, 0], R_mj[:, 1]])
    fovy = 2 * math.degrees(math.atan(h / (2 * cfg["fy"])))
    print()
    print("=== 中央相机相对桌面（桌心为原点，z 从桌面算起）===")
    print("  位置 = [%.4f, %.4f, %.4f] m   离桌面 %.1f cm"
          % (C[0], C[1], C[2], C[2] * 100))
    print("  光轴 = %s   俯角 %.1f°" % (np.round(fwd, 4), pitch))
    print("  MuJoCo: pos=\"%.4f %.4f %.4f\"  xyaxes=\"%s\""
          % (C[0], C[1], SURFACE_Z + C[2], " ".join("%.6f" % v for v in xyaxes)))
    print("          fovy=%.2f  (focalpixel=\"%.2f %.2f\" principalpixel=\"%.1f %.1f\")"
          % (fovy, cfg["fx"], cfg["fy"], w / 2 - cfg["cx"], h / 2 - cfg["cy"]))

    out = ROOT / "outputs/calibration/central_camera_extrinsics.yaml"
    yaml.safe_dump({
        "source": "PnP(board flat & parallel on table) + measured board corner",
        "board_lower_left_cm_from_table_lower_left": list(BOARD_LL_CM),
        "board_frame_origin_corner": name,
        "pnp_rms_px": rms,
        "board_span_px": span,
        "camera_position_world_m": [float(v) for v in C],
        "camera_height_above_table_m": float(C[2]),
        "camera_optical_axis_world": [float(v) for v in fwd],
        "mujoco": {
            "pos": "%.4f %.4f %.4f" % (C[0], C[1], SURFACE_Z + C[2]),
            "xyaxes": " ".join("%.6f" % v for v in xyaxes),
            "fovy": fovy,
            "focalpixel": [float(cfg["fx"]), float(cfg["fy"])],
            "principalpixel": [float(w / 2 - cfg["cx"]), float(h / 2 - cfg["cy"])],
        },
    }, open(out, "w", encoding="utf-8"), sort_keys=False, allow_unicode=True)
    print("\n已写入", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
