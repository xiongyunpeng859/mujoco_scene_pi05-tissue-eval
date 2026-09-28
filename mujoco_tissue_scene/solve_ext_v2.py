#!/usr/bin/env python3
"""解中央相机外参，并用"投影桌面四角是否落在黑色桌面上"做判别 + 可视化验证。"""
import math, sys
from pathlib import Path
import cv2, numpy as np, yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import capture_hand_eye_dataset as M   # noqa

INTR = ROOT / "outputs/calibration/camera_intrinsics_uvc_20260918_065532/camera_intrinsics.yaml"
FRAME = Path("/tmp/ext_frame.png")
BOARD_LL_CM = (55.0, 29.5)
TABLE_CM = (120.0, 75.0)
SQ_CM = 1.5
COLS, ROWS = 11, 8

cfg = yaml.safe_load(INTR.read_text())
K = np.array(cfg["camera_matrix"], float); dist = np.array(cfg["distortion_coefficients"], float)
img = cv2.imread(str(FRAME)); h, w = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

det = M.CheckerboardDetector(COLS, ROWS, SQ_CM / 100)
corners, obj = det.detect(gray)
ok, rvec, tvec = cv2.solvePnP(obj, corners.reshape(-1, 1, 2), K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
R, _ = cv2.Rodrigues(rvec); t = tvec.reshape(3)
print("板: %d 角点 跨度 %.0fpx PnP RMS %.3fpx 距离 %.3fm" % (
    len(corners), max(np.ptp(corners[:,0]), np.ptp(corners[:,1])),
    float(np.sqrt(np.mean(np.sum((cv2.projectPoints(obj,rvec,tvec,K,dist)[0].reshape(-1,2)-corners)**2,axis=1)))),
    float(np.linalg.norm(t))))

def to_world(x_cm, y_cm):
    return np.array([x_cm/100 - TABLE_CM[0]/200, y_cm/100 - TABLE_CM[1]/200, 0.0])

ll = (BOARD_LL_CM[0] + SQ_CM, BOARD_LL_CM[1] + SQ_CM)
W_CM = (COLS-1)*SQ_CM; H_CM = (ROWS-1)*SQ_CM
cands = [
    ("ll", to_world(ll[0], ll[1]),                 np.array([1.,0,0]),  np.array([0,1.,0])),
    ("lr", to_world(ll[0]+W_CM, ll[1]),            np.array([-1.,0,0]), np.array([0,1.,0])),
    ("ul", to_world(ll[0], ll[1]+H_CM),            np.array([1.,0,0]),  np.array([0,-1.,0])),
    ("ur", to_world(ll[0]+W_CM, ll[1]+H_CM),       np.array([-1.,0,0]), np.array([0,-1.,0])),
]
T_cb = np.eye(4); T_cb[:3,:3] = R; T_cb[:3,3] = t

# 桌面四角（世界）
tw, td = TABLE_CM[0]/200, TABLE_CM[1]/200
table_world = np.array([[-tw,-td,0],[tw,-td,0],[tw,td,0],[-tw,td,0]], float)

def project(P_world, T_wc):
    T_cw = np.linalg.inv(T_wc)
    pts = (T_cw[:3,:3] @ P_world.T).T + T_cw[:3,3]
    if np.any(pts[:,2] <= 0.01):
        return None
    px, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), K, dist)
    return px.reshape(-1,2)

print()
print("%-4s %-34s %-8s %-10s %s" % ("角", "相机位置(m)", "z>0", "桌面投影", "投影区内平均灰度(越低越是桌面)"))
best = None
for name, origin, xb, yb in cands:
    zb = np.cross(xb, yb)
    T_wb = np.eye(4); T_wb[:3,:3] = np.column_stack([xb,yb,zb]); T_wb[:3,3] = origin
    T_wc = T_wb @ np.linalg.inv(T_cb)
    C = T_wc[:3,3]
    px = project(table_world, T_wc)
    if px is None:
        print("%-4s %-34s %-8s %-10s %s" % (name, np.round(C,3), C[2]>0, "在相机后", "-"))
        continue
    inside = (px[:,0] > -200).all() and (px[:,0] < w+200).all() and (px[:,1] > -200).all() and (px[:,1] < h+200).all()
    mask = np.zeros((h,w), np.uint8); cv2.fillConvexPoly(mask, px.astype(np.int32), 255)
    area = int((mask>0).sum())
    mg = float(gray[mask>0].mean()) if area > 0 else -1
    in_frame = bool(cv2.rectangle.__self__ is not None)
    print("%-4s %-34s %-8s %-10s %.1f  (面积 %d px, %s)"
          % (name, np.round(C,3), C[2]>0, "框内" if inside else "出框", mg, area,
             "有效" if area > 2000 else "太小/无效"))
    if C[2] > 0 and area > 2000:
        if best is None or mg < best[2]:
            best = (name, T_wc, mg, C, px)

if best is None:
    print("\n❌ 没有物理合理的解"); raise SystemExit(1)
name, T_wc, mg, C, px = best
print("\n✅ 选定: 板原点=%s，相机 pos=%s，投影区平均灰度 %.1f" % (name, np.round(C,3), mg))

# 可视化：把投影的桌面轮廓和板轮廓画上
vis = img.copy()
cv2.polylines(vis, [px.astype(np.int32)], True, (0,255,255), 2)
board_world = np.array([[0,0,0],[W_CM/100,0,0],[W_CM/100,H_CM/100,0],[0,H_CM/100,0]], float)
board_world += np.array([origin[0]*0, 0, 0])
# 板在世界里的四角
bx, by = best[0], None
# 用板坐标系原点 + 轴方向重算板四角
for nm, org, xbv, ybv in cands:
    if nm == name:
        bpts = np.array([org, org + xbv*(W_CM/100), org + xbv*(W_CM/100) + ybv*(H_CM/100), org + ybv*(H_CM/100)])
        break
bpx = project(bpts, T_wc)
if bpx is not None:
    cv2.polylines(vis, [bpx.astype(np.int32)], True, (0,255,0), 2)
cv2.imwrite("/tmp/ext_verify.png", vis)

fwd = T_wc[:3,:3].T @ np.array([0,0,1.0])
pitch = math.degrees(math.asin(max(-1,min(1,-fwd[2]))))
R_mj = T_wc[:3,:3] @ np.diag([1.,-1.,-1.])
xyaxes = np.concatenate([R_mj[:,0], R_mj[:,1]])
fovy = 2*math.degrees(math.atan(h/(2*cfg["fy"])))
print()
print("=== 中央相机相对桌面 ===")
print("  位置 = [%.4f, %.4f, %.4f] m  离桌面 %.1f cm" % (C[0], C[1], C[2], C[2]*100))
print("  光轴 = %s  俯角 %.1f°" % (np.round(fwd,4), pitch))
print("  MuJoCo: pos=\"%.4f %.4f %.4f\"  xyaxes=\"%s\"  fovy=%.2f"
      % (C[0], C[1], 0.75+C[2], " ".join("%.6f"%v for v in xyaxes), fovy))
yaml.safe_dump({
    "source": "PnP(board flat & parallel on table) + table-corner projection check",
    "board_origin_corner": name, "pnp_rms_px": None,
    "camera_position_world_m": [float(v) for v in C],
    "camera_height_above_table_m": float(C[2]),
    "optical_axis_world": [float(v) for v in fwd],
    "mujoco": {"pos": "%.4f %.4f %.4f" % (C[0], C[1], 0.75+C[2]),
               "xyaxes": " ".join("%.6f"%v for v in xyaxes), "fovy": fovy,
               "focalpixel": [float(cfg["fx"]), float(cfg["fy"])],
               "principalpixel": [float(w/2-cfg["cx"]), float(h/2-cfg["cy"])]},
}, open(ROOT/"outputs/calibration/central_camera_extrinsics.yaml","w",encoding="utf-8"),
   sort_keys=False, allow_unicode=True)
print("\n可视化: /tmp/ext_verify.png")
