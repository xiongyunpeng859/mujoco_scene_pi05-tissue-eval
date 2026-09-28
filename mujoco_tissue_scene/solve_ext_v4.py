#!/usr/bin/env python3
"""连续扫描标定板相对桌子的偏航角，用"投影桌子远边 = 观测远边"定出相机外参。

板的偏航角不需要用户测量：扫描 0~360°，取投影远边与观测直线偏差最小的解。
覆盖两个分支（板坐标系 z 朝上 / 朝下），因为检测器的角点标号可能是镜像的。
"""
import math, sys
from pathlib import Path
import cv2, numpy as np, yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import capture_hand_eye_dataset as M

INTR = ROOT/"outputs/calibration/camera_intrinsics_uvc_20260918_065532/camera_intrinsics.yaml"
cfg = yaml.safe_load(INTR.read_text())
K = np.array(cfg["camera_matrix"], float); dist = np.array(cfg["distortion_coefficients"], float)
img = cv2.imread("/tmp/ext_frame.png"); h, w = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ---- 观测到的桌子远边 ----
newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0)
ug = cv2.cvtColor(cv2.undistort(img, K, dist, None, newK), cv2.COLOR_BGR2GRAY)
pts=[]
for x in range(15, w-15, 3):
    col = ug[:, x]
    for y in range(50, h-30):
        if col[y]<100 and col[y+1]<100 and col[y+4]<100 and col[y-6]>110:
            pts.append((x,y)); break
xs = np.array([p[0] for p in pts], float); ys = np.array([p[1] for p in pts], float)
rng = np.random.default_rng(0); best=None
for _ in range(3000):
    i,j = rng.choice(len(xs), 2, replace=False)
    if abs(xs[i]-xs[j]) < 60: continue
    a=(ys[j]-ys[i])/(xs[j]-xs[i]); b=ys[i]-a*xs[i]
    inl = np.abs(ys-(a*xs+b)) < 2.0
    if best is None or inl.sum() > best[0]: best=(int(inl.sum()), inl)
a_obs, b_obs = np.polyfit(xs[best[1]], ys[best[1]], 1)
print("观测远边: 内点 %d/%d  y=%.4fx+%.1f  残差 %.2fpx"
      % (best[0], len(xs), a_obs, b_obs, float(np.std(ys[best[1]]-(a_obs*xs[best[1]]+b_obs)))))

# ---- 板 PnP ----
det = M.CheckerboardDetector(11, 8, 0.015)
corners, obj = det.detect(gray)
ok, rvec, tvec = cv2.solvePnP(obj, corners.reshape(-1,1,2), K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
R,_ = cv2.Rodrigues(rvec); t = tvec.reshape(3)
T_cb = np.eye(4); T_cb[:3,:3]=R; T_cb[:3,3]=t
pnp_rms = float(np.sqrt(np.mean(np.sum((cv2.projectPoints(obj,rvec,tvec,K,dist)[0].reshape(-1,2)-corners)**2,axis=1))))
print("板: 跨度 %.0fpx  PnP RMS %.3fpx" % (max(np.ptp(corners[:,0]),np.ptp(corners[:,1])), pnp_rms))

TABLE=(1.20,0.75); LL=(55.0,29.5); SQ=0.015; W_CM=15.0; H_CM=10.5
def to_world(x,y): return np.array([x/100-TABLE[0]/200, y/100-TABLE[1]/200, 0.0])
tw, td = TABLE[0]/2, TABLE[1]/2
far = np.array([[-tw, td,0],[tw, td,0]], float)
near = np.array([[-tw,-td,0],[tw,-td,0]], float)

def proj_pts(P, T_wc):
    T_cw = np.linalg.inv(T_wc)
    p = (T_cw[:3,:3] @ P.T).T + T_cw[:3,3]
    if np.any(p[:,2] <= 0.02): return None
    px,_ = cv2.projectPoints(p, np.zeros(3), np.zeros(3), K, dist)
    return px.reshape(-1,2)

def T_world_board(psi, up):
    c, s = math.cos(psi), math.sin(psi)
    xb = np.array([c, s, 0.0])
    yb = np.array([-s, c, 0.0]) if up else np.array([s, -c, 0.0])
    zb = np.cross(xb, yb)
    # 板坐标系原点（第一个内角点）= 用户量的左下角外沿 + 1 格（沿板自身两轴）
    org_cm = np.array(LL) + SQ*100*(xb[:2] + yb[:2])
    T = np.eye(4); T[:3,:3] = np.column_stack([xb,yb,zb])
    T[:3,3] = to_world(org_cm[0], org_cm[1])
    return T

results=[]
for up in (True, False):
    for k in range(720):
        psi = k*math.pi/360
        T_wb = T_world_board(psi, up)
        T_wc = T_wb @ np.linalg.inv(T_cb)
        C = T_wc[:3,3]
        if C[2] <= 0.05: continue
        pf = proj_pts(far, T_wc); pn = proj_pts(near, T_wc)
        if pf is None or pn is None: continue
        if pn[:,1].mean() <= pf[:,1].mean(): continue        # 近边必须在下方
        xs_ = np.linspace(max(pf[:,0].min(),3), min(pf[:,0].max(),w-3), 25)
        if len(xs_)<2 or xs_.max()-xs_.min() < 80: continue
        pa = np.polyfit(pf[:,0], pf[:,1], 1)
        dev = float(np.mean(np.abs(np.polyval(pa, xs_) - (a_obs*xs_+b_obs))))
        results.append((dev, math.degrees(psi), up, C, T_wc, pf, pn, pa))

results.sort(key=lambda r: r[0])
print()
print("扫描 %d 个可行解，偏差最小的 5 个：" % len(results))
for r in results[:5]:
    print("  偏航 %6.1f°  z%s  相机=[%.3f, %.3f, %.3f]  远边偏差 %6.1f px"
          % (r[1], "上" if r[2] else "下", r[3][0], r[3][1], r[3][2], r[0]))
if not results:
    print("❌ 无可行解"); raise SystemExit(1)
dev, psi_deg, up, C, T_wc, pf, pn, pa = results[0]
print()
print("✅ 最优: 板偏航 %.1f°（z%s），相机 pos=[%.4f, %.4f, %.4f] 离桌面 %.1f cm，远边偏差 %.1f px"
      % (psi_deg, "上" if up else "下", C[0], C[1], C[2], C[2]*100, dev))

fwd = T_wc[:3,:3].T @ np.array([0,0,1.0])
R_mj = T_wc[:3,:3] @ np.diag([1.,-1.,-1.])
xyaxes = np.concatenate([R_mj[:,0], R_mj[:,1]])
fovy = 2*math.degrees(math.atan(h/(2*cfg["fy"])))
print("  光轴 = %s" % np.round(fwd,4))
print("  MuJoCo: pos=\"%.4f %.4f %.4f\"  xyaxes=\"%s\"  fovy=%.2f"
      % (C[0], C[1], 0.75+C[2], " ".join("%.6f"%v for v in xyaxes), fovy))

vis = img.copy()
P = np.array([[-tw,-td,0],[tw,-td,0],[tw,td,0],[-tw,td,0]], float)
cv2.polylines(vis, [proj_pts(P,T_wc).astype(np.int32)], True, (0,255,0), 2)
cv2.polylines(vis, [pf.astype(np.int32)], False, (0,255,255), 3)
cv2.imwrite("/tmp/ext_final.png", vis)
yaml.safe_dump({
    "source": "PnP(board flat on table) + yaw scan matched to observed table far edge",
    "board_yaw_deg_relative_to_table_x": psi_deg,
    "board_frame_z_up": bool(up),
    "far_edge_line_residual_px": dev,
    "pnp_rms_px": pnp_rms,
    "camera_position_world_m": [float(v) for v in C],
    "camera_height_above_table_m": float(C[2]),
    "optical_axis_world": [float(v) for v in fwd],
    "mujoco": {"pos": "%.4f %.4f %.4f" % (C[0], C[1], 0.75+C[2]),
               "xyaxes": " ".join("%.6f"%v for v in xyaxes), "fovy": fovy,
               "focalpixel": [float(cfg["fx"]), float(cfg["fy"])],
               "principalpixel": [float(w/2-cfg["cx"]), float(h/2-cfg["cy"])]},
}, open(ROOT/"outputs/calibration/central_camera_extrinsics.yaml","w",encoding="utf-8"),
   sort_keys=False, allow_unicode=True)
print("\n可视化 /tmp/ext_final.png")
