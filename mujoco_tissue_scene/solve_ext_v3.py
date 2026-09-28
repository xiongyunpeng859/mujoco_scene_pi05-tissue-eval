#!/usr/bin/env python3
"""枚举板坐标系 8 种可能朝向，用"桌子远边"判别并解出相机外参。

判别依据（两条硬几何约束，不依赖灰度阈值调参）：
  1. 投影出的"桌子远边"必须与图像里观测到的"黑板→墙"边界直线重合；
  2. 投影出的"近边"必须比"远边"更靠图像下方（否则就是上下颠倒的解）。
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

# ---------- 观测到的桌子远边直线 ----------
newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0)
ug = cv2.cvtColor(cv2.undistort(img, K, dist, None, newK), cv2.COLOR_BGR2GRAY)
pts = []
for x in range(15, w-15, 3):
    col = ug[:, x]
    for y in range(50, h-30):
        if col[y] < 100 and col[y+1] < 100 and col[y+4] < 100 and col[y-6] > 110:
            pts.append((x, y)); break
xs = np.array([p[0] for p in pts], float); ys = np.array([p[1] for p in pts], float)
rng = np.random.default_rng(0)
best = None
for _ in range(3000):
    i, j = rng.choice(len(xs), 2, replace=False)
    if abs(xs[i]-xs[j]) < 60: continue
    a = (ys[j]-ys[i])/(xs[j]-xs[i]); b = ys[i]-a*xs[i]
    d = np.abs(ys-(a*xs+b)); inl = d < 2.0
    if best is None or inl.sum() > best[0]: best = (int(inl.sum()), inl)
n_inl, inl = best
a_obs, b_obs = np.polyfit(xs[inl], ys[inl], 1)
resid = float(np.std(ys[inl]-(a_obs*xs[inl]+b_obs)))
print("观测到的桌子远边: 内点 %d/%d  y = %.4f x + %.1f   残差 %.2f px"
      % (n_inl, len(xs), a_obs, b_obs, resid))

# ---------- 板 PnP ----------
det = M.CheckerboardDetector(11, 8, 0.015)
corners, obj = det.detect(gray)
ok, rvec, tvec = cv2.solvePnP(obj, corners.reshape(-1,1,2), K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
R,_ = cv2.Rodrigues(rvec); t = tvec.reshape(3)
T_cb = np.eye(4); T_cb[:3,:3]=R; T_cb[:3,3]=t
print("板: 跨度 %.0fpx, PnP RMS %.3fpx" % (
    max(np.ptp(corners[:,0]), np.ptp(corners[:,1])),
    float(np.sqrt(np.mean(np.sum((cv2.projectPoints(obj,rvec,tvec,K,dist)[0].reshape(-1,2)-corners)**2,axis=1))))))

TABLE=(1.20,0.75); LL=(55.0+1.5, 29.5+1.5); W_CM=15.0; H_CM=10.5
def to_world(x,y): return np.array([x/100-TABLE[0]/200, y/100-TABLE[1]/200, 0.0])
X=np.array([1.,0,0]); Y=np.array([0,1.,0])
# 8 种：长边沿 X 或沿 Y；原点在 4 个角
cands=[]
for long_axis in ("X","Y"):
    for sx in (+1,-1):
        for name,ox,oy in (("ll",0,0),("lr",1,0),("ul",0,1),("ur",1,1)):
            d1 = (X if long_axis=="X" else Y)
            d2 = (Y if long_axis=="X" else X)
            xb = sx*((d1 if ox==0 else -d1))
            yb = (d2 if oy==0 else -d2)
            org = to_world(LL[0] + ox*W_CM, LL[1] + oy*H_CM)
            cands.append(("长边沿%s_%s_sx%+d" % (long_axis, name, sx), org, xb, yb))

tw, td = TABLE[0]/2, TABLE[1]/2
far = np.array([[-tw, td,0],[tw, td,0]], float)      # 远边两端
near = np.array([[-tw,-td,0],[tw,-td,0]], float)     # 近边两端

def proj(P, T_wc):
    T_cw = np.linalg.inv(T_wc)
    p = (T_cw[:3,:3] @ P.T).T + T_cw[:3,3]
    if np.any(p[:,2] <= 0.02): return None
    px,_ = cv2.projectPoints(p, np.zeros(3), np.zeros(3), K, dist)
    return px.reshape(-1,2)

print()
print("%-22s %-30s %-8s %-9s %s" % ("候选","相机位置(m)","近>远","远边偏差","备注"))
rows=[]
for name, org, xb, yb in cands:
    zb = np.cross(xb, yb)
    T_wb = np.eye(4); T_wb[:3,:3]=np.column_stack([xb,yb,zb]); T_wb[:3,3]=org
    T_wc = T_wb @ np.linalg.inv(T_cb)
    C = T_wc[:3,3]
    pf, pn = proj(far, T_wc), proj(near, T_wc)
    if pf is None or pn is None:
        print("%-22s %-30s %-8s %-9s %s" % (name, np.round(C,3), "-", "-", "在相机后"))
        continue
    flip_ok = bool(pn[:,1].mean() > pf[:,1].mean())
    # 远边偏差：把投影远边与观测直线的 y 差（在若干 x 上采样）
    xsam = np.linspace(max(pf[:,0].min(),5), min(pf[:,0].max(),w-5), 20)
    if len(xsam) < 2 or xsam.max()-xsam.min() < 30:
        dmean = 999
    else:
        pa = np.polyfit(pf[:,0], pf[:,1], 1)
        dmean = float(np.mean(np.abs(np.polyval(pa, xsam) - (a_obs*xsam+b_obs))))
    rows.append((dmean, name, C, flip_ok, T_wc, pf, pn))
    print("%-22s %-30s %-8s %-9.1f %s" % (name, np.round(C,3), flip_ok, dmean,
          "z>0" if C[2]>0 else "z<0(坏)"))

good = [r for r in rows if r[3] and r[2][2] > 0.05]
good.sort(key=lambda r: r[0])
print()
if not good:
    print("❌ 没有同时满足两条约束的候选"); raise SystemExit(1)
dmean, name, C, flip_ok, T_wc, pf, pn = good[0]
print("✅ 选定 %s  相机 pos=[%.4f, %.4f, %.4f]（离桌面 %.1f cm）远边偏差 %.1f px"
      % (name, C[0], C[1], C[2], C[2]*100, dmean))

fwd = T_wc[:3,:3].T @ np.array([0,0,1.0])
yaw = math.degrees(math.atan2(fwd[1], fwd[0]))
R_mj = T_wc[:3,:3] @ np.diag([1.,-1.,-1.])
xyaxes = np.concatenate([R_mj[:,0], R_mj[:,1]])
fovy = 2*math.degrees(math.atan(h/(2*cfg["fy"])))
print("  光轴 = %s   水平方位 %.1f°（+x 为 0°）" % (np.round(fwd,4), yaw))
print("  MuJoCo: pos=\"%.4f %.4f %.4f\"  xyaxes=\"%s\"  fovy=%.2f"
      % (C[0], C[1], 0.75+C[2], " ".join("%.6f"%v for v in xyaxes), fovy))

vis = img.copy()
cv2.polylines(vis, [pf.astype(np.int32)], False, (0,255,255), 3)
cv2.polylines(vis, [pn.astype(np.int32)], False, (0,165,255), 3)
for i in range(4):
    P = np.array([[ -tw,-td,0],[tw,-td,0],[tw,td,0],[-tw,td,0]], float)
px = proj(P, T_wc)
cv2.polylines(vis, [px.astype(np.int32)], True, (0,255,0), 2)
cv2.imwrite("/tmp/ext_final.png", vis)
yaml.safe_dump({
    "source": "PnP(board flat on table) + table far-edge line match",
    "assumed_board_long_edge_axis": name,
    "far_edge_line_residual_px": dmean,
    "observed_far_edge_line": {"slope": float(a_obs), "intercept": float(b_obs)},
    "camera_position_world_m": [float(v) for v in C],
    "camera_height_above_table_m": float(C[2]),
    "optical_axis_world": [float(v) for v in fwd],
    "mujoco": {"pos": "%.4f %.4f %.4f" % (C[0], C[1], 0.75+C[2]),
               "xyaxes": " ".join("%.6f"%v for v in xyaxes), "fovy": fovy,
               "focalpixel": [float(cfg["fx"]), float(cfg["fy"])],
               "principalpixel": [float(w/2-cfg["cx"]), float(h/2-cfg["cy"])]},
}, open(ROOT/"outputs/calibration/central_camera_extrinsics.yaml","w",encoding="utf-8"),
   sort_keys=False, allow_unicode=True)
print("\n可视化 /tmp/ext_final.png（黄=投影远边，橙=投影近边，绿=投影桌面轮廓）")
