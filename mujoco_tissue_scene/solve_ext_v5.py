#!/usr/bin/env python3
"""偏航角扫描 v5：修正"哪条桌边被观测到"。

相机可能朝 +y 或 -y 看，观测到的那条"桌子/墙"边界对应的世界边也不同。
对每条候选边都做投影，取偏差最小者，并用"另一条边的位置"做一致性检查。
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

newK,_ = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 0)
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
print("观测边界: 内点 %d/%d  y=%.4fx+%.1f (残差 %.2fpx)" % (best[0], len(xs), a_obs, b_obs,
      float(np.std(ys[best[1]]-(a_obs*xs[best[1]]+b_obs)))))

det = M.CheckerboardDetector(11, 8, 0.015)
corners, obj = det.detect(gray)
ok, rvec, tvec = cv2.solvePnP(obj, corners.reshape(-1,1,2), K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
R,_ = cv2.Rodrigues(rvec); t = tvec.reshape(3)
T_cb = np.eye(4); T_cb[:3,:3]=R; T_cb[:3,3]=t
pnp_rms = float(np.sqrt(np.mean(np.sum((cv2.projectPoints(obj,rvec,tvec,K,dist)[0].reshape(-1,2)-corners)**2,axis=1))))
print("板: 跨度 %.0fpx PnP RMS %.3fpx" % (max(np.ptp(corners[:,0]),np.ptp(corners[:,1])), pnp_rms))

TABLE=(1.20,0.75); LL=(55.0,29.5); SQ=0.015
def to_world(x,y): return np.array([x/100-TABLE[0]/200, y/100-TABLE[1]/200, 0.0])
tw, td = TABLE[0]/2, TABLE[1]/2
edges = {"y=+37.5cm": np.array([[-tw, td,0],[tw, td,0]], float),
         "y=-37.5cm": np.array([[-tw,-td,0],[tw,-td,0]], float)}
tablew = np.array([[-tw,-td,0],[tw,-td,0],[tw,td,0],[-tw,td,0]], float)

def proj_pts(P, T_wc):
    T_cw = np.linalg.inv(T_wc)
    p = (T_cw[:3,:3] @ P.T).T + T_cw[:3,3]
    if np.any(p[:,2] <= 0.02): return None
    px,_ = cv2.projectPoints(p, np.zeros(3), np.zeros(3), K, dist)
    return px.reshape(-1,2)

def T_world_board(psi, up):
    c,s = math.cos(psi), math.sin(psi)
    xb = np.array([c,s,0.]); yb = np.array([-s,c,0.]) if up else np.array([s,-c,0.])
    org = np.array(LL) + SQ*100*(xb[:2]+yb[:2])
    T = np.eye(4); T[:3,:3] = np.column_stack([xb,yb,np.cross(xb,yb)]); T[:3,3] = to_world(org[0],org[1])
    return T

res=[]
for up in (True, False):
    for k in range(720):
        psi = k*math.pi/360
        T_wc = T_world_board(psi, up) @ np.linalg.inv(T_cb)
        C = T_wc[:3,3]
        if C[2] <= 0.05: continue
        for ename, E in edges.items():
            pe = proj_pts(E, T_wc)
            if pe is None: continue
            xs_ = np.linspace(max(pe[:,0].min(),3), min(pe[:,0].max(),w-3), 25)
            if len(xs_)<2 or xs_.max()-xs_.min() < 80: continue
            pa = np.polyfit(pe[:,0], pe[:,1], 1)
            dev = float(np.mean(np.abs(np.polyval(pa, xs_) - (a_obs*xs_+b_obs))))
            # 另一条边：应当在相机后、或投影在匹配边之下（更近）
            other = [v for kk,v in edges.items() if kk != ename][0]
            po = proj_pts(other, T_wc)
            consistent = True
            if po is not None and po[:,1].mean() <= pe[:,1].mean():
                consistent = False          # 另一条边反而更远 → 不合理
            res.append((dev, math.degrees(psi), up, ename, C, T_wc, pe, consistent))
res.sort(key=lambda r: r[0])
print()
print("偏差最小的 6 个（* = 另一条边位置也自洽）：")
for r in res[:6]:
    print("  偏航%6.1f° z%-2s 匹配%s 相机=[%6.3f,%6.3f,%5.3f] 偏差%6.1fpx %s"
          % (r[1], "上" if r[2] else "下", r[3], r[4][0], r[4][1], r[4][2], r[0],
             "*" if r[7] else " "))
ok_res = [r for r in res if r[7]]
pick = ok_res[0] if ok_res else res[0]
dev, psi_deg, up, ename, C, T_wc, pe, _ = pick
print()
print("✅ 选定: 板偏航 %.1f°，匹配 %s，相机 pos=[%.4f, %.4f, %.4f]（离桌面 %.1f cm），偏差 %.1f px"
      % (psi_deg, ename, C[0], C[1], C[2], C[2]*100, dev))
fwd = T_wc[:3,:3].T @ np.array([0,0,1.])
R_mj = T_wc[:3,:3] @ np.diag([1.,-1.,-1.])
xyaxes = np.concatenate([R_mj[:,0], R_mj[:,1]])
fovy = 2*math.degrees(math.atan(h/(2*cfg["fy"])))
print("  光轴 = %s" % np.round(fwd,4))
print("  MuJoCo: pos=\"%.4f %.4f %.4f\"  xyaxes=\"%s\"  fovy=%.2f"
      % (C[0],C[1],0.75+C[2]," ".join("%.6f"%v for v in xyaxes),fovy))
vis = img.copy()
pt = proj_pts(tablew, T_wc)
cv2.polylines(vis,[pt.astype(np.int32)],True,(0,255,0),2)
cv2.polylines(vis,[pe.astype(np.int32)],False,(0,255,255),3)
cv2.imwrite("/tmp/ext_final.png", vis)
yaml.safe_dump({
 "source":"PnP(board flat) + yaw scan matched to observed table/wall boundary",
 "board_yaw_deg":psi_deg, "board_z_up":bool(up), "matched_table_edge":ename,
 "boundary_residual_px":dev, "pnp_rms_px":pnp_rms,
 "camera_position_world_m":[float(v) for v in C],
 "camera_height_above_table_m":float(C[2]),
 "optical_axis_world":[float(v) for v in fwd],
 "mujoco":{"pos":"%.4f %.4f %.4f"%(C[0],C[1],0.75+C[2]),
           "xyaxes":" ".join("%.6f"%v for v in xyaxes),"fovy":fovy,
           "focalpixel":[float(cfg["fx"]),float(cfg["fy"])],
           "principalpixel":[float(w/2-cfg["cx"]),float(h/2-cfg["cy"])]},
}, open(ROOT/"outputs/calibration/central_camera_extrinsics.yaml","w",encoding="utf-8"),
   sort_keys=False, allow_unicode=True)
print("\n可视化 /tmp/ext_final.png")
