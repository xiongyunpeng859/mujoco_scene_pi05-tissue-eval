#!/usr/bin/env python3
"""用桌子三条边解相机外参（不依赖标定板的摆放测量）。

原理：图像上取一条边的若干点 → 过光心的射线 → 与桌面平面(z=0)求交 → 得到世界点；
     残差 = 该世界点到"这条边应有的世界直线"的距离。
     三条边（远边 + 左右侧边）联合最小二乘，同时可放开主点 cx/cy 一起优化。
参数：相机位姿 6 个 + 可选主点 2 个。
"""
import math, sys
from pathlib import Path
import cv2, numpy as np, yaml
from scipy.optimize import least_squares

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
img = cv2.imread("/tmp/ext_frame.png"); h, w = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cfg = yaml.safe_load((ROOT/"outputs/calibration/camera_intrinsics_uvc_20260918_065532/camera_intrinsics.yaml").read_text())
fx0, fy0 = float(cfg["fx"]), float(cfg["fy"])
cx0, cy0 = float(cfg["cx"]), float(cfg["cy"])
dist = np.array(cfg["distortion_coefficients"], float)

# ---------- 观测：远边 + 左右侧边 ----------
newK,_ = cv2.getOptimalNewCameraMatrix(np.array([[fx0,0,cx0],[0,fy0,cy0],[0,0,1.]]), dist, (w,h), 0)
ug = cv2.cvtColor(cv2.undistort(img, np.array([[fx0,0,cx0],[0,fy0,cy0],[0,0,1.]]), dist, None, newK), cv2.COLOR_BGR2GRAY)

# 远边（墙→桌 的转变）
far=[]
for x in range(15, w-15, 3):
    col = ug[:, x]
    for y in range(50, h-30):
        if col[y]<100 and col[y+1]<100 and col[y+4]<100 and col[y-6]>110:
            far.append((x,y)); break
far=np.array(far,float)

# 侧边：逐行的暗区左右边界（对离群稳健，只取靠边缘的）
dark = (ug < 90).astype(np.uint8)
dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((9,9),np.uint8))
n,lbl,st,_ = cv2.connectedComponentsWithStats(dark)
t = 1+int(np.argmax(st[1:,cv2.CC_STAT_AREA]))
left_pts=[]; right_pts=[]
for y in range(215, h-5, 5):
    xs = np.where(lbl[y]==t)[0]
    if len(xs)==0: continue
    left_pts.append((xs.min(), y)); right_pts.append((xs.max(), y))
left_pts=np.array(left_pts,float); right_pts=np.array(right_pts,float)

def ransac_line(P, tol=2.5, iters=4000, seed=0):
    rng=np.random.default_rng(seed); best=None
    for _ in range(iters):
        i,j=rng.choice(len(P),2,replace=False)
        if abs(P[i,0]-P[j,0])<25 and abs(P[i,1]-P[j,1])<40: continue
        a=(P[j,1]-P[i,1])/(P[j,0]-P[i,0]+1e-9); b=P[i,1]-a*P[i,0]
        d=np.abs(P[:,1]-(a*P[:,0]+b)); inl=d<tol
        if best is None or inl.sum()>best[0]: best=(int(inl.sum()),inl,a,b)
    n_in,inl,_,_ = best
    a,b = np.polyfit(P[inl,0], P[inl,1], 1)
    return P[inl], a, b, n_in

far_in, aF, bF, nF = ransac_line(far, 2.0)
lf_in, aL, bL, nL = ransac_line(left_pts, 3.0, seed=1)
rt_in, aR, bR, nR = ransac_line(right_pts, 3.0, seed=2)
print("观测: 远边 %d/%d 点  y=%.4fx+%.1f" % (nF, len(far), aF, bF))
print("      左边 %d/%d 点  y=%.4fx+%.1f" % (nL, len(left_pts), aL, bL))
print("      右边 %d/%d 点  y=%.4fx+%.1f" % (nR, len(right_pts), aR, bR))

TABLE=(1.20,0.75); tw,td = TABLE[0]/2, TABLE[1]/2

def sample_line(a,b,n=25,x0=5,x1=w-5):
    xs=np.linspace(x0,x1,n); ys=a*xs+b
    m=(ys>=0)&(ys<h-1)
    return np.stack([xs[m],ys[m]],1)

def residuals(params, free_pp):
    rv = params[:3]; C = params[3:6]
    cx, cy = (params[6], params[7]) if free_pp else (cx0, cy0)
    R,_ = cv2.Rodrigues(rv)                 # world -> cam? 我们用 R 表示 cam->world
    R_wc = R
    K = np.array([[fx0,0,cx],[0,fy0,cy],[0,0,1.]])
    Kinv = np.linalg.inv(K)
    res=[]
    for pts, axis, target in ((sample_line(aF,bF),'y',None),
                              (sample_line(aL,bL),'x',None),
                              (sample_line(aR,bR),'x',None)):
        for (u,v) in pts:
            d_cam = Kinv @ np.array([u,v,1.0])
            d_world = R_wc @ d_cam
            if abs(d_world[2]) < 1e-6: res.append(50.0); continue
            s = -C[2]/d_world[2]
            if s <= 0: res.append(50.0); continue
            P = C + s*d_world
            if axis=='y':
                res.append(P[1] - td if False else None)
            else:
                res.append(None)
    return res
# 上面的残差写法太绕，改为直接构造目标列表
def make_res(free_pp):
    def f(params):
        rv = params[:3]; C = params[3:6]
        cx, cy = (params[6], params[7]) if free_pp else (cx0, cy0)
        R_wc,_ = cv2.Rodrigues(rv)
        K = np.array([[fx0,0,cx],[0,fy0,cy],[0,0,1.]])
        Kinv = np.linalg.inv(K)
        out=[]
        for pts, kind in ((sample_line(aF,bF),'far'),
                          (sample_line(aL,bL),'left'),
                          (sample_line(aR,bR),'right')):
            for (u,v) in pts:
                d = R_wc @ (Kinv @ np.array([u,v,1.0]))
                if abs(d[2]) < 1e-6: out.append(0.3); continue
                s = -C[2]/d[2]
                if s <= 0.02: out.append(0.3); continue
                P = C + s*d
                if kind=='far':   out.append(P[1] + td)   # 远边 y=-37.5（相机在 +y 侧看 -y）
                elif kind=='left':out.append(P[0] - tw)
                else:             out.append(P[0] + tw)
        return np.array(out)
    return f

# 初值：来自标定板法（相机在桌 +y 侧、离桌面 0.56m、朝 -y 俯视）
C0 = np.array([0.30, 0.45, 0.56])
fwd0 = np.array([-0.06, -0.50, -0.86]); fwd0/=np.linalg.norm(fwd0)
tmp = np.array([0,0,-1.0])
right0 = np.cross(tmp, fwd0); right0/=np.linalg.norm(right0)
up0 = np.cross(fwd0, right0)
R0 = np.column_stack([right0, -up0, fwd0])       # OpenCV: x右 y下 z前
rv0,_ = cv2.Rodrigues(cv2.Rodrigues(R0)[0])

for free_pp in (False, True):
    p0 = np.concatenate([rv0.ravel(), C0] + ([cx0, cy0] if free_pp else []))
    try:
        sol = least_squares(make_res(free_pp), p0, method='lm', max_nfev=20000)
    except Exception as e:
        print("free_pp=%s 优化失败: %s" % (free_pp, e)); continue
    r = make_res(free_pp)(sol.x)
    rms = float(np.sqrt(np.mean(r**2)))
    rv = sol.x[:3]; C = sol.x[3:6]
    cx, cy = (sol.x[6], sol.x[7]) if free_pp else (cx0, cy0)
    R_wc,_ = cv2.Rodrigues(rv)
    fwd = R_wc @ np.array([0,0,1.0])
    print()
    print("=== free_pp=%s ===" % free_pp)
    print("  残差 RMS = %.3f m  (中位 %.3f m, 最大 %.3f m)" % (rms, float(np.median(np.abs(r))), float(np.max(np.abs(r)))))
    print("  相机位置 = [%.4f, %.4f, %.4f]  离桌面 %.1f cm" % (C[0],C[1],C[2],C[2]*100))
    print("  光轴 = %s   俯角 %.1f°" % (np.round(fwd,4), math.degrees(math.asin(min(1,abs(fwd[2]))))))
    print("  主点 = (%.1f, %.1f)  [原标定 (%.1f, %.1f)]" % (cx, cy, cx0, cy0))
    R_mj = R_wc @ np.diag([1.,-1.,-1.])
    xy = np.concatenate([R_mj[:,0], R_mj[:,1]])
    fovy = 2*math.degrees(math.atan(h/(2*fy0)))
    print("  MuJoCo: pos=\"%.4f %.4f %.4f\" xyaxes=\"%s\" fovy=%.2f"
          % (C[0],C[1],0.75+C[2], " ".join("%.6f"%v for v in xy), fovy))
    yaml.safe_dump({"source":"table 3-edge ray-plane least squares",
      "free_principal_point":free_pp, "residual_rms_m":rms,
      "camera_position_world_m":[float(v) for v in C],
      "camera_height_above_table_m":float(C[2]),
      "focal_length_px":[fx0,fy0],"principal_point_px":[float(cx),float(cy)],
      "mujoco":{"pos":"%.4f %.4f %.4f"%(C[0],C[1],0.75+C[2]),
                "xyaxes":" ".join("%.6f"%v for v in xy),"fovy":fovy}},
      open(ROOT/"outputs/calibration/central_camera_extrinsics.yaml","w",encoding="utf-8"),
      sort_keys=False, allow_unicode=True)
