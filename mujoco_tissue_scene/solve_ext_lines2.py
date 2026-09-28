#!/usr/bin/env python3
"""桌子三边解外参 v2：侧边用 x = a*y + b 参数化；稳健损失；合理初值。"""
import math, sys
from pathlib import Path
import cv2, numpy as np, yaml
from scipy.optimize import least_squares

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
img = cv2.imread("/tmp/ext_frame.png"); h, w = img.shape[:2]
cfg = yaml.safe_load((ROOT/"outputs/calibration/camera_intrinsics_uvc_20260918_065532/camera_intrinsics.yaml").read_text())
fx0, fy0, cx0, cy0 = float(cfg["fx"]), float(cfg["fy"]), float(cfg["cx"]), float(cfg["cy"])
dist = np.array(cfg["distortion_coefficients"], float)
K0 = np.array([[fx0,0,cx0],[0,fy0,cy0],[0,0,1.]])
newK,_ = cv2.getOptimalNewCameraMatrix(K0, dist, (w,h), 0)
ug = cv2.cvtColor(cv2.undistort(img, K0, dist, None, newK), cv2.COLOR_BGR2GRAY)

# 远边
far=[]
for x in range(10, w-10, 2):
    col = ug[:, x]
    for y in range(40, h-30):
        if col[y]<100 and col[y+1]<100 and col[y+4]<100 and col[y-6]>110:
            far.append((x,y)); break
far=np.array(far,float)

# 侧边：逐行取最左/最右暗像素（背景亮、桌子暗）
dark = ug < 95
lp=[]; rp=[]
for y in range(210, h-2, 3):
    row = np.where(dark[y])[0]
    if len(row) < 20: continue
    lp.append((row.min(), y)); rp.append((row.max(), y))
lp=np.array(lp,float); rp=np.array(rp,float)

def ransac_v(P, tol=2.5, iters=6000, seed=0):
    """拟合 x = a*y + b（近垂直线）"""
    rng=np.random.default_rng(seed); best=None
    for _ in range(iters):
        i,j=rng.choice(len(P),2,replace=False)
        if abs(P[i,1]-P[j,1])<40: continue
        a=(P[j,0]-P[i,0])/(P[j,1]-P[i,1]); b=P[i,0]-a*P[i,1]
        d=np.abs(P[:,0]-(a*P[:,1]+b)); inl=d<tol
        if best is None or inl.sum()>best[0]: best=(int(inl.sum()),inl)
    a,b=np.polyfit(P[best[1],1], P[best[1],0], 1)   # x = a*y+b
    return P[best[1]], a, b, best[0]

far_in, aF, bF, nF = ransac_v(far, 2.0)
lf_in, aL, bL, nL = ransac_v(lp, 2.5, seed=1)
rt_in, aR, bR, nR = ransac_v(rp, 2.5, seed=2)
print("观测(全部按 x=a*y+b): 远边 %d/%d, 左边 %d/%d, 右边 %d/%d"
      % (nF,len(far),nL,len(lp),nR,len(rp)))
print("  远边 x=%.4f y+%.1f ; 左边 x=%.4f y+%.1f ; 右边 x=%.4f y+%.1f" % (aF,bF,aL,bL,aR,bR))

TABLE=(1.20,0.75); tw,td = TABLE[0]/2, TABLE[1]/2
def pts_on_v(a,b,xlim=(2,w-2),n=25):
    ys=np.linspace(210, h-3, n)
    xs=a*ys+b
    m=(xs>xlim[0])&(xs<xlim[1])
    return np.stack([xs[m],ys[m]],1)

def make_res(pp_free):
    def f(p):
        rv=p[:3]; C=p[3:6]
        cx,cy=(p[6],p[7]) if pp_free else (cx0,cy0)
        R,_=cv2.Rodrigues(rv); Kinv=np.linalg.inv(np.array([[fx0,0,cx],[0,fy0,cy],[0,0,1.]]))
        out=[]
        for pts,kind in ((far_in,'far'),(pts_on_v(aL,bL),'left'),(pts_on_v(aR,bR),'right')):
            for (u,v) in pts:
                d=R@(Kinv@np.array([u,v,1.0]))
                if abs(d[2])<1e-9: out.append(0.5); continue
                s=-C[2]/d[2]
                if s<=0.05: out.append(0.5); continue
                P=C+s*d
                out.append(P[1]+td if kind=='far' else (P[0]-tw if kind=='left' else P[0]+tw))
        return np.array(out)
    return f

results={}
for pp_free in (False, True):
    best=None
    for C0 in (np.array([0.0,0.45,0.56]), np.array([0.3,0.45,0.56]), np.array([-0.2,0.5,0.7]),
               np.array([0.0,0.2,0.56]), np.array([0.15,0.55,0.45])):
        for pitch in (50,60,70):
            for yaw in (0,180):
                th=math.radians(pitch); ya=math.radians(yaw)
                fwd=np.array([math.cos(th)*math.sin(ya), -math.cos(th)*math.cos(ya) if yaw==0 else math.cos(th)*math.cos(ya), -math.sin(th)])
                fwd/=np.linalg.norm(fwd)
                right=np.cross([0,0,-1.],fwd); right/=np.linalg.norm(right)
                up=np.cross(fwd,right)
                R0=np.column_stack([right,-up,fwd])
                rv0=cv2.Rodrigues(R0)[0].ravel()
                p0=np.concatenate([rv0,C0]+([cx0,cy0] if pp_free else []))
                try:
                    sol=least_squares(make_res(pp_free),p0,method='lm',max_nfev=8000)
                except Exception:
                    continue
                r=make_res(pp_free)(sol.x); rms=float(np.sqrt(np.mean(r**2)))
                if np.any(np.abs(r) > 0.49):      # 含无效射线，丢弃
                    continue
                if best is None or rms<best[0]: best=(rms,sol.x)
    results[pp_free]=best
    print()
    print("=== 主点%s ===" % ("自由" if pp_free else "固定"))
    if best is None:
        print("  没找到有效解"); continue
    rms,p=best; C=p[3:6]
    cx,cy=(p[6],p[7]) if pp_free else (cx0,cy0)
    R,_=cv2.Rodrigues(p[:3]); fwd=R@np.array([0,0,1.])
    print("  残差 RMS = %.4f m  相机=[%.3f, %.3f, %.3f] 离桌面 %.1f cm"
          % (rms, C[0],C[1],C[2], C[2]*100))
    print("  光轴=%s 俯角 %.1f°  主点=(%.1f, %.1f)" % (np.round(fwd,3),
          math.degrees(math.asin(min(1,abs(fwd[2])))), cx, cy))
    R_mj=R@np.diag([1.,-1.,-1.]); xy=np.concatenate([R_mj[:,0],R_mj[:,1]])
    fovy=2*math.degrees(math.atan(h/(2*fy0)))
    print("  MuJoCo pos=\"%.4f %.4f %.4f\" xyaxes=\"%s\" fovy=%.2f"
          % (C[0],C[1],0.75+C[2]," ".join("%.6f"%v for v in xy),fovy))
    yaml.safe_dump({"source":"table 3-edge ray-plane LSQ","free_principal_point":bool(pp_free),
      "residual_rms_m":rms,"camera_position_world_m":[float(v) for v in C],
      "camera_height_above_table_m":float(C[2]),"principal_point_px":[float(cx),float(cy)],
      "mujoco":{"pos":"%.4f %.4f %.4f"%(C[0],C[1],0.75+C[2]),
                "xyaxes":" ".join("%.6f"%v for v in xy),"fovy":fovy}},
      open(ROOT/"outputs/calibration/central_camera_extrinsics.yaml","w",encoding="utf-8"),
      sort_keys=False,allow_unicode=True)
