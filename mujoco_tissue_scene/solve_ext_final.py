#!/usr/bin/env python3
"""中央相机外参（最终版）。

约束（全部来自实测）：
  · 桌面 120x75cm，桌心为世界原点，z 从桌面算起
  · 相机在桌面上方 60cm（用户卷尺实测）
  · 20x20cm 白纸，左下角距桌面左下角 (55.0, 29.5)cm，平放且与桌边平行
  · 板平放在纸上 → 板平面法线 = 世界上方向（PnP 精确给出，RMS ~0.1px）

未知：绕竖直轴的偏航 ψ + 水平位置 (Cx, Cy)  —— 3 个
观测：白纸 4 角在图像中的亚像素位置（8 个方程）
板上的棋盘格 88 点用于独立交叉验证。
"""
import math, sys
from pathlib import Path
import cv2, numpy as np, yaml
from scipy.optimize import least_squares

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import capture_hand_eye_dataset as M

cfg = yaml.safe_load((ROOT/"outputs/calibration/camera_intrinsics_uvc_20260918_065532/camera_intrinsics.yaml").read_text())
K = np.array(cfg["camera_matrix"], float); dist = np.array(cfg["distortion_coefficients"], float)
img = cv2.imread("/tmp/ext_frame.png"); h, w = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

TABLE=(1.20,0.75); tw,td=TABLE[0]/2,TABLE[1]/2
CAM_H = 0.60                      # 用户实测：镜头到桌面 60cm
PAPER=0.20; PAPER_LL=(55.0,29.5)
def to_world(x,y): return np.array([x/100-tw, y/100-td, 0.0])

# ---- 板 PnP → 世界上方向在相机系 ----
det = M.CheckerboardDetector(11,8,0.015)
corners,obj = det.detect(gray)
ok,rv,tv = cv2.solvePnP(obj, corners.reshape(-1,1,2), K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
R_pat,_ = cv2.Rodrigues(rv); T_cam_pat = np.eye(4); T_cam_pat[:3,:3]=R_pat; T_cam_pat[:3,3]=tv.reshape(3)
n_raw = R_pat[:,2]
# PnP 给的 R[:,2] 是"标号 +z"方向，它指向背离相机的一侧（因为检测器角点标号是镜像的）。
# 物理上的"世界上方"应指向相机：用 n·(板→相机) > 0 判定并取反。
up_from_board = -tv.reshape(3); up_from_board /= np.linalg.norm(up_from_board)
n_up_cam = n_raw if float(n_raw @ up_from_board) > 0 else -n_raw
print("板法线(标号+z)=%s → 取%s 作为世界上方" % (np.round(n_raw,3),
      "原值" if np.allclose(n_up_cam,n_raw) else "反值"))
pnp_rms = float(np.sqrt(np.mean(np.sum((cv2.projectPoints(obj,rv,tv,K,dist)[0].reshape(-1,2)-corners)**2,axis=1))))
print("板 PnP: %d 角点  RMS %.3fpx  板平面垂直距离 %.3f m" % (len(corners), pnp_rms, abs(float(n_up_cam@tv.reshape(3)))))

# ---- 白纸四角 ----
bright=(gray>150).astype(np.uint8)*255
bright=cv2.morphologyEx(bright,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
cnts,_=cv2.findContours(bright,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cands=[]
for c in cnts:
    a=cv2.contourArea(c)
    if a<3000: continue
    ap=cv2.approxPolyDP(c,0.02*cv2.arcLength(c,True),True)
    if len(ap)!=4 or not cv2.isContourConvex(ap): continue
    if cv2.boundingRect(ap)[1] < 180: continue
    cands.append((a,ap.reshape(-1,2).astype(np.float32)))
if not cands:
    print("❌ 没找到白纸四边形"); raise SystemExit(1)
cands.sort(key=lambda t:-t[0])
for a,p in cands[:4]:
    print("  白纸候选: 面积 %7.0f  角点 %s" % (a, np.round(p,0).astype(int).tolist()))
paper = cv2.cornerSubPix(gray, cands[0][1].reshape(-1,1,2), (5,5), (-1,-1),
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,40,0.01)).reshape(-1,2)
c0=paper.mean(axis=0)
paper=paper[np.argsort(np.arctan2(paper[:,1]-c0[1], paper[:,0]-c0[0]))]   # 顺时针
print("  白纸角点(亚像素):"); [print("     ", np.round(p,2)) for p in paper]

WL=to_world(*PAPER_LL); WR=to_world(PAPER_LL[0]+PAPER*100, PAPER_LL[1])
UR=to_world(PAPER_LL[0]+PAPER*100, PAPER_LL[1]+PAPER*100); UL=to_world(PAPER_LL[0], PAPER_LL[1]+PAPER*100)
world_cycles=[np.array([WL,WR,UR,UL]),np.array([WR,UR,UL,WL]),
              np.array([UR,UL,WL,WR]),np.array([UL,WL,WR,UR])]

# ---- 参数化：R_cw = Rot(n_cam, ψ) · R_ref，R_ref 把世界 z 映射到 n_cam ----
a = np.array([1.,0,0]) if abs(n_up_cam[0])<0.9 else np.array([0,1.,0])
u = np.cross(n_up_cam,a); u/=np.linalg.norm(u)
v = np.cross(n_up_cam,u)
R_ref = np.column_stack([u,v,n_up_cam])          # 世界→相机 的参考旋转

def R_of(psi):
    K_ = np.array([[0,-n_up_cam[2],n_up_cam[1]],[n_up_cam[2],0,-n_up_cam[0]],[-n_up_cam[1],n_up_cam[0],0]])
    Rz = np.eye(3)+math.sin(psi)*K_+(1-math.cos(psi))*(K_@K_)
    return Rz @ R_ref

def make_res(Wc):
    def f(p):
        psi, Cx, Cy = p
        R_wc = R_of(psi).T                        # 相机→世界
        C = np.array([Cx,Cy,CAM_H])
        Rt = np.eye(4); Rt[:3,:3]=R_wc; Rt[:3,3]=C
        T_cw = np.linalg.inv(Rt)
        pts=(T_cw[:3,:3]@Wc.T).T+T_cw[:3,3]
        if np.any(pts[:,2]<=0.05): return np.full(8,10.0)
        px,_=cv2.projectPoints(pts,np.zeros(3),np.zeros(3),K,dist)
        return (px.reshape(-1,2)-paper).ravel()
    return f

print()
best=None
for i,Wc in enumerate(world_cycles):
    # 初值：水平位置取纸中心正上方附近的若干点
    for Cx0 in (-0.3,0.0,0.3):
        for Cy0 in (-0.3,0.0,0.3):
            for psi0 in np.linspace(0,2*math.pi,9):
                try:
                    sol=least_squares(make_res(Wc),[psi0,Cx0,Cy0],method='lm',max_nfev=4000)
                except Exception:
                    continue
                r=make_res(Wc)(sol.x); rms=float(np.sqrt(np.mean(r**2)))
                if best is None or rms<best[0]:
                    best=(rms,i,sol.x,Wc)
if best is None:
    print("❌ 拟合失败"); raise SystemExit(1)
rms,idx,p,Wc = best
psi,Cx,Cy = p
print("✅ 最佳：图像-世界对应 #%d，白纸四角重投影 RMS = %.3f px" % (idx, rms))
print("   偏航 ψ = %.2f°   相机水平位置 = (%.4f, %.4f)  高度 %.2f m" % (math.degrees(psi)%360, Cx, Cy, CAM_H))

R_wc = R_of(psi).T; C=np.array([Cx,Cy,CAM_H])
fwd = R_wc@np.array([0,0,1.])
print("   光轴 = %s   俯角 %.1f°" % (np.round(fwd,4), math.degrees(math.asin(min(1,abs(fwd[2]))))))
print("   相机到板原点距离 %.3f m（PnP 说 %.3f m）" % (np.linalg.norm(C-tv.reshape(3)), np.linalg.norm(tv)))

# ---- 交叉验证：投影桌子远边 / 墙脚线 / 花纹 ----
T_world_cam=np.eye(4); T_world_cam[:3,:3]=R_wc; T_world_cam[:3,3]=C
T_cw=np.linalg.inv(T_world_cam)
def proj(Pw):
    pts=(T_cw[:3,:3]@np.asarray(Pw,float).T).T+T_cw[:3,3]
    if np.any(pts[:,2]<=0.05): return None
    px,_=cv2.projectPoints(pts,np.zeros(3),np.zeros(3),K,dist)
    return px.reshape(-1,2)
vis=img.copy()
pf=proj(np.array([[-tw,td,0],[tw,td,0]],float))
if pf is not None: cv2.line(vis,tuple(pf[0].astype(int)),tuple(pf[1].astype(int)),(0,255,0),3)
pw=proj(np.array([[-tw,td+0.17,-0.75],[tw,td+0.17,-0.75]],float))
if pw is not None: cv2.line(vis,tuple(pw[0].astype(int)),tuple(pw[1].astype(int)),(255,0,255),3)
pp=proj(obj)
if pp is not None: cv2.polylines(vis,[pp.astype(np.int32)],True,(0,255,255),2)
papr=proj(np.array([WL,WR,UR,UL],float))
if papr is not None: cv2.polylines(vis,[papr.astype(np.int32)],True,(255,255,0),2)
cv2.imwrite("/tmp/ext_final2.png",vis)
# 与观测边界的偏差
newK,_=cv2.getOptimalNewCameraMatrix(K,dist,(w,h),0)
ug=cv2.cvtColor(cv2.undistort(img,K,dist,None,newK),cv2.COLOR_BGR2GRAY)
ys=[]
for x in range(10,w-10,2):
    col=ug[:,x]
    for y in range(40,h-30):
        if col[y]<100 and col[y+1]<100 and col[y+4]<100 and col[y-6]>110:
            ys.append((x,y)); break
ys=np.array(ys,float); a_obs,b_obs=np.polyfit(ys[:,0],ys[:,1],1)
if pw is not None:
    xs=np.linspace(max(pw[:,0].min(),5),min(pw[:,0].max(),w-5),20)
    pa=np.polyfit(pw[:,0],pw[:,1],1)
    print("   观测边界 vs 投影墙脚线(y=+54.5cm, z=-75cm) 平均差 %.1f px" % float(np.mean(np.abs(np.polyval(pa,xs)-(a_obs*xs+b_obs)))))
if pf is not None:
    xs=np.linspace(max(pf[:,0].min(),5),min(pf[:,0].max(),w-5),20)
    pa=np.polyfit(pf[:,0],pf[:,1],1)
    print("   观测边界 vs 投影桌子远边(y=+37.5cm, z=0)  平均差 %.1f px" % float(np.mean(np.abs(np.polyval(pa,xs)-(a_obs*xs+b_obs)))))
R_mj=R_wc@np.diag([1.,-1.,-1.]); xy=np.concatenate([R_mj[:,0],R_mj[:,1]])
fovy=2*math.degrees(math.atan(h/(2*cfg["fy"])))
print("   MuJoCo: pos=\"%.4f %.4f %.4f\"  xyaxes=\"%s\"  fovy=%.2f"
      % (C[0],C[1],0.75+CAM_H," ".join("%.6f"%v for v in xy),fovy))
yaml.safe_dump({"source":"paper 4-corner LSQ with user-measured camera height 0.60 m",
  "camera_height_above_table_m":CAM_H,"paper_corner_rms_px":rms,
  "yaw_deg":math.degrees(psi)%360,"camera_position_world_m":[float(v) for v in C],
  "optical_axis_world":[float(v) for v in fwd],
  "mujoco":{"pos":"%.4f %.4f %.4f"%(C[0],C[1],0.75+CAM_H),
            "xyaxes":" ".join("%.6f"%v for v in xy),"fovy":fovy,
            "focalpixel":[float(cfg["fx"]),float(cfg["fy"])],
            "principalpixel":[float(w/2-cfg["cx"]),float(h/2-cfg["cy"])]}},
  open(ROOT/"outputs/calibration/central_camera_extrinsics.yaml","w",encoding="utf-8"),
  sort_keys=False,allow_unicode=True)
print("\n可视化 /tmp/ext_final2.png  绿=桌远边 品红=墙脚线 黄=花纹 青=白纸")
