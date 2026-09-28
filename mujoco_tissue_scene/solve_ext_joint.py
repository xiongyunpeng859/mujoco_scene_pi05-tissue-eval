#!/usr/bin/env python3
"""中央相机外参：纸四角 + 花纹 88 点联合平差。

已知（实测）：桌面 120x75cm；白纸 20x20cm，左下角 (55.0, 29.5)cm，平放且与桌边平行。
未知（一起解）：相机位姿 6 + 花纹在纸内的印偏/朝向 3 = 9
观测：纸四角 8 个方程 + 花纹 88 点 176 个方程 = 184
高度和 60cm 的实测值只作对照，不强制。
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
PAPER=0.20; PAPER_LL=(55.0,29.5)
def to_world(x,y): return np.array([x/100-tw, y/100-td, 0.0])

# 纸四角
bright=(gray>150).astype(np.uint8)*255
bright=cv2.morphologyEx(bright,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
cnts,_=cv2.findContours(bright,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cands=[]
for c in cnts:
    a=cv2.contourArea(c)
    if a<3000: continue
    ap=cv2.approxPolyDP(c,0.02*cv2.arcLength(c,True),True)
    if len(ap)!=4 or not cv2.isContourConvex(ap): continue
    if cv2.boundingRect(ap)[1]<180: continue
    cands.append((a,ap.reshape(-1,2).astype(np.float32)))
cands.sort(key=lambda t:-t[0])
paper=cv2.cornerSubPix(gray,cands[0][1].reshape(-1,1,2),(5,5),(-1,-1),
      (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,40,0.01)).reshape(-1,2)
c0=paper.mean(axis=0); paper=paper[np.argsort(np.arctan2(paper[:,1]-c0[1],paper[:,0]-c0[0]))]
print("白纸四角(px): %s" % np.round(paper,2).tolist())

# 花纹
det=M.CheckerboardDetector(11,8,0.015)
corners,obj=det.detect(gray)
print("花纹内角点: %d 个" % len(corners))

WL=to_world(*PAPER_LL); WR=to_world(PAPER_LL[0]+20,PAPER_LL[1])
UR=to_world(PAPER_LL[0]+20,PAPER_LL[1]+20); UL=to_world(PAPER_LL[0],PAPER_LL[1]+20)
ex=(WR-WL); ex/=np.linalg.norm(ex); ey=(UL-WL); ey/=np.linalg.norm(ey)
T_world_paper=np.eye(4); T_world_paper[:3,:3]=np.column_stack([ex,ey,np.cross(ex,ey)])
T_world_paper[:3,3]=WL
paper_world_cyclic=[np.array([WL,WR,UR,UL]),np.array([WR,UR,UL,WL]),
                    np.array([UR,UL,WL,WR]),np.array([UL,WL,WR,UR])]

def pack(rv,C,off,psi_pat):
    # 高度用 exp(u) 参数化，恒为正 → 强制相机在桌面上方，排除镜像解
    return np.concatenate([rv,[C[0],C[1]], [math.log(max(1e-3,C[2]))], off,[psi_pat]])
def unpack(p):
    rv=p[:3]; C=np.array([p[3],p[4],math.exp(p[5])]); off=p[6:8]; psi_pat=p[8]
    return rv,C,off,psi_pat

def residuals(p):
    rv,C,off,psi_pat = unpack(p)
    R,_=cv2.Rodrigues(rv); T=np.eye(4); T[:3,:3]=R; T[:3,3]=C
    Tcw=np.linalg.inv(T)
    res=[]
    # 纸四角
    Wc = CUR_WORLD
    pts=(Tcw[:3,:3]@Wc.T).T+Tcw[:3,3]
    if np.any(pts[:,2]<=0.05): return np.full(8+2*len(obj),50.0)
    px,_=cv2.projectPoints(pts,np.zeros(3),np.zeros(3),K,dist)
    res.append((px.reshape(-1,2)-paper).ravel()*3.0)   # 纸角权重×3
    # 花纹（在纸坐标系里：先平移 off，再绕纸法线转 psi_pat）
    cp,sp=math.cos(psi_pat),math.sin(psi_pat)
    Rpp=np.array([[cp,-sp,0],[sp,cp,0],[0,0,1.]])
    T_pat_in_paper=np.eye(4); T_pat_in_paper[:3,:3]=Rpp; T_pat_in_paper[:3,3]=[off[0],off[1],0]
    W_pat = T_world_paper @ T_pat_in_paper
    Pw=(W_pat[:3,:3]@obj.T).T+W_pat[:3,3]
    q=(Tcw[:3,:3]@Pw.T).T+Tcw[:3,3]
    if np.any(q[:,2]<=0.05): return np.full(8+2*len(obj),50.0)
    px2,_=cv2.projectPoints(q,np.zeros(3),np.zeros(3),K,dist)
    res.append((px2.reshape(-1,2)-corners).ravel())
    return np.concatenate(res)

best=None
for ci,CUR in enumerate(paper_world_cyclic):
    globals()['CUR_WORLD']=CUR
    # 初值：由花纹 PnP 给姿态；位置取纸中心上方
    ok,rv0,tv0=cv2.solvePnP(obj,corners.reshape(-1,1,2),K,dist,flags=cv2.SOLVEPNP_ITERATIVE)
    # 相机世界位置初值：纸中心正上方 0.56m 附近
    ctr=(WL+UR)/2
    for dz in (0.50,0.56,0.65):
        for dx in (-0.15,0,0.15):
            for dy in (-0.15,0,0.15):
                C0=np.array([ctr[0]+dx,ctr[1]+dy,dz])
                # 初值旋转：光轴指向纸中心
                zc = ctr - C0; zc = zc/np.linalg.norm(zc)
                xc = np.cross(np.array([0,0,1.0]), zc); n_=np.linalg.norm(xc)
                if n_ < 1e-6: continue
                xc/=n_
                yc = np.cross(zc, xc)
                R_wc0 = np.column_stack([xc,yc,zc])
                rv_init = cv2.Rodrigues(R_wc0.T)[0].ravel()
                for off_y in (0.0325,0.0):
                    for pp in (0.0, math.pi/2, math.pi, -math.pi/2):
                        p0=pack(rv_init,C0,np.array([0.01,off_y]),pp)
                        try:
                            sol=least_squares(residuals,p0,method='lm',max_nfev=8000)
                        except Exception:
                            continue
                        r=residuals(sol.x)
                        if np.any(np.abs(r)>=49): continue
                        rms=float(np.sqrt(np.mean(r**2)))
                        if best is None or rms<best[0]:
                            best=(rms,ci,sol.x,CUR)
if best is None:
    print("❌ 联合平差未收敛"); raise SystemExit(1)
rms,ci,p,CUR=best
rv,C,off,psi_pat = unpack(p)
R,_=cv2.Rodrigues(rv)
print()
print("✅ 联合平差成功（纸角对应 #%d）：总残差 RMS = %.3f px" % (ci, rms))
print("   相机位置 = [%.4f, %.4f, %.4f] m   离桌面 %.1f cm  (你实测 60cm)" % (C[0],C[1],C[2],C[2]*100))
fwd=R@np.array([0,0,1.])
print("   光轴 = %s  俯角 %.1f°" % (np.round(fwd,4), math.degrees(math.asin(min(1,abs(fwd[2]))))))
print("   花纹印偏（纸坐标系）= [%.2f, %.2f] cm，花纹相对纸的转角 %.2f°"
      % (off[0]*100, off[1]*100, math.degrees(psi_pat)))
print("   （纸 20x20、花纹 18x13.5 居中时印偏应为 (1.0, 3.25)cm 或 (3.25, 1.0)cm）")

# 交叉验证
T=np.eye(4); T[:3,:3]=R; T[:3,3]=C; Tcw=np.linalg.inv(T)
def proj(Pw):
    q=(Tcw[:3,:3]@np.asarray(Pw,float).T).T+Tcw[:3,3]
    if np.any(q[:,2]<=0.05): return None
    px,_=cv2.projectPoints(q,np.zeros(3),np.zeros(3),K,dist); return px.reshape(-1,2)
vis=img.copy()
pf=proj(np.array([[-tw,td,0],[tw,td,0]],float))
if pf is not None: cv2.line(vis,tuple(pf[0].astype(int)),tuple(pf[1].astype(int)),(0,255,0),3)
pw=proj(np.array([[-tw,td+0.17,-0.75],[tw,td+0.17,-0.75]],float))
if pw is not None: cv2.line(vis,tuple(pw[0].astype(int)),tuple(pw[1].astype(int)),(255,0,255),3)
pp=proj(obj); 
if pp is not None: cv2.polylines(vis,[pp.astype(np.int32)],True,(0,255,255),2)
papr=proj(CUR)
if papr is not None: cv2.polylines(vis,[papr.astype(np.int32)],True,(255,255,0),2)
cv2.imwrite("/tmp/ext_final3.png",vis)
newK,_=cv2.getOptimalNewCameraMatrix(K,dist,(w,h),0)
ug=cv2.cvtColor(cv2.undistort(img,K,dist,None,newK),cv2.COLOR_BGR2GRAY)
ys=[]
for x in range(10,w-10,2):
    col=ug[:,x]
    for y in range(40,h-30):
        if col[y]<100 and col[y+1]<100 and col[y+4]<100 and col[y-6]>110:
            ys.append((x,y)); break
ys=np.array(ys,float); a_o,b_o=np.polyfit(ys[:,0],ys[:,1],1)
for tag,pline in (("墙脚线(桌边外17cm, 地面高度)",pw),("桌子远边",pf)):
    if pline is None: continue
    xs=np.linspace(max(pline[:,0].min(),5),min(pline[:,0].max(),w-5),20)
    pa=np.polyfit(pline[:,0],pline[:,1],1)
    print("   观测边界 vs 投影%s : 平均差 %.1f px" % (tag, float(np.mean(np.abs(np.polyval(pa,xs)-(a_o*xs+b_o))))))
R_mj=R@np.diag([1.,-1.,-1.]); xy=np.concatenate([R_mj[:,0],R_mj[:,1]])
fovy=2*math.degrees(math.atan(h/(2*cfg["fy"])))
print("   MuJoCo: pos=\"%.4f %.4f %.4f\" xyaxes=\"%s\" fovy=%.2f"
      % (C[0],C[1],0.75+C[2]," ".join("%.6f"%v for v in xy),fovy))
yaml.safe_dump({"source":"joint LSQ: paper 4 corners + pattern 88 corners",
 "residual_rms_px":rms,"camera_position_world_m":[float(v) for v in C],
 "camera_height_above_table_m":float(C[2]),"optical_axis_world":[float(v) for v in fwd],
 "pattern_offset_in_paper_cm":[float(off[0]*100),float(off[1]*100)],
 "pattern_yaw_in_paper_deg":math.degrees(psi_pat),
 "user_measured_height_m":0.60,
 "mujoco":{"pos":"%.4f %.4f %.4f"%(C[0],C[1],0.75+C[2]),
           "xyaxes":" ".join("%.6f"%v for v in xy),"fovy":fovy,
           "focalpixel":[float(cfg["fx"]),float(cfg["fy"])],
           "principalpixel":[float(w/2-cfg["cx"]),float(h/2-cfg["cy"])]}},
 open(ROOT/"outputs/calibration/central_camera_extrinsics.yaml","w",encoding="utf-8"),
 sort_keys=False,allow_unicode=True)
print("\n可视化 /tmp/ext_final3.png")
