#!/usr/bin/env python3
"""中央相机外参（最终）：solvePnPGeneric 取 IPPE 两解 → 选相机在桌面上方的 → 联合精修。"""
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
PAPER_LL=(55.0,29.5)
def to_world(x,y): return np.array([x/100-tw, y/100-td, 0.0])

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
det=M.CheckerboardDetector(11,8,0.015)
corners,obj=det.detect(gray)

WL=to_world(*PAPER_LL); WR=to_world(PAPER_LL[0]+20,PAPER_LL[1])
UR=to_world(PAPER_LL[0]+20,PAPER_LL[1]+20); UL=to_world(PAPER_LL[0],PAPER_LL[1]+20)
# 图像 y 轴向下、世界 y 轴向上 → 图像里的"顺时针"在世界里是"逆时针"。
# 因此世界角点顺序必须取 WL→UL→UR→WR（及其循环），才能与图像顺序同向。
cycles=[np.array([WL,UL,UR,WR]),np.array([UL,UR,WR,WL]),
        np.array([UR,WR,WL,UL]),np.array([WR,WL,UL,UR])]

print("用 solvePnPGeneric 枚举纸角对应与两个平面解：")
inits=[]
for ci,Wc in enumerate(cycles[:3]+[np.array([UL,WL,WR,UR])]):
    n,rv,tv,err = cv2.solvePnPGeneric(Wc.astype(np.float64), paper.reshape(-1,1,2), K, dist,
                                      flags=cv2.SOLVEPNP_IPPE)
    for k in range(n):
        C = -cv2.Rodrigues(rv[k])[0].T @ tv[k].reshape(3)     # 世界位置（世界=纸所在平面坐标）
        Cw = np.array([C[0],C[1],C[2]])
        e0 = float(np.ravel(err[k])[0])
        print("  对应#%d 解%d: 相机世界位置=[%.3f, %.3f, %.3f] 重新投影误差 %.3f"
              % (ci,k,Cw[0],Cw[1],Cw[2], e0))
        if Cw[2] > 0.15:            # 相机必须在桌面上方
            inits.append((e0, ci, k, rv[k].ravel(), Cw, Wc))

if not inits:
    print("❌ 两个解都没有相机在桌面上方的情况"); raise SystemExit(1)
inits.sort(key=lambda t:t[0])
err0, ci, k, rv0, C0, Wc = inits[0]
print("\n选定初值：对应#%d 解%d，相机=[%.3f, %.3f, %.3f]（离桌面 %.1f cm）"
      % (ci,k,C0[0],C0[1],C0[2],C0[2]*100))

ex=(Wc[1]-Wc[0]); ex/=np.linalg.norm(ex); ey=(Wc[3]-Wc[0]); ey/=np.linalg.norm(ey)
T_world_paper=np.eye(4); T_world_paper[:3,:3]=np.column_stack([ex,ey,np.cross(ex,ey)])
T_world_paper[:3,3]=Wc[0]

def residuals(p):
    rv=p[:3]; C=p[3:6]; off=p[6:8]; psi=p[8]
    R,_=cv2.Rodrigues(rv)
    T=np.eye(4); T[:3,:3]=R; T[:3,3]=C; Tcw=np.linalg.inv(T)
    out=[]
    pts=(Tcw[:3,:3]@Wc.T).T+Tcw[:3,3]
    if np.any(pts[:,2]<=0.05): return np.full(8+176,30.0)
    px,_=cv2.projectPoints(pts,np.zeros(3),np.zeros(3),K,dist)
    out.append((px.reshape(-1,2)-paper).ravel()*3.0)
    cp,sp=math.cos(psi),math.sin(psi)
    Tpp=np.eye(4); Tpp[:3,:3]=np.array([[cp,-sp,0],[sp,cp,0],[0,0,1.]]); Tpp[:3,3]=[off[0],off[1],0]
    Wp=T_world_paper@Tpp
    Pw=(Wp[:3,:3]@obj.T).T+Wp[:3,3]
    q=(Tcw[:3,:3]@Pw.T).T+Tcw[:3,3]
    if np.any(q[:,2]<=0.05): return np.full(8+176,30.0)
    px2,_=cv2.projectPoints(q,np.zeros(3),np.zeros(3),K,dist)
    out.append((px2.reshape(-1,2)-corners).ravel())
    return np.concatenate(out)

best=None
for oy in (0.0325, 0.01, 0.0):
    for pp in (0.0, math.pi/2, math.pi, -math.pi/2):
        p0=np.concatenate([rv0, C0, [0.01, oy, pp]])
        try:
            sol=least_squares(residuals,p0,method='lm',max_nfev=20000)
        except Exception:
            continue
        r=residuals(sol.x)
        if np.any(np.abs(r)>=29): continue
        rms=float(np.sqrt(np.mean(r**2)))
        if best is None or rms<best[0]: best=(rms,sol.x)
if best is None:
    print("❌ 精修未收敛"); raise SystemExit(1)
rms,p=best
rv=p[:3]; C=p[3:6]; off=p[6:8]; psi=p[8]
R,_=cv2.Rodrigues(rv); fwd=R@np.array([0,0,1.])
print()
print("✅ 精修完成：总残差 RMS = %.3f px" % rms)
print("   相机位置 = [%.4f, %.4f, %.4f] m   离桌面 %.1f cm  （你实测 60cm）"
      % (C[0],C[1],C[2],C[2]*100))
print("   光轴 = %s   俯角 %.1f°" % (np.round(fwd,4), math.degrees(math.asin(min(1,abs(fwd[2]))))))
print("   花纹印偏（纸坐标系）= [%.2f, %.2f] cm，相对纸转角 %.2f°"
      % (off[0]*100, off[1]*100, math.degrees(psi)))
print("   预期印偏：纸 20x20、花纹 18x13.5 居中 → (1.0, 3.25) 或 (3.25, 1.0) cm")
R_mj=R@np.diag([1.,-1.,-1.]); xy=np.concatenate([R_mj[:,0],R_mj[:,1]])
fovy=2*math.degrees(math.atan(h/(2*cfg["fy"])))
print("   MuJoCo: pos=\"%.4f %.4f %.4f\"  xyaxes=\"%s\"  fovy=%.2f"
      % (C[0],C[1],0.75+C[2]," ".join("%.6f"%v for v in xy),fovy))

# 可视化
T=np.eye(4); T[:3,:3]=R; T[:3,3]=C; Tcw=np.linalg.inv(T)
def proj(Pw):
    q=(Tcw[:3,:3]@np.asarray(Pw,float).T).T+Tcw[:3,3]
    if np.any(q[:,2]<=0.05): return None
    px,_=cv2.projectPoints(q,np.zeros(3),np.zeros(3),K,dist); return px.reshape(-1,2)
vis=img.copy()
for Pw,col in ((np.array([[-tw,td,0],[tw,td,0]],float),(0,255,0)),
               (np.array([[-tw,td+0.17,-0.75],[tw,td+0.17,-0.75]],float),(255,0,255)),
               (Wc,(255,255,0))):
    p_=proj(Pw)
    if p_ is not None:
        if len(p_)==2: cv2.line(vis,tuple(p_[0].astype(int)),tuple(p_[1].astype(int)),col,3)
        else: cv2.polylines(vis,[p_.astype(np.int32)],True,col,2)
p_=proj(obj)
if p_ is not None: cv2.polylines(vis,[p_.astype(np.int32)],True,(0,255,255),2)
cv2.imwrite("/tmp/ext_final4.png",vis)
yaml.safe_dump({"source":"IPPE two-solution + joint refinement (paper+pattern)",
 "residual_rms_px":rms,"camera_position_world_m":[float(v) for v in C],
 "camera_height_above_table_m":float(C[2]),
 "user_measured_height_m":0.60,"optical_axis_world":[float(v) for v in fwd],
 "pattern_offset_in_paper_cm":[float(off[0]*100),float(off[1]*100)],
 "pattern_yaw_in_paper_deg":math.degrees(psi),
 "mujoco":{"pos":"%.4f %.4f %.4f"%(C[0],C[1],0.75+C[2]),
           "xyaxes":" ".join("%.6f"%v for v in xy),"fovy":fovy,
           "focalpixel":[float(cfg["fx"]),float(cfg["fy"])],
           "principalpixel":[float(w/2-cfg["cx"]),float(h/2-cfg["cy"])]}},
 open(ROOT/"outputs/calibration/central_camera_extrinsics.yaml","w",encoding="utf-8"),
 sort_keys=False,allow_unicode=True)
print("\n可视化 /tmp/ext_final4.png（绿=桌远边 品红=墙脚线 黄=纸 青=花纹）")
