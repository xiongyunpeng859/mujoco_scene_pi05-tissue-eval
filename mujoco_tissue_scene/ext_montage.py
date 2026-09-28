#!/usr/bin/env python3
"""把偏航扫描的前几名候选画成 2x2 拼图，肉眼挑出与画面真正吻合的那个。
同时叠上"观测边界线"作参考（青色）。
"""
import math, sys
from pathlib import Path
import cv2, numpy as np, yaml

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import capture_hand_eye_dataset as M

cfg = yaml.safe_load((ROOT/"outputs/calibration/camera_intrinsics_uvc_20260918_065532/camera_intrinsics.yaml").read_text())
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
xs=np.array([p[0] for p in pts],float); ys=np.array([p[1] for p in pts],float)
rng=np.random.default_rng(0); best=None
for _ in range(3000):
    i,j=rng.choice(len(xs),2,replace=False)
    if abs(xs[i]-xs[j])<60: continue
    a=(ys[j]-ys[i])/(xs[j]-xs[i]); b=ys[i]-a*xs[i]
    inl=np.abs(ys-(a*xs+b))<2.0
    if best is None or inl.sum()>best[0]: best=(int(inl.sum()),inl)
a_obs,b_obs=np.polyfit(xs[best[1]],ys[best[1]],1)

det=M.CheckerboardDetector(11,8,0.015)
corners,obj=det.detect(gray)
ok,rvec,tvec=cv2.solvePnP(obj,corners.reshape(-1,1,2),K,dist,flags=cv2.SOLVEPNP_ITERATIVE)
R,_=cv2.Rodrigues(rvec); t=tvec.reshape(3)
T_cb=np.eye(4); T_cb[:3,:3]=R; T_cb[:3,3]=t

TABLE=(1.20,0.75); LL=(55.0,29.5); SQ=0.015
def to_world(x,y): return np.array([x/100-TABLE[0]/200,y/100-TABLE[1]/200,0.])
tw,td=TABLE[0]/2,TABLE[1]/2
edges={"y=+37.5":np.array([[-tw,td,0],[tw,td,0]],float),
       "y=-37.5":np.array([[-tw,-td,0],[tw,-td,0]],float)}
tablew=np.array([[-tw,-td,0],[tw,-td,0],[tw,td,0],[-tw,td,0]],float)

def proj(P,T_wc):
    T_cw=np.linalg.inv(T_wc)
    p=(T_cw[:3,:3]@P.T).T+T_cw[:3,3]
    if np.any(p[:,2]<=0.02): return None
    px,_=cv2.projectPoints(p,np.zeros(3),np.zeros(3),K,dist)
    return px.reshape(-1,2)

def Twb(psi,up):
    c,s=math.cos(psi),math.sin(psi)
    xb=np.array([c,s,0.]); yb=np.array([-s,c,0.]) if up else np.array([s,-c,0.])
    org=np.array(LL)+SQ*100*(xb[:2]+yb[:2])
    T=np.eye(4); T[:3,:3]=np.column_stack([xb,yb,np.cross(xb,yb)]); T[:3,3]=to_world(org[0],org[1])
    return T

res=[]
for up in (True,False):
    for k in range(720):
        psi=k*math.pi/360
        T_wc=Twb(psi,up)@np.linalg.inv(T_cb)
        C=T_wc[:3,3]
        if C[2]<=0.05: continue
        for en,E in edges.items():
            pe=proj(E,T_wc)
            if pe is None: continue
            xsam=np.linspace(max(pe[:,0].min(),3),min(pe[:,0].max(),w-3),25)
            if len(xsam)<2 or xsam.max()-xsam.min()<80: continue
            pa=np.polyfit(pe[:,0],pe[:,1],1)
            dev=float(np.mean(np.abs(np.polyval(pa,xsam)-(a_obs*xsam+b_obs))))
            res.append((dev,math.degrees(psi),up,en,C,T_wc))
res.sort(key=lambda r:r[0])

tiles=[]
for rank,(dev,psi,up,en,C,T_wc) in enumerate(res[:4]):
    vis=img.copy()
    cv2.line(vis,(0,int(b_obs)),(w,int(a_obs*w+b_obs)),(255,255,0),2)   # 观测边界=青
    pt=proj(tablew,T_wc)
    if pt is not None:
        cv2.polylines(vis,[np.nan_to_num(pt).astype(np.int32)],True,(0,255,0),2)
    pe=proj(edges[en],T_wc)
    if pe is not None:
        cv2.polylines(vis,[np.nan_to_num(pe).astype(np.int32)],False,(0,0,255),3)
    cv2.putText(vis,"#%d yaw%.0f z%s %s dev%.0fpx C=[%.2f,%.2f,%.2f]"
                %(rank+1,psi,"up" if up else "dn",en,dev,C[0],C[1],C[2]),
                (5,18),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1)
    small=cv2.resize(vis,(w//2,h//2))
    tiles.append(small)
grid=np.zeros((h,w,3),np.uint8)
th, tw2 = tiles[0].shape[:2]
grid[:th,:tw2]=tiles[0]; grid[:th,tw2:tw2*2]=tiles[1]
grid[th:th*2,:tw2]=tiles[2]; grid[th:th*2,tw2:tw2*2]=tiles[3]
cv2.imwrite("/tmp/ext_montage.png",grid)
print("前 4 名：")
for rank,(dev,psi,up,en,C,T_wc) in enumerate(res[:4]):
    print("  #%d yaw=%6.1f° z=%s match=%s dev=%.1fpx  C=[%.3f, %.3f, %.3f]"
          %(rank+1,psi,"up" if up else "down",en,dev,C[0],C[1],C[2]))
print("\n拼图已存 /tmp/ext_montage.png   青=观测边界  绿=投影桌面轮廓  红=匹配的那条边")
yaml.safe_dump({"candidates":[{"yaw_deg":r[1],"z_up":bool(r[2]),"edge":r[3],
   "dev_px":r[0],"camera":[float(v) for v in r[4]]} for r in res[:8]]},
   open("/tmp/ext_cands.yaml","w"),allow_unicode=True)
