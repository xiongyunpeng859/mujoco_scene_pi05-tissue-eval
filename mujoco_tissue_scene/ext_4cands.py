#!/usr/bin/env python3
"""把 4 个候选解的桌面四角投影画在同一张图上，直接肉眼判别。"""
import sys
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

det = M.CheckerboardDetector(11, 8, 0.015)
corners, obj = det.detect(gray)
ok, rvec, tvec = cv2.solvePnP(obj, corners.reshape(-1,1,2), K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
R,_ = cv2.Rodrigues(rvec); t = tvec.reshape(3)
T_cb = np.eye(4); T_cb[:3,:3]=R; T_cb[:3,3]=t

TABLE = (1.20, 0.75); LL=(55.0+1.5, 29.5+1.5); W_CM=(11-1)*1.5; H_CM=(8-1)*1.5
def to_world(x,y): return np.array([x/100-TABLE[0]/200, y/100-TABLE[1]/200, 0.0])
cands = [("ll", to_world(LL[0],LL[1]), np.array([1.,0,0]), np.array([0,1.,0])),
         ("lr", to_world(LL[0]+W_CM,LL[1]), np.array([-1.,0,0]), np.array([0,1.,0])),
         ("ul", to_world(LL[0],LL[1]+H_CM), np.array([1.,0,0]), np.array([0,-1.,0])),
         ("ur", to_world(LL[0]+W_CM,LL[1]+H_CM), np.array([-1.,0,0]), np.array([0,-1.,0]))]
tw, td = TABLE[0]/2, TABLE[1]/2
table_w = np.array([[-tw,-td,0],[tw,-td,0],[tw,td,0],[-tw,td,0]], float)

def proj(P, T_wc):
    T_cw = np.linalg.inv(T_wc)
    p = (T_cw[:3,:3] @ P.T).T + T_cw[:3,3]
    if np.any(p[:,2] <= 0.01): return None
    px,_ = cv2.projectPoints(p, np.zeros(3), np.zeros(3), K, dist)
    return px.reshape(-1,2)

colors = [(0,0,255),(0,255,0),(255,0,0),(0,255,255)]
vis = img.copy()
for i,(name,org,xb,yb) in enumerate(cands):
    T_wb = np.eye(4); T_wb[:3,:3]=np.column_stack([xb,yb,np.cross(xb,yb)]); T_wb[:3,3]=org
    T_wc = T_wb @ np.linalg.inv(T_cb)
    px = proj(table_w, T_wc)
    if px is None:
        print("%s: 桌面在相机后" % name); continue
    cv2.polylines(vis, [px.astype(np.int32)], True, colors[i], 2)
    c = px.mean(axis=0).astype(int)
    cv2.putText(vis, name, (int(c[0])-10, int(c[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[i], 3)
    print("%-3s 相机z=%+.3f  桌面投影角点=%s" % (name, T_wc[2,3], np.round(px,0).astype(int).tolist()))
cv2.imwrite("/tmp/ext_4cands.png", vis)
print("\n颜色: ll=红  lr=绿  ul=蓝  ur=黄   已存 /tmp/ext_4cands.png")
