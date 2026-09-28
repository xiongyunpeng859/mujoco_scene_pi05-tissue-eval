"""Conditional planar camera fit using green tray; not a metric calibration.

Assumes scene tray dimensions, orientation and the existing camera focal length.
Writes a separate candidate config, never the default or raw data.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml


def tray_corners(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([35,60,35]), np.array([90,255,255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No green tray detected")
    contour = max(contours, key=cv2.contourArea)
    quad = cv2.approxPolyDP(contour, .035*cv2.arcLength(contour,True),True).reshape(-1,2)
    if len(quad) != 4 or cv2.contourArea(contour)<1000:
        raise ValueError("Cannot reliably find tray quadrilateral")
    quad = quad.astype(np.float64)
    # TL, TR, BR, BL in image; correspondence to world is an explicit assumption.
    sums, diffs = quad.sum(1), quad[:,1]-quad[:,0]
    return quad[[np.argmin(sums),np.argmin(diffs),np.argmax(sums),np.argmax(diffs)]]


def run(alignment, config, output):
    if output.exists():
        raise FileExistsError(output)
    report = json.loads((alignment/"report.json").read_text())
    quads=[]
    excluded=[]
    for entry in report["comparison_frames"]:
        # Comparison images have a 20px title, then the original 640x480 frame.
        image=cv2.imread(str(alignment/entry["comparison"]))[20:500,:640]
        try:
            quads.append(tray_corners(image))
        except ValueError as error:
            excluded.append({"frame":entry["comparison"],"reason":str(error)})
    if len(quads)<4:
        raise ValueError("Too few reliably detected tray frames")
    pixels=np.median(quads,axis=0)
    settings=yaml.safe_load(config.read_text())
    tray=settings["tray"]
    x,y=tray["center"]
    # Detect green interior, not white outer border.
    width,depth=np.asarray(tray["size"][:2])-2*tray["wall_thickness"]
    z=settings["table"]["surface_z"]+.01
    points=np.array([[x-width/2,y+depth/2,z],[x+width/2,y+depth/2,z],
                     [x+width/2,y-depth/2,z],[x-width/2,y-depth/2,z]],dtype=np.float64)
    f=240/np.tan(np.deg2rad(settings["cameras"]["central"]["fovy"])/2)
    intrinsic=np.array([[f,0,320],[0,f,240],[0,0,1]],dtype=np.float64)
    ok,rotation,translation=cv2.solvePnP(points,pixels,intrinsic,None,flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError("PnP failed")
    matrix=cv2.Rodrigues(rotation)[0]
    position=(-matrix.T@translation).ravel()
    if position[2]<=z:
        raise ValueError("Rejected below-table camera solution")
    projected=cv2.projectPoints(points,rotation,translation,intrinsic,None)[0].reshape(-1,2)
    settings["cameras"]["central"]["position"]=position.tolist()
    settings["cameras"]["central"]["xyaxes"]=np.r_[matrix[0],-matrix[1]].tolist()
    settings["cameras"]["central"]["target"]=(position+matrix[2]).tolist()
    output.mkdir(parents=True)
    (output/"scene_candidate.yaml").write_text(yaml.safe_dump(settings,sort_keys=False))
    result={"status":"CONDITIONAL_CENTRAL_CAMERA_CANDIDATE", "frames_used":len(quads),
            "excluded_frames":excluded,
            "image_corners":pixels.tolist(),"position":position.tolist(),
            "corner_fit_rmse_pixels":float(np.sqrt(np.mean((pixels-projected)**2))),
            "tray_corner_variability_pixels":np.std(quads,axis=0).tolist(),
            "assumptions":["Tray dimensions/location/orientation remain estimates",
                           "Focal length retained from estimated 52 degree vertical FOV",
                           "Distortion assumed zero; corner correspondence assumed",
                           "A single plane does not uniquely calibrate intrinsics/extrinsics",
                           "Wrist camera and joint axes not fitted by this script"]}
    (output/"fit_report.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-dir",type=Path,required=True)
    parser.add_argument("--config",type=Path,default=Path(__file__).resolve().parent/"configs/scene.yaml")
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    run(args.alignment_dir,args.config,args.output_dir)
