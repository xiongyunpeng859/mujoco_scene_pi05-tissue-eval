#!/usr/bin/env bash
# One-shot status dump for the sim-to-real MuJoCo tissue scene.
# Run:  bash /workspace/shared/mujoco_tissue_scene/status.sh
set -u

ROOT=/workspace/shared/mujoco_tissue_scene
PY_RENDER=/opt/miniconda3/envs/turbovla-libero/bin/python       # has mujoco 3.11.0
PY_CV=/opt/miniconda3/envs/arm-hand-teleop/bin/python           # has opencv + scipy
DATASET=/workspace/shared/new_program_qiuzhi/without_tactile/pi05_normal_recovery_merged_214eps

echo "==================================================================="
echo " MuJoCo sim-to-real tissue scene - status"
echo "==================================================================="
echo "project root : $ROOT"
echo "host         : $(hostname)  ($(hostname -I 2>/dev/null | awk '{print $1}'))"
echo "date         : $(date '+%Y-%m-%d %H:%M:%S')"
echo "render env   : $PY_RENDER  -> mujoco $($PY_RENDER -c 'import mujoco;print(mujoco.__version__)' 2>/dev/null || echo MISSING)"
echo "vision env   : $PY_CV  -> $($PY_CV -c 'import cv2,scipy;print("opencv",cv2.__version__)' 2>/dev/null || echo MISSING)"
echo

echo "--- key files -----------------------------------------------------"
for f in scene.py configs/scene.yaml README_SIM2REAL.md ENVIRONMENT.md status.sh \
         sim_env.py action_layout.py dataset_io.py analyze_lerobot_dataset.py \
         replay_check.py align_with_dataset.py match_appearance.py \
         measure_layout.py sim_real_align.py hand_control.py; do
  if [ -f "$ROOT/$f" ]; then
    printf "  %-28s %8s B   %s\n" "$f" "$(stat -c%s "$ROOT/$f")" "$(stat -c%y "$ROOT/$f" | cut -d. -f1)"
  else
    printf "  %-28s %s\n" "$f" "(absent)"
  fi
done
echo

echo "--- calibration inputs -------------------------------------------"
for f in outputs/calibration/central_camera_extrinsics.yaml \
         outputs/calibration/hand_eye_left_realsense_20260918_062627/T_eef_camera.yaml; do
  [ -f "$ROOT/$f" ] && echo "  [ok] $f" || echo "  [--] $f"
done
ls -d "$ROOT"/outputs/calibration/*/ 2>/dev/null | sed 's|^|  dir |'
echo

echo "--- effective scene config (after apply_measured_layout) ---------"
"$PY_RENDER" - "$ROOT" <<'PY'
import sys, yaml, math
root = sys.argv[1]
sys.path.insert(0, root)
cfg = yaml.safe_load(open(root + "/configs/scene.yaml"))
try:
    import scene
    cfg = scene.apply_measured_layout(cfg)
    tray_left = scene.tray_left_edge_cm(cfg)
except Exception as exc:                      # pragma: no cover
    scene, tray_left = None, float("nan")
    print("  (scene.py import failed: %s)" % exc)
size = cfg["table"]["size"]
def cm(p):
    return (round((p[0] + size[0] / 2) * 100, 2), round((p[1] + size[1] / 2) * 100, 2))
print("  table        : %.0f x %.0f cm, surface_z = %.3f m%s"
      % (size[0] * 100, size[1] * 100, cfg["table"]["surface_z"],
         "" if cfg["table"].get("surface_z_measured") else "  (still an estimate)"))
print("  arm base     : %s cm, euler.z = %.4f rad" % (cm(cfg["arm"]["position"]), cfg["arm"]["euler"][2]))
tray = cfg["tray"]
print("  green box    : centre %s cm  size %.0fx%.0fx%.0f mm  yaw %.1f deg"
      % (cm(tray["center"]), tray["size"][0] * 1000, tray["size"][1] * 1000,
         tray["size"][2] * 1000, math.degrees(float(tray.get("yaw", 0.0)))))
print("                 left edge %.2f cm" % tray_left)
for box in cfg["boxes"]:
    print("  %-13s: %s cm  yaw %.1f deg" % (box["name"], cm(box["position"]), math.degrees(box["yaw"])))
rand = cfg.get("box_randomization", {})
if rand.get("enabled"):
    region = rand["region_xy_cm_from_left_bottom"]
    print("  bag spawn    : x %s  y %s  yaw jitter %.0f deg  min gap %.0f cm  seed %s"
          % (region["x"], region["y"], rand.get("yaw_jitter_deg", 0),
             rand.get("min_separation_cm", 0), rand.get("seed")))
else:
    print("  bag spawn    : DISABLED (boxes stay at the configured positions)")
central = cfg["cameras"]["central"]
intr = central["intrinsics"]
print("  central cam  : pos %s  fx=%.3f fy=%.3f cx=%.3f cy=%.3f  fovy=%.2f deg"
      % ([round(v, 4) for v in central["position"]], intr["fx"], intr["fy"],
         intr["cx"], intr["cy"], central["fovy"]))
print("                 intrinsics_calibrated=%s  extrinsics_calibrated=%s"
      % (central.get("intrinsics_calibrated"), central.get("extrinsics_calibrated")))
wrist = cfg["wrist_camera"]
print("  wrist cam    : parent %s  pos %s  hand_eye_calibrated=%s"
      % (wrist["parent_body"], [round(v, 5) for v in wrist["position"]],
         wrist.get("hand_eye_calibrated")))
he = wrist.get("hand_eye", {})
if he:
    print("                 %s, %s samples, translation RMS %.4f m / rotation RMS %.3f deg"
          % (he.get("method"), he.get("sample_count"), he.get("translation_rms_m", 0),
             he.get("rotation_rms_deg", 0)))
control = cfg.get("control", {})
if control:
    print("  control      : physics %s Hz (dt %s s), control %s Hz, action %s"
          % (control.get("physics_frequency_hz"), control.get("physics_timestep_s"),
             control.get("control_frequency_hz"), control.get("action_space")))
PY
echo

echo "--- latest render / alignment outputs ----------------------------"
for d in outputs/calibrated_20260918 outputs/measured_placement \
         outputs/alignment_measured outputs/alignment_20260918; do
  if [ -d "$ROOT/$d" ]; then
    echo "  $d/"
    ls -1 "$ROOT/$d" | sed 's|^|      |'
  fi
done
echo

echo "--- how to use ---------------------------------------------------"
cat <<EOF
  build + check                : $PY_RENDER scene.py --check
  build + check + render       : MUJOCO_GL=osmesa $PY_RENDER scene.py --render --output-dir outputs/run_XXX
  render measured bag layout   : MUJOCO_GL=osmesa $PY_RENDER scene.py --render --no-randomize --output-dir outputs/measured_placement
  reproducible random layout   : MUJOCO_GL=osmesa $PY_RENDER scene.py --render --seed 20260918 --output-dir outputs/run_XXX
  measure bags from a photo    : $PY_CV measure_layout.py --frame /tmp/frame.png --mode both --annotate /tmp/measured.png
  real vs sim overlay          : $PY_CV sim_real_align.py --real /tmp/frame.png --sim outputs/run_XXX/central.png --out outputs/alignment_XXX
  grab one central frame       : $PY_CV -c "import cv2;c=cv2.VideoCapture('/dev/video12',cv2.CAP_V4L2);c.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'));[c.read() for _ in range(20)];ok,f=c.read();cv2.imwrite('/tmp/frame.png',f)"
  environment contract + demo  : $PY_RENDER sim_env.py --demo --episode 0
  dataset statistics           : $PY_RENDER analyze_lerobot_dataset.py --dataset $DATASET
  replay fidelity / gains      : $PY_RENDER replay_check.py --dataset $DATASET --episodes 0 1 2 [--sweep]
  sim vs real dataset frame    : $PY_RENDER align_with_dataset.py --dataset $DATASET --episode 0 --frame 0 --out outputs/alignment_dataset --place-bags-from-image
  appearance measurement       : $PY_CV match_appearance.py --real /tmp/frame.png --sim outputs/run_XXX/central.png
  report / archive / index     : $ROOT/reports/
  this status script           : $ROOT/status.sh
EOF
echo "--- reports/ archive ---------------------------------------------"
if [ -d "$ROOT/reports" ]; then
  find "$ROOT/reports" -type f | sed "s|$ROOT/reports/|  |" | sort
else
  echo "  (no reports/ directory)"
fi
echo "==================================================================="
