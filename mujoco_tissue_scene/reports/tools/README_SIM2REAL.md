# README_SIM2REAL — sim-to-real tissue scene

This directory is the MuJoCo scene that reproduces the real robot cell (qiuzhi O10 left arm
+ OmniHand left hand, black 120x75cm tabletop, green box, three blue tissue bags).

**Run this for a full status dump (paths, calibrated values, commands):**

```bash
bash /workspace/shared/mujoco_tissue_scene/status.sh
```

## Files

| file | role |
|---|---|
| `scene.py` | generators the MJCF from the config; `--check`, `--render`, `--viewer`, `--seed`, `--no-randomize` |
| `configs/scene.yaml` | **single source of truth**: table, tray (the green box), bags + spawn region, both cameras, arm/hand, control |
| `measure_layout.py` | measures object poses on the tabletop from one central-camera photo (no tape measure) |
| `sim_real_align.py` | writes `real.png` / `sim.png` / `blend.png` / `overlay.png` / `alignment.json` |
| `update_scene_config.py` | older helper that folds calibration YAMLs into `configs/scene.yaml` |
| `capture_hand_eye_dataset.py` | capture + gating + robust hand-eye / intrinsics solver, with a web UI |

## Environments

* `MUJOCO_GL=osmesa /opt/miniconda3/envs/turbovla-libero/bin/python scene.py --check --render`
* camera work: `/opt/miniconda3/envs/arm-hand-teleop/bin/python` (opencv + scipy)

## Vocabulary (avoid confusion)

* `tray` in the config **is the green box** on the table (white walls, green inner floor, 21x20x7.5cm).
* `boxes` in the config **are the three blue tissue bags** (12x8.5x6.5cm).
* There is no other tray. The right-hand second arm is deliberately not modelled.

## Known still-estimated values

Table height 0.75m, arm base `euler.z`, tray wall thickness 8mm, bag mass 50g, friction
coefficients, arm dynamics. Everything else (both cameras, table size, arm base xy, green box
pose, bag spawn region) is measured and cross-checked.

## Full status dump

```bash
bash /workspace/shared/mujoco_tissue_scene/status.sh
```

Everything for this work lives on this host: the scene itself in this directory, and the
report, tools archive, alignment images and measurement evidence under `reports/`
(start at `reports/START_HERE.md`).  No copies are kept anywhere else.
