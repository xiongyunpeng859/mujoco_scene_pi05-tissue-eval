# START HERE — MuJoCo sim-to-real tissue scene

主机：`fmc3-robotics-01002` / `192.168.1.104`，用户 `fmc3-6`
**所有内容都在主机上，本地电脑不留任何副本。**

## 现状（重要）

环境本体可用，但**任务在仿真里还复现不了**，因此**还不能采数据**。
原因与下一步见 `ENVIRONMENT.md` §8。

## 环境已经搭好

完整的用法、观测/动作契约、参数依据、验证结果与已知差距：**`ENVIRONMENT.md`**（工程根目录与 `reports/` 各一份）。

```bash
cd /workspace/shared/mujoco_tissue_scene
export MUJOCO_GL=osmesa
PY=/opt/miniconda3/envs/turbovla-libero/bin/python
$PY sim_env.py --demo --episode 0     # 回放真机 episode 并报跟踪误差
$PY scene.py --render --output-dir outputs/run_XXX
```

## 一条命令看全部

```bash
bash /workspace/shared/mujoco_tissue_scene/status.sh
```

打印：工程路径、主机、python 环境、关键文件与时间戳、标定输入、**当前生效的场景参数**
（桌面 / 臂底座 / 绿盒子位姿 / 纸巾袋随机区域 / 两台相机 / 控制频率）、
最近的渲染与对齐产物、`reports/` 归档清单、以及全部常用命令。

## 位置速查

| 东西 | 路径（主机上） |
|---|---|
| **工程本体** | `/workspace/shared/mujoco_tissue_scene/` |
| 场景生成器 | `.../scene.py` |
| 场景参数（单一真源） | `.../configs/scene.yaml` |
| 状态脚本 | `.../status.sh` |
| 工程说明 | `.../README_SIM2REAL.md` |
| **交付物归档** | `.../reports/` |
| 重建报告 | `.../reports/SIM2REAL_RECONSTRUCTION_REPORT.md` |
| 标定工具说明 | `.../reports/CALIBRATION_TOOL_GUIDE.md` |
| 渲染用 python | `/opt/miniconda3/envs/turbovla-libero/bin/python`（mujoco 3.11.0） |
| 视觉用 python | `/opt/miniconda3/envs/arm-hand-teleop/bin/python`（opencv 4.12 + scipy） |
| 数据集 | `/workspace/shared/new_program_qiuzhi/without_tactile/pi05_normal_recovery_merged_214eps` |
| 机器人 SDK 工程 | `/workspace/shared/o10-openpi-demo/arm-hand-teleop-o10-openpi-demo-stable` |

## reports/ 归档结构

```
reports/
├─ START_HERE.md                       ← 本文件
├─ SIM2REAL_RECONSTRUCTION_REPORT.md   ← 主报告（A/B/C 分类 + 验证结果）
├─ CALIBRATION_TOOL_GUIDE.md           ← 标定工具用法
├─ alignment/    real.png sim.png blend.png overlay.png alignment.json + sim_measured_placement.png
├─ measurement/  实测标注图 + measurement.json
├─ calibration/  T_eef_camera.yaml camera_intrinsics.yaml central_camera_extrinsics.yaml
│                中央相机外参验证图 / 棋盘格实拍 / 腕相机当前视角 / GO-NOGO 横幅
├─ real_frames/  数据集抽帧
├─ sim_preview/  仿真渲染（overview / central / left_wrist）
└─ tools/        measure_layout.py sim_real_align.py capture_hand_eye_dataset.py scene.py scene.yaml
```

## 常用命令

```bash
ROOT=/workspace/shared/mujoco_tissue_scene
PY_RENDER=/opt/miniconda3/envs/turbovla-libero/bin/python
PY_CV=/opt/miniconda3/envs/arm-hand-teleop/bin/python

# 构建 + 物理自检
$PY_RENDER $ROOT/scene.py --check

# 构建 + 渲染（overview / central / left_wrist）
MUJOCO_GL=osmesa $PY_RENDER $ROOT/scene.py --render --output-dir $ROOT/outputs/run_XXX

# 用实测的袋子位置渲染（做对齐验证时用这个）
MUJOCO_GL=osmesa $PY_RENDER $ROOT/scene.py --render --no-randomize --output-dir $ROOT/outputs/measured_placement

# 可复现的随机布局
MUJOCO_GL=osmesa $PY_RENDER $ROOT/scene.py --render --seed 20260918 --output-dir $ROOT/outputs/run_XXX

# 抓一帧中央相机
$PY_CV -c "import cv2;c=cv2.VideoCapture('/dev/video12',cv2.CAP_V4L2);c.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'));[c.read() for _ in range(20)];ok,f=c.read();cv2.imwrite('/tmp/frame.png',f)"

# 从照片实测物体位姿（输出 cm 与 yaw + 标注图）
$PY_CV $ROOT/measure_layout.py --frame /tmp/frame.png --mode both --annotate /tmp/measured.png

# real / sim / overlay 对齐
$PY_CV $ROOT/sim_real_align.py --real /tmp/frame.png \
       --sim $ROOT/outputs/run_XXX/central.png --out $ROOT/outputs/alignment_XXX
```

## 相机

- 中央相机 = UVC `LRCP 500W`，设备节点 `/dev/video12`（`/dev/video13` 是第二节点）。
- 腕部相机 = RealSense D405，序列号 `260322276846`。
- 两台相机都**无 GUI**，采集/预览走 web 界面（`capture_hand_eye_dataset.py`）。

## 场景既定口径（别再问一遍）

- 配置里的 `tray` = 桌上那个**绿色盒子**（白盒壁 + 绿内底，21×20×7.5cm），位置**固定**。
- 配置里的 `boxes` = **蓝色纸巾袋**，共 3 个，每次都变，**永远在绿盒子左边**。
- **右侧第二只机械臂不建模**。
- 不要擅自把动作改成 delta 或笛卡尔控制；保持 16 维绝对关节位置。

## 仍是估计值（别当实测用）

桌面高度 0.75m、臂底座 `euler.z`、托盘壁厚 8mm、纸巾袋质量 50g、摩擦系数、机器人动力学参数。
其余（两台相机、桌面尺寸、臂底座 xy、绿盒子位姿、袋子随机区域）都已实测并交叉验证。
