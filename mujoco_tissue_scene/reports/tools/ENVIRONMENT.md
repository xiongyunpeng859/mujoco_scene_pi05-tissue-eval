# 仿真环境说明（已搭建完成）

主机 `fmc3-robotics-01002`，工程 `/workspace/shared/mujoco_tissue_scene`。
本文件描述**这个仿真环境现在长什么样、怎么用、被什么验证过、还差什么**。

---

## 1. 怎么用

```bash
cd /workspace/shared/mujoco_tissue_scene
export MUJOCO_GL=osmesa
PY=/opt/miniconda3/envs/turbovla-libero/bin/python

# 作为 pi0.5 环境用
$PY - <<'EOF'
from sim_env import TissueSceneEnv
env = TissueSceneEnv()                      # 观测/动作与数据集逐字段一致
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action16)   # 16 维绝对关节位置
EOF

# 自检：回放真机 episode 0 并报跟踪误差
$PY sim_env.py --demo --episode 0 --seed 0

# 只渲染
$PY scene.py --render --output-dir outputs/run_XXX                  # 随机袋位
$PY scene.py --render --no-randomize --output-dir outputs/measured  # 实测袋位
$PY scene.py --check                                                # 只做物理自检
```

## 2. 观测与动作契约（与数据集 `meta/info.json` 完全一致）

| 键 | dtype | shape | 来源 |
|---|---|---|---|
| `observation.images.top` | uint8 | (480, 640, 3) | 中央相机 UVC `LRCP 500W`（`/dev/video12`） |
| `observation.images.left` | uint8 | (480, 640, 3) | 左腕 RealSense D405（`sn 260322276846`） |
| `observation.state` | float32 | (16,) | 16 个关节绝对位置 |

动作 = **16 维绝对关节位置目标**，顺序与数据集逐维相同：

```
0..5    joint1..joint6                              机械臂
6..15   thumb_cm_roll, thumb_cm_yaw, thumb_cm_pitch,
        index_mp_yaw, index_mp_pitch, middle_mp_pitch,
        ring_mp_yaw, ring_mp_pitch, pinky_mp_yaw, pinky_mp_pitch
```

映射的唯一真源是 `action_layout.py`（`DATASET_NAMES` / `SIM_JOINTS` / `ACTUATORS` 逐位对齐）。
`step()` 不做任何 reshape、不换算成 delta、不转笛卡尔。

- 物理 500 Hz（dt 0.002），控制 30 Hz，`step()` 精确推进 1/30 s（16/17 步交替）。
- 镜头畸变按标定系数施加（真机视频带畸变，MuJoCo 渲染的是理想针孔）。
- 任务字符串：`pick up the tissue pack and place it on the right side`。

## 3. 场景参数（全部实测或数据驱动）

| 项 | 值 | 依据 |
|---|---|---|
| 桌面 | 120 × 75 cm，面高 0.75 m（高度仍是估计） | 用户卷尺 |
| 中央相机 | `fx=248.786 fy=248.894 cx=307.069 cy=246.547`，内参 RMS 0.087px；外参平差 RMS 0.157px | 棋盘格 + 纸角点 |
| 腕相机 | hand-eye `T_eef_camera`，39 帧，平移 RMS 4.6mm / 旋转 1.44° | 手眼标定 |
| 绿盒子(tray) | 中心 (71.25, 40.61) cm，yaw −91.2°，21×20×7.5 cm | 绿内底正面 PnP，角点 RMS 1.6px |
| **机械臂底座** | **(95.0, 15.0) cm，yaw 1.7738 rad** | 数据集 episode 首帧轮廓匹配 |
| 纸巾袋随机区 | x 6..52 cm，y 43..72 cm，yaw 抖动 ±25°，最小间距 15 cm | 用户摆的三个典型位置实测 |

`configs/scene.yaml` 是唯一真源；`scene.py` 只读不算。

## 4. 已做的验证（都是可复现的实验，不是断言）

| 验证 | 方法 | 结果 |
|---|---|---|
| 相机外参 | 三个互相独立的物体 | 白纸差 0.2cm；绿盒子两次独立解差 0.3cm；桌边在叠加图上重合 |
| 16 维语义与符号 | 数据集每一维的 [min,max] 对仿真关节限位 | 16/16 全部落在限位内（含拇指 pitch 的负区间） |
| 控制保真度 | 把真机 episode 的 action 序列回放 | 法兰相对真机轨迹 **RMS 3.8mm / 峰值 11.7mm / 末态 0.8mm**；无执行器饱和 |
| 执行器增益 | kp×kv×阻尼 27 组扫描 | kp=100/kv=20 最优；kp=300/1000 因 ±80N·m 饱和反而变差 |
| 臂底座位置 | 5 个候选 + 12 组细化，各自渲染后比轮廓 | (95,15)/yaw 1.7738 最优，匹配度 22.5/33.1%，是原配置的 2.7 倍 |
| 光照与材质 | 真机帧各区域实测色 → 逐轮按残差修正 | 墙 ±2、桌面 ±3、绿内底 ±2.5；整图 mean\|diff\| **100.6 → 38.0** |
| 端到端对齐 | 数据集 episode 0 首帧 vs 仿真渲染 | mean\|diff\| 41.2，轮廓匹配 27.5/46.2% |
| 观测契约 | 与数据集 `info.json` 逐字段比对 | 键名/dtype/shape 完全一致 |

复现命令见 `reports/tools/` 下各脚本的 docstring；每一项都留了 JSON 结果在 `reports/`。

## 5. 这一轮修掉的问题

| 问题 | 原来 | 现在 |
|---|---|---|
| **臂底座左右基准反了** | `measured_layout` 写 (25,10)cm，机械臂渲染在画面左下 | (95,15)cm、yaw 镜像为 π−1.3678；匹配度 8.2→22.5% |
| **5 个手关节被硬锁** | `scene.py` 用 equality 把它们钉死在 (−0.03,1.51,0,0,0) | 自由关节 + 执行器保持，重置时取该 episode 的实际值，16 维动作完全可表示 |
| **手部动作只接受二值** | `hand_control.command()` 只认 "open"/"closed" | 接受长度 10 的连续绝对位置 |
| **腕相机光心偏 16mm** | 相机放在 mount 的 `0 0 -0.016` | 光心在原点，外壳后移 |
| **托盘不支持 yaw** | 5 块 geom 写死在 worldbody | `body + euler` |
| **外观全暗** | 只有一盏直射光、无 ambient，墙渲染 ≈90 | ambient 0.80 + diffuse 0.35 + 实测材质，墙 ≈208 |
| **物体位置写死** | 3 个固定坐标 | 绿盒子固定 + 袋子区域内随机（seed 可复现） |
| **起姿写死** | 固定 `initial_qpos` | 从 214 个 episode 首帧采样 + ±0.01 rad 抖动 |

## 6. 仍然是估计值（别当实测用）

桌面高度 0.75 m、托盘壁厚 8 mm、纸巾袋质量 50 g、摩擦系数、机器人动力学参数。
另外真机画面有**渐晕和阴影**（角落更暗），仿真没有，这是整图残差的一个固定来源。

## 7. 关键文件

| 文件 | 作用 |
|---|---|
| `configs/scene.yaml` | 唯一真源：几何/相机/材质/光照/控制/随机化 |
| `scene.py` | 由配置生成 MJCF；`--check --render --seed --no-randomize` |
| `sim_env.py` | pi0.5 兼容环境（观测/动作/畸变/重置采样） |
| `action_layout.py` | 16 维 ↔ 仿真关节/执行器 的唯一映射 |
| `dataset_io.py` | 读 LeRobot parquet 成 per-episode 数组 |
| `analyze_lerobot_dataset.py` | 数据集逐维统计（仿真的对齐目标） |
| `replay_check.py` | 回放真机轨迹测保真度 + 增益扫描 |
| `align_with_dataset.py` | 数据集真机帧 vs 仿真渲染的对齐与指标 |
| `match_appearance.py` | 光照/材质对齐的测量工具 |
| `measure_layout.py` | 从照片实测物体位姿（bags / tray） |
| `arm_base_search.py` / `arm_base_refine*.py` | 判定臂底座位置的那组实验 |
| `status.sh` | 一条命令打印全部状态 |
