# Sim-to-Real Environment Reconstruction Report

> **环境已经搭建完成**：可用的 pi0.5 兼容 mu 环境、全部实测参数、验证结果与用法，
> 见 `ENVIRONMENT.md`（同一目录）。本报告是 A/B/C 勘察与分类记录，保留原始结论与后续修订。

**目标**：把 MuJoCo 中的纸巾抓取场景，改造成尽可能匹配真机（求之 O10 左臂 + OmniHand 左手 + 黑桌 + 绿色托盘 + 蓝色纸巾袋）的 manipulation 仿真环境。

**本轮范围**：勘察 + A/B/C 分类（已完成）→ 你确认参数后动手修改场景（进行中，见 §0.5）。

> **术语对齐（容易混，先说清）**：配置里的 `tray`（"托盘"）**就是桌上那个"绿色盒子"**——白色盒壁 + 绿色内底，21×20×7.5cm。本报告后文统一叫"绿盒子 / tray"。
> 配置里的 `boxes`（3 个 12×8.5×6.5cm 的盒子）**就是蓝色纸巾袋**。桌上没有别的托盘。

- 勘察日期：2026-09-18
- 仿真工程：`/workspace/shared/mujoco_tissue_scene`（主机 `fmc3-robotics-01002` / `192.168.1.104`）
- 数据集：`/workspace/shared/new_program_qiuzhi/without_tactile/pi05_normal_recovery_merged_214eps`
- 机器人 SDK 工程：`/workspace/shared/o10-openpi-demo/arm-hand-teleop-o10-openpi-demo-stable`
- 渲染环境：`/opt/miniconda3/envs/turbovla-libero`（mujoco 3.11.0 / Python 3.10.20）

---

## 0. 结论摘要

| 维度 | 现状判定 |
|---|---|
| 机器人几何 | **基本可信**。6 关节 URDF 导入，关节数/顺序/限位与数据集数值全部自洽（已逐维验证） |
| 机器人控制 | **方向正确**。绝对关节位置控制（position actuator），16 维 action，与真机 `action_control_mode: joint` 一致 |
| 手部动作 | ⚠️ **过度简化**。真机动作是 0.7 rad 内 ~1/255 量化的连续值（222 个不同值），仿真只接受 open/closed 二值 |
| 桌面环境 | ✅ **已实测并交叉验证**。桌 120×75cm、臂底座 (25,10)cm、绿盒子中心 (71.3, 40.6)cm / yaw −91°；**桌高 0.75m 仍是估计值** |
| 中央相机 | ✅ **已标定，并被三个独立对象交叉验证**：白纸 0.2cm、绿盒子两次独立解差 0.3cm、桌面远近边在叠加图上完全重合 |
| 腕部相机 | ✅ **内参与 hand-eye 齐备**（39 帧，平移 RMS 4.6mm / 旋转 1.4°）；并修掉了仿真里光心沿光轴多偏 16mm 的 bug |
| 目标物体 | ✅ **位置已由照片实测**（3 个纸巾袋），并按要求改成"绿盒子左侧区域内随机采样"。尺寸/质量/摩擦仍是估计值 |
| 场景完整性 | ⚠️ 光照/材质与真机差距大（灰度均差 ≈100）。**右侧第二只机械臂按你的要求不建模** |

**一句话**：运动学和控制这条线已经可以直接用；**相机（中央+左腕）与物体布局已经测完并交叉验证**；剩下的是**外观/光照、第二只臂、以及若干物理量（桌高、质量、摩擦、动力学）**。

---

## 0.5 本轮更新（已完成并验证）

### 0.5.1 中央相机外参的三重独立验证

相机标定不能自证——用同一份数据解出来的参数去反投影同一份数据永远是对的。所以做了三次**互相独立**的检验：

| 检验 | 做法 | 结果 |
|---|---|---|
| 白纸 | 用标定好的相机把实测白纸位置 (55, 29.5)cm 投影回图像，再反投影回来 | 反投影得 x 55.1–74.9 / y 29.7–49.4cm，与卷尺值差 **0.2cm** |
| 绿盒子 | 换成完全不同的物体：对绿色内底（19.4×18.4cm）做正面 PnP，与配置里早先独立记录的 (73.5, 40.0)cm 比较 | 得 (71.25, 40.61)cm，差 **2.3cm**（差值里还含 90° 转角与"外壁 vs 内底"的差别） |
| 桌面边线 | 把 y=75cm 的远边投影到图像，与真机桌面/墙交界比较 | 叠加图上红（仿真）绿（真机）**几乎完全重合** |

另外托盘位姿在两帧不同照片上独立各解一次：(71.25, 40.61, −91.2°) 与 (71.19, 40.35, −89.6°)，中心差 **0.3cm**，角点 RMS 1.6px / 1.8px。

### 0.5.2 修掉的两个真实代码缺陷

| 位置 | 问题 | 影响 | 修复 |
|---|---|---|---|
| `scene.py` 腕相机制作 | 相机被放在 mount 的 `0 0 -0.016`，即沿**光轴再前移 16mm**；但 `wrist_camera.position` 是 hand-eye 标定给出的**光心** | 仿真里腕相机光心比标定值偏 16mm（标定本身精度只有 4.6mm） | 光心放 mount 原点，外壳几何往后移 16mm |
| `scene.py` 托盘制作 | 托盘 5 块 geom 直接写进 worldbody，按轴对齐摆放，**不支持 yaw** | 实测盒子转了 90°，仿真里是正的 | 改成 `body name="tray"` + `euler=[0,0,yaw]`；`in_tray` 判断也改为在托盘局部系里做 |

### 0.5.3 物体布局按你的口径重做

你的口径：**绿盒子位置固定；3 个纸巾袋每次都变，但永远在绿盒子左边**。所以配置里不再写死三个坐标，而是"固定盒子 + 区域随机"：

```yaml
box_randomization:
  region_xy_cm_from_left_bottom: {x: [6.0, 52.0], y: [43.0, 72.0]}
  yaw_jitter_deg: 25.0
  min_separation_cm: 15.0
  seed: 20260918
```

区域来自你摆的三个典型位置的实测值（`measure_layout.py` 从照片量出，不用卷尺）：

| 位置 | 实测中心 (cm) | yaw | 接触边拟合 RMS |
|---|---|---|---|
| 最左 / 最远 | (5.53, 70.2) | 15.6° | 0.86 cm |
| 中间 / 最远 | (44.76, 71.33) | −1.6° | 0.49 cm |
| 最靠近盒子 | (51.12, 43.11) | 89.1°（低置信） | 0.05 cm |

`scene.py` 新增 `apply_box_randomization()` 与 `--seed / --no-randomize`：同 seed 完全可复现，且会强制"袋子在托盘左边缘外、不出桌、彼此 ≥15cm"。

### 0.5.4 仍然只是估计值的量（不要当实测用）

桌面高度 0.75m、臂底座朝向 `euler.z`、托盘壁厚 0.008m、纸巾袋质量 50g、摩擦系数、机器人动力学参数、托盘内底离桌面的 8mm。

---

## 1. 证据来源

| 项 | 路径 / 命令 |
|---|---|
| 场景生成器 | `mujoco_tissue_scene/scene.py`（461 行） |
| 场景参数 | `mujoco_tissue_scene/configs/scene.yaml` |
| 手部姿态源 | `hand_control.py` → `configs/reset_poses/o10_dual_reset.json` |
| 臂 URDF | `.../qiuzhi/lerobot_play_1.0.4/.../urdf/play_e2/urdf/play_e2.urdf` |
| 手 URDF | `.../yudie/vendor_sdk/Omnihand-2025-SDK-dev_xuqigui/assets/urdf/omnihand_left.urdf` |
| 真机推理配置 | `.../configs/left_arm/o10_left_infer.yaml` |
| 相机注册表 | `.../configs/cameras/o10_cameras.yaml`、`o10_right_wrist_intrinsics.yaml`、`o10_right_wrist_hand_eye.yaml` |
| 数据集元数据 | `pi05_normal_recovery_merged_214eps/meta/{info,stats}.json` + 逐 episode parquet |
| 真机画面 | 从 `videos/observation.images.{top,left}/chunk-000/file-000.mp4` 抽帧 |

---

## A 类：已可从现有代码 / asset / 数据集自动获得

### A1. 机械臂运动学（已逐维验证）

`play_e2.urdf` 共 14 link、13 joint。其中**臂本体 6 个 revolute**：

| joint | axis | lower | upper | 数据集 state 实测范围 | 是否在限位内 |
|---|---|---|---|---|---|
| joint1 | 0 0 1 | -3.1416 | 2.0944 | [-1.1542, 1.2797] | ✅ |
| joint2 | 0 0 1 | -2.9671 | 0.17453 | [-1.9072, 0.1200] | ✅ |
| joint3 | 0 0 1 | -0.087266 | 3.1416 | [-0.0521, 1.6573] | ✅ |
| joint4 | 0 0 1 | -3.0107 | 3.0107 | [-2.9589, 2.6488] | ✅ |
| joint5 | 0 0 1 | -1.7628 | 1.7628 | [-1.5509, 1.1069] | ✅ |
| joint6 | 0 0 1 | -3.0107 | 3.0107 | [-2.9391, 2.9292] | ✅ |

→ **关节顺序 joint1..joint6、正负方向、限位与真机数据完全相容**，无需改动。

URDF 里还有**原装平行夹爪链**（`arm_connect_joint` → `eef_connect_base_link` → `e2_base_link` → `e2_gear_joint`/`e2_left_joint`/`e2_right_joint`(mimic×0.5) → `e2_virtual_link`，以及 `link6 → end_link`）。
`scene.py:124-129` 用 `arm_only=True` **只保留 base_link + link1..link6**，所以：

> ⚠️ **仿真里的 end-effector frame = `link6`**，而 URDF 另有 `end_link`（fixed）和原夹爪 TCP。真机 SDK 的 EEF/TCP 参考到底是 `link6`、`end_link` 还是夹爪中心——**需要确认**，因为它直接决定手部安装和 `T_ee_camera` 的基准（见 §5.4）。

### A2. 数据集与控制的对应关系

| 项 | 值 | 来源 |
|---|---|---|
| 格式 | LeRobot **v3.0** | `meta/info.json` |
| episodes / frames | 214 / 53,227 | 同上 |
| fps | **30** | 同上 |
| robot_type | `pico_follower_single_arm_agibot_o10` | 同上 |
| state / action 维度 | **16**（6 臂 + 10 手） | 同上 |
| state/action 语义 | **绝对关节位置**（字段名均为 `*.pos`） | 同上 |
| 控制模式 | `action_control_mode: joint`，`include_eef_pose: false` | `o10_left_infer.yaml` |
| 推理频率 | `fps: 30` | 同上 |
| action chunk | `actions_per_chunk: 50`，`chunk_size_threshold: 0.8` | 同上 |
| 相机 key | 16 维版：`observation.images.top`(中央) / `observation.images.left`(左腕) | `meta/info.json` |
| 相机 key 映射 | `top → base_0_rgb`，`left → left_wrist_0_rgb` | `o10_left_infer.yaml` 的 `observation_rename_map` |
| 图像 | 640×480×3，h264，yuv420p，30fps，RGB | `meta/info.json` |
| 任务文本 | 2 个：task0 正常抓取（108 集），task1 抓取失败重抓（106 集） | `pi05_merge_report.json` |
| episode 长度 | 185–892 帧，mean 248.7，median 236.5（6.2–29.7 s） | 逐 episode parquet |

**仿真侧对应关系（已一致，无需改）**：
- `scene.py:271-273` 6 个 `position` actuator → `joint1..joint6`
- `scene.py:274-276` 10 个 `position` actuator → 手部 10 关节
- `nu = 16` ✅ 与数据集 16 维 action 对齐
- `hand_control.py:6-8` 的 `HAND_JOINTS` 顺序 = 数据集手部 10 维顺序

### A3. 手部：关节定义与 open/close 数值（全部已验证）

手 URDF 共 24 link：**10 个主动 revolute + 4 个 mimic（dip）+ 1 个固定（middle_abad）**。

| # | 数据集名 | URDF joint | axis | 限位 | 真机 action 值 | 真机 state 范围 | 一致 |
|---|---|---|---|---|---|---|---|
| 6 | thumb_cm_roll | `L_thumb_roll_joint` | 1 0 0 | [-0.8727, 0.1745] | 恒定 **-0.03** | [-0.0542, 0.0241] | ✅ |
| 7 | thumb_cm_yaw | `L_thumb_abad_joint` | 0 0 1 | [0, 1.7453] | 恒定 **1.51** | [1.3623, 1.6400] | ✅ |
| 8 | thumb_cm_pitch | `L_thumb_mcp_joint` | 1 0 0 | [-0.8552, 0] | **-0.7**（开 0） | [-0.7003, 0] | ✅ |
| 9 | index_mp_yaw | `L_index_abad_joint` | 1 0 0 | [0, 0.2094] | 恒定 **0** | [0, 0.0451] | ✅ |
| 10 | index_mp_pitch | `L_index_pip_joint` | 0 1 0 | [0, 1.5708] | **0.7**（开 0） | [0.0129, 0.7042] | ✅ |
| 11 | middle_mp_pitch | `L_middle_pip_joint` | 0 1 0 | [0, 1.5708] | **0.7** | [0.0118, 0.7042] | ✅ |
| 12 | ring_mp_yaw | `L_ring_abad_joint` | 1 0 0 | [-0.1745, 0] | 恒定 **0** | [-0.0048, 0] | ✅ |
| 13 | ring_mp_pitch | `L_ring_pip_joint` | 0 1 0 | [0, 1.5708] | **0.7** | [0.0096, 0.7042] | ✅ |
| 14 | pinky_mp_yaw | `L_pinky_abad_joint` | 1 0 0 | [-0.1745, 0] | 恒定 **0** | [-0.0034, 0] | ✅ |
| 15 | pinky_mp_pitch | `L_pinky_pip_joint` | 0 1 0 | [0, 1.5708] | **0.7** | [0.0107, 0.7042] | ✅ |

- **gripper open/close 数值（实测，非猜测）**，来自 `o10_dual_reset.json` → `gestures.cylindrical_straight.left`：
  - open = `[-0.03, 1.51, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`
  - closed = `[-0.03, 1.51, -0.7, 0.0, 0.7, 0.7, 0.0, 0.7, 0.0, 0.7]`
  - `scene.py:364-366` 已用这组值做 keyframe ✅
- 中指的 `middle_abad` 在 URDF 里本来就是 **fixed** —— 与数据集**没有** `middle_mp_yaw` 这一维完全吻合 ✅
- 4 个 `dip` 由 mimic 驱动（×1.097 / thumb ×1.3, ×1.33），`scene.py:202-209` 已转成 MuJoCo equality ✅

### A4. 桌面布局（已由你实测的部分）

`configs/scene.yaml → measured_layout`：桌 120×75cm，臂底座 (25,10)cm，托盘左下 (63,30)cm，托盘 21×20cm。
`scene.py:65-105` 会把这些 cm 值换算到以桌心为原点的 MuJoCo 坐标：
臂底座 → `[-0.35, -0.275]`，托盘中心 → `[0.135, 0.025]`。这部分**可直接用**。

### A5. 腕部相机内参

`scene.py:54-62` 已实现 OpenCV → MuJoCo 的转换（`focalpixel` + `principalpixel` 带符号偏移）。
左腕厂家内参（序列号 260322276846，640×480）：`fx=394.4324, fy=393.4235, cx=318.9330, cy=250.6992`，畸变 `inverse_brown_conrady` 已记录但**未施加**（MuJoCo 原生是针孔）。

### A6. 频率与时间基准

| 项 | 值 | 位置 |
|---|---|---|
| MuJoCo physics timestep | 0.002 s（**500 Hz**） | `scene.py:285` |
| 真机控制 / 相机 / policy | **30 Hz** | `o10_left_infer.yaml` |
| action chunk | 50 步 @30Hz（=1.667 s），消费阈值 0.8 | 同上 |
| 500Hz↔30Hz 调度 | 16/17 步交替（30 次控制精确 = 1 s） | README 已有说明 |

→ **不因 timestep 改变 π0.5 action 物理含义**这一条，README 和 `test_torch_rollout.py` 已按此实现 ✅

### A7. 真机在线相机配置（用于对齐仿真相机）

```yaml
left_wrist: intelrealsense, serial "260322276846", 640x480@30, use_depth: false, rgb
right_wrist: intelrealsense, serial "409122272886"
top:        opencv(UVC), /dev/v4l/by-id/usb-LRCP_ZAWK_LRCP_500W_SN0001-video-index0,
            640x480@30, fourcc: MJPG
```

---

## B 类：可从数据集估计 / 已有部分证据，但**建议用真实标定确认**

### B1. 中央相机内外参（最高优先级）
- **数据集内不存在任何标定元数据**（已检查 `meta/`、全目录搜 `calib|intrinsic|extrinsic` → 0 命中）。
- 顶置相机是 UVC 设备（LRCP 500W / ZAWK SN0001），**不通过 V4L2 暴露厂家内参**，故 `scene.yaml` 里 `fov: 52` 是猜的。
- **但**：桌面（120×75cm 已知）+ 绿色托盘（21×20cm 已知）都是**尺寸已知的平面矩形**，因此可以从真机 top 视频里做 **PnP 联合求解**（外参 + 焦距，主点取图像中心），再用棋盘格独立验证。
  `fit_central_camera.py` 已尝试过这条路，自评"四角 RMSE 2.79 px 但其他物体仍不匹配，不能当作正确标定" —— 这个自我否定的结论是**对的**，因为当时用了估计的托盘尺寸和估计 FOV。

### B2. 左腕 hand-eye（`T_ee_camera`）
- **现状：不存在左腕标定结果。** 工程里只有 `configs/cameras/o10_right_wrist_hand_eye.yaml`（右腕，serial 409122272886，method tsai，20 样本）。
- 而且那份右腕标定**质量很差**：`translation_rms_m = 0.0296`（**29.6 mm**）、`translation_max_m = 0.0585`、`rotation_rms = 6.26°`、`rotation_max = 12.35°`。手眼标定做到 3cm RMS 基本不可用，**不要照搬到左腕**。
- `outputs/calibration/left_wrist_checkerboard_20260917_012442`：16/16 张棋盘格检测成功、内参算出来了，但 `arm_joint_positions_rad: null` —— **每张图都没有配对的机械臂关节角**，所以这份数据**无法进入 hand-eye 求解器**（README 自己也这么写了）。
- 结论：需要一个**图像与关节角同步记录**的新采集流程。

### B3. 物体参数（尺寸 / 质量 / 摩擦）
- 当前值全部是估计：纸巾袋 `0.12×0.095×0.065 m`、`50 g`、摩擦 `[0.8, 0.005, 0.0001]`；托盘高 0.045/壁厚 0.008。
- 数据集**给不出**绝对尺寸或质量（只有 RGB + 关节角）。
- 可从数据估计的：**物体在画面中的像素区域**（见 B4），以及抓取时的关节范围（见 B5）。

### B4. 物体初始位置分布
- 数据集 state 里**没有物体位姿**，无法直接得到世界坐标。
- 可估计的是**像素区域**（需中央相机标定后才能映射到桌面坐标）。
- 但**机器人起始姿态分布可以直接统计**（可直接用于 sim reset 随机化）：

| 关节 | first-frame min | max | mean | std |
|---|---|---|---|---|
| joint1 | -0.5228 | 0.6003 | 0.0458 | 0.2123 |
| joint2 | -1.5951 | 0.0016 | -0.5236 | 0.5604 |
| joint3 | -0.0179 | 1.0657 | 0.2647 | 0.3217 |
| joint4 | -2.5904 | 2.2032 | -0.1182 | 0.9720 |
| joint5 | -0.5056 | 0.5503 | -0.0132 | 0.2403 |
| joint6 | -2.1429 | 2.6625 | 0.1022 | 0.9725 |

  → **注意：真机录制起点不是固定 reset 位**（joint4/joint6 跨度近 4.8 rad）。当前 `scene.py` 用固定 `initial_qpos` 是不符合数据分布的。
- 单 episode 内关节行程（mean over 214 集）：j1 1.12 / j2 1.47 / j3 1.08 / j4 2.07 / j5 0.65 / j6 2.04 rad。
- 手部起始：thumb_cm_pitch 的 5% 分位 = -0.699 → **约一半 episode（106 个 recovery）起手就是闭合状态**。

### B5. 5 个"被锁定"关节的真实柔顺偏差 ⚠️
当前 `scene.py:210-215` 把 open==closed 的 5 个关节用 **hard equality 约束锁死**：

| 关节 | action（锁死值） | 真机 state 实际范围 | 偏差 |
|---|---|---|---|
| thumb_cm_roll | -0.03 | [-0.0542, 0.0241] | 0.078 rad (4.5°) |
| thumb_cm_yaw | 1.51 | [1.3623, 1.6400] | **0.278 rad (15.9°)** |
| index_mp_yaw | 0 | [0, 0.0451] | 0.045 rad |
| ring_mp_yaw | 0 | [-0.0048, 0] | 0.005 rad |
| pinky_mp_yaw | 0 | [-0.0034, 0] | 0.003 rad |

→ 真机这几维**是有被动柔顺/漂移的**（拇指 yaw 尤其明显）。硬锁会让 **sim state 与 real state 在这 5 维上必然不一致**（直接违反你要求的"state 完全一致"）。
建议改为：**保留关节自由度、不给 actuator、加弱弹簧/阻尼**，或在 reset 时按实测分布采样。这属于参数问题，不必现在决定。

### B6. 场景视觉内容（从真机抽帧得到的新证据）
真机 top 画面（episode 0，t=0s / 4s / 8s）显示：

| 真机看到的 | 当前仿真 | 差异 |
|---|---|---|
| 黑桌面，画面下方约 2/3 | ✅ 有 | 一致 |
| 浅灰墙 + 线缆（画面上方 1/3） | ✅ 有 `background_wall` | 颜色接近，无细节 |
| **绿底白边托盘位于桌面中部偏左** | 位置由实测换算成右半桌 | ⚠️ 需相机标定才能判定 |
| **3 个蓝色纸巾袋在桌子左侧、成一排** | target 在托盘内 + 2 个 distractor 在远侧 | ❌ 初始布局不符 |
| **黑色臂身（带 FMC 标识）+ 银白灵巧手在画面左侧** | 仿真是银白色臂 | ⚠️ 视觉差异（`scene.py:200-201` 把非视觉 mesh 统一刷成银色） |
| **第二只白色机械臂（画面右侧）** | ❌ 未建模 | 真机画面里存在 |
| **右桌沿外的蓝色塑料筐** | ❌ 未建模 | 真机画面里存在 |
| **桌面中前部一张带深色十字的白纸/垫** | ❌ 未建模 | 真机画面里存在，且可能是标定物 |
| 纸巾袋有印刷标签 | 用 `_label` geom 近似 ✅ | 大体一致 |

→ 这些差异会**直接进入策略的输入图像**，因此不是纯外观问题。

---

## C 类：必须由你额外测量 / 标定才能得到

### C1. 需要你测量的物理量

| # | 参数 | 当前值（估计） | 为什么必须测 | 怎么测 |
|---|---|---|---|---|
| 1 | **桌面高度 `TABLE_HEIGHT`** | 0.75 m（猜测） | 决定整个场景的 z 基准，错一点全部错 | 卷尺量桌面到地面 |
| 2 | **托盘几何**：高度、壁厚、内底高度 | 0.045 / 0.008 m | 物体落入托盘的高度和碰撞 | 卡尺量绿托盘外高、壁厚 |
| 3 | **纸巾袋尺寸** | 0.12×0.095×0.065 m | 抓取几何、碰撞体、质量分布 | 卡尺量单个袋的长宽高（含/不含封口） |
| 4 | **纸巾袋质量** | 50 g | 抓取力、摩擦标定 | 电子秤 |
| 5 | **目标物体起始位置与朝向** | 托盘内（照片布局） | 真机是**在桌子左侧**，必须给坐标 | 以"从中央相机看的桌面左下角"为原点，量 3 个袋的中心 xy + 偏航 |
| 6 | **托盘位置复测** | 左下角 (63,30)cm | 与画面投影不自洽（见 §5.1） | 复测时**同时说明"左右"是相对相机还是相对你站的位置** |
| 7 | **臂底座位置复测** | (25,10)cm | 同上 | 同上，并注明是底座中心还是安装孔 |
| 8 | **臂底座朝向** | yaw 78.36°（估计） | 决定整个工作空间朝向 | 量底座某条边相对桌边的夹角 |
| 9 | **第二只臂 / 蓝色筐 / 白纸十字** 的位置尺寸 | 无 | 会进画面，影响策略 | 至少给蓝筐外形与位置；白纸十字若有用途请给边长 |

### C2. 需要你做标定的项

| # | 项 | 输入 | 输出 | 现状 |
|---|---|---|---|---|
| 10 | **中央相机内参** | 7×8 内角点棋盘格，在桌面多个位置/姿态拍 15–20 张（就用顶置相机） | `fx, fy, cx, cy` | 不存在 |
| 11 | **中央相机外参** | 用上面内参 + 桌面四角 + 托盘四角做 PnP | 相机 position + 姿态（或 roll/pitch/yaw/quat） | 估算值，且不自洽 |
| 12 | **左腕 hand-eye `T_ee_camera`** | 棋盘格/AprilTag 固定桌面，机械臂走 15–25 个姿态，**每个姿态同时记录 6 个关节角 + 一张腕部图** | 4×4 `T_ee_camera` | **不存在**；已有 checkerboard 数据因缺关节角而作废 |
| 13 | **EEF 基准定义** | 确认 SDK 的 TCP 是 `link6` / `end_link` / 夹爪中心哪一个 | 明确 `T_ee_camera` 挂在哪 | 未知 |
| 14 | 手部安装朝向 | 手法兰相对 link6 的旋转 | 决定腕部相机朝向（README §"已证实的问题"指出当前相机朝下、明显不对） | 估计值 |

---

## 5. 已发现的具体不一致（按影响排序）

### 5.1 ~~中央相机投影与桌面测量不自洽~~ ✅ 已解决
当时用的是"`fov: 52` 的估计相机 + 估计位置"，结论不可靠。中央相机完成棋盘格标定 + 纸角点外参后：
- 实测内参 `fx=248.79` 对应垂直 FOV **87.9°**，而原来的 `fov: 52` 差了近一倍 —— 这就是当时投影对不上的根因。
- 标定后的相机把实测白纸投到真机白纸位置差 **0.2cm**、把绿盒子投到真机绿盒子位置差 **2.3cm**、把桌面远/近边投到与真机边界**重合**（见 `alignment/overlay.png`）。

另外当时"真机画面里托盘中心约在 305px"这一观察也不成立：那帧里盒子已被移动到画面左上角，不是它的标准位置。

### 5.2 手部动作不是严格二值 ⚠️
数据集 action 第 8/10/11/13/15 维共 **222 个不同取值**，步长 ≈ 0.7/255 ≈ 0.00274 rad，范围 [0, ±0.7]。
→ 真机接受的是**连续量化**的关节位置目标，不是 open/closed。当前仿真 `hand_control.command()` 只接受 `{"open","closed"}`，会**丢掉真机 action 的连续部分**。
（你要求的"action 定义完全一致"在这一项上目前不满足。）

### 5.3 ~~目标物体初始布局反了~~ ✅ 已解决
按你的口径重做：绿盒子固定、3 个纸巾袋在**绿盒子左侧区域随机**（见 §0.5.3）。
实测三个典型位置为 (5.53, 70.2)、(44.76, 71.33)、(51.12, 43.11) cm，随机区域取它们的包围盒 x∈[6,52]、y∈[43,72]，并带 ±25° 的 yaw 抖动。
`--no-randomize` 时用实测的那三个位置，此时 sim/real 叠加图上袋子轮廓正好落在真机袋子上。

### 5.4 EEF frame 不明确
`arm_only=True` 砍掉了原夹爪链，仿真 EEF = `link6`。但 URDF 还有 `end_link` 和夹爪 TCP，SDK 的 EEF 定义未确认。
→ 手部安装 + `T_ee_camera` 的基准若不统一，手眼标定结果无法直接用。

### 5.5 机器人起始姿态被写成固定值
`scene.py:358-363` 用固定 `initial_qpos` 校验并写 keyframe，而实测首帧分布很宽（§B4）。
→ 应按分布随机初始化（你要求的第 6 条）。

### 5.6 动力学参数不可信（预期之中）
`gravcomp=1` + `kp=100/kv=20` + `forcerange ±80`，手部 `kp=3/kv=0.1`，全部是估计值；URDF 的 `effort/velocity` 都是 0。按你说的"第一阶段不追求精度"，这一块**先不动**，留作后续 system identification。

---

## 6. 对照你的 12 条要求

| # | 要求 | 现状 | 缺什么 |
|---|---|---|---|
| 1 | 机器人模型（关节/顺序/限位/方向/gripper/actuator） | ✅ 已核实一致 | 仅 §5.2 连续手部动作 + §5.4 EEF 定义 |
| 2 | 桌面坐标系（TABLE_*, ROBOT_BASE_*） | ⚠️ 已有换算，但桌高未测、朝向是估计 | C1-1/7/8，且需统一输出到独立配置 |
| 3 | 中央相机 | ❌ 无标定，参数散在 `scene.yaml` 的一个块里 | C2-10/11；需要独立 camera config 文件 |
| 4 | 腕部相机（T_ee_camera） | ⚠️ 结构已对（相机是 link6 后代，随臂运动 ✅），外参无 | C2-12/13/14；需要 hand-eye 采集脚本 |
| 5 | 目标物体（纸巾袋 / 绿色目标盒） | ⚠️ 有几何，全靠估计 | C1-2/3/4/5/6 |
| 6 | 物体初始位置分布 | ❌ 固定位置 | C1-5；机器人起始分布已在 §B4 统计好 |
| 7 | 物理参数 | ⚠️ 全是估计 | 第一阶段可保持，后续辨识 |
| 8 | 控制频率与时间 | ✅ 30Hz / 500Hz / chunk 50 已正确 | 无 |
| 9 | π0.5 observation/action 兼容 | ⚠️ `test_torch_rollout.py` 已能跑通路 | 需 action 连续化（§5.2）+ 相机对齐 |
| 10 | Sim/Real 对齐验证工具 | ⚠️ 已有 `align_real_dataset.py` + 18 张并排图 | 缺 `overlay.png`；且需标定后才能有意义 |
| 11 | 数据集自动分析工具 | ✅ 本轮已做（统计脚本可脚本化沉淀） | 需要落成正式脚本 |
| 12 | 集中配置文件 `sim_real_config.yaml` | ❌ 参数在 `configs/scene.yaml` + `hand_control.py` 常量里 | 待重构（等你确认参数后） |

---

## 7. 我需要你提供的东西（按优先级）

**第 1 优先（不给我做不了任何相机相关的对齐）**
1. 中央相机棋盘格标定数据（§C2-10）：顶置相机，7×8 内角点，**实测方格边长 mm**，15–20 张不同位置/姿态
2. 复测并说明：桌面尺寸 L/W/H（含高度）、臂底座位置与朝向、托盘位置 —— **并明确"左右"的参考系**
3. 确认 SDK 的 EEF/TCP 参考是哪个 link（§C2-13）

**第 2 优先（决定物体与 reset）**
4. 纸巾袋实测尺寸 + 质量
5. 真机上 3 个纸巾袋的起始中心坐标与偏航（以桌面左下角为原点）
6. 托盘高度、壁厚

**第 3 优先（决定场景完整度）**
7. 蓝筐尺寸/位置；第二只机械臂是否需要建模（如需要，它的 URDF 在哪）
8. 桌上白纸/十字是什么、是否标定物

**已有但需要重建的**
9. 左腕 hand-eye 采集：`capture_left_wrist_checkerboard.py` 已经能拍图，但**必须同时记录 6 个机械臂关节角**才能用于求解；现有 16 张图因缺关节角作废

---

## 8. 确认后的执行计划（现在不动手）

1. ~~新建 `configs/sim_real_config.yaml`~~ → 已按"扩展 `configs/scene.yaml`"实现（相机内参/外参、`wrist_camera.hand_eye`、`box_randomization`、`control`、`calibration_summary` 都在里面，单一真源）
2. ~~拆分出独立的相机标定文件~~ ✅ 相机标定已在 `configs/scene.yaml` 的 `cameras.central` / `wrist_camera` 内，且 `scene.py` 只读不算
3. ~~修 §5.1 的相机不自洽~~ ✅ 已用标定值替换，并被三个独立对象验证
4. ~~改物体初始布局~~ ✅ 绿盒子固定 + 纸巾袋在盒子左侧区域随机（`apply_box_randomization()`）
5. 手部动作从二值放开为 [0, 0.7] 连续目标（对齐真机 action 空间）—— **未做**
6. 5 个被动关节从硬 equality 改成弱弹簧/自由（对齐真机 state）—— **未做**
7. ~~扩展对齐脚本输出 `real.png` / `sim.png` / `overlay.png`~~ ✅ 新增 `sim_real_align.py`，产出在 `alignment/`
8. 把本轮的统计脚本沉淀成正式的 `analyze_lerobot_dataset.py` —— **未做**
9. 最后才跑 π0.5 端到端 sim 推理验证 —— **未做**

### 新增待办（本轮发现）

10. **光照与材质对齐**：仿真墙/桌面明显偏暗（灰度均差 ≈100），真机墙 ≈210、桌面 ≈77。当前 `light` 只有一盏 + 材质常数，需要按真机画面调光照与 `rgba`。
11. ~~第二只机械臂~~ → **按你的要求不建模**（中央相机里会看到它，忽略即可）
12. **腕相机畸变未在渲染中应用**（MuJoCo 是针孔原生），两台相机口径一致，暂不影响对齐。
13. 渲染对比时把机械臂摆到真机记录的关节角（用数据集某帧的 state），否则臂部轮廓必然不匹配。

---

## 附：本轮产物

> 所有产物都在**主机** `/workspace/shared/mujoco_tissue_scene/` 下，本地不留任何副本。

### 入口（先看这个）

| 用途 | 路径 / 命令（都在主机上） |
|---|---|
| **一条命令看全部状态** | `bash /workspace/shared/mujoco_tissue_scene/status.sh` |
| 索引（工程在哪、产物在哪、既定口径） | `/workspace/shared/mujoco_tissue_scene/reports/START_HERE.md` |
| 工程说明 | `/workspace/shared/mujoco_tissue_scene/README_SIM2REAL.md` |
| 本报告 | `/workspace/shared/mujoco_tissue_scene/reports/SIM2REAL_RECONSTRUCTION_REPORT.md` |

`status.sh` 会打印：工程路径、主机、python 环境、关键文件与时间戳、标定输入、
**当前生效的场景参数**（桌面/臂底座/绿盒子位姿/袋子随机区域/两台相机/控制频率）、
最近渲染与对齐产物、全部常用命令。

### 文件（相对 `reports/`）

- 本报告：`SIM2REAL_RECONSTRUCTION_REPORT.md`
- 标定工具说明：`CALIBRATION_TOOL_GUIDE.md`
- 抽帧证据：`real_frames/ep000_top_t0.png`、`ep000_top_t4.png`、`ep000_top_t8.png`、`ep109_top_t0.png`、`ep109_left_t0.png`
- 仿真现状渲染：`sim_preview/{overview,central,left_wrist}.png`
- **工具**：`tools/measure_layout.py`（`--mode bags|tray|both`，直接出 cm 与 yaw，带标注图）、`tools/sim_real_align.py`（输出 `real.png`/`sim.png`/`blend.png`/`overlay.png`/`alignment.json`）、`tools/capture_hand_eye_dataset.py`、`tools/scene.py`、`tools/scene.yaml`
- **对齐结果**：`alignment/{overlay,blend,real,sim}.png`、`alignment/alignment.json`、`alignment/sim_measured_placement.png`
- **实测标注**：`measurement/{real_frame_with_bags,measured_annotation,tray_pose_fit}.png`、`measurement/measurement.json`
- **标定输入与证据**：`calibration/{T_eef_camera.yaml,camera_intrinsics.yaml,central_camera_extrinsics.yaml,central_camera_extrinsics_verification.png,checkerboard_capture_12x9.png,wrist_camera_now.png,go_nogo_banner_NOT_READY.jpg}`
