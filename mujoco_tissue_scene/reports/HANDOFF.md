# MuJoCo 仿真环境 — 工作交接文档

> 写给接手的 AI。目标：让你不用重走弯路就能接着做。
> 所有产物都在**主机** `/workspace/shared/mujoco_tissue_scene/`，本地不留副本。

---

## 0. 一句话现状

**仿真环境本体（相机/几何/物体/控制接口）已经搭好并被多项独立实验验证；但"抓取"这一步在仿真里复现不了 —— 手能碰到纸巾袋，袋子却不会被拿起来，因此任务无法完成，也就还不该开始采集数据。**

---

## 1. 任务是什么

把 MuJoCo 里的纸巾抓取场景改造成匹配真机的 manipulation 环境，用于 π0.5 相关的仿真评测 / 数据生成。

- 真机：求之 O10 左臂 + OmniHand 左手，黑色桌面 120×75cm，桌上有一个**绿色盒子**（配置里叫 `tray`）和 **3 个蓝色纸巾袋**（配置里叫 `boxes`）。
- 任务字符串（来自数据集）：`pick up the tissue pack and place it on the right side`。
- 口径（用户明确过，不要再问）：
  - 绿盒子位置**固定**；纸巾袋每次都变，**永远在绿盒子左边**。
  - **右侧第二只机械臂不建模**（真机画面右下的那只银色臂是闲置臂，不是受控臂）。
  - 不要擅自把动作改成 delta 或笛卡尔控制，保持 **16 维绝对关节位置**。

---

## 2. 访问与环境

```bash
# 局域网
plink -ssh -batch -hostkey SHA256:3fBjBj5fEIHhGtnecYc/V3Jzy/R7RhKk4zboNqguIQM \
      -pw 521521 fmc3-6@192.168.1.104
# 外网（Tailscale）
plink -ssh -batch -hostkey SHA256:3fBjBj5fEIHhGtnecYc/V3Jzy/R7RhKk4zboNqguIQM \
      -pw 521521 fmc3-6@100.87.220.18
```

主机 `fmc3-robotics-01002`，**负载常年在 20~35**，SSH 偶尔被掐（重试即可），渲染慢，建议 `nice -n 10`。

| 用途 | 解释器 |
|---|---|
| 渲染 / MuJoCo（3.11.0） | `/opt/miniconda3/envs/turbovla-libero/bin/python` |
| OpenCV / scipy / 标定 | `/opt/miniconda3/envs/arm-hand-teleop/bin/python` |

渲染必须 `export MUJOCO_GL=osmesa`。

关键外部路径：

- 工程：`/workspace/shared/mujoco_tissue_scene`
- 训练数据集（214 eps，含 recovery）：`/workspace/shared/new_program_qiuzhi/without_tactile/pi05_normal_recovery_merged_214eps`
- **全成功数据集（108 eps）**：`/workspace/shared/new_program_qiuzhi/without_tactile/success_episode/pick_up_the_tissue_pack_and_place_it_on_the_right_side_20260908_merged_all_108eps`
- 臂 URDF：`/workspace/shared/o10-openpi-demo/arm-hand-teleop-o10-openpi-demo-stable/qiuzhi/lerobot_play_1.0.4/x86/noble/lerobot_play-1.0.4-py3-none-any/lerobot_play/urdf/play_e2/urdf/play_e2.urdf`
- 手 URDF：`.../yudie/vendor_sdk/Omnihand-2025-SDK-dev_xuqigui/assets/urdf/omnihand_left.urdf`
- **复位手势文件**（很重要）：`.../configs/reset_poses/o10_dual_reset.json`

---

## 3. 一键了解状态

```bash
bash /workspace/shared/mujoco_tissue_scene/status.sh
```

打印工程路径、生效参数、关键文件、最近产物、全部常用命令。
另有 `ENVIRONMENT.md`（工程根 + `reports/`）记录环境说明与 §9/§10 的抓取诊断。
全部归档在 `reports/`（约 82 个文件，8.5MB），工具源码在 `reports/tools/`。

---

## 4. 已完成的（每条都有可复现实验）

### 4.1 相机标定（中央 + 腕部）

| 项 | 值 | 精度 |
|---|---|---|
| 中央相机内参 | `fx=248.786 fy=248.894 cx=307.069 cy=246.547`，畸变 `[-0.0130,-0.0292,0.00055,0.000065,0.00827]` | 10 视图 RMS 0.087px；卷尺独立复核差 3.5% |
| 中央相机外参 | 位置 `[0.0561,-0.0927,1.315]`，`xyaxes` 见配置；离桌面 56.5cm | 88 点联合平差 RMS 0.157px |
| 腕相机 | 厂家内参 + hand-eye `T_eef_camera`（基准 `link6`） | 39 帧，平移 RMS 4.63mm / 旋转 RMS 1.44° |

**外参被三个互相独立的对象交叉验证**（不是自证）：
白纸反投影 vs 卷尺实测差 **0.2cm**；绿盒子 PnP vs 配置旧值差 2.3cm；桌面远/近边在叠加图上与真机**重合**。

### 4.2 场景几何

| 项 | 值 | 依据 |
|---|---|---|
| 桌面 | 120×75cm，面高 0.75m | 卷尺；**面高仍是估计值** |
| 绿盒子 `tray` | 中心 **(71.25, 40.61) cm**，yaw **−91.2°**，21×20×7.5cm，壁厚 8mm（估计） | 绿内底正面 PnP，角点 RMS 1.6px；两帧独立解差 0.3cm |
| 机械臂底座 | **(25.0, 10.0) cm**，yaw **1.90 rad** | 见 §5.2，**yaw 是拟合值，有已知残差** |
| 纸巾袋随机区 | x∈[6,52] cm，y∈[43,72] cm，yaw 抖动 ±25°，最小间距 15cm，seed 可复现 | 用户摆的三个典型位置实测 |

### 4.3 外观（光照/材质）

从真机帧各区域实测色反推，逐轮按残差修正。**整图 mean|diff| 从 100.6 降到 38.0**，
墙面 ±2、桌面 ±3、绿内底 ±2.5（0-255 尺度）。配置在 `lighting:` / `materials:`。
仍存差异源于真机画面有**渐晕和阴影**，仿真没有。

### 4.4 π0.5 兼容环境 `sim_env.py`

```python
from sim_env import TissueSceneEnv
env = TissueSceneEnv()
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action16)
```

- 观测键名/dtype/shape 与数据集 `meta/info.json` **逐字段一致**：
  `observation.images.top` uint8(480,640,3)、`observation.images.left` uint8(480,640,3)、
  `observation.state` float32(16,)。无任何适配层、不改 delta/笛卡尔。
- 物理 500Hz（dt 0.002），控制 30Hz，`step()` 精确推进 1/30s（16/17 步交替）。
- 按标定系数施加**镜头畸变**（真机视频带畸变，MuJoCo 渲染理想针孔）。
- `reset()` 从 214 个 episode 首帧采样起姿（+±0.01rad 抖动），并重新采样三个袋子位置（不重建模型，写 free joint qpos）。
- `success()` 判据：目标袋中心落在托盘内且高于盘底。

### 4.5 16 维动作空间对齐

唯一映射真源是 `action_layout.py`。数据集 16 维语义：

```
0..5   joint1..joint6
6..15  thumb_cm_roll, thumb_cm_yaw, thumb_cm_pitch, index_mp_yaw, index_mp_pitch,
       middle_mp_pitch, ring_mp_yaw, ring_mp_pitch, pinky_mp_yaw, pinky_mp_pitch
```

**验证：16/16 维的真机 [min,max] 全部落在仿真关节限位内**（含拇指 pitch 的负区间
`[-0.8552, 0]` 对上真机 `[-0.7003,-0.0007]`）。映射和符号正确。

数据集实测（`analyze_lerobot_dataset.py`）：
- 6 个臂关节连续，步长 ≈0.00038。
- 5 个 "pitch" 手关节（dim 8,10,11,13,15）在 0 与 ±0.7 之间跳变，**最大步长恰好 0.70000**。
- 5 个 "yaw/roll" 手关节（dim 6,7,9,12,14）**同一 episode 内步长恒为 0**（置一次后保持），
  跨 episode 会变（如 thumb_cm_yaw 1.362~1.640）。

### 4.6 控制保真度

回放真机 episode 的 action 序列，**法兰相对真机记录轨迹 RMS 3.8mm / 峰值 11.7mm / 末态 0.8mm**，无执行器饱和。
增益扫描（27 组 kp×kv×阻尼）确认 **kp=100 / kv=20 / forcerange 80 就是最优**；
调高（300/1000）会因力矩饱和反而变差（20~58mm）。关节阻尼从 1.0 降到 **0.1** 有小幅改善。

### 4.7 修掉的代码缺陷

| 问题 | 原来 | 现在 |
|---|---|---|
| 腕相机光心沿光轴多偏 16mm | `<camera pos="0 0 -0.016">`（而 `position` 是标定光心） | 光心在 mount 原点，外壳几何后移 |
| 托盘不支持 yaw | 5 块 geom 写死在 worldbody | `body name="tray"` + `euler` |
| 5 个 yaw/roll 手关节被 equality 硬锁 | 钉死在 `(-0.03,1.51,0,0,0)`，数据集动作无法表示 | 自由关节 + 执行器保持，重置时取该 episode 实际值（误差 0.00000） |
| 手部动作只接受 open/closed 二值 | 丢掉真机连续量化 | `hand_control.command()` 接受长度 10 的连续绝对位置 |
| 执行器参数/材质/光照写死 | 硬编码 | 全部进 `configs/scene.yaml` |

### 4.8 单测

`pytest tests` → **16 passed**。（我改过 `tests/test_scene.py` 里一条断言：腕相机支架
现在挂在 `link6` 而不是手部 body，因为 `T_eef_camera` 的基准就是 `link6`。）

---

## 5. 没做到的 / 卡点

### 5.1 核心卡点：仿真里抓不住纸巾袋 ⛔

**回放数据集里一个真机成功的 episode（episode 0：从左侧抓一个袋子放进绿盒子），仿真结果：**

```
手到最近袋子最小距离   0.034 m
手-袋接触采样          260 次（frame 95 单独 139 次）
袋子抬升量             -0.0005 m   ← 纹丝不动
袋子进托盘             没有一个
```

**把对齐彻底排除的隔离实验**（`grasp_isolate2.py`）：把袋子**精确放到闭合后
"拇指指尖 / 四指指尖质心"的中点**，长轴沿该连线，再驱动真机的抬升段 —— 仍然不跟随。
所以**不是手没对准，是手扣不住这个物体**。

**已排除的原因：**

| 假设 | 实验 | 结果 |
|---|---|---|
| 臂伸不到 | `arm_reach.py` 采样关节限位 | 最大伸展 法兰 0.673m / 手掌 0.768m；以 (25,10) 为底座能到离袋子 0.001m ❌不是这个原因 |
| 手部数值单位不对 | 对比 `o10_dual_reset.json` | 数据集手部值**逐位等于 `cylindrical_straight` 手势**，就是弧度 ❌不是这个原因 |
| 物体太大/太刚 | `bag_collision_sweep.py` 扫 7 组（缩放 60~80% + 柔化 + 加摩擦） | **全部更差**，接触掉到 0、rise 全负 ❌不是这个原因 |
| 袋子需要可变形 | 已加 `soft_body`（`<flexcomp dim=3>`） | 能编译（nflex=3、72 顶点），但 MuJoCo 警告 **没有弹性回复力**；`young`/`poisson` 在 3.11 里**不能通过 flexcomp 传**，只能手写 `<flex>` ⚠️未完成 |

**手部几何实测**（`grip_geometry.py`）：

| | 拇指-食指 | 拇指-中指 | 指尖簇尺寸 |
|---|---|---|---|
| 手 OPEN（记录手势） | 20.79 cm | 21.22 cm | 11.89 × 17.33 × 9.42 cm |
| 手 CLOSED（记录手势） | 11.39 cm | 11.87 cm | 8.85 × 9.26 × 5.62 cm |
| 真机抓取帧 119 实际值 | 14.58 cm | 14.69 cm | 9.86 × 12.15 × 6.56 cm |

袋子 12.0 × 8.5 × 6.5 cm。**"闭合"时拇指到手指还有 11.4~12.5cm**，
而真机数据里手指最多只弯到 **33~37°**（关节量程 90°）。

**8 种手势的闭合值**（`o10_dual_reset.json`，可能有用）：

```
cylindrical_straight closed  [..., -0.7, 0, 0.7, 0.7, 0, 0.7, 0, 0.7]        ← 数据集用的就是这个，手指只弯 40°
cylindrical          closed  [..., -0.834, 0, 0.05, 0.057, 0, 1.463, 0, 1.469] ← 无名指/小指弯 84°
fist                 closed  [..., -0.905, 0, 1.48, 1.48, 0, 1.48, 0, 1.48]     ← 四指全弯 85°
```

### 5.2 臂底座 yaw 有未解释的残差 ⚠️

- 位置 `(25,10)cm` 已确认（遮掉真机闲置臂区域后扫 x/y，25,10 最优）。
- **yaw 是任务级拟合值 1.90 rad**（判据：回放时手到袋子的最近距离 / 接触次数）：

  | yaw | 最近距离 | 接触 |
  |---|---|---|
  | 1.3678（原配置） | 0.132 m | 0 |
  | 1.5708（=π/2，"垂直于安装边"的预期值） | 0.067 m | 0 |
  | **1.90** | **0.034 m** | **260** |
  | 2.25 / 2.40 | 0.070 / 0.088 m | 0 |

- **用户指出朝向不该拟合**：底座是垂直于安装边拧上去的，所以 yaw 应由安装几何定死。
  但我从 URDF 算**伸展包络各方位几乎相同**（0.643~0.656m，差 0.2%），
  所以"最够得着的方向"是噪声，**推不出朝向**。→ 这 19° 的差还是悬案。

### 5.3 其他未完成项

- **袋子初始位置的对应关系**：真机首帧里检测器只认出 **2 个**袋子，第 3 个
  （`distractor_right`）仍在用随机位置。如果 episode 抓的是那第 3 个，手就会伸向空处。
- **`<flex>` 手写未做**：要让软体真正有弹性，需要绕过 flexcomp，按 `flex_count`
  生成顶点数组（6×4×3=72）和四面体单元数组（30 格 × 6 = 180），写
  `<flex dim="3" young=... poisson=... damping=... vertex=... element=.../>`。
- **数据采集链路完全没搭**：没有 rollout 循环、没有 LeRobot v3 写出（parquet + 两路 mp4 + meta）、
  没有 episode 终止/成功判据、没有 π0.5 策略接入。
- 真机畸变以外的其他相机效应（渐晕、噪声）未建模。
- `soft_bag_test.py` 里有个笔误：`model.flex_nvert` 应为 `model.flex_vertnum`。

---

## 6. ⚠️ 用户最新补充（很可能改变方向）

> "现实中虽然纸巾袋会有轻微变形，但是变形并不明显，因为**纸巾袋很厚**。"

**含义：变形量很小，所以"软体"未必是正解。** 抓取更可能是**靠摩擦/夹持一个接近刚性的厚块**。
如果这样，重点应该转向：

1. **手的碰撞几何**（我怀疑的最大嫌疑，还没查）：`scene.py` 里手的碰撞来自导入 mesh，
   设了 `contype=2/conaffinity=1/condim=4/group=3`。需要确认这些碰撞体是不是**真实的手指面**，
   还是粗糙的凸包/退化几何 —— 如果接触只发生在指尖的圆角上，物体就会"挤出去"。
2. **接触参数**：`default` 里 `friction="0.8 0.005 0.0001"`、`solref="0.01 1"`、
   `solimp="0.95 0.99 0.001"` 对**手-袋**接触是否合适（condim、滑动/扭转摩擦）。
3. **臂/手 body 全部开了 `gravcomp="1"`**（完全重力补偿，`scene.py` 里注释写"preview control"）。
   这会让臂"失重"，可能使接触预载不真实 —— 值得试关掉对比。
4. **闭合手势是否与真机一致**：确认采集时操作员用的是 `cylindrical_straight`（数据里就是它），
   而不是别的。若真机其实用了 `cylindrical`（84°），那仿真就该用那组值。

---

## 7. 建议的下一步（按性价比排序）

1. **查手的碰撞几何**：把手的每个碰撞 geom 的类型/尺寸/位置打出来，与视觉 mesh 对比；
   或用 MuJoCo 的接触可视化在抓取瞬间看接触点落在手指的哪个部位。
   若接触只在指尖，就需要给指腹加显式的碰撞面（pad）。
2. **用 108 个全成功 episode 做多帧 yaw 复核**（`success_episode/..._all_108eps`）：
   取每个 episode 首帧（用户说每次都从同一位姿出发），遮掉闲置臂区域，联合估计底座 yaw，
   看 1.90 vs π/2 到底哪个对。
3. **确认第 3 个袋子的位置**：改进检测器或人工标注真机首帧的三个袋子位置，
   让仿真里三个位置都有袋子。
4. 上面三步做完再回到抓取测试。**只有抓取能稳定复现，才值得搭数据采集链路。**

---

## 8. 复现命令速查

```bash
ROOT=/workspace/shared/mujoco_tissue_scene
cd $ROOT && export MUJOCO_GL=osmesa
PY=/opt/miniconda3/envs/turbovla-libero/bin/python
CV=/opt/miniconda3/envs/arm-hand-teleop/bin/python
D=/workspace/shared/new_program_qiuzhi/without_tactile/pi05_normal_recovery_merged_214eps

bash status.sh                                   # 全部状态
$PY scene.py --check                             # 构建 + 物理自检
$PY scene.py --render --no-randomize --output-dir outputs/measured   # 渲染（实测袋位）
$PY analyze_lerobot_dataset.py --dataset $D      # 数据集逐维统计
$PY replay_check.py --dataset $D --episodes 0 1 2 --sweep            # 保真度 + 增益扫描
$PY sim_env.py --demo --episode 0                # 环境自检 + 回放
$PY align_with_dataset.py --dataset $D --episode 0 --frame 0 \
      --out outputs/alignment_dataset --place-bags-from-image        # 真机帧 vs 仿真
$PY grasp_isolate2.py                            # 抓取隔离实验（关键诊断）
$PY grip_geometry.py                             # 手部握持几何
$PY arm_reach.py                                 # 臂可达范围
$PY arm_base_controlled.py --stage yaw           # 底座 yaw 任务级搜索
$CV match_appearance.py --real /tmp/frame.png --sim outputs/run/central.png
$PY -m pytest tests -q                           # 16 个单测
# 抓一帧中央相机（设备是 /dev/video12）
$CV -c "import cv2;c=cv2.VideoCapture('/dev/video12',cv2.CAP_V4L2);c.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'));[c.read() for _ in range(20)];ok,f=c.read();cv2.imwrite('/tmp/frame.png',f)"
```

---

## 9. 踩过的坑（省你时间）

1. **`shell` 传参**：远端命令里的 `$`、引号会被 PowerShell 吃掉 —— **一律写成脚本文件再 `pscp` 过去跑**。
2. **PowerShell `-replace` 默认不区分大小写**：我用它把 `align.DATASET` 替换成 `D` 时，
   连带把 `align.dataset_frame` 也改坏了。用 `-creplace`。
3. **PowerShell 变量名大小写不敏感**：循环里的 `$t` 会覆盖目标主机变量 `$T`。
4. **`pscp` 会把中文文件名写坏**（非法 UTF-8 字节）——用 ASCII 名，或用 python 按大小重命名。
5. **`mj_printSchema` 签名是 `(flg_html, flg_pad)`**，不是文档里那种。
6. **MuJoCo flexcomp 只接受 `name/type/count/spacing/pos/dim/radius/mass/rgba`**；
   `young/poisson/damping/friction/contype/solref/solimp` 全部会被拒（属于 `<flex>`）。
   **另外：自由体必须自带 `<inertial>`**，否则报 "mass and inertia of moving bodies must be larger than mjMINVAL"。
7. **`diaginertia` 必须满足三角不等式** `A+B>=C`，否则编译报错。
8. **真机场景里有第二只闲置臂在画面右下**：我曾用"边缘轮廓匹配"判定底座镜像到 95cm，
   结果仿真的臂匹配上了那只**闲置臂**的边缘，把错误答案抬高了 2.7 倍。
   **任何基于整幅图像的匹配都要先把闲置臂区域遮掉**（`arm_base_controlled.py` 里有 `IDLE_ARM_BOX`）。
9. **自证陷阱**：用同一份数据解标定、再用同一份数据反投影验证，永远是"对的"。
   必须换**独立对象**验证（本项目用了白纸 / 绿盒子 / 桌边三个）。
10. **真机 episode 末帧机械臂回到起始位姿**（这是 `pi05_normal_recovery` 的特征），
    所以"末帧和首帧看起来一样"不代表没动过。

---

## 10. 关键配置位置

- `configs/scene.yaml` — **唯一真源**：`table` / `measured_layout` / `tray` / `boxes` /
  `box_randomization` / `lighting` / `materials` / `cameras` / `wrist_camera` / `arm` / `hand` /
  `control` / `env` / `reset` / `calibration_summary`
- `scene.py` — 由配置生成 MJCF；CLI：`--check --render --viewer --seed --no-randomize --target-on-table`
- `action_layout.py` — 16 维 ↔ 仿真关节/执行器的**唯一映射**
- `sim_env.py` — π0.5 兼容环境
- `hand_control.py` — 手部命令（接受手势名或 10 个连续值）
