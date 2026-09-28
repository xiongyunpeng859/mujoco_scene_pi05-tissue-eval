# 三袋连续抓取与外观随机化（2026-09-20）

> 后续更新：旧速度批次已停止，最新改为 1.5 倍速度并逐集实际复位，375 帧/12.5 秒。见 [最新说明](SPEED_AND_HOME_20260920.md)。下文保留初版结果。

## 用户确认的数据语义

- 1000 条 = 1000 次成功的单袋抓放，每条 450 帧，30 fps，15 秒。
- 每轮随机摆放 3 袋，按桌面前后位置依次选择；抓完三袋才重新生成。
- 每袋落盒后，在两条数据之间清空盒内该袋，桌面剩余袋子保持原位。
- 到绿色盒子上方直接张手，张手阶段六个机械臂关节命令固定，不再垂直下放。
- 重置与清盒发生在 episode 之间，不作为训练动作记录。

## 外观与训练建议

每轮固定一种外观，下轮重采样：机械臂颜色、三袋独立颜色、桌面 plain/wood/fabric/stone/stripes 程序纹理和轻微光照变化。约 25% 轮次保留近真机外观。绿色盒子与标定相机固定，物理参数不随外观变化。

多物体场景能提供遮挡及目标选择训练，外观随机化有助于减少对颜色与背景的依赖；不能保证少量真机微调即可达到高成功率。应保留独立真机验证集，混合真机训练数据。当前选择策略为从前往后，不包含语言指定颜色目标的任务。推荐按完整三袋轮次划分训练/验证，避免同轮外观泄漏。

## 已完成验证

- 原直接释放版随机测试 10/10 成功。
- 三袋无渲染验证 6/6；双相机预览采集 6/6。
- 批量采集与合并端到端 smoke：2 轮、6 条、2700 帧；原生 LeRobotDataset/PyAV 读取通过。
- 8 项原有回归测试通过；新增三袋生命周期与外观测试修正坐标刷新后通过。
- 原生读取检查每条数据首末帧、双相机尺寸、动作形状、末尾 padding；视频检查 H264、640×480、30 fps、450 帧。
- 写出真实 LeRobot v3.0，逐集编码释放内存；记录 round、spawn 与 appearance 元数据。
- 批量合并支持中断恢复，恢复时校验源代码哈希与采集配置。

## 视频

`outputs/domain_randomization/video_rounds_v3/dual_camera_two_rounds.mp4`

90 秒，45 秒处切换外观；另有 central_two_rounds.mp4 与 wrist_two_rounds.mp4。

## 正式 1000 条任务

目录：`outputs/tissue_pick_place_dr_1000_20260920/`

2026-09-20 已启动，4 个工作进程；运行状态以 `progress.json` 为准，不把启动当作完成。日志 `collection.log`，最终数据集 `dataset/`。采满 334 个完整成功轮次后选取前 1000 条，自动合并统计并运行训练端验证；仅验证通过才标记 complete。`dataset/VALIDATION.json` 记录最终结果。

```bash
/opt/miniconda3/envs/turbovla-libero/bin/python -u reports/tools/collect_randomized_dataset.py \
  --output outputs/tissue_pick_place_dr_1000_20260920 --episodes 1000 --workers 4 --seed 620260920
```

同命令可以恢复已停止的任务；文件锁防止重复运行。失败轮次保留诊断但不并入数据集。每轮失败重试上限 3 次，总轮次尝试上限 800，磁盘剩余低于 20 GiB 时停止。

最终 `meta/recommended_split.json` 给出按轮划分的建议训练/验证 episode 范围。旧 `outputs/lerobot_sim/` 不可混入这批数据。
