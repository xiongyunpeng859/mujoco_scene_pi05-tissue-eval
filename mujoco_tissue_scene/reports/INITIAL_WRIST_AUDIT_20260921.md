# 初始腕部朝向核对

## 结论与边界

未发现初始姿态把腕部圆盖翻到下方的证据。使用当前 URDF 场景和真实成功数据 episode 0 首帧关节状态，从两侧、上方和后方渲染，圆盖朝上，手指向前，腕部相机在下方。没有修改关节零位、安装变换或采集规则。

脚本控制器的 home 来自 108 条成功真机数据 episode 0 的 observation.state[0]，不是任意指定姿态。六臂关节角（度）约为 [1.082, 0.076, -0.295, 0.229, 0.120, -0.055]。用户投诉的旧试采 episode 003 首帧约为 [1.154, 0.001, 0.072, 0.290, -0.187, 0.153]，也接近相同零位；其最大起始角差小于 0.37°。

这不证明整个仿真和现实标定正确。真机和试采顶视画面的构图、机械臂在画面中的位置仍有明显不同，不能当作像素对齐对照。圆盖朝向正常也不能解释或否定后续抓取的大幅腕部旋转：此前 episode 003 的 joint5 全程相对起点最大偏转约 101.2°，是另外的运动规划问题。

## 证据

- `outputs/initial_wrist_audit_20260921/wrist_views.jpg`：四视角诊断图，使用真机首帧状态。
- `outputs/initial_wrist_audit_20260921/pilot_003_initial_wrist.jpg`：使用投诉试采首帧状态的上方诊断图（沿用诊断场景，非原视频截图）。
- `outputs/initial_wrist_audit_20260921/recorded_first_frames.jpg`：真机 episode 0 和试采 episode 003 的原始第一帧，相机/构图不同。
- `outputs/initial_wrist_audit_20260921/evidence.json`：关节角和各连杆变换。
- 复现工具 `reports/tools/audit_initial_wrist.py`；使用 turbovla-libero 环境、MUJOCO_GL=egl。

## 按用户要求移除旧批次

确认采集进程已退出后，将旧规则批次 `outputs/tissue_pick_place_visual_dr_planned_500_20260921` 整个移出采集目录至 `/workspace/shared/.dataset_trash/tissue_pick_place_visual_dr_planned_500_20260921`。

原目录已不存在；隔离目录约 413 MB，含 2 轮/6 条完成数据和未完成规划产物，可恢复，非永久擦除。系统 gio 回收站不可用，故采用独立隔离目录。未删除真机数据、早先试采或诊断视频。采集仍暂停。
