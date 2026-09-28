# 1.5 倍动作速度与逐集复位

用户新增要求：动作加快至 1.5 倍，每个 episode 结束必须复位。

实现：保持控制/视频 30 Hz 与物理步长 0.0005 s，缩短各抓放阶段轨迹；抓放从 450 帧/15 秒变为 300 帧/10 秒。这是仿真中实际加速，不是视频播放加速。

每条新增 return_home（60 帧/2 秒）和 home_settle（15 帧/0.5 秒），总计 375 帧/12.5 秒。返回目标为真机示教起始的 16 维状态，连续执行和记录，禁止在 episode 内瞬移复位。到盒子上方直接松手的要求保持不变。

成功验收除抓持、完整落盒、关节跟踪与命令连续性外，还要求复位末 5 帧全部关节误差不超过 0.03 rad，回程/停稳期间机械臂与手不接触外部物体。最后 15 帧须持续落盒成功。原生训练读取也核验末帧动作及观测的复位姿态。

旧速度批次 `outputs/tissue_pick_place_dr_1000_20260920/` 已停止并保留。新批次使用独立目录，避免混入缺少复位的数据。

## 最终验证

初始的简单时间压缩出现了命令峰值超限；改用正弦加减速段加匀速中段，并在反馈 IK 后限制机械臂单帧命令变化为 0.14 rad，原有 0.15 rad 验收上限不变。12 次最终测试只有 3 帧触发限制，总时长不变。

- 最终 12 次随机测试：12/12 成功，4 个完整三袋轮次；最大机械臂跟踪误差 0.483254 rad（上限 0.5），全部 16 维复位误差最大 0.016597 rad（约 0.95 度，上限 0.03）。
- 每条均 375 帧，直接松手阶段机械臂命令变化范围为 0；复位阶段外部接触为 0。
- 最终双相机试采：6/6，2250 帧；每集视频、动作边界 padding、末帧复位动作与观测均通过原生 LeRobotDataset 验证。
- 4 项释放/物理回归测试、1 项三袋生命周期测试通过。
- 采集器仅接受三次首次抓取全部成功的完整轮次；任何失败均排除整轮，避免把盒内重抓等恢复动作当作桌面抓取数据。已检查完整轮次接受、含失败轮次拒绝。

证据：`outputs/speed_1p5_home/final_test12/result.json`、`outputs/speed_1p5_home/final_preview/dataset/VALIDATION.json`。

视频：`outputs/speed_1p5_home/final_preview/dual_camera_two_rounds.mp4`，75 秒、1280×480、30 fps；37.5 秒处切换下一轮外观，每 12.5 秒完成一次含复位的抓放。

## 正式采集

新任务已启动：`outputs/tissue_pick_place_dr_1000_speed1p5_home_20260920/`。实时状态见 `progress.json`，日志 `collection.log`；最终 `dataset/` 必须在状态 complete 且 `VALIDATION.json` passed 为 true 后作为完整 1000 条数据集使用。预期 375000 帧，约 208 分钟仿真视频（每个相机）。

```bash
/opt/miniconda3/envs/turbovla-libero/bin/python -u reports/tools/collect_randomized_dataset.py \
  --output outputs/tissue_pick_place_dr_1000_speed1p5_home_20260920 \
  --episodes 1000 --workers 4 --seed 620260920 --speed 1.5
```

完整轮次三条之间保留桌面袋子；每条物理复位完成后，才清除箱内目标。外观仍只在新一轮采样。训练数据每集 `meta/collection/episode-*.json` 记录 speed_multiplier、return_home、home_state、home_error_rad。
