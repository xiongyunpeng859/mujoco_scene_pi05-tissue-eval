# 包装视觉与接触变化小样

输出：`outputs/wrapper_contact_preview_20260920/`。仅预览，未修改已有 1000 条数据，未启动采集/训练。

`comparison.mp4` 是四组顶视同步对照，`comparison.jpg` 是关键帧，各组 `dual.mp4` 为顶视与腕视。视频由每 12 个物理控制帧抽取的预览图生成，2.5 fps，非完整 30 fps 训练视频。控制仍为 30 Hz，speed=1.5，375 帧并真实复位。

| 组 | 摩擦第一分量 | Young 模量 | 质量 | 本次结果 |
|---|---:|---:|---:|---|
| baseline | 4.0 | 100 kPa | 50 g | 成功 |
| printed | 4.0 | 100 kPa | 50 g | 成功，与基线物理指标一致 |
| slippery | 1.2 | 100 kPa | 50 g | 失败，未持续抓持且未满足完整落盒 |
| soft_heavy | 4.0 | 50 kPa | 80 g | 持续抓持，但未满足完整落盒 |

均为相同 seed=7、相同初始位置和控制策略；反馈控制允许后续动作随物理状态变化，因此不是逐帧固定动作回放。仅各 1 次，不代表成功率。参数是探索性压力测试，未经过真机材料测量。soft_heavy 同时改变两个参数，不能独立归因。

视觉：程序化印刷面板、文字、条码样式、封边和褶皱明暗，配合材质高光；直接贴在可形变 flex 表面，非后期视频覆盖。褶皱纹理只是视觉明暗，不是薄膜几何；不能称为照片级真实包装。可选 `boxes[].wrapper_visual`，默认场景不启用，不改变碰撞或质量。物理参数通过现有质量/弹性/接触字段真正参与仿真。

复现（输出目录必须不存在）：

```bash
MUJOCO_GL=osmesa /opt/miniconda3/envs/turbovla-libero/bin/python reports/tools/preview_wrapper_contact.py --output outputs/new_wrapper_contact_preview
```

失败样例只作观察，不能混入 success-only 数据。先确认视觉方向，再单独标定摩擦、质量和弹性范围，并增加多种种子验证。
