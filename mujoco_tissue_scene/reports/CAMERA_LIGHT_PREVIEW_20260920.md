# 相机与光照预览

独立输出 `outputs/camera_light_preview_20260920/`，不修改原始数据或标定配置。
固定位置/控制种子 7，单袋抓放，接触参数保持基准，实际控制 30 Hz、speed=1.5、逐集复位。

四组：baseline（原始）、camera_only（仅相机）、warm_dim（仅暖暗光）、combined_cool（相机＋冷亮光＋印刷包装）。

- 顶视相机世界坐标偏移 `[15,-10,10]` mm，绕世界轴 Euler XYZ `[1.5,-2,1]` 度旋转；fx/fy ×1.03，cx +4 px，cy -3 px。
- 腕视相机相对标定光学坐标偏移 `[3,-2,2]` mm，局部 Euler XYZ `[1,-1,.5]` 度旋转；fx/fy ×.98，同样主点偏移。使用可选 render_perturbation，仅移动渲染光学坐标，不移动碰撞、手或支架；仍继承腕部运动。
- 暖暗光：ambient/diffuse/specular ×0.75×RGB `[1.10,1,.86]`；冷亮光 ×1.12×`[.90,.96,1.07]`，裁到 [0,1]。光源位置偏移 `[.15,-.12,0]` m，方向 `[.10,.15,-1]`。保持原始无影模式，本批没有加入阴影、模糊和噪声。
- 每集内参数固定。畸变系数未增加扰动，沿用现有渲染畸变管线；内参通过配置同时进入渲染和畸变映射。

compile_check 阶段逐组检查 body_mass、body_inertia、qpos0、flex_friction、flex_stiffness、actuator_gainprm 与基线严格一致。包装材料回归测试通过。结果见 summary.json，各仅一次，非成功率统计。

`comparison.mp4` 为四组顶视对照，各组 `dual.mp4` 为顶视/腕视；`comparison.jpg` 为关键帧。每 3 帧抽样，10 fps，末帧额外保留，视频约 12.6 秒；这是预览，不是训练视频。

复现：

```bash
MUJOCO_GL=osmesa /opt/miniconda3/envs/turbovla-libero/bin/python reports/tools/preview_camera_light.py --output outputs/new_camera_light_preview
```
