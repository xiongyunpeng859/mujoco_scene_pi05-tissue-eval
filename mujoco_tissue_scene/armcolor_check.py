#!/usr/bin/env python3
"""Verify the arm colour properly, without a threshold that biases the sample."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/workspace/shared/mujoco_tissue_scene")
sys.path.insert(0, str(ROOT))
import scene                            # noqa: E402


def render(xml, camera="central"):
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        renderer.update_scene(data, camera=camera)
        return cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)


def main() -> int:
    out = ROOT / "outputs/armcolor_check"
    out.mkdir(parents=True, exist_ok=True)
    with_arm = render(scene.build(ROOT / "configs/scene.yaml", out / "a.xml", with_hand=True))
    no_arm = render(scene.build(ROOT / "configs/scene.yaml", out / "b.xml", with_hand=False))
    cv2.imwrite(str(out / "with_arm.png"), with_arm)
    cv2.imwrite(str(out / "no_arm.png"), no_arm)

    difference = np.abs(with_arm.astype(np.float32) - no_arm.astype(np.float32)).max(2)
    print("arm/background difference: >10 %d px, >25 %d px, >60 %d px"
          % ((difference > 10).sum(), (difference > 25).sum(), (difference > 60).sum()))
    for threshold in (5, 10, 25):
        mask = difference > threshold
        if mask.sum():
            print("  mean BGR inside the mask at threshold %-3d : %s   (n=%d)"
                  % (threshold, np.round(with_arm[mask].mean(0), 1).tolist(), mask.sum()))
    print()
    print("  darkest 20%% of the arm pixels  :",
          np.round(np.sort(with_arm[difference > 10].mean(1))[:max(1, (difference > 10).sum() // 5)].mean(), 1))
    print("  brightest 20%% of the arm pixels:",
          np.round(np.sort(with_arm[difference > 10].mean(1))[-max(1, (difference > 10).sum() // 5):].mean(), 1))
    print()
    print("reference: real controlled arm links measured from the 40-frame consensus = BGR ~(79, 78, 73)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
