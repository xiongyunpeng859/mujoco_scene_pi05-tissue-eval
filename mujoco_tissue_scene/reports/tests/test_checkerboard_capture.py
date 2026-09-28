from collections import deque
import unittest

import cv2
import numpy as np

from capture_left_wrist_checkerboard import (
    assess_detection,
    detect_checkerboard,
    repeated_arm_pose_rejection,
    robot_pose_or_rejection,
)


class CheckerboardCaptureTests(unittest.TestCase):
    def test_camera_only_save_requires_explicit_diagnostic_override(self):
        pose, rejection = robot_pose_or_rejection(
            None, allow_camera_only_save=False
        )
        self.assertIsNone(pose)
        self.assertIn("robot pose unavailable", rejection)

        pose, rejection = robot_pose_or_rejection(
            None, allow_camera_only_save=True
        )
        self.assertIsNone(rejection)
        self.assertIsNone(pose["arm_joint_positions_rad"])

    def test_repeated_arm_pose_is_rejected(self):
        records = [{
            "sample_index": 1,
            "arm_joint_positions_rad": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        }]
        repeated = {
            "arm_joint_positions_rad": [0.01, 0.1, 0.2, 0.3, 0.4, 0.5],
        }
        changed = {
            "arm_joint_positions_rad": [0.06, 0.1, 0.2, 0.3, 0.4, 0.5],
        }
        rejection = repeated_arm_pose_rejection(
            repeated, records, min_joint_change_rad=0.05
        )
        self.assertIn("arm pose repeats sample 001", rejection)
        self.assertIsNone(repeated_arm_pose_rejection(
            changed, records, min_joint_change_rad=0.05
        ))

    def test_complete_board_detection_and_quality(self):
        columns, rows = 7, 8
        square = 40
        board = np.full(((rows + 1) * square, (columns + 1) * square), 255, np.uint8)
        for row in range(rows + 1):
            for column in range(columns + 1):
                if (row + column) % 2 == 0:
                    cv2.rectangle(
                        board,
                        (column * square, row * square),
                        ((column + 1) * square, (row + 1) * square),
                        0,
                        -1,
                    )
        image = cv2.copyMakeBorder(board, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=180)
        found, corners = detect_checkerboard(image, (columns, rows))
        self.assertTrue(found)
        history = deque([corners.copy() for _ in range(4)], maxlen=5)
        quality = assess_detection(
            image, corners, history,
            sharpness_min=10.0, margin_min=10.0,
            stability_max=1.0, stability_frames=5,
        )
        self.assertTrue(quality.accepted)
        self.assertGreater(quality.coverage, 0.2)

    def test_partial_or_blank_image_is_rejected(self):
        image = np.full((480, 640), 127, np.uint8)
        found, corners = detect_checkerboard(image, (7, 8))
        self.assertFalse(found)
        quality = assess_detection(
            image, corners, deque(maxlen=5),
            sharpness_min=10.0, margin_min=10.0,
            stability_max=1.0, stability_frames=5,
        )
        self.assertFalse(quality.accepted)
        self.assertTrue(
            any("complete 7x8 pattern not found" in reason for reason in quality.reasons)
        )

    def test_low_contrast_board_uses_robust_detection(self):
        columns, rows = 7, 8
        square = 32
        board = np.full(((rows + 1) * square, (columns + 1) * square), 145, np.uint8)
        for row in range(rows + 1):
            for column in range(columns + 1):
                if (row + column) % 2 == 0:
                    cv2.rectangle(
                        board,
                        (column * square, row * square),
                        ((column + 1) * square, (row + 1) * square),
                        105,
                        -1,
                    )
        board = cv2.GaussianBlur(board, (5, 5), 1.2)
        image = cv2.copyMakeBorder(board, 30, 30, 40, 40, cv2.BORDER_CONSTANT, value=130)
        found, corners = detect_checkerboard(image, (columns, rows))
        self.assertTrue(found)
        self.assertEqual(corners.shape, (columns * rows, 2))


if __name__ == "__main__":
    unittest.main()
