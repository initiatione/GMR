from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from general_motion_retargeting.motion_retarget_options import (
    calibrate_human_robot_hit_frame,
    resolve_actual_human_height,
)


def _quat_identity() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _sample_official_frame() -> dict:
    quat = _quat_identity()
    return {
        "Hips": [np.array([0.0, 0.0, 0.9], dtype=np.float64), quat.copy()],
        "LeftUpLeg": [np.array([0.1, 0.0, 0.9], dtype=np.float64), quat.copy()],
        "RightUpLeg": [np.array([-0.1, 0.0, 0.9], dtype=np.float64), quat.copy()],
        "LeftHand": [np.array([0.6, 0.0, 1.3], dtype=np.float64), quat.copy()],
    }


def test_resolve_actual_human_height_ignores_loader_height_for_official_profile_by_default() -> None:
    actual_height = resolve_actual_human_height(
        loader_human_height=1.41,
        source_profile="human_robot_hit",
    )

    assert actual_height is None


def test_resolve_actual_human_height_keeps_lafan1_loader_height() -> None:
    actual_height = resolve_actual_human_height(
        loader_human_height=1.75,
        source_profile="lafan1",
    )

    assert actual_height == 1.75


def test_calibrate_human_robot_hit_frame_builds_root_frame_from_hips_and_world_up() -> None:
    frame = _sample_official_frame()
    calibrated = calibrate_human_robot_hit_frame(frame)
    root_rotation = R.from_quat(calibrated["Hips"][1], scalar_first=True)

    # The synthetic official frame has the left hip on +X and right hip on -X.
    # Calibration should make the root frame's local +Y axis point toward the
    # human left side, while +Z stays world-up for MuJoCo/T800.
    assert np.allclose(root_rotation.apply([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    assert np.allclose(root_rotation.apply([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]))


def test_calibrate_human_robot_hit_frame_does_not_mutate_input_frame() -> None:
    frame = _sample_official_frame()
    original_hips_quat = frame["Hips"][1].copy()

    calibrated = calibrate_human_robot_hit_frame(frame)

    assert calibrated is not frame
    assert np.allclose(frame["Hips"][1], original_hips_quat)
    assert not np.allclose(calibrated["Hips"][1], original_hips_quat)
