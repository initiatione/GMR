from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from general_motion_retargeting.utils.bvh_profile_adapter import (
    detect_bvh_unit_divisor,
    estimate_height_from_raw_global_positions,
)


def test_estimate_height_from_raw_global_positions_uses_head_and_lowest_foot() -> None:
    positions_by_body = {
        "Head": np.asarray([[0.0, 0.0, 55.0], [0.0, 0.0, 56.0], [0.0, 0.0, 57.0]]),
        "LeftFoot": np.asarray([[0.0, 0.0, 2.0], [0.0, 0.0, 3.0], [0.0, 0.0, 4.0]]),
        "RightFoot": np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]]),
    }

    height = estimate_height_from_raw_global_positions(positions_by_body)

    assert height == 55.8


def test_detect_bvh_unit_divisor_keeps_lafan1_centimeter_scale() -> None:
    assert detect_bvh_unit_divisor(raw_height=155.0, detected_profile="lafan1_official") == 100.0


def test_detect_bvh_unit_divisor_uses_inches_for_small_human_robot_hit_skeletons() -> None:
    assert detect_bvh_unit_divisor(raw_height=57.0, detected_profile="human_robot_hit") == 39.37


def test_detect_bvh_unit_divisor_falls_back_to_centimeters_when_unknown() -> None:
    assert detect_bvh_unit_divisor(raw_height=57.0, detected_profile="unknown") == 100.0
