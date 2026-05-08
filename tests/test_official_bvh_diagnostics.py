from __future__ import annotations

from pathlib import Path
import sys
import types

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_bvh_source_motion import build_source_motion_report
from scripts.summarize_gmr_debug_log import summarize_debug_records


def _frame(root_z: float, left_foot_z: float, right_foot_z: float) -> dict:
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return {
        "Hips": [np.array([0.0, 0.0, root_z], dtype=np.float64), quat],
        "Spine2": [np.array([0.0, 0.0, root_z + 0.4], dtype=np.float64), quat],
        "Head": [np.array([0.0, 0.0, root_z + 0.8], dtype=np.float64), quat],
        "LeftFootMod": [np.array([0.0, 0.1, left_foot_z], dtype=np.float64), quat],
        "RightFootMod": [np.array([0.0, -0.1, right_foot_z], dtype=np.float64), quat],
    }


def test_build_source_motion_report_summarizes_core_body_heights() -> None:
    report = build_source_motion_report(
        frames=[
            _frame(root_z=0.8, left_foot_z=0.0, right_foot_z=0.02),
            _frame(root_z=0.9, left_foot_z=-0.01, right_foot_z=0.03),
        ],
        fps=120.0,
        source_path="official.bvh",
    )

    assert report["source_path"] == "official.bvh"
    assert report["frame_count"] == 2
    assert report["fps"] == 120.0
    assert report["bodies"]["Hips"]["z_min"] == 0.8
    assert report["bodies"]["LeftFootMod"]["z_min"] == -0.01
    assert report["quaternion_norm_error_max"] == 0.0


def test_summarize_debug_records_ranks_position_and_orientation_errors() -> None:
    records = [
        {
            "final_error1": 2.0,
            "final_error2": 1.5,
            "task_table1": {
                "LeftLeg": {"world_position_error_norm_m": 0.2, "world_orientation_error_deg": 80.0},
                "LeftHand": {"world_position_error_norm_m": 0.8, "world_orientation_error_deg": 10.0},
            },
        },
        {
            "final_error1": 4.0,
            "final_error2": 3.5,
            "task_table1": {
                "LeftLeg": {"world_position_error_norm_m": 0.4, "world_orientation_error_deg": 100.0},
                "LeftHand": {"world_position_error_norm_m": 1.0, "world_orientation_error_deg": 20.0},
            },
        },
    ]

    summary = summarize_debug_records(records)

    assert summary["frames"] == 2
    assert summary["final_error1"]["mean"] == 3.0
    assert summary["top_position_errors"][0]["body"] == "LeftHand"
    assert summary["top_orientation_errors"][0]["body"] == "LeftLeg"


def test_summarize_debug_records_ranks_world_position_axis_errors() -> None:
    records = [
        {
            "task_table1": {
                "LeftArm": {"world_position_error_vector_m": [0.30, -0.02, 0.01]},
                "RightArm": {"world_position_error_vector_m": [0.02, -0.40, 0.03]},
            },
        },
        {
            "task_table1": {
                "LeftArm": {"world_position_error_vector_m": [0.20, -0.01, 0.02]},
                "RightArm": {"world_position_error_vector_m": [0.01, -0.20, 0.04]},
            },
        },
    ]

    summary = summarize_debug_records(records)

    assert summary["top_position_axis_errors"]["x"][0]["body"] == "LeftArm"
    assert summary["top_position_axis_errors"]["y"][0]["body"] == "RightArm"
    assert summary["top_position_axis_errors"]["z"][0]["body"] == "RightArm"


def test_collect_task_debug_info_reports_world_vectors_and_axis_alignment() -> None:
    sys.modules.setdefault("mink", types.SimpleNamespace())
    sys.modules.setdefault("mujoco", types.SimpleNamespace())
    from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting

    class DummyTask:
        def compute_error(self, configuration):
            return np.array([0.1, -0.2, 0.3, 0.01, -0.02, 0.03], dtype=np.float64)

    retargeter = GeneralMotionRetargeting.__new__(GeneralMotionRetargeting)
    retargeter.configuration = object()
    retargeter.get_robot_body_pose = lambda frame_name: (
        np.array([1.25, 1.5, 3.75], dtype=np.float64),
        R.from_euler("z", 90, degrees=True).as_quat(scalar_first=True),
    )

    info = retargeter.collect_task_debug_info(
        body_to_task={"LeftArm": DummyTask()},
        body_to_frame={"LeftArm": "LINK_SHOULDER_YAW_L"},
        latest_targets={
            "LeftArm": {
                "frame_name": "LINK_SHOULDER_YAW_L",
                "target_pos": np.array([1.0, 2.0, 3.0], dtype=np.float64),
                "target_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            }
        },
    )

    left_arm = info["LeftArm"]
    assert left_arm["world_position_error_vector_m"] == [0.25, -0.5, 0.75]
    np.testing.assert_allclose(
        left_arm["axis_alignment_current_in_target"],
        R.from_euler("z", 90, degrees=True).as_matrix(),
        atol=1e-7,
    )
