from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

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
