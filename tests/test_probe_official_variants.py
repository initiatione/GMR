from __future__ import annotations

import copy
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.probe_official_frame_scale_variants import (
    build_variant_config,
    prepare_official_probe_frames,
)


def _sample_config() -> dict:
    return {
        "ik_match_table1": {
            "LINK_BASE": ["Hips", 0, 10, [0.0, 0.0, 0.08], [1.0, 0.0, 0.0, 0.0]],
            "LINK_WRIST_END_L": ["LeftHand", 0, 0, [0.0, 0.0, -0.1], [1.0, 0.0, 0.0, 0.0]],
            "LINK_WRIST_END_R": ["RightHand", 0, 0, [0.0, 0.0, -0.1], [1.0, 0.0, 0.0, 0.0]],
            "LINK_ELBOW_PITCH_L": ["LeftForeArm", 0, 0, [0.0, 0.03, 0.0], [1.0, 0.0, 0.0, 0.0]],
        },
        "ik_match_table2": {
            "LINK_BASE": ["Hips", 100, 5, [0.0, 0.0, 0.08], [1.0, 0.0, 0.0, 0.0]],
            "LINK_WRIST_END_L": ["LeftHand", 10, 0, [0.0, 0.0, -0.1], [1.0, 0.0, 0.0, 0.0]],
            "LINK_WRIST_END_R": ["RightHand", 10, 0, [0.0, 0.0, -0.1], [1.0, 0.0, 0.0, 0.0]],
            "LINK_ELBOW_PITCH_L": ["LeftForeArm", 10, 0, [0.0, 0.03, 0.0], [1.0, 0.0, 0.0, 0.0]],
        },
    }


def test_prepare_official_probe_frames_matches_cli_profile_preprocessing(monkeypatch) -> None:
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    raw_frames = [
        {
            "Hips": [np.array([0.0, 0.0, 0.9], dtype=np.float64), quat.copy()],
            "LeftUpLeg": [np.array([0.1, 0.0, 0.9], dtype=np.float64), quat.copy()],
            "RightUpLeg": [np.array([-0.1, 0.0, 0.9], dtype=np.float64), quat.copy()],
        }
    ]

    def fake_loader(bvh_file: str, format: str) -> tuple[list[dict], float]:
        assert bvh_file == "official.bvh"
        assert format == "lafan1"
        return copy.deepcopy(raw_frames), 1.41

    monkeypatch.setattr(
        "scripts.probe_official_frame_scale_variants.load_bvh_file",
        fake_loader,
    )

    frames, actual_human_height, loader_height = prepare_official_probe_frames(
        Path("official.bvh"),
        frame_count=1,
    )

    root_rotation = R.from_quat(frames[0]["Hips"][1], scalar_first=True)
    assert actual_human_height is None
    assert loader_height == 1.41
    assert np.allclose(root_rotation.apply([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    assert np.allclose(root_rotation.apply([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]))


def test_upper_body_zero_offset_variant_clears_official_arm_and_hand_offsets() -> None:
    config = build_variant_config(_sample_config(), "no_height_zero_upper_offsets")

    for table_name in ["ik_match_table1", "ik_match_table2"]:
        table = config[table_name]
        assert table["LINK_WRIST_END_L"][3] == [0.0, 0.0, 0.0]
        assert table["LINK_WRIST_END_R"][3] == [0.0, 0.0, 0.0]
        assert table["LINK_ELBOW_PITCH_L"][3] == [0.0, 0.0, 0.0]
        assert table["LINK_BASE"][3] == [0.0, 0.0, 0.08]


def test_arm_chain_priority_variant_prefers_arm_chain_over_wrist_endpoints() -> None:
    config = build_variant_config(_sample_config(), "no_height_arm_chain_priority")

    assert config["ik_match_table2"]["LINK_ELBOW_PITCH_L"][1] == 20
    assert config["ik_match_table2"]["LINK_WRIST_END_L"][1] == 3
    assert config["ik_match_table2"]["LINK_BASE"][1] == 100
