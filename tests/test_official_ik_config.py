from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from general_motion_retargeting.params import IK_CONFIG_DICT


MANUAL_OFFICIAL_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "general_motion_retargeting"
    / "ik_configs"
    / "bvh_human_robot_hit_to_t800--manual.json"
)


def test_official_human_robot_hit_config_disables_untrusted_non_root_orientations() -> None:
    config = json.loads(IK_CONFIG_DICT["bvh_human_robot_hit"]["t800"].read_text(encoding="utf-8"))
    orientation_disabled_bodies = {
        "Spine2",
        "Head",
        "LeftUpLeg",
        "RightUpLeg",
        "LeftLeg",
        "RightLeg",
        "LeftFootMod",
        "RightFootMod",
        "LeftArm",
        "RightArm",
        "LeftForeArm",
        "RightForeArm",
        "LeftHand",
        "RightHand",
    }

    for table_name in ["ik_match_table1", "ik_match_table2"]:
        for entry in config[table_name].values():
            body_name = entry[0]
            if body_name in orientation_disabled_bodies:
                assert entry[2] == 0, f"{table_name}:{body_name} should be position-driven for official BVH"


def test_official_human_robot_hit_config_uses_calibrated_root_orientation_guide() -> None:
    config = json.loads(IK_CONFIG_DICT["bvh_human_robot_hit"]["t800"].read_text(encoding="utf-8"))

    assert config["ik_match_table1"]["LINK_BASE"][2] > 0
    assert config["ik_match_table2"]["LINK_BASE"][2] > 0


def test_official_human_robot_hit_root_rot_offset_preserves_calibrated_pelvis_frame() -> None:
    config = json.loads(IK_CONFIG_DICT["bvh_human_robot_hit"]["t800"].read_text(encoding="utf-8"))

    assert config["ik_match_table1"]["LINK_BASE"][4] == [1.0, 0.0, 0.0, 0.0]
    assert config["ik_match_table2"]["LINK_BASE"][4] == [1.0, 0.0, 0.0, 0.0]


def test_manual_official_human_robot_hit_config_uses_expected_rot_offsets_for_manual_groups() -> None:
    config = json.loads(MANUAL_OFFICIAL_CONFIG.read_text(encoding="utf-8"))
    z_plus_90 = [0.7071068, 0.0, 0.0, 0.7071068]
    z_plus_90_bodies = {
        "LINK_TORSO_YAW",
        "LINK_HEAD_YAW",
        "LINK_HIP_YAW_L",
        "LINK_KNEE_PITCH_L",
        "LINK_ANKLE_ROLL_L",
        "LINK_HIP_YAW_R",
        "LINK_KNEE_PITCH_R",
        "LINK_ANKLE_ROLL_R",
    }

    for table_name in ["ik_match_table1", "ik_match_table2"]:
        for robot_body in z_plus_90_bodies:
            assert config[table_name][robot_body][4] == z_plus_90


def test_manual_official_human_robot_hit_config_keeps_table_rot_offsets_synchronized() -> None:
    config = json.loads(MANUAL_OFFICIAL_CONFIG.read_text(encoding="utf-8"))

    assert set(config["ik_match_table1"]) == set(config["ik_match_table2"])
    for robot_body in config["ik_match_table1"]:
        assert config["ik_match_table1"][robot_body][4] == config["ik_match_table2"][robot_body][4]
