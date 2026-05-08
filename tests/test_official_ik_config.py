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

MILD_TWO_STAGE_OFFICIAL_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "general_motion_retargeting"
    / "ik_configs"
    / "bvh_human_robot_hit_to_t800--mild_two_stage.json"
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


def test_manual_official_human_robot_hit_config_keeps_table_pos_offsets_synchronized() -> None:
    config = json.loads(MANUAL_OFFICIAL_CONFIG.read_text(encoding="utf-8"))

    assert set(config["ik_match_table1"]) == set(config["ik_match_table2"])
    for robot_body in config["ik_match_table1"]:
        assert config["ik_match_table1"][robot_body][3] == config["ik_match_table2"][robot_body][3]


def test_manual_official_human_robot_hit_config_uses_symmetric_limb_pos_offsets() -> None:
    config = json.loads(MANUAL_OFFICIAL_CONFIG.read_text(encoding="utf-8"))
    symmetric_pairs = [
        ("LINK_HIP_YAW_L", "LINK_HIP_YAW_R"),
        ("LINK_KNEE_PITCH_L", "LINK_KNEE_PITCH_R"),
        ("LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"),
        ("LINK_SHOULDER_YAW_L", "LINK_SHOULDER_YAW_R"),
        ("LINK_ELBOW_PITCH_L", "LINK_ELBOW_PITCH_R"),
        ("LINK_WRIST_END_L", "LINK_WRIST_END_R"),
    ]

    for table_name in ["ik_match_table1", "ik_match_table2"]:
        for left_body, right_body in symmetric_pairs:
            left_offset = config[table_name][left_body][3]
            right_offset = config[table_name][right_body][3]

            assert config[table_name][left_body][1] == config[table_name][right_body][1]
            assert config[table_name][left_body][2] == config[table_name][right_body][2]
            assert left_offset[0] == right_offset[0]
            assert left_offset[1] == -right_offset[1]
            assert left_offset[2] == right_offset[2]


def test_manual_official_human_robot_hit_config_uses_low_weight_head_position_task() -> None:
    config = json.loads(MANUAL_OFFICIAL_CONFIG.read_text(encoding="utf-8"))

    assert config["ik_match_table1"]["LINK_HEAD_YAW"] == [
        "Head",
        0,
        0,
        [0.0, 0.0, 0.05],
        [0.7071068, 0.0, 0.0, 0.7071068],
    ]


def test_manual_official_human_robot_hit_config_uses_foot_orientation_tasks_without_overconstraining() -> None:
    config = json.loads(MANUAL_OFFICIAL_CONFIG.read_text(encoding="utf-8"))

    for table_name in ["ik_match_table1", "ik_match_table2"]:
        assert config[table_name]["LINK_ANKLE_ROLL_L"][1] == 50
        assert config[table_name]["LINK_ANKLE_ROLL_R"][1] == 50
        assert config[table_name]["LINK_ANKLE_ROLL_L"][2] == 5
        assert config[table_name]["LINK_ANKLE_ROLL_R"][2] == 5
        assert config[table_name]["LINK_ANKLE_ROLL_L"][3] == [0.0, 0.0, -0.01]
        assert config[table_name]["LINK_ANKLE_ROLL_R"][3] == [0.0, 0.0, -0.01]
    assert config["ik_match_table2"]["LINK_HEAD_YAW"] == [
        "Head",
        1,
        0,
        [0.0, 0.0, 0.05],
        [0.7071068, 0.0, 0.0, 0.7071068],
    ]


def test_upperbody_core_candidate_alias_uses_dedicated_config_and_transparent_t800_model() -> None:
    from general_motion_retargeting.params import ROBOT_XML_DICT

    expected_configs = {
        "t800_transparent_upperbody_core_candidate": "bvh_human_robot_hit_to_t800--manual_upperbody_core_candidate.json",
    }

    for robot_name, config_name in expected_configs.items():
        candidate_path = IK_CONFIG_DICT["bvh_human_robot_hit"][robot_name]
        assert candidate_path.name == config_name
        assert candidate_path.exists()
        assert ROBOT_XML_DICT[robot_name].name == "t800_full_gmr_transparent.xml"


def test_mild_two_stage_transparent_alias_is_primary_official_visual_route() -> None:
    assert (
        IK_CONFIG_DICT["bvh_human_robot_hit"]["t800_transparent"].name
        == "bvh_human_robot_hit_to_t800--mild_two_stage.json"
    )
    assert IK_CONFIG_DICT["bvh_human_robot_hit"]["t800_transparent"].exists()


def test_mild_two_stage_config_has_runtime_required_metadata() -> None:
    config = json.loads(MILD_TWO_STAGE_OFFICIAL_CONFIG.read_text(encoding="utf-8"))

    assert config["robot_root_name"] == "LINK_BASE"
    assert config["human_root_name"] == "Hips"
    assert config["ground_height"] == 0.0
    assert config["human_height_assumption"] == 1.8
    assert config["use_ik_match_table1"] is True
    assert config["use_ik_match_table2"] is True


def test_mild_two_stage_config_only_changes_weights_from_manual_baseline() -> None:
    manual = json.loads(MANUAL_OFFICIAL_CONFIG.read_text(encoding="utf-8"))
    mild = json.loads(MILD_TWO_STAGE_OFFICIAL_CONFIG.read_text(encoding="utf-8"))

    manual_probe = json.loads(json.dumps(manual))
    for table_name in ["ik_match_table1", "ik_match_table2"]:
        for robot_body in manual_probe[table_name]:
            manual_probe[table_name][robot_body][1] = mild[table_name][robot_body][1]
            manual_probe[table_name][robot_body][2] = mild[table_name][robot_body][2]

    assert manual_probe == mild


def test_manual_transparent_alias_remains_available_as_fallback() -> None:
    assert (
        IK_CONFIG_DICT["bvh_human_robot_hit"]["t800_transparent_manual"].name
        == "bvh_human_robot_hit_to_t800--manual.json"
    )
    assert IK_CONFIG_DICT["bvh_human_robot_hit"]["t800_transparent_manual"].exists()
