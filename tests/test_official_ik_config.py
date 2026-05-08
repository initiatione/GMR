from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from general_motion_retargeting.params import IK_CONFIG_DICT


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
