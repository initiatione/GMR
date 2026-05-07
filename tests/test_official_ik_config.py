from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from general_motion_retargeting.params import IK_CONFIG_DICT


def test_official_human_robot_hit_config_disables_untrusted_lower_body_orientations() -> None:
    config = json.loads(IK_CONFIG_DICT["bvh_human_robot_hit"]["t800"].read_text(encoding="utf-8"))
    orientation_disabled_bodies = {
        "Hips",
        "Spine2",
        "Head",
        "LeftUpLeg",
        "RightUpLeg",
        "LeftLeg",
        "RightLeg",
        "LeftFootMod",
        "RightFootMod",
    }

    for table_name in ["ik_match_table1", "ik_match_table2"]:
        for entry in config[table_name].values():
            body_name = entry[0]
            if body_name in orientation_disabled_bodies:
                assert entry[2] == 0, f"{table_name}:{body_name} should be position-driven for official BVH"


def test_official_human_robot_hit_config_keeps_shoulder_orientation_guides() -> None:
    config = json.loads(IK_CONFIG_DICT["bvh_human_robot_hit"]["t800"].read_text(encoding="utf-8"))
    shoulder_entries = [
        entry
        for table in [config["ik_match_table1"], config["ik_match_table2"]]
        for entry in table.values()
        if entry[0] in {"LeftArm", "RightArm"}
    ]

    assert shoulder_entries
    assert all(entry[2] > 0 for entry in shoulder_entries)
