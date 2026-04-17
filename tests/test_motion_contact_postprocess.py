from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from general_motion_retargeting.motion_contact_postprocess import (
    CONTACT_AWARE_PROFILE_PRESETS,
    ContactAwarePostprocessConfig,
    apply_contact_aware_postprocess,
    build_contact_aware_config,
)


TEST_XML = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="contact_postprocess_test">
  <worldbody>
    <geom name="floor" type="plane" pos="0 0 0" size="0 0 1"/>
    <body name="base" pos="0 0 0">
      <freejoint/>
      <body name="left_foot" pos="0 0.1 0">
        <geom name="left_foot_box" type="box" size="0.1 0.05 0.02" pos="0 0 0" contype="1" conaffinity="1"/>
      </body>
      <body name="right_foot" pos="0 -0.1 0">
        <geom name="right_foot_box" type="box" size="0.1 0.05 0.02" pos="0 0 0" contype="1" conaffinity="1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

TEST_XML_FALLBACK_NAMES = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="contact_postprocess_fallback_test">
  <worldbody>
    <geom name="floor" type="plane" pos="0 0 0" size="0 0 1"/>
    <body name="base" pos="0 0 0">
      <freejoint/>
      <body name="foot_a" pos="0 0.2 0">
        <geom name="foot_geom_a" type="box" size="0.1 0.05 0.02" pos="0 0 0" contype="1" conaffinity="1"/>
      </body>
      <body name="foot_b" pos="0 -0.2 0">
        <geom name="foot_geom_b" type="box" size="0.1 0.05 0.02" pos="0 0 0" contype="1" conaffinity="1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _write_test_xml(tmp_path: Path) -> Path:
    xml_path = tmp_path / "contact_postprocess_test.xml"
    xml_path.write_text(TEST_XML, encoding="utf-8")
    return xml_path


def _write_fallback_test_xml(tmp_path: Path) -> Path:
    xml_path = tmp_path / "contact_postprocess_fallback_test.xml"
    xml_path.write_text(TEST_XML_FALLBACK_NAMES, encoding="utf-8")
    return xml_path


def test_contact_aware_postprocess_locks_stance_xy_and_removes_penetration(tmp_path: Path) -> None:
    xml_path = _write_test_xml(tmp_path)
    motion = {
        "fps": 30,
        "root_pos": np.array(
            [
                [0.000, 0.000, 0.010],
                [0.002, 0.000, 0.012],
                [0.004, 0.000, 0.009],
                [0.006, 0.000, 0.011],
            ],
            dtype=np.float64,
        ),
        "root_rot": np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64), (4, 1)),
        "dof_pos": np.zeros((4, 0), dtype=np.float64),
    }

    processed, stats = apply_contact_aware_postprocess(
        motion_data=motion,
        model_or_path=xml_path,
        config=ContactAwarePostprocessConfig(
            stance_height_threshold=0.03,
            stance_speed_threshold=0.10,
            stance_min_frames=2,
            ground_clearance=0.002,
            ground_mode="per_frame",
            root_z_smoothing_window=1,
        ),
        inplace=False,
    )

    assert np.allclose(processed["root_pos"][:, 0], processed["root_pos"][0, 0])
    assert stats["stance"]["stance_frames_left"] == 4
    assert stats["grounding_after_smoothing"]["after_penetrating_frames"] == 0
    assert stats["grounding_after_smoothing"]["after_min_support_z"] >= 0.002 - 1e-9


def test_contact_aware_postprocess_smooths_root_z_then_regrounds(tmp_path: Path) -> None:
    xml_path = _write_test_xml(tmp_path)
    motion = {
        "fps": 30,
        "root_pos": np.array(
            [
                [0.000, 0.000, 0.010],
                [0.000, 0.000, 0.040],
                [0.000, 0.000, 0.010],
                [0.000, 0.000, 0.040],
                [0.000, 0.000, 0.010],
            ],
            dtype=np.float64,
        ),
        "root_rot": np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64), (5, 1)),
        "dof_pos": np.zeros((5, 0), dtype=np.float64),
    }

    processed, stats = apply_contact_aware_postprocess(
        motion_data=motion,
        model_or_path=xml_path,
        config=ContactAwarePostprocessConfig(
            stance_height_threshold=0.03,
            stance_speed_threshold=0.10,
            stance_min_frames=2,
            ground_clearance=0.002,
            ground_mode="per_frame",
            root_z_smoothing_window=3,
        ),
        inplace=False,
    )

    assert processed["root_pos"].shape == motion["root_pos"].shape
    assert stats["grounding_after_smoothing"]["after_penetrating_frames"] == 0
    assert stats["smoothed_root_z_shift_max"] >= stats["smoothed_root_z_shift_median"]


def test_contact_aware_postprocess_falls_back_to_default_pose_y_when_names_are_ambiguous(tmp_path: Path) -> None:
    xml_path = _write_fallback_test_xml(tmp_path)
    motion = {
        "fps": 30,
        "root_pos": np.array([[0.0, 0.0, 0.01], [0.002, 0.0, 0.012]], dtype=np.float64),
        "root_rot": np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64), (2, 1)),
        "dof_pos": np.zeros((2, 0), dtype=np.float64),
    }

    processed, stats = apply_contact_aware_postprocess(
        motion_data=motion,
        model_or_path=xml_path,
        config=ContactAwarePostprocessConfig(
            stance_height_threshold=0.03,
            stance_speed_threshold=0.10,
            stance_min_frames=1,
            ground_clearance=0.002,
            ground_mode="per_frame",
            root_z_smoothing_window=1,
        ),
        inplace=False,
    )

    assert processed["root_pos"].shape == motion["root_pos"].shape
    assert stats["stance"]["stance_frames_left"] >= 1
    assert stats["stance"]["stance_frames_right"] >= 1


def test_build_contact_aware_config_balanced_matches_current_default_behavior() -> None:
    cfg = build_contact_aware_config(profile="balanced")
    assert cfg == CONTACT_AWARE_PROFILE_PRESETS["balanced"]


def test_build_contact_aware_config_allows_expert_overrides() -> None:
    cfg = build_contact_aware_config(
        profile="conservative",
        ground_mode="per_frame",
        root_z_smoothing_window=9,
    )
    assert cfg.stance_height_threshold == CONTACT_AWARE_PROFILE_PRESETS["conservative"].stance_height_threshold
    assert cfg.ground_mode == "per_frame"
    assert cfg.root_z_smoothing_window == 9
