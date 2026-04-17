from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from general_motion_retargeting.motion_grounding import align_motion_root_to_ground, compute_support_min_z


TEST_XML = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="grounding_test">
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


def _write_test_xml(tmp_path: Path) -> Path:
    xml_path = tmp_path / "grounding_test.xml"
    xml_path.write_text(TEST_XML, encoding="utf-8")
    return xml_path


def test_compute_support_min_z_reads_box_bottom(tmp_path: Path) -> None:
    xml_path = _write_test_xml(tmp_path)
    min_z = compute_support_min_z(
        model_or_path=xml_path,
        root_pos=np.array([[0.0, 0.0, 0.01]], dtype=np.float64),
        root_rot=np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
        dof_pos=np.zeros((1, 0), dtype=np.float64),
    )
    assert min_z.shape == (1,)
    assert np.isclose(min_z[0], -0.01)


def test_align_motion_root_to_ground_per_frame_removes_penetration(tmp_path: Path) -> None:
    xml_path = _write_test_xml(tmp_path)
    motion = {
        "fps": 30,
        "root_pos": np.array([[0.0, 0.0, 0.01], [0.0, 0.0, 0.03]], dtype=np.float64),
        "root_rot": np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
        "dof_pos": np.zeros((2, 0), dtype=np.float64),
    }

    grounded, stats = align_motion_root_to_ground(
        motion_data=motion,
        model_or_path=xml_path,
        clearance=0.002,
        mode="per_frame",
        inplace=False,
    )

    assert np.isclose(grounded["root_pos"][0, 2], 0.022)
    assert np.isclose(grounded["root_pos"][1, 2], 0.03)
    assert stats["after_penetrating_frames"] == 0
    assert stats["after_min_support_z"] >= 0.002 - 1e-9


def test_align_motion_root_to_ground_global_uses_single_shift(tmp_path: Path) -> None:
    xml_path = _write_test_xml(tmp_path)
    motion = {
        "fps": 30,
        "root_pos": np.array([[0.0, 0.0, 0.01], [0.0, 0.0, 0.03]], dtype=np.float64),
        "root_rot": np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
        "dof_pos": np.zeros((2, 0), dtype=np.float64),
    }

    grounded, stats = align_motion_root_to_ground(
        motion_data=motion,
        model_or_path=xml_path,
        clearance=0.002,
        mode="global",
        inplace=False,
    )

    assert np.allclose(grounded["root_pos"][:, 2], np.array([0.022, 0.042]))
    assert stats["applied_shift_max"] == stats["applied_shift_median"]
