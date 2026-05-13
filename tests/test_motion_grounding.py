from __future__ import annotations

from pathlib import Path
import sys

import mujoco as mj
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from general_motion_retargeting.motion_grounding import (
    align_motion_root_to_ground,
    compute_support_min_z,
    find_support_geom_ids,
    save_grounding_diagnostics_plot,
)


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

T800_XML = Path(__file__).resolve().parents[1] / "assets" / "t800" / "mujoco" / "t800_full_gmr.xml"


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


def test_t800_support_geoms_match_training_ankle_roll_foot_boxes() -> None:
    model = mj.MjModel.from_xml_path(str(T800_XML))

    support_ids = find_support_geom_ids(model)
    support_bodies = {
        mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom_id])
        for geom_id in support_ids
    }
    support_groups = {int(model.geom_group[geom_id]) for geom_id in support_ids}
    fallback_bodies = {
        mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom_id])
        for geom_id in range(model.ngeom)
        if int(model.geom_group[geom_id]) == 4
    }

    assert support_bodies == {"LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"}
    assert support_groups == {3}
    assert len(support_ids) == 2
    assert {"LINK_ANKLE_PITCH_L", "LINK_ANKLE_PITCH_R"}.issubset(fallback_bodies)
    assert "LINK_ANKLE_PITCH_L" not in support_bodies
    assert "LINK_ANKLE_PITCH_R" not in support_bodies


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


def test_align_motion_root_to_ground_smooth_per_frame_is_gated_and_smoother_than_hard_snap(tmp_path: Path) -> None:
    xml_path = _write_test_xml(tmp_path)
    original_root_z = np.array([0.02, 0.05, 0.022, 0.20, 0.024], dtype=np.float64)
    motion = {
        "fps": 30,
        "root_pos": np.column_stack(
            [
                np.zeros_like(original_root_z),
                np.zeros_like(original_root_z),
                original_root_z,
            ]
        ),
        "root_rot": np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64), (original_root_z.size, 1)),
        "dof_pos": np.zeros((original_root_z.size, 0), dtype=np.float64),
    }

    global_grounded, global_stats = align_motion_root_to_ground(
        motion_data=motion,
        model_or_path=xml_path,
        clearance=0.0,
        mode="global",
        inplace=False,
    )
    smooth_grounded, smooth_stats = align_motion_root_to_ground(
        motion_data=motion,
        model_or_path=xml_path,
        clearance=0.0,
        mode="smooth_per_frame",
        inplace=False,
        smooth_window=3,
        smooth_contact_threshold=0.04,
    )

    applied_shift = smooth_grounded["root_pos"][:, 2] - original_root_z
    exact_signed_shift = 0.0 - (original_root_z - 0.02)
    after_min_z = compute_support_min_z(
        model_or_path=xml_path,
        root_pos=smooth_grounded["root_pos"],
        root_rot=smooth_grounded["root_rot"],
        dof_pos=smooth_grounded["dof_pos"],
    )

    assert np.allclose(global_grounded["root_pos"][:, 2], original_root_z)
    assert smooth_stats["mode"] == "smooth_per_frame"
    assert smooth_stats["smooth_window"] == 3
    assert smooth_stats["smooth_contact_threshold"] == 0.04
    assert smooth_stats["smooth_contact_candidate_frames"] == 4
    assert smooth_stats["after_min_support_z"] >= -1e-9
    assert np.any(applied_shift < 0.0)
    assert np.isclose(applied_shift[3], 0.0)
    assert not np.allclose(applied_shift, applied_shift[0])
    assert np.max(np.abs(np.diff(applied_shift))) < np.max(np.abs(np.diff(exact_signed_shift)))
    assert np.median(after_min_z) < global_stats["after_median_support_z"]


def test_align_motion_root_to_ground_contact_lowfreq_limits_root_z_pumping(tmp_path: Path) -> None:
    xml_path = _write_test_xml(tmp_path)
    original_root_z = np.array([0.010, 0.040, 0.012, 0.038, 0.011, 0.039, 0.013], dtype=np.float64)
    motion = {
        "fps": 30,
        "root_pos": np.column_stack(
            [
                np.zeros_like(original_root_z),
                np.zeros_like(original_root_z),
                original_root_z,
            ]
        ),
        "root_rot": np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64), (original_root_z.size, 1)),
        "dof_pos": np.zeros((original_root_z.size, 0), dtype=np.float64),
    }

    per_frame_grounded, per_frame_stats = align_motion_root_to_ground(
        motion_data=motion,
        model_or_path=xml_path,
        clearance=0.0,
        mode="per_frame",
        inplace=False,
    )
    lowfreq_grounded, lowfreq_stats = align_motion_root_to_ground(
        motion_data=motion,
        model_or_path=xml_path,
        clearance=0.0,
        mode="contact_lowfreq",
        inplace=False,
        smooth_window=5,
        smooth_contact_threshold=0.04,
        max_shift_step=0.006,
    )

    per_frame_shift = per_frame_grounded["root_pos"][:, 2] - original_root_z
    lowfreq_shift = lowfreq_grounded["root_pos"][:, 2] - original_root_z
    after_min_z = compute_support_min_z(
        model_or_path=xml_path,
        root_pos=lowfreq_grounded["root_pos"],
        root_rot=lowfreq_grounded["root_rot"],
        dof_pos=lowfreq_grounded["dof_pos"],
    )

    assert lowfreq_stats["mode"] == "contact_lowfreq"
    assert lowfreq_stats["max_shift_step"] == 0.006
    assert lowfreq_stats["after_penetrating_frames"] == 0
    assert lowfreq_stats["after_min_support_z"] >= -1e-9
    assert np.max(np.abs(np.diff(lowfreq_shift))) <= 0.006 + 1e-9
    assert np.max(np.abs(np.diff(lowfreq_shift))) < np.max(np.abs(np.diff(per_frame_shift)))
    assert lowfreq_stats["applied_shift_max_step"] < per_frame_stats["applied_shift_max_step"]
    assert np.std(lowfreq_shift) < np.std(per_frame_shift)


def test_save_grounding_diagnostics_plot_writes_png(tmp_path: Path) -> None:
    xml_path = _write_test_xml(tmp_path)
    original_root_z = np.array([0.010, 0.040, 0.012, 0.038], dtype=np.float64)
    motion = {
        "fps": 30,
        "root_pos": np.column_stack(
            [
                np.zeros_like(original_root_z),
                np.zeros_like(original_root_z),
                original_root_z,
            ]
        ),
        "root_rot": np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64), (original_root_z.size, 1)),
        "dof_pos": np.zeros((original_root_z.size, 0), dtype=np.float64),
    }

    _, stats = align_motion_root_to_ground(
        motion_data=motion,
        model_or_path=xml_path,
        clearance=0.0,
        mode="contact_lowfreq",
        inplace=False,
        smooth_window=3,
        max_shift_step=0.006,
        return_diagnostics=True,
    )
    diagnostics = stats["diagnostics"]
    plot_path = tmp_path / "grounding_curves.png"

    save_grounding_diagnostics_plot(plot_path, diagnostics, title="test plot")

    assert plot_path.exists()
    assert plot_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
