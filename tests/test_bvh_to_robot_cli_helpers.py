from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bvh_to_robot import build_motion_data_from_qpos_list, build_retargeter, maybe_step_viewer, slice_motion_frames
from general_motion_retargeting.motion_retarget_options import resolve_ik_safety_break
from general_motion_retargeting.retarget_config import iter_retarget_config_entries
from general_motion_retargeting.params import IK_CONFIG_DICT


def test_slice_motion_frames_applies_start_end_and_step() -> None:
    frames = list(range(10))

    selected = slice_motion_frames(frames, frame_start=2, frame_end=9, frame_step=3)

    assert selected == [2, 5, 8]


def test_slice_motion_frames_accepts_open_end() -> None:
    frames = list(range(6))

    selected = slice_motion_frames(frames, frame_start=3, frame_end=None, frame_step=2)

    assert selected == [3, 5]


@pytest.mark.parametrize(
    ("frame_start", "frame_end", "frame_step"),
    [
        (-1, None, 1),
        (0, 10, 1),
        (4, 2, 1),
        (0, None, 0),
    ],
)
def test_slice_motion_frames_rejects_invalid_ranges(
    frame_start: int,
    frame_end: int | None,
    frame_step: int,
) -> None:
    frames = list(range(5))

    with pytest.raises(ValueError):
        slice_motion_frames(frames, frame_start=frame_start, frame_end=frame_end, frame_step=frame_step)


def test_human_robot_hit_t800_uses_promoted_official_ik_config() -> None:
    config_path = IK_CONFIG_DICT["bvh_human_robot_hit"]["t800"]

    assert config_path.name == "bvh_human_robot_hit_to_t800--mild_two_stage.json"


def test_maybe_step_viewer_skips_when_viewer_is_none() -> None:
    maybe_step_viewer(
        viewer=None,
        qpos=[0.0] * 32,
        human_motion_data={},
        rate_limit=False,
    )


def test_bvh_to_robot_script_help_runs_from_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "scripts/bvh_to_robot.py", "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
    assert "--bvh_file" in result.stdout
    assert "--max_iter" in result.stdout
    assert "contact_lowfreq" in result.stdout
    assert "--foot-ground-max-shift-step" in result.stdout
    assert "--foot-ground-plot-path" in result.stdout


def test_ground_robot_motion_script_help_exposes_plot_and_lowfreq_mode() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "scripts/ground_robot_motion.py", "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
    assert "--plot_path" in result.stdout
    assert "--max_shift_step" in result.stdout
    assert "contact_lowfreq" in result.stdout


def test_build_retargeter_forwards_max_iter(monkeypatch) -> None:
    captured_kwargs = {}

    class FakeGMR:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setattr("scripts.bvh_to_robot.GMR", FakeGMR)

    retargeter = build_retargeter(
        source_profile="human_robot_hit",
        robot="t800_transparent",
        actual_human_height=1.6,
        debug_log_path="debug.jsonl",
        debug_log_every_n=5,
        disable_ik_safety_break=True,
        max_iter=20,
    )

    assert isinstance(retargeter, FakeGMR)
    assert captured_kwargs["src_human"] == "bvh_human_robot_hit"
    assert captured_kwargs["tgt_robot"] == "t800_transparent"
    assert captured_kwargs["actual_human_height"] is None
    assert captured_kwargs["debug_log_path"] == "debug.jsonl"
    assert captured_kwargs["debug_log_every_n"] == 5
    assert captured_kwargs["ik_safety_break"] is False
    assert captured_kwargs["max_iter"] == 20


def test_build_motion_data_from_qpos_list_keeps_all_frames() -> None:
    qpos_list = [
        np.arange(32, dtype=np.float64),
        np.arange(32, dtype=np.float64) + 1.0,
        np.arange(32, dtype=np.float64) + 2.0,
    ]

    motion_data = build_motion_data_from_qpos_list(qpos_list, motion_fps=120)

    assert motion_data["fps"] == 120
    assert motion_data["root_pos"].shape == (3, 3)
    assert motion_data["root_rot"].shape == (3, 4)
    assert motion_data["dof_pos"].shape == (3, 25)
    assert motion_data["root_lin_vel"].shape == (3, 3)
    assert motion_data["root_ang_vel"].shape == (3, 3)
    assert motion_data["dof_vel"].shape == (3, 25)
    assert np.allclose(motion_data["root_rot"][0], np.array([4.0, 5.0, 6.0, 3.0]))
    assert np.allclose(motion_data["root_lin_vel"][1], np.array([120.0, 120.0, 120.0]))
    assert np.allclose(motion_data["dof_vel"][1], np.full(25, 120.0))


def test_iter_retarget_config_entries_keeps_zero_weight_offset_carriers() -> None:
    table = {
        "LINK_BASE": [
            "Hips",
            0,
            0,
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    }

    entries = list(iter_retarget_config_entries(table))

    assert entries == [
        (
            "LINK_BASE",
            "Hips",
            0,
            0,
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            False,
        )
    ]


def test_resolve_ik_safety_break_defaults_to_true() -> None:
    assert resolve_ik_safety_break(disable_ik_safety_break=False) is True


def test_resolve_ik_safety_break_can_disable_pre_solve_limit_exception() -> None:
    assert resolve_ik_safety_break(disable_ik_safety_break=True) is False
