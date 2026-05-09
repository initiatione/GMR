from __future__ import annotations

from pathlib import Path
import pickle
import subprocess
import sys
import textwrap

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from general_motion_retargeting.motion_quality import (
    MotionQualityConfig,
    RepairSpec,
    apply_repair_specs,
    audit_motion_quality,
    build_motion_data,
    build_review_commands,
)


def _basic_motion(dof_pos: np.ndarray, fps: int = 10) -> dict:
    frame_count = dof_pos.shape[0]
    dof_vel = (
        np.gradient(np.asarray(dof_pos, dtype=np.float64), 1.0 / fps, axis=0)
        if frame_count >= 2
        else np.zeros_like(dof_pos, dtype=np.float64)
    )
    return {
        "fps": fps,
        "root_pos": np.column_stack(
            [
                np.linspace(0.0, 0.1, frame_count),
                np.zeros(frame_count),
                np.full(frame_count, 0.8),
            ]
        ),
        "root_rot": np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (frame_count, 1)),
        "dof_pos": np.asarray(dof_pos, dtype=np.float64),
        "dof_vel": dof_vel,
        "root_lin_vel": np.zeros((frame_count, 3), dtype=np.float64),
        "root_ang_vel": np.zeros((frame_count, 3), dtype=np.float64),
        "local_body_pos": None,
        "link_body_list": None,
    }


def test_audit_motion_quality_reports_jump_velocity_limits_and_review_windows() -> None:
    dof_pos = np.zeros((6, 2), dtype=np.float64)
    dof_pos[:, 0] = [0.0, 0.1, 0.85, 0.2, 0.3, 0.4]
    dof_pos[:, 1] = [0.0, 0.49, 0.50, 0.49, 0.0, 0.0]
    motion = _basic_motion(dof_pos, fps=10)

    report = audit_motion_quality(
        motion,
        config=MotionQualityConfig(
            robot="t800",
            joint_names=["hip", "knee"],
            joint_lower_limits=np.array([-1.0, -0.5]),
            joint_upper_limits=np.array([1.0, 0.5]),
            jump_threshold_rad=0.6,
            velocity_threshold_rad_s=5.0,
            limit_margin_rad=0.02,
            review_padding_frames=2,
        ),
    )

    assert report["schema"]["frame_count"] == 6
    assert report["summary"]["qpos_jump_count"] >= 2
    assert report["summary"]["velocity_spike_count"] >= 1
    assert report["summary"]["limit_pressure_count"] == 3
    first_jump = report["issues"]["qpos_jumps"][0]
    assert first_jump["joint_name"] == "hip"
    assert first_jump["frame"] == 2
    assert first_jump["time_sec"] == pytest.approx(0.2)
    assert first_jump["review_window"] == {"start_frame": 0, "end_frame": 5, "start_sec": 0.0, "end_sec": 0.5}


def test_audit_motion_quality_rejects_too_few_frames() -> None:
    motion = _basic_motion(np.zeros((1, 1), dtype=np.float64), fps=10)

    with pytest.raises(ValueError, match="at least 2 frames"):
        audit_motion_quality(motion)


def test_build_review_commands_points_to_vis_robot_motion_with_frame_window() -> None:
    report = {
        "motion_path": "motions/sample.pkl",
        "robot": "t800",
        "review_windows": [
            {"start_frame": 10, "end_frame": 20, "start_sec": 1.0, "end_sec": 2.0, "reasons": ["qpos_jump"]}
        ],
    }

    commands = build_review_commands(report)

    assert commands == [
        "python scripts/vis_robot_motion.py --robot t800 --robot_motion_path motions/sample.pkl "
        "--frame_start 10 --frame_end 20"
    ]


def test_build_review_commands_quotes_paths_with_spaces() -> None:
    report = {
        "motion_path": r"motions\space path\sample motion.pkl",
        "robot": "t800",
        "review_windows": [
            {"start_frame": 10, "end_frame": 20, "start_sec": 1.0, "end_sec": 2.0, "reasons": ["qpos_jump"]}
        ],
    }

    command = build_review_commands(report)[0]

    assert '"motions\\space path\\sample motion.pkl"' in command


def test_audit_review_windows_do_not_expand_to_every_limit_pressure_frame() -> None:
    dof_pos = np.zeros((100, 1), dtype=np.float64)
    dof_pos[10:90, 0] = 0.5
    motion = _basic_motion(dof_pos, fps=10)

    report = audit_motion_quality(
        motion,
        config=MotionQualityConfig(
            robot="t800",
            joint_names=["knee"],
            joint_lower_limits=np.array([-0.5]),
            joint_upper_limits=np.array([0.5]),
            jump_threshold_rad=1.0,
            velocity_threshold_rad_s=100.0,
            limit_margin_rad=0.01,
            review_padding_frames=2,
        ),
    )

    assert report["summary"]["limit_pressure_count"] == 80
    assert report["review_windows"]
    assert report["review_windows"][0]["end_frame"] < 90


def test_audit_candidate_collisions_include_body_names(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    model_path = tmp_path / "collision_test.xml"
    model_path.write_text(
        textwrap.dedent(
                """
                <mujoco>
                  <worldbody>
                    <body name="root" pos="0 0 0">
                      <freejoint/>
                      <body name="base" pos="0 0 0">
                        <geom name="base_geom" type="sphere" size="0.2" contype="1" conaffinity="1"/>
                      </body>
                      <body name="arm" pos="0.3 0 0">
                        <inertial pos="0 0 0" mass="0.1" diaginertia="0.001 0.001 0.001"/>
                        <joint name="hinge" type="hinge" axis="0 0 1" range="-1 1"/>
                      </body>
                      <body name="target" pos="0.1 0 0">
                        <geom name="target_geom" type="sphere" size="0.2" contype="1" conaffinity="1"/>
                      </body>
                    </body>
                  </worldbody>
              <actuator>
                <motor joint="hinge"/>
              </actuator>
            </mujoco>
            """
        ).strip(),
        encoding="utf-8",
    )
    motion = _basic_motion(np.zeros((2, 1), dtype=np.float64), fps=10)

    report = audit_motion_quality(
        motion,
        config=MotionQualityConfig(
            robot="unit",
            joint_names=["hinge"],
            model_path=model_path,
            collision_stride=1,
            max_collision_pairs=1,
        ),
    )

    issue = report["issues"]["candidate_collisions"][0]
    assert issue["geom_a"] == "base_geom"
    assert issue["geom_b"] == "target_geom"
    assert issue["body_a"] == "base"
    assert issue["body_b"] == "target"


def test_apply_repair_specs_interpolates_bounded_joint_window_and_reports_metrics() -> None:
    dof_pos = np.zeros((5, 2), dtype=np.float64)
    dof_pos[:, 0] = [0.0, 0.1, 3.0, 0.3, 0.4]
    motion = _basic_motion(dof_pos, fps=10)
    motion["local_body_pos"] = np.ones((5, 2, 3), dtype=np.float64)
    motion["link_body_list"] = ["hip", "knee"]

    repaired, report = apply_repair_specs(
        motion,
        [
            RepairSpec(
                frame_start=1,
                frame_end=4,
                joint_names=["hip"],
                method="linear_interpolate",
                rationale="single-frame IK branch jump",
            )
        ],
        joint_names=["hip", "knee"],
    )

    assert np.allclose(repaired["dof_pos"][:, 0], [0.0, 0.1, 0.2, 0.3, 0.4])
    assert repaired["local_body_pos"] is None
    assert repaired["link_body_list"] is None
    assert repaired["derived_fields_invalidated"] is True
    assert report["repairs"][0]["before_max_abs_delta_rad"] == pytest.approx(2.9)
    assert report["repairs"][0]["after_max_abs_delta_rad"] == pytest.approx(0.1)
    assert report["diagnostic_only_final_npz_smoothing"] is False


def test_apply_repair_specs_rejects_excessive_correction() -> None:
    dof_pos = np.zeros((5, 1), dtype=np.float64)
    dof_pos[:, 0] = [0.0, 0.1, 10.0, 0.3, 0.4]
    motion = _basic_motion(dof_pos, fps=10)

    with pytest.raises(ValueError, match="exceeds max_correction_rad"):
        apply_repair_specs(
            motion,
            [
                RepairSpec(
                    frame_start=1,
                    frame_end=4,
                    joint_names=["hip"],
                    method="linear_interpolate",
                    rationale="too large to repair automatically",
                    max_correction_rad=1.0,
                )
            ],
            joint_names=["hip"],
        )


def test_apply_repair_specs_rejects_degraded_continuity() -> None:
    dof_pos = np.zeros((5, 1), dtype=np.float64)
    dof_pos[:, 0] = [0.0, 0.1, 0.2, 0.3, 10.0]
    motion = _basic_motion(dof_pos, fps=10)

    with pytest.raises(ValueError, match="degrades continuity"):
        apply_repair_specs(
            motion,
            [
                RepairSpec(
                    frame_start=1,
                    frame_end=4,
                    joint_names=["hip"],
                    method="linear_interpolate",
                    rationale="bad anchor would create a bigger post-window jump",
                    max_correction_rad=1.0,
                )
            ],
            joint_names=["hip"],
        )


def test_audit_robot_motion_help_runs_from_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "scripts/audit_robot_motion.py", "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
    assert "--robot" in result.stdout
    assert "--input" in result.stdout


def test_repair_robot_motion_cli_applies_json_spec(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    motion_path = tmp_path / "source.pkl"
    output_path = tmp_path / "repaired.pkl"
    spec_path = tmp_path / "repairs.json"
    motion = _basic_motion(np.array([[0.0], [0.1], [2.0], [0.3], [0.4]], dtype=np.float64), fps=10)
    motion_path.write_bytes(pickle.dumps(motion))
    spec_path.write_text(
        """
{
  "repairs": [
    {
      "frame_start": 1,
      "frame_end": 4,
      "joint_names": ["J00"],
      "method": "linear_interpolate",
      "rationale": "test repair"
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/repair_robot_motion.py",
            "--input",
            str(motion_path),
            "--output",
            str(output_path),
            "--repair_spec",
            str(spec_path),
            "--joint_names",
            "J00",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
    repaired = pickle.loads(output_path.read_bytes())
    assert np.allclose(repaired["dof_pos"][:, 0], [0.0, 0.1, 0.2, 0.3, 0.4])


def test_repair_robot_motion_cli_refuses_existing_output_without_overwrite(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    motion_path = tmp_path / "source.pkl"
    output_path = tmp_path / "repaired.pkl"
    spec_path = tmp_path / "repairs.json"
    motion = _basic_motion(np.array([[0.0], [0.1], [2.0], [0.3], [0.4]], dtype=np.float64), fps=10)
    motion_path.write_bytes(pickle.dumps(motion))
    output_path.write_bytes(b"existing")
    spec_path.write_text(
        '{"repairs":[{"frame_start":1,"frame_end":4,"joint_names":["J00"],'
        '"method":"linear_interpolate","rationale":"test"}]}',
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/repair_robot_motion.py",
            "--input",
            str(motion_path),
            "--output",
            str(output_path),
            "--repair_spec",
            str(spec_path),
            "--joint_names",
            "J00",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        timeout=15,
    )

    assert result.returncode != 0
    assert "Output file already exists" in result.stderr
    assert output_path.read_bytes() == b"existing"


def test_repair_robot_motion_cli_overwrite_replaces_existing_output(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    motion_path = tmp_path / "source.pkl"
    output_path = tmp_path / "repaired.pkl"
    spec_path = tmp_path / "repairs.json"
    motion = _basic_motion(np.array([[0.0], [0.1], [2.0], [0.3], [0.4]], dtype=np.float64), fps=10)
    motion_path.write_bytes(pickle.dumps(motion))
    output_path.write_bytes(b"existing")
    spec_path.write_text(
        '{"repairs":[{"frame_start":1,"frame_end":4,"joint_names":["J00"],'
        '"method":"linear_interpolate","rationale":"test"}]}',
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/repair_robot_motion.py",
            "--input",
            str(motion_path),
            "--output",
            str(output_path),
            "--repair_spec",
            str(spec_path),
            "--joint_names",
            "J00",
            "--overwrite",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
    repaired = pickle.loads(output_path.read_bytes())
    assert np.allclose(repaired["dof_pos"][:, 0], [0.0, 0.1, 0.2, 0.3, 0.4])


def test_repair_robot_motion_cli_rejects_empty_joint_repair(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    motion_path = tmp_path / "source.pkl"
    output_path = tmp_path / "repaired.pkl"
    spec_path = tmp_path / "repairs.json"
    motion = _basic_motion(np.array([[0.0], [0.1], [2.0], [0.3], [0.4]], dtype=np.float64), fps=10)
    motion_path.write_bytes(pickle.dumps(motion))
    spec_path.write_text(
        '{"repairs":[{"frame_start":1,"frame_end":4,"joint_names":[],'
        '"method":"linear_interpolate","rationale":"bad"}]}',
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/repair_robot_motion.py",
            "--input",
            str(motion_path),
            "--output",
            str(output_path),
            "--repair_spec",
            str(spec_path),
            "--joint_names",
            "J00",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        timeout=15,
    )

    assert result.returncode != 0
    assert "at least one joint name" in result.stderr
    assert not output_path.exists()


def test_repair_robot_motion_cli_refuses_to_overwrite_input(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    motion_path = tmp_path / "source.pkl"
    spec_path = tmp_path / "repairs.json"
    motion = _basic_motion(np.array([[0.0], [0.1], [2.0], [0.3], [0.4]], dtype=np.float64), fps=10)
    motion_path.write_bytes(pickle.dumps(motion))
    spec_path.write_text(
        '{"repairs":[{"frame_start":1,"frame_end":4,"joint_names":["J00"],'
        '"method":"linear_interpolate","rationale":"test"}]}',
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/repair_robot_motion.py",
            "--input",
            str(motion_path),
            "--output",
            str(motion_path),
            "--repair_spec",
            str(spec_path),
            "--joint_names",
            "J00",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        timeout=15,
    )

    assert result.returncode != 0
    assert "Refusing to overwrite input file" in result.stderr
