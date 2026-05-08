from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.probe_official_mild_two_stage_ik import (
    FULL_MOTION_SPECS,
    REPO_ROOT,
    VARIANT_PATCHES,
    apply_weight_patch,
    assert_weight_only_changes,
    build_variant_configs,
    build_report_from_results,
    evaluate_candidate_acceptance,
    filter_comparison_rows,
    validate_reuse_report,
)


def _minimal_config() -> dict:
    entry = ["Human", 1, 2, [0.1, 0.2, 0.3], [1.0, 0.0, 0.0, 0.0]]
    return {
        "human_scale_table": {"Human": 1.0},
        "ik_match_table1": {
            "RobotA": copy.deepcopy(entry),
            "RobotB": copy.deepcopy(entry),
        },
        "ik_match_table2": {
            "RobotA": copy.deepcopy(entry),
            "RobotB": copy.deepcopy(entry),
        },
    }


def test_apply_weight_patch_changes_only_position_and_orientation_weights() -> None:
    base_config = _minimal_config()
    variant = apply_weight_patch(
        base_config,
        {
            "ik_match_table1": {"RobotA": (3, 4)},
            "ik_match_table2": {"RobotB": (5, 6)},
        },
    )

    assert variant["ik_match_table1"]["RobotA"][1:3] == [3, 4]
    assert variant["ik_match_table2"]["RobotB"][1:3] == [5, 6]
    assert variant["ik_match_table1"]["RobotA"][0] == "Human"
    assert variant["ik_match_table1"]["RobotA"][3] == [0.1, 0.2, 0.3]
    assert variant["ik_match_table1"]["RobotA"][4] == [1.0, 0.0, 0.0, 0.0]
    assert base_config["ik_match_table1"]["RobotA"][1:3] == [1, 2]
    assert_weight_only_changes(base_config, variant)


def test_assert_weight_only_changes_rejects_offset_changes() -> None:
    base_config = _minimal_config()
    variant = copy.deepcopy(base_config)
    variant["ik_match_table1"]["RobotA"][3] = [9.0, 0.2, 0.3]

    try:
        assert_weight_only_changes(base_config, variant)
    except ValueError as exc:
        assert "non-weight field changed" in str(exc)
    else:
        raise AssertionError("Expected non-weight changes to be rejected")


def test_build_variant_configs_writes_outside_formal_ik_config_dir(tmp_path: Path) -> None:
    base_config_path = tmp_path / "bvh_human_robot_hit_to_t800--manual.json"
    output_dir = tmp_path / "debug_logs" / "ik_weight_trials"
    base_config_path.write_text(json.dumps(_minimal_config()), encoding="utf-8")

    paths = build_variant_configs(
        base_config_path=base_config_path,
        output_dir=output_dir,
        variant_patches={
            "baseline_current_manual": {},
            "candidate": {"ik_match_table1": {"RobotA": (7, 8)}},
        },
    )

    assert set(paths) == {"baseline_current_manual", "candidate"}
    assert paths["candidate"].parent == output_dir
    assert paths["candidate"].name == "candidate.json"
    assert json.loads(base_config_path.read_text(encoding="utf-8")) == _minimal_config()


def test_build_variant_configs_rejects_formal_ik_config_output_dir(tmp_path: Path) -> None:
    base_config_path = tmp_path / "bvh_human_robot_hit_to_t800--manual.json"
    base_config_path.write_text(json.dumps(_minimal_config()), encoding="utf-8")
    formal_ik_config_dir = REPO_ROOT / "general_motion_retargeting" / "ik_configs"

    try:
        build_variant_configs(
            base_config_path=base_config_path,
            output_dir=formal_ik_config_dir,
            variant_patches={"candidate": {"ik_match_table1": {"RobotA": (7, 8)}}},
        )
    except ValueError as exc:
        assert "outside GMR/general_motion_retargeting/ik_configs" in str(exc)
    else:
        raise AssertionError("Expected formal ik_configs output directory to be rejected")


def test_default_variants_keep_feet_near_manual_baseline() -> None:
    aggressive_variants = {
        name
        for name, patch in VARIANT_PATCHES.items()
        if any(
            body_name in {"LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"}
            for table_patch in patch.values()
            for body_name in table_patch
        )
    }

    assert aggressive_variants == {"mild_foot_rot4"}
    assert VARIANT_PATCHES["mild_foot_rot4"]["ik_match_table1"]["LINK_ANKLE_ROLL_L"] == (50, 4)
    assert VARIANT_PATCHES["mild_foot_rot4"]["ik_match_table2"]["LINK_ANKLE_ROLL_R"] == (50, 4)


def test_full_motion_specs_cover_complete_official_bvh_files() -> None:
    assert FULL_MOTION_SPECS["zhiquan_full"].frame_start == 0
    assert FULL_MOTION_SPECS["zhiquan_full"].frame_end == 2828
    assert FULL_MOTION_SPECS["kick_540_full"].frame_start == 0
    assert FULL_MOTION_SPECS["kick_540_full"].frame_end == 3920


def test_run_variant_motion_forwards_max_iter(monkeypatch, tmp_path: Path) -> None:
    from scripts import probe_official_mild_two_stage_ik as probe

    captured_kwargs = {}

    class FakeGMR:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

        def retarget(self, frame, **_kwargs):
            return [0.0] * 32

    monkeypatch.setattr(probe, "GMR", FakeGMR)
    monkeypatch.setattr(probe, "load_bvh_file", lambda *_args, **_kwargs: ([{"Hips": ([0, 0, 0], [1, 0, 0, 0])}], 1.75))
    monkeypatch.setattr(probe, "calibrate_human_robot_hit_frames", lambda frames: frames)
    monkeypatch.setattr(probe, "slice_motion_frames", lambda frames, **_kwargs: frames)
    monkeypatch.setattr(probe, "extract_key_metrics", lambda _debug_log: {"frames_sampled": 1})

    config_path = tmp_path / "candidate.json"
    config_path.write_text(json.dumps(_minimal_config()), encoding="utf-8")

    motion_spec = probe.MotionSpec(
        name="tiny",
        bvh_file=tmp_path / "tiny.bvh",
        frame_start=0,
        frame_end=1,
        debug_log_every_n=1,
    )

    probe.run_variant_motion(
        variant_config_path=config_path,
        motion_spec=motion_spec,
        output_dir=tmp_path,
        robot="t800_transparent",
        max_iter=20,
    )

    assert captured_kwargs["max_iter"] == 20


def _comparison_row(
    variant: str,
    motion: str,
    *,
    final2_p95: float,
    qpos_step_abs_max: float = 0.5,
    qpos_step_abs_p95: float = 0.02,
    joint_limit_hit_count: int = 100,
    left_foot_p95_mm: float = 4.0,
    right_foot_p95_mm: float = 4.0,
    left_foot_ori_p95_deg: float = 5.0,
    right_foot_ori_p95_deg: float = 5.0,
    root_z_min: float = 0.5,
    root_z_max: float = 1.0,
) -> dict:
    return {
        "variant": variant,
        "motion": motion,
        "final2_p95": final2_p95,
        "qpos_step_abs_max": qpos_step_abs_max,
        "qpos_step_abs_p95": qpos_step_abs_p95,
        "joint_limit_hit_count": joint_limit_hit_count,
        "left_foot_p95_mm": left_foot_p95_mm,
        "right_foot_p95_mm": right_foot_p95_mm,
        "left_foot_ori_p95_deg": left_foot_ori_p95_deg,
        "right_foot_ori_p95_deg": right_foot_ori_p95_deg,
        "root_z_min": root_z_min,
        "root_z_max": root_z_max,
    }


def _baseline_rows() -> list[dict]:
    return [
        _comparison_row("baseline_current_manual", "zhiquan_full", final2_p95=4.0),
        _comparison_row("baseline_current_manual", "kick_540_full", final2_p95=5.0),
    ]


def test_evaluate_candidate_acceptance_rejects_lower_error_with_qpos_jump_risk() -> None:
    rows = _baseline_rows() + [
        _comparison_row("risky", "zhiquan_full", final2_p95=2.0, qpos_step_abs_max=0.6),
        _comparison_row("risky", "kick_540_full", final2_p95=3.0, qpos_step_abs_max=0.5),
    ]

    acceptance = evaluate_candidate_acceptance(rows)

    assert acceptance["candidates"]["risky"]["accepted"] is False
    assert "qpos_step_abs_max" in acceptance["candidates"]["risky"]["reject_reasons"][0]
    assert acceptance["selected_candidate"] is None


def test_evaluate_candidate_acceptance_rejects_foot_orientation_degradation() -> None:
    rows = _baseline_rows() + [
        _comparison_row("risky", "zhiquan_full", final2_p95=2.0, left_foot_ori_p95_deg=6.2),
        _comparison_row("risky", "kick_540_full", final2_p95=3.0),
    ]

    acceptance = evaluate_candidate_acceptance(rows)

    assert acceptance["candidates"]["risky"]["accepted"] is False
    assert any("left_foot_ori_p95_deg" in reason for reason in acceptance["candidates"]["risky"]["reject_reasons"])
    assert acceptance["selected_candidate"] is None


def test_evaluate_candidate_acceptance_selects_safe_candidate_across_complete_motions() -> None:
    rows = _baseline_rows() + [
        _comparison_row(
            "safe",
            "zhiquan_full",
            final2_p95=2.0,
            qpos_step_abs_max=0.45,
            joint_limit_hit_count=80,
            left_foot_p95_mm=5.5,
        ),
        _comparison_row(
            "safe",
            "kick_540_full",
            final2_p95=3.0,
            qpos_step_abs_max=0.45,
            joint_limit_hit_count=80,
            right_foot_ori_p95_deg=4.0,
        ),
    ]

    acceptance = evaluate_candidate_acceptance(rows)

    assert acceptance["candidates"]["safe"]["accepted"] is True
    assert acceptance["candidates"]["safe"]["reject_reasons"] == []
    assert acceptance["selected_candidate"] == "safe"


def test_build_report_from_results_preserves_existing_rows_when_reused() -> None:
    rows = _baseline_rows() + [
        _comparison_row("safe", "zhiquan_full", final2_p95=2.0, joint_limit_hit_count=80),
        _comparison_row("safe", "kick_540_full", final2_p95=3.0, joint_limit_hit_count=80),
    ]

    report = build_report_from_results(
        manifest={"variants": {}},
        results={},
        existing_rows=rows,
    )

    assert report["comparison_rows"] == rows
    assert report["acceptance"]["selected_candidate"] == "safe"


def _manifest(base_hash: str = "base-a", candidate_hash: str = "candidate-a") -> dict:
    return {
        "base_config_sha256": base_hash,
        "variants": {
            "baseline_current_manual": {"config_sha256": base_hash},
            "safe": {"config_sha256": candidate_hash},
        },
        "motions": {
            "zhiquan_full": {},
            "kick_540_full": {},
        },
    }


def test_validate_reuse_report_rejects_missing_hashes_by_default() -> None:
    current_manifest = _manifest()
    old_report = {"manifest": {"variants": {"safe": {}}, "motions": {"zhiquan_full": {}, "kick_540_full": {}}}}

    validation = validate_reuse_report(
        current_manifest=current_manifest,
        existing_report=old_report,
        selected_variants=["baseline_current_manual", "safe"],
        selected_motions=["zhiquan_full", "kick_540_full"],
    )

    assert validation["verified"] is False
    assert any("base_config_sha256" in error for error in validation["errors"])


def test_validate_reuse_report_accepts_matching_hashes() -> None:
    current_manifest = _manifest()
    existing_report = {
        "manifest": _manifest(),
        "comparison_rows": _baseline_rows()
        + [
            _comparison_row("safe", "zhiquan_full", final2_p95=2.0),
            _comparison_row("safe", "kick_540_full", final2_p95=3.0),
        ],
    }

    validation = validate_reuse_report(
        current_manifest=current_manifest,
        existing_report=existing_report,
        selected_variants=["baseline_current_manual", "safe"],
        selected_motions=["zhiquan_full", "kick_540_full"],
    )

    assert validation["verified"] is True
    assert validation["errors"] == []


def test_validate_reuse_report_rejects_variant_hash_mismatch() -> None:
    current_manifest = _manifest(candidate_hash="candidate-a")
    existing_report = {"manifest": _manifest(candidate_hash="candidate-b")}

    validation = validate_reuse_report(
        current_manifest=current_manifest,
        existing_report=existing_report,
        selected_variants=["baseline_current_manual", "safe"],
        selected_motions=["zhiquan_full", "kick_540_full"],
    )

    assert validation["verified"] is False
    assert any("safe" in error and "config_sha256" in error for error in validation["errors"])


def test_filter_comparison_rows_keeps_selected_variants_and_motions_only() -> None:
    rows = _baseline_rows() + [
        _comparison_row("safe", "zhiquan_full", final2_p95=2.0),
        _comparison_row("safe", "kick_540_full", final2_p95=3.0),
        _comparison_row("other", "zhiquan_full", final2_p95=1.0),
    ]

    filtered = filter_comparison_rows(rows, selected_variants=["safe"], selected_motions=["zhiquan_full"])

    assert {(row["variant"], row["motion"]) for row in filtered} == {
        ("baseline_current_manual", "zhiquan_full"),
        ("safe", "zhiquan_full"),
    }
