from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.motion_retarget_options import (
    calibrate_human_robot_hit_frames,
    resolve_actual_human_height,
    resolve_ik_safety_break,
    resolve_max_iter,
)
from general_motion_retargeting.params import IK_CONFIG_DICT
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from scripts.bvh_to_robot import slice_motion_frames
from scripts.summarize_gmr_debug_log import read_debug_records, summarize_debug_records


@dataclass(frozen=True)
class MotionSpec:
    name: str
    bvh_file: Path
    frame_start: int
    frame_end: int
    debug_log_every_n: int = 10


FULL_MOTION_SPECS = {
    "zhiquan_full": MotionSpec(
        name="zhiquan_full",
        bvh_file=WORKSPACE_ROOT / "official_data" / "1.2 题目拳击数据" / "zhiquan_quanji_001.bvh",
        frame_start=0,
        frame_end=2828,
        debug_log_every_n=10,
    ),
    "kick_540_full": MotionSpec(
        name="kick_540_full",
        bvh_file=WORKSPACE_ROOT / "official_data" / "1.2 题目拳击数据" / "540huixuantitui_001.bvh",
        frame_start=0,
        frame_end=3920,
        debug_log_every_n=10,
    ),
}


WeightPatch = dict[str, dict[str, tuple[float, float]]]

# mild_two_stage 的命名含义：
# - mild：只温和加强核心和上肢，不把脚/髋/全身都拉成硬跟踪，避免高动态动作里抖动和限位压力变大。
# - two_stage：同时利用 GMR 的两张 IK table。table1 给一点姿态预对齐，table2 再做最终位置/姿态跟踪。
#
# 这里的数值只改每个 IK target 的 position/orientation 权重，不改 manual 基线里的 offset/rot_offset。
# 这样做是为了让候选之间可追溯：如果视觉变好或变坏，优先归因到权重，而不是混入新的坐标轴手调。
MILD_TWO_STAGE_PATCH: WeightPatch = {
    "ik_match_table1": {
        "LINK_TORSO_YAW": (0, 4),
        "LINK_HEAD_YAW": (0, 2),
        "LINK_SHOULDER_YAW_L": (0, 4),
        "LINK_SHOULDER_YAW_R": (0, 4),
        "LINK_ELBOW_PITCH_L": (0, 2),
        "LINK_ELBOW_PITCH_R": (0, 2),
        "LINK_WRIST_END_L": (0, 2),
        "LINK_WRIST_END_R": (0, 2),
    },
    "ik_match_table2": {
        "LINK_TORSO_YAW": (2, 4),
        "LINK_HEAD_YAW": (3, 2),
        "LINK_SHOULDER_YAW_L": (8, 2),
        "LINK_SHOULDER_YAW_R": (8, 2),
        "LINK_ELBOW_PITCH_L": (12, 2),
        "LINK_ELBOW_PITCH_R": (12, 2),
        "LINK_WRIST_END_L": (12, 2),
        "LINK_WRIST_END_R": (12, 2),
    },
}


VARIANT_PATCHES: dict[str, WeightPatch] = {
    "baseline_current_manual": {},
    "mild_two_stage": MILD_TWO_STAGE_PATCH,
    "mild_head_torso_plus": {
        "ik_match_table1": {
            **MILD_TWO_STAGE_PATCH["ik_match_table1"],
            "LINK_TORSO_YAW": (0, 5),
            "LINK_HEAD_YAW": (0, 3),
        },
        "ik_match_table2": {
            **MILD_TWO_STAGE_PATCH["ik_match_table2"],
            "LINK_TORSO_YAW": (3, 5),
            "LINK_HEAD_YAW": (4, 3),
        },
    },
    "mild_elbow_wrist_plus": {
        "ik_match_table1": {
            **MILD_TWO_STAGE_PATCH["ik_match_table1"],
            "LINK_ELBOW_PITCH_L": (0, 3),
            "LINK_ELBOW_PITCH_R": (0, 3),
            "LINK_WRIST_END_L": (0, 3),
            "LINK_WRIST_END_R": (0, 3),
        },
        "ik_match_table2": {
            **MILD_TWO_STAGE_PATCH["ik_match_table2"],
            "LINK_ELBOW_PITCH_L": (15, 3),
            "LINK_ELBOW_PITCH_R": (15, 3),
            "LINK_WRIST_END_L": (15, 3),
            "LINK_WRIST_END_R": (15, 3),
        },
    },
    "mild_head_torso_elbow_wrist_plus": {
        "ik_match_table1": {
            **MILD_TWO_STAGE_PATCH["ik_match_table1"],
            "LINK_TORSO_YAW": (0, 5),
            "LINK_HEAD_YAW": (0, 3),
            "LINK_ELBOW_PITCH_L": (0, 3),
            "LINK_ELBOW_PITCH_R": (0, 3),
            "LINK_WRIST_END_L": (0, 3),
            "LINK_WRIST_END_R": (0, 3),
        },
        "ik_match_table2": {
            **MILD_TWO_STAGE_PATCH["ik_match_table2"],
            "LINK_TORSO_YAW": (3, 5),
            "LINK_HEAD_YAW": (4, 3),
            "LINK_ELBOW_PITCH_L": (15, 3),
            "LINK_ELBOW_PITCH_R": (15, 3),
            "LINK_WRIST_END_L": (15, 3),
            "LINK_WRIST_END_R": (15, 3),
        },
    },
    "mild_foot_rot4": {
        "ik_match_table1": {
            **MILD_TWO_STAGE_PATCH["ik_match_table1"],
            "LINK_ANKLE_ROLL_L": (50, 4),
            "LINK_ANKLE_ROLL_R": (50, 4),
        },
        "ik_match_table2": {
            **MILD_TWO_STAGE_PATCH["ik_match_table2"],
            "LINK_ANKLE_ROLL_L": (50, 4),
            "LINK_ANKLE_ROLL_R": (50, 4),
        },
    },
}


KEY_BODIES = [
    "Hips",
    "Spine2",
    "Head",
    "LeftFootMod",
    "RightFootMod",
    "LeftArm",
    "RightArm",
    "LeftForeArm",
    "RightForeArm",
    "LeftHand",
    "RightHand",
]


ACCEPTANCE_LIMITS = {
    # 单帧 qpos 最大跳变是这里最敏感的“奇异/近奇异”代理指标。
    # IK error 变小但 qpos 跳变变大，通常说明机器人在某些帧用了很激烈的关节补偿，不适合直接进训练。
    "qpos_step_abs_max_ratio": 1.05,
    "qpos_step_abs_p95_ratio": 1.20,
    # 不允许候选比 manual baseline 命中更多关节限位。限位压力上升时，viewer 里未必第一眼明显，
    # 但后续导入训练和高速回放时很容易变成抖动、穿模或策略难学。
    "joint_limit_hit_ratio": 1.00,
    # 脚端指标单独设门槛，是因为脚掌是否稳定会直接影响 grounding、接触和训练质量；
    # 不能只因为手臂误差下降，就接受脚底明显变差的候选。
    "foot_position_p95_increase_mm": 2.0,
    "foot_orientation_p95_increase_deg": 1.0,
    # root 高度范围异常扩大，常见原因是 IK 在某些高动态帧拉身体补偿四肢目标。
    "root_z_extra_range_m": 0.05,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_weight_patch(base_config: dict[str, Any], patch: WeightPatch) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    for table_name, table_patch in patch.items():
        if table_name not in {"ik_match_table1", "ik_match_table2"}:
            raise ValueError(f"Unsupported IK table in patch: {table_name}")
        for robot_body, (position_cost, orientation_cost) in table_patch.items():
            if robot_body not in config[table_name]:
                raise KeyError(f"{table_name}:{robot_body} does not exist in base config")
            config[table_name][robot_body][1] = position_cost
            config[table_name][robot_body][2] = orientation_cost
    return config


def assert_weight_only_changes(base_config: dict[str, Any], variant_config: dict[str, Any]) -> None:
    # trial 脚本只负责筛“权重候选”。如果 offset/rot_offset 也变了，
    # 那就已经是另一类人工 IK config 调参，必须单独命名和复核，不能混进 mild_two_stage 的统计结论。
    base_probe = copy.deepcopy(base_config)
    variant_probe = copy.deepcopy(variant_config)
    for table_name in ["ik_match_table1", "ik_match_table2"]:
        for robot_body in base_probe[table_name]:
            base_probe[table_name][robot_body][1] = variant_probe[table_name][robot_body][1]
            base_probe[table_name][robot_body][2] = variant_probe[table_name][robot_body][2]
    if base_probe != variant_probe:
        raise ValueError("Candidate config contains a non-weight field changed from the manual baseline.")


def build_variant_configs(
    base_config_path: Path,
    output_dir: Path,
    variant_patches: dict[str, WeightPatch],
) -> dict[str, Path]:
    formal_ik_config_dir = REPO_ROOT / "general_motion_retargeting" / "ik_configs"
    output_dir = output_dir.resolve()
    if output_dir == formal_ik_config_dir.resolve() or formal_ik_config_dir.resolve() in output_dir.parents:
        raise ValueError("Trial configs must be written outside GMR/general_motion_retargeting/ik_configs.")

    base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    variant_paths = {}
    for variant_name, patch in variant_patches.items():
        variant = apply_weight_patch(base_config, patch)
        assert_weight_only_changes(base_config, variant)
        variant_path = output_dir / f"{variant_name}.json"
        variant_path.write_text(json.dumps(variant, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        variant_paths[variant_name] = variant_path
    return variant_paths


def build_manifest(
    base_config_path: Path,
    output_dir: Path,
    selected_patches: dict[str, WeightPatch],
    variant_paths: dict[str, Path],
    selected_motions: list[str],
) -> dict[str, Any]:
    base_config_hash = file_sha256(base_config_path)
    return {
        "base_config": str(base_config_path),
        "base_config_sha256": base_config_hash,
        "output_dir": str(output_dir),
        "variants": {
            name: {
                "config_path": str(path),
                "config_sha256": file_sha256(path),
                "patch": selected_patches[name],
            }
            for name, path in variant_paths.items()
        },
        "motions": {
            name: FULL_MOTION_SPECS[name].__dict__ | {"bvh_file": str(FULL_MOTION_SPECS[name].bvh_file)}
            for name in selected_motions
        },
    }


def _summary(values: list[float]) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "p95": None, "max": None}
    return {
        "mean": round(float(np.mean(arr)), 6),
        "p95": round(float(np.percentile(arr, 95)), 6),
        "max": round(float(np.max(arr)), 6),
    }


def extract_key_metrics(debug_log: Path) -> dict[str, Any]:
    records = read_debug_records(debug_log)
    summary = summarize_debug_records(records, top_n=12)
    final_error1 = [float(record["final_error1"]) for record in records if record.get("final_error1") is not None]
    final_error2 = [float(record["final_error2"]) for record in records if record.get("final_error2") is not None]
    pos_by_body = {body: [] for body in KEY_BODIES}
    ori_by_body = {body: [] for body in KEY_BODIES}
    limit_margins = []
    limit_hit_count = 0

    for record in records:
        task_table2 = record.get("task_table2") or {}
        for body in KEY_BODIES:
            task_info = task_table2.get(body)
            if not task_info:
                continue
            pos_error = task_info.get("world_position_error_norm_m")
            ori_error = task_info.get("world_orientation_error_deg")
            if pos_error is not None:
                pos_by_body[body].append(float(pos_error))
            if ori_error is not None:
                ori_by_body[body].append(float(ori_error))

        for joint_info in (record.get("joint_limits") or {}).values():
            margin = joint_info.get("margin_min_rad")
            if margin is None:
                continue
            margin = float(margin)
            limit_margins.append(margin)
            if margin <= 1e-4:
                limit_hit_count += 1

    return {
        "frames_sampled": len(records),
        "summary": summary,
        "final_error1": _summary(final_error1),
        "final_error2": _summary(final_error2),
        "position_error_m": {body: _summary(values) for body, values in pos_by_body.items()},
        "orientation_error_deg": {body: _summary(values) for body, values in ori_by_body.items()},
        "joint_limits": {
            "hit_count": limit_hit_count,
            "min_margin_rad": None if not limit_margins else round(float(np.min(limit_margins)), 9),
        },
    }


def run_variant_motion(
    variant_config_path: Path,
    motion_spec: MotionSpec,
    output_dir: Path,
    robot: str = "t800_transparent",
    disable_ik_safety_break: bool = True,
    max_iter: int | None = None,
) -> dict[str, Any]:
    frames, actual_human_height = load_bvh_file(str(motion_spec.bvh_file), format="lafan1")
    frames = slice_motion_frames(
        frames,
        frame_start=motion_spec.frame_start,
        frame_end=motion_spec.frame_end,
        frame_step=1,
    )
    frames = calibrate_human_robot_hit_frames(frames)

    original_config_path = IK_CONFIG_DICT["bvh_human_robot_hit"][robot]
    IK_CONFIG_DICT["bvh_human_robot_hit"][robot] = variant_config_path
    debug_log = output_dir / f"{variant_config_path.stem}_{motion_spec.name}.jsonl"
    try:
        retargeter = GMR(
            src_human="bvh_human_robot_hit",
            tgt_robot=robot,
            actual_human_height=resolve_actual_human_height(actual_human_height, "human_robot_hit"),
            debug_log_path=str(debug_log),
            debug_log_every_n=motion_spec.debug_log_every_n,
            ik_safety_break=resolve_ik_safety_break(disable_ik_safety_break),
            max_iter=resolve_max_iter(max_iter),
            verbose=False,
        )
        qpos_list = []
        for frame_index, frame in enumerate(frames):
            qpos_list.append(retargeter.retarget(frame, frame_index=motion_spec.frame_start + frame_index))
    finally:
        IK_CONFIG_DICT["bvh_human_robot_hit"][robot] = original_config_path

    metrics = extract_key_metrics(debug_log)
    qpos = np.asarray(qpos_list, dtype=np.float64)
    qpos_step = np.diff(qpos, axis=0) if len(qpos) > 1 else np.empty((0, qpos.shape[1] if qpos.ndim == 2 else 0))
    metrics["debug_log"] = str(debug_log)
    metrics["motion"] = motion_spec.name
    metrics["bvh_file"] = str(motion_spec.bvh_file)
    metrics["frame_start"] = motion_spec.frame_start
    metrics["frame_end"] = motion_spec.frame_end
    metrics["debug_log_every_n"] = motion_spec.debug_log_every_n
    metrics["qpos"] = {
        "frames": int(qpos.shape[0]),
        "abs_max": None if qpos.size == 0 else round(float(np.max(np.abs(qpos))), 6),
        "root_z_min": None if qpos.size == 0 else round(float(np.min(qpos[:, 2])), 6),
        "root_z_max": None if qpos.size == 0 else round(float(np.max(qpos[:, 2])), 6),
        "step_abs_max": None if qpos_step.size == 0 else round(float(np.max(np.abs(qpos_step))), 6),
        "step_abs_p95": None if qpos_step.size == 0 else round(float(np.percentile(np.abs(qpos_step), 95)), 6),
    }
    return metrics


def _metric_value(metrics: dict[str, Any], path: tuple[str, ...]) -> float | None:
    current: Any = metrics
    for key in path:
        if current is None:
            return None
        current = current.get(key)
    if current is None:
        return None
    return float(current)


def build_comparison_rows(results: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for variant_name, motion_results in results.items():
        for motion_name, metrics in motion_results.items():
            row = {
                "variant": variant_name,
                "motion": motion_name,
                "frames_sampled": metrics["frames_sampled"],
                "final2_p95": _metric_value(metrics, ("final_error2", "p95")),
                "final2_max": _metric_value(metrics, ("final_error2", "max")),
                "head_p95_mm": _metric_value(metrics, ("position_error_m", "Head", "p95")),
                "left_hand_p95_mm": _metric_value(metrics, ("position_error_m", "LeftHand", "p95")),
                "right_hand_p95_mm": _metric_value(metrics, ("position_error_m", "RightHand", "p95")),
                "left_forearm_p95_mm": _metric_value(metrics, ("position_error_m", "LeftForeArm", "p95")),
                "right_forearm_p95_mm": _metric_value(metrics, ("position_error_m", "RightForeArm", "p95")),
                "left_foot_p95_mm": _metric_value(metrics, ("position_error_m", "LeftFootMod", "p95")),
                "right_foot_p95_mm": _metric_value(metrics, ("position_error_m", "RightFootMod", "p95")),
                "left_foot_ori_p95_deg": _metric_value(metrics, ("orientation_error_deg", "LeftFootMod", "p95")),
                "right_foot_ori_p95_deg": _metric_value(metrics, ("orientation_error_deg", "RightFootMod", "p95")),
                "joint_limit_hit_count": metrics["joint_limits"]["hit_count"],
                "joint_limit_min_margin_rad": metrics["joint_limits"]["min_margin_rad"],
                "qpos_abs_max": metrics["qpos"]["abs_max"],
                "qpos_step_abs_p95": metrics["qpos"]["step_abs_p95"],
                "qpos_step_abs_max": metrics["qpos"]["step_abs_max"],
                "root_z_min": metrics["qpos"]["root_z_min"],
                "root_z_max": metrics["qpos"]["root_z_max"],
            }
            for key in [
                "head_p95_mm",
                "left_hand_p95_mm",
                "right_hand_p95_mm",
                "left_forearm_p95_mm",
                "right_forearm_p95_mm",
                "left_foot_p95_mm",
                "right_foot_p95_mm",
            ]:
                if row[key] is not None:
                    row[key] = round(row[key] * 1000.0, 3)
            for key in ["final2_p95", "final2_max", "left_foot_ori_p95_deg", "right_foot_ori_p95_deg"]:
                if row[key] is not None:
                    row[key] = round(row[key], 6)
            rows.append(row)
    return rows


def _row_by_variant_motion(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row["variant"]), str(row["motion"])): row for row in rows}


def _ratio(candidate_value: float | int | None, baseline_value: float | int | None) -> float:
    if candidate_value is None or baseline_value is None:
        return float("inf")
    candidate = float(candidate_value)
    baseline = float(baseline_value)
    if baseline == 0.0:
        return 1.0 if candidate <= 0.0 else float("inf")
    return candidate / baseline


def _increase(candidate_value: float | int | None, baseline_value: float | int | None) -> float:
    if candidate_value is None or baseline_value is None:
        return float("inf")
    return float(candidate_value) - float(baseline_value)


def _round_metric(value: float) -> float | str:
    if not np.isfinite(value):
        return "inf"
    return round(float(value), 6)


def evaluate_candidate_acceptance(
    rows: list[dict[str, Any]],
    baseline_variant: str = "baseline_current_manual",
) -> dict[str, Any]:
    row_map = _row_by_variant_motion(rows)
    motions = sorted({str(row["motion"]) for row in rows if row["variant"] == baseline_variant})
    variants = sorted({str(row["variant"]) for row in rows if row["variant"] != baseline_variant})
    candidates: dict[str, Any] = {}

    for variant in variants:
        reject_reasons: list[str] = []
        motion_checks: dict[str, Any] = {}
        score_parts: list[float] = []

        for motion in motions:
            baseline = row_map.get((baseline_variant, motion))
            candidate = row_map.get((variant, motion))
            if baseline is None:
                reject_reasons.append(f"{motion}: missing baseline row")
                continue
            if candidate is None:
                reject_reasons.append(f"{motion}: missing candidate row")
                continue

            final2_ratio = _ratio(candidate.get("final2_p95"), baseline.get("final2_p95"))
            qpos_step_abs_max_ratio = _ratio(candidate.get("qpos_step_abs_max"), baseline.get("qpos_step_abs_max"))
            qpos_step_abs_p95_ratio = _ratio(candidate.get("qpos_step_abs_p95"), baseline.get("qpos_step_abs_p95"))
            joint_limit_hit_ratio = _ratio(candidate.get("joint_limit_hit_count"), baseline.get("joint_limit_hit_count"))
            left_foot_position_increase_mm = _increase(candidate.get("left_foot_p95_mm"), baseline.get("left_foot_p95_mm"))
            right_foot_position_increase_mm = _increase(candidate.get("right_foot_p95_mm"), baseline.get("right_foot_p95_mm"))
            left_foot_orientation_increase_deg = _increase(
                candidate.get("left_foot_ori_p95_deg"),
                baseline.get("left_foot_ori_p95_deg"),
            )
            right_foot_orientation_increase_deg = _increase(
                candidate.get("right_foot_ori_p95_deg"),
                baseline.get("right_foot_ori_p95_deg"),
            )
            root_z_min_extra_drop_m = _increase(baseline.get("root_z_min"), candidate.get("root_z_min"))
            root_z_max_extra_lift_m = _increase(candidate.get("root_z_max"), baseline.get("root_z_max"))

            motion_checks[motion] = {
                "final2_ratio": _round_metric(final2_ratio),
                "qpos_step_abs_max_ratio": _round_metric(qpos_step_abs_max_ratio),
                "qpos_step_abs_p95_ratio": _round_metric(qpos_step_abs_p95_ratio),
                "joint_limit_hit_ratio": _round_metric(joint_limit_hit_ratio),
                "left_foot_position_increase_mm": _round_metric(left_foot_position_increase_mm),
                "right_foot_position_increase_mm": _round_metric(right_foot_position_increase_mm),
                "left_foot_orientation_increase_deg": _round_metric(left_foot_orientation_increase_deg),
                "right_foot_orientation_increase_deg": _round_metric(right_foot_orientation_increase_deg),
                "root_z_min_extra_drop_m": _round_metric(root_z_min_extra_drop_m),
                "root_z_max_extra_lift_m": _round_metric(root_z_max_extra_lift_m),
            }

            if final2_ratio >= 1.0:
                reject_reasons.append(f"{motion}: final2_p95 ratio {_round_metric(final2_ratio)} is not improved")
            if qpos_step_abs_max_ratio > ACCEPTANCE_LIMITS["qpos_step_abs_max_ratio"]:
                reject_reasons.append(
                    f"{motion}: qpos_step_abs_max ratio {_round_metric(qpos_step_abs_max_ratio)} "
                    f"> {ACCEPTANCE_LIMITS['qpos_step_abs_max_ratio']}"
                )
            if qpos_step_abs_p95_ratio > ACCEPTANCE_LIMITS["qpos_step_abs_p95_ratio"]:
                reject_reasons.append(
                    f"{motion}: qpos_step_abs_p95 ratio {_round_metric(qpos_step_abs_p95_ratio)} "
                    f"> {ACCEPTANCE_LIMITS['qpos_step_abs_p95_ratio']}"
                )
            if joint_limit_hit_ratio > ACCEPTANCE_LIMITS["joint_limit_hit_ratio"]:
                reject_reasons.append(
                    f"{motion}: joint_limit_hit_count ratio {_round_metric(joint_limit_hit_ratio)} "
                    f"> {ACCEPTANCE_LIMITS['joint_limit_hit_ratio']}"
                )
            for metric_name, increase_value, limit in [
                ("left_foot_p95_mm", left_foot_position_increase_mm, ACCEPTANCE_LIMITS["foot_position_p95_increase_mm"]),
                ("right_foot_p95_mm", right_foot_position_increase_mm, ACCEPTANCE_LIMITS["foot_position_p95_increase_mm"]),
                (
                    "left_foot_ori_p95_deg",
                    left_foot_orientation_increase_deg,
                    ACCEPTANCE_LIMITS["foot_orientation_p95_increase_deg"],
                ),
                (
                    "right_foot_ori_p95_deg",
                    right_foot_orientation_increase_deg,
                    ACCEPTANCE_LIMITS["foot_orientation_p95_increase_deg"],
                ),
            ]:
                if increase_value > limit:
                    reject_reasons.append(f"{motion}: {metric_name} increase {_round_metric(increase_value)} > {limit}")
            if root_z_min_extra_drop_m > ACCEPTANCE_LIMITS["root_z_extra_range_m"]:
                reject_reasons.append(
                    f"{motion}: root_z_min extra drop {_round_metric(root_z_min_extra_drop_m)}m "
                    f"> {ACCEPTANCE_LIMITS['root_z_extra_range_m']}m"
                )
            if root_z_max_extra_lift_m > ACCEPTANCE_LIMITS["root_z_extra_range_m"]:
                reject_reasons.append(
                    f"{motion}: root_z_max extra lift {_round_metric(root_z_max_extra_lift_m)}m "
                    f"> {ACCEPTANCE_LIMITS['root_z_extra_range_m']}m"
                )

            score_parts.append(
                final2_ratio
                + 0.20 * qpos_step_abs_max_ratio
                + 0.10 * qpos_step_abs_p95_ratio
                + 0.10 * joint_limit_hit_ratio
            )

        accepted = len(reject_reasons) == 0 and len(motion_checks) == len(motions)
        candidates[variant] = {
            "accepted": accepted,
            "reject_reasons": reject_reasons,
            "score": None if not accepted or not score_parts else round(float(np.mean(score_parts)), 6),
            "motion_checks": motion_checks,
        }

    accepted_candidates = [
        (variant, info["score"])
        for variant, info in candidates.items()
        if info["accepted"] and info["score"] is not None
    ]
    accepted_candidates.sort(key=lambda item: (float(item[1]), item[0]))

    best_numeric_by_motion: dict[str, str | None] = {}
    for motion in motions:
        motion_candidates = [
            row
            for row in rows
            if row["motion"] == motion and row["variant"] != baseline_variant and row.get("final2_p95") is not None
        ]
        motion_candidates.sort(key=lambda row: (float(row["final2_p95"]), str(row["variant"])))
        best_numeric_by_motion[motion] = None if not motion_candidates else str(motion_candidates[0]["variant"])

    return {
        "baseline_variant": baseline_variant,
        "limits": ACCEPTANCE_LIMITS,
        "motions": motions,
        "best_numeric_by_motion": best_numeric_by_motion,
        "candidates": candidates,
        "selected_candidate": None if not accepted_candidates else accepted_candidates[0][0],
    }


def filter_comparison_rows(
    rows: list[dict[str, Any]],
    selected_variants: list[str],
    selected_motions: list[str],
    baseline_variant: str = "baseline_current_manual",
) -> list[dict[str, Any]]:
    allowed_variants = set(selected_variants) | {baseline_variant}
    allowed_motions = set(selected_motions)
    return [
        row
        for row in rows
        if str(row.get("variant")) in allowed_variants and str(row.get("motion")) in allowed_motions
    ]


def validate_reuse_report(
    current_manifest: dict[str, Any],
    existing_report: dict[str, Any],
    selected_variants: list[str],
    selected_motions: list[str],
    baseline_variant: str = "baseline_current_manual",
) -> dict[str, Any]:
    errors: list[str] = []
    existing_manifest = existing_report.get("manifest") or {}

    current_base_hash = current_manifest.get("base_config_sha256")
    existing_base_hash = existing_manifest.get("base_config_sha256")
    if not existing_base_hash:
        errors.append("reused report manifest is missing base_config_sha256")
    elif existing_base_hash != current_base_hash:
        errors.append(
            f"base_config_sha256 mismatch: current={current_base_hash}, reused={existing_base_hash}"
        )

    existing_variants = existing_manifest.get("variants") or {}
    current_variants = current_manifest.get("variants") or {}
    for variant in set(selected_variants) | {baseline_variant}:
        current_variant = current_variants.get(variant) or {}
        existing_variant = existing_variants.get(variant) or {}
        current_hash = current_variant.get("config_sha256")
        existing_hash = existing_variant.get("config_sha256")
        if not existing_hash:
            errors.append(f"reused report manifest is missing {variant} config_sha256")
        elif existing_hash != current_hash:
            errors.append(
                f"{variant} config_sha256 mismatch: current={current_hash}, reused={existing_hash}"
            )

    existing_motions = set((existing_manifest.get("motions") or {}).keys())
    for motion in selected_motions:
        if motion not in existing_motions:
            errors.append(f"reused report manifest is missing motion {motion}")

    existing_rows = existing_report.get("comparison_rows") or []
    row_keys = {(str(row.get("variant")), str(row.get("motion"))) for row in existing_rows}
    for motion in selected_motions:
        if (baseline_variant, motion) not in row_keys:
            errors.append(f"reused report rows are missing {baseline_variant}/{motion}")
        for variant in selected_variants:
            if (variant, motion) not in row_keys:
                errors.append(f"reused report rows are missing {variant}/{motion}")

    return {"verified": len(errors) == 0, "errors": errors}


def write_visual_review_commands(
    output_dir: Path,
    variant_config_path: Path,
    candidate_name: str,
    robot: str,
) -> Path:
    command_file = output_dir / f"{candidate_name}_visual_review_commands.ps1"
    lines = [
        "# Run from D:\\human_robot\\GMR",
        "# These commands monkeypatch the official BVH route to use the trial config without changing params.py.",
    ]
    for motion_spec in FULL_MOTION_SPECS.values():
        runner = output_dir / f"visual_{candidate_name}_{motion_spec.name}.py"
        runner.write_text(
            "\n".join(
                [
                    "from pathlib import Path",
                    "import runpy",
                    "import sys",
                    f"sys.path.insert(0, r'{REPO_ROOT}')",
                    "from general_motion_retargeting.params import IK_CONFIG_DICT",
                    f"IK_CONFIG_DICT['bvh_human_robot_hit']['{robot}'] = Path(r'{variant_config_path}')",
                    "sys.argv = [",
                    "    'bvh_to_robot.py',",
                    f"    '--bvh_file', r'{motion_spec.bvh_file}',",
                    f"    '--robot', '{robot}',",
                    "    '--format', 'lafan1',",
                    "    '--source_profile', 'human_robot_hit',",
                    "    '--motion_fps', '120',",
                    "    '--disable_ik_safety_break',",
                    "    '--rate_limit',",
                    "]",
                    f"runpy.run_path(r'{REPO_ROOT / 'scripts' / 'bvh_to_robot.py'}', run_name='__main__')",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        lines.append(f"& D:\\MiniConda\\envs\\robot\\python.exe \"{runner}\"")
    command_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return command_file


def build_report_from_results(
    manifest: dict[str, Any],
    results: dict[str, dict[str, dict[str, Any]]],
    existing_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rows = existing_rows if existing_rows is not None else build_comparison_rows(results)
    acceptance = evaluate_candidate_acceptance(rows) if rows else None
    return {
        "manifest": manifest,
        "comparison_rows": rows,
        "acceptance": acceptance,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full-action mild two-stage IK weight trials for official BVH to T800.")
    parser.add_argument(
        "--base_config",
        type=Path,
        default=REPO_ROOT / "general_motion_retargeting" / "ik_configs" / "bvh_human_robot_hit_to_t800--manual.json",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=WORKSPACE_ROOT / "debug_logs" / "ik_weight_trials" / "mild_two_stage_search",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(VARIANT_PATCHES),
        choices=list(VARIANT_PATCHES),
    )
    parser.add_argument(
        "--motions",
        nargs="+",
        default=list(FULL_MOTION_SPECS),
        choices=list(FULL_MOTION_SPECS),
    )
    parser.add_argument("--robot", default="t800_transparent")
    parser.add_argument(
        "--max_iter",
        type=int,
        default=None,
        help="Optional IK iteration cap per stage for trial runs. Defaults to the current GMR solver limit of 10.",
    )
    parser.add_argument("--skip_runs", action="store_true", default=False)
    parser.add_argument(
        "--reuse_report",
        type=Path,
        default=None,
        help="Reuse an existing full_action_mild_two_stage_compare.json and refresh acceptance/visual commands without rerunning IK.",
    )
    parser.add_argument(
        "--allow_stale_reuse_report",
        action="store_true",
        default=False,
        help="Allow reuse of an old report without matching baseline/candidate sha256 metadata. This is for read-only analysis only.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_patches = {name: VARIANT_PATCHES[name] for name in args.variants}
    variant_paths = build_variant_configs(args.base_config, args.output_dir, selected_patches)
    manifest = build_manifest(args.base_config, args.output_dir, selected_patches, variant_paths, args.motions)

    results: dict[str, dict[str, dict[str, Any]]] = {}
    existing_rows = None
    if args.reuse_report is not None:
        existing_report = json.loads(args.reuse_report.read_text(encoding="utf-8"))
        results = existing_report.get("results", {})
        reuse_validation = validate_reuse_report(
            current_manifest=manifest,
            existing_report=existing_report,
            selected_variants=args.variants,
            selected_motions=args.motions,
        )
        if not reuse_validation["verified"] and not args.allow_stale_reuse_report:
            raise SystemExit(
                "Refusing to reuse report because its manifest does not match the current generated configs:\n"
                + "\n".join(f"- {error}" for error in reuse_validation["errors"])
                + "\nRe-run full diagnostics, or pass --allow_stale_reuse_report for read-only stale analysis."
            )
        existing_rows = filter_comparison_rows(
            existing_report.get("comparison_rows", []),
            selected_variants=args.variants,
            selected_motions=args.motions,
        )
        manifest["reuse_validation"] = reuse_validation
        manifest["allow_stale_reuse_report"] = args.allow_stale_reuse_report
        if existing_report.get("manifest"):
            manifest["reused_report"] = str(args.reuse_report)
            manifest["reused_manifest"] = existing_report["manifest"]
    elif not args.skip_runs:
        for variant_name, variant_path in variant_paths.items():
            results[variant_name] = {}
            for motion_name in args.motions:
                metrics = run_variant_motion(
                    variant_config_path=variant_path,
                    motion_spec=FULL_MOTION_SPECS[motion_name],
                    output_dir=args.output_dir,
                    robot=args.robot,
                    max_iter=args.max_iter,
                )
                results[variant_name][motion_name] = metrics

    report = build_report_from_results(manifest=manifest, results=results, existing_rows=existing_rows)
    rows = report["comparison_rows"]
    acceptance = report["acceptance"]
    report_path = args.output_dir / "full_action_mild_two_stage_compare.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    best_candidate = None
    if acceptance:
        best_candidate = acceptance["selected_candidate"]
        if best_candidate:
            command_path = write_visual_review_commands(
                output_dir=args.output_dir,
                variant_config_path=variant_paths[best_candidate],
                candidate_name=best_candidate,
                robot=args.robot,
            )
            report["visual_review_commands"] = str(command_path)
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "report": str(report_path),
                "rows": rows,
                "acceptance": acceptance,
                "selected_candidate": best_candidate,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
