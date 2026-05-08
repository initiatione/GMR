from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _summary(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "p95": None, "max": None}
    return {
        "mean": round(float(np.mean(arr)), 6),
        "p95": round(float(np.percentile(arr, 95)), 6),
        "max": round(float(np.max(arr)), 6),
    }


def _task_entries(record: dict) -> list[tuple[str, dict]]:
    entries = []
    for table_name in ["task_table1", "task_table2"]:
        table = record.get(table_name) or {}
        for body_name, task_info in table.items():
            entries.append((body_name, task_info))
    return entries


def summarize_debug_records(records: list[dict], top_n: int = 8) -> dict:
    pos_by_body: dict[str, list[float]] = {}
    pos_axis_by_body: dict[str, list[list[float]]] = {}
    ori_by_body: dict[str, list[float]] = {}
    pre_pos_by_body: dict[str, list[float]] = {}
    post_pos_by_body: dict[str, list[float]] = {}
    limit_margin_by_joint: dict[str, list[float]] = {}
    final_error1 = []
    final_error2 = []

    for record in records:
        if record.get("final_error1") is not None:
            final_error1.append(float(record["final_error1"]))
        if record.get("final_error2") is not None:
            final_error2.append(float(record["final_error2"]))

        for body_name, task_info in _task_entries(record):
            pos = task_info.get("world_position_error_norm_m")
            ori = task_info.get("world_orientation_error_deg")
            if pos is not None:
                pos_by_body.setdefault(body_name, []).append(float(pos))
            pos_vector = task_info.get("world_position_error_vector_m")
            if pos_vector is not None:
                pos_axis_by_body.setdefault(body_name, []).append([float(v) for v in pos_vector])
            if ori is not None:
                ori_by_body.setdefault(body_name, []).append(float(ori))

        for pre_table_name, post_table_name in [
            ("task_table1_pre_ik", "task_table1"),
            ("task_table2_pre_ik", "task_table2"),
        ]:
            pre_table = record.get(pre_table_name) or {}
            post_table = record.get(post_table_name) or {}
            for body_name, pre_info in pre_table.items():
                pre_pos = pre_info.get("world_position_error_norm_m")
                post_pos = (post_table.get(body_name) or {}).get("world_position_error_norm_m")
                if pre_pos is not None:
                    pre_pos_by_body.setdefault(body_name, []).append(float(pre_pos))
                if post_pos is not None:
                    post_pos_by_body.setdefault(body_name, []).append(float(post_pos))

        for joint_name, joint_info in (record.get("joint_limits") or {}).items():
            margin = joint_info.get("margin_min_rad")
            if margin is not None:
                limit_margin_by_joint.setdefault(joint_name, []).append(float(margin))

    def ranked(metric_by_body: dict[str, list[float]], key_name: str) -> list[dict]:
        rows = []
        for body_name, values in metric_by_body.items():
            stats = _summary(values)
            rows.append(
                {
                    "body": body_name,
                    f"{key_name}_mean": stats["mean"],
                    f"{key_name}_p95": stats["p95"],
                    f"{key_name}_max": stats["max"],
                }
            )
        return sorted(rows, key=lambda row: row[f"{key_name}_mean"] or 0.0, reverse=True)[:top_n]

    def ranked_axis_errors() -> dict:
        axis_names = ["x", "y", "z"]
        ranked_by_axis = {axis_name: [] for axis_name in axis_names}
        for body_name, vectors in pos_axis_by_body.items():
            arr = np.asarray(vectors, dtype=np.float64)
            if arr.size == 0:
                continue
            for axis_index, axis_name in enumerate(axis_names):
                signed_values = arr[:, axis_index]
                abs_values = np.abs(signed_values)
                ranked_by_axis[axis_name].append(
                    {
                        "body": body_name,
                        "mean_abs_error_m": round(float(np.mean(abs_values)), 6),
                        "mean_signed_error_m": round(float(np.mean(signed_values)), 6),
                        "p95_abs_error_m": round(float(np.percentile(abs_values, 95)), 6),
                        "max_abs_error_m": round(float(np.max(abs_values)), 6),
                    }
                )
        for axis_name in axis_names:
            ranked_by_axis[axis_name] = sorted(
                ranked_by_axis[axis_name],
                key=lambda row: row["mean_abs_error_m"],
                reverse=True,
            )[:top_n]
        return ranked_by_axis

    def ranked_pre_post_position_improvement() -> list[dict]:
        rows = []
        for body_name in sorted(set(pre_pos_by_body) | set(post_pos_by_body)):
            pre_stats = _summary(pre_pos_by_body.get(body_name, []))
            post_stats = _summary(post_pos_by_body.get(body_name, []))
            pre_mean = pre_stats["mean"]
            post_mean = post_stats["mean"]
            improvement = None
            if pre_mean is not None and post_mean is not None:
                improvement = round(float(pre_mean - post_mean), 6)
            rows.append(
                {
                    "body": body_name,
                    "pre_position_error_m_mean": pre_mean,
                    "post_position_error_m_mean": post_mean,
                    "improvement_m_mean": improvement,
                }
            )
        return sorted(rows, key=lambda row: row["improvement_m_mean"] or 0.0)[:top_n]

    def ranked_joint_limit_margins() -> list[dict]:
        rows = []
        for joint_name, margins in limit_margin_by_joint.items():
            arr = np.asarray(margins, dtype=np.float64)
            if arr.size == 0:
                continue
            rows.append(
                {
                    "joint": joint_name,
                    "min_margin_rad": round(float(np.min(arr)), 6),
                    "mean_margin_rad": round(float(np.mean(arr)), 6),
                }
            )
        return sorted(rows, key=lambda row: row["min_margin_rad"])[:top_n]

    return {
        "frames": int(len(records)),
        "final_error1": _summary(final_error1),
        "final_error2": _summary(final_error2),
        "top_position_errors": ranked(pos_by_body, "position_error_m"),
        "top_position_axis_errors": ranked_axis_errors(),
        "top_orientation_errors": ranked(ori_by_body, "orientation_error_deg"),
        "worst_pre_post_position_improvements": ranked_pre_post_position_improvement(),
        "closest_joint_limits": ranked_joint_limit_margins(),
    }


def read_debug_records(jsonl_path: str | Path) -> list[dict]:
    records = []
    with Path(jsonl_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize GMR debug jsonl task errors.")
    parser.add_argument("--debug_log", required=True, type=str)
    parser.add_argument("--json_out", default=None, type=str)
    parser.add_argument("--top_n", default=8, type=int)
    args = parser.parse_args()

    summary = summarize_debug_records(read_debug_records(args.debug_log), top_n=args.top_n)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
