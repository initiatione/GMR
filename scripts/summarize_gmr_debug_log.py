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
    ori_by_body: dict[str, list[float]] = {}
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
            if ori is not None:
                ori_by_body.setdefault(body_name, []).append(float(ori))

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

    return {
        "frames": int(len(records)),
        "final_error1": _summary(final_error1),
        "final_error2": _summary(final_error2),
        "top_position_errors": ranked(pos_by_body, "position_error_m"),
        "top_orientation_errors": ranked(ori_by_body, "orientation_error_deg"),
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
