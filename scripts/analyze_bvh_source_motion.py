from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from general_motion_retargeting.utils.lafan1 import load_bvh_file
from general_motion_retargeting.utils.bvh_profile_adapter import inspect_bvh_profile


DEFAULT_BODIES = [
    "Hips",
    "Spine2",
    "Head",
    "LeftUpLeg",
    "RightUpLeg",
    "LeftLeg",
    "RightLeg",
    "LeftFootMod",
    "RightFootMod",
    "LeftArm",
    "RightArm",
    "LeftForeArm",
    "RightForeArm",
    "LeftHand",
    "RightHand",
]


def _round_float(value: float) -> float:
    return round(float(value), 6)


def _body_summary(frames: list[dict], body_name: str) -> dict | None:
    positions = []
    quat_norm_errors = []
    for frame in frames:
        if body_name not in frame:
            continue
        pos, quat = frame[body_name]
        positions.append(np.asarray(pos, dtype=np.float64))
        quat_norm_errors.append(abs(np.linalg.norm(np.asarray(quat, dtype=np.float64)) - 1.0))

    if not positions:
        return None

    pos_arr = np.asarray(positions, dtype=np.float64)
    return {
        "frames_present": int(pos_arr.shape[0]),
        "x_min": _round_float(np.min(pos_arr[:, 0])),
        "x_max": _round_float(np.max(pos_arr[:, 0])),
        "y_min": _round_float(np.min(pos_arr[:, 1])),
        "y_max": _round_float(np.max(pos_arr[:, 1])),
        "z_min": _round_float(np.min(pos_arr[:, 2])),
        "z_max": _round_float(np.max(pos_arr[:, 2])),
        "travel_xy": _round_float(np.linalg.norm(pos_arr[-1, :2] - pos_arr[0, :2])),
        "quat_norm_error_max": _round_float(max(quat_norm_errors)),
    }


def build_source_motion_report(
    frames: list[dict],
    fps: float,
    source_path: str,
    bodies: list[str] | None = None,
    profile: dict | None = None,
    human_height: float | None = None,
) -> dict:
    selected_bodies = DEFAULT_BODIES if bodies is None else bodies
    body_reports = {}
    quat_errors = []

    for body_name in selected_bodies:
        body_report = _body_summary(frames, body_name)
        if body_report is None:
            continue
        body_reports[body_name] = body_report
        quat_errors.append(body_report["quat_norm_error_max"])

    foot_z_values = []
    for foot_name in ["LeftFootMod", "RightFootMod", "LeftToe", "RightToe", "LeftToeBase", "RightToeBase"]:
        if foot_name not in body_reports:
            continue
        foot_z_values.append(body_reports[foot_name]["z_min"])

    return {
        "source_path": str(source_path),
        "frame_count": int(len(frames)),
        "fps": float(fps),
        "human_height": None if human_height is None else _round_float(human_height),
        "profile": profile,
        "bodies": body_reports,
        "foot_z_min": None if not foot_z_values else _round_float(min(foot_z_values)),
        "quaternion_norm_error_max": 0.0 if not quat_errors else _round_float(max(quat_errors)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize loaded BVH source motion before GMR IK.")
    parser.add_argument("--bvh_file", required=True, type=str)
    parser.add_argument("--format", choices=["lafan1", "nokov"], default="lafan1")
    parser.add_argument("--json_out", default=None, type=str)
    args = parser.parse_args()

    frames, human_height = load_bvh_file(args.bvh_file, format=args.format)
    profile = inspect_bvh_profile(args.bvh_file)
    report = build_source_motion_report(
        frames=frames,
        fps=float(profile["fps"]),
        source_path=args.bvh_file,
        profile=profile,
        human_height=human_height,
    )

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
