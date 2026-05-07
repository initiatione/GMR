from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from general_motion_retargeting.utils.bvh_profile_adapter import (
    adapt_frame_for_gmr,
    inspect_bvh_profile,
    read_bvh_with_joint_orders,
)
import general_motion_retargeting.utils.lafan_vendor.utils as lafan_utils
from scripts.summarize_gmr_debug_log import read_debug_records, summarize_debug_records


def load_probe_frames(bvh_file: Path, unit_divisor: float | None, frame_count: int) -> list[dict]:
    if unit_divisor is None:
        frames, _ = load_bvh_file(str(bvh_file), format="lafan1")
        return frames[:frame_count]

    profile = inspect_bvh_profile(bvh_file)
    data = read_bvh_with_joint_orders(bvh_file, start=0, end=frame_count)
    global_data = lafan_utils.quat_fk(data.quats, data.pos, data.parents)

    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)

    frames = []
    for frame_index in range(data.pos.shape[0]):
        frame = {}
        for joint_index, bone in enumerate(data.bones):
            orientation = lafan_utils.quat_mul(rotation_quat, global_data[0][frame_index, joint_index])
            position = global_data[1][frame_index, joint_index] @ rotation_matrix.T / unit_divisor
            frame[bone] = [position, orientation]

        frame = adapt_frame_for_gmr(frame, profile["detected_profile"])
        frame["LeftFootMod"] = [frame["LeftFoot"][0], frame["LeftToe"][1]]
        frame["RightFootMod"] = [frame["RightFoot"][0], frame["RightToe"][1]]
        frames.append(frame)

    return frames


def run_probe_case(
    bvh_file: Path,
    robot: str,
    unit_divisor: float | None,
    actual_human_height: float | None,
    frame_count: int,
    debug_log: Path,
) -> dict:
    frames = load_probe_frames(bvh_file, unit_divisor=unit_divisor, frame_count=frame_count)
    retargeter = GMR(
        src_human="bvh_human_robot_hit",
        tgt_robot=robot,
        actual_human_height=actual_human_height,
        debug_log_path=str(debug_log),
        debug_log_every_n=1,
        verbose=False,
    )

    qpos_list = []
    for frame_index, frame in enumerate(frames):
        qpos_list.append(retargeter.retarget(frame, frame_index=frame_index))

    summary = summarize_debug_records(read_debug_records(debug_log), top_n=5)
    summary["unit_divisor"] = unit_divisor
    summary["actual_human_height"] = actual_human_height
    summary["qpos_root_z_minmax"] = [
        round(float(np.min([qpos[2] for qpos in qpos_list])), 6),
        round(float(np.max([qpos[2] for qpos in qpos_list])), 6),
    ]

    first_record = read_debug_records(debug_log)[0]
    summary["first_targets"] = {
        body_name: first_record["task_table1"][body_name]["target_pos"]
        for body_name in [
            "Hips",
            "Spine2",
            "Head",
            "LeftFootMod",
            "RightFootMod",
            "LeftHand",
            "RightHand",
        ]
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe official BVH unit scale choices through short-window GMR retargeting.")
    parser.add_argument("--bvh_file", required=True, type=Path)
    parser.add_argument("--robot", default="t800")
    parser.add_argument("--frame_count", default=20, type=int)
    parser.add_argument("--output_json", required=True, type=Path)
    parser.add_argument("--debug_dir", required=True, type=Path)
    args = parser.parse_args()

    cases = [
        ("auto_loader", None, 1.75),
        ("cm_keep175", 100.0, 1.75),
        ("inch_keep175", 39.37, 1.75),
        ("inch_no_height", 39.37, None),
        ("inch_est145", 39.37, 1.45),
    ]

    args.debug_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for case_name, unit_divisor, actual_human_height in cases:
        debug_log = args.debug_dir / f"probe_{case_name}_{args.bvh_file.stem}_{args.frame_count}f.jsonl"
        results[case_name] = run_probe_case(
            bvh_file=args.bvh_file,
            robot=args.robot,
            unit_divisor=unit_divisor,
            actual_human_height=actual_human_height,
            frame_count=args.frame_count,
            debug_log=debug_log,
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
