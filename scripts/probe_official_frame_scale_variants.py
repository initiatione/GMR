from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.motion_retarget_options import (
    calibrate_human_robot_hit_frames,
    resolve_actual_human_height,
)
from general_motion_retargeting.params import IK_CONFIG_DICT
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from scripts.bvh_to_robot import build_motion_data_from_qpos_list
from scripts.summarize_gmr_debug_log import read_debug_records, summarize_debug_records


ARM_BODIES = {
    "LeftArm",
    "RightArm",
    "LeftForeArm",
    "RightForeArm",
    "LeftHand",
    "RightHand",
}


def prepare_official_probe_frames(bvh_file: Path, frame_count: int) -> tuple[list[dict], float | None, float]:
    frames, loader_height = load_bvh_file(str(bvh_file), format="lafan1")
    frames = calibrate_human_robot_hit_frames(frames[:frame_count])
    actual_human_height = resolve_actual_human_height(loader_height, "human_robot_hit")
    return frames, actual_human_height, loader_height


def build_variant_config(base_config: dict, variant: str) -> dict:
    config = copy.deepcopy(base_config)

    if variant in {
        "no_height_no_arm_rot",
        "height175_no_arm_rot",
        "loader_height_no_arm_rot",
        "no_height_position_only_all",
    }:
        for table_name in ["ik_match_table1", "ik_match_table2"]:
            for entry in config[table_name].values():
                if entry[0] in ARM_BODIES:
                    entry[2] = 0
                if variant == "no_height_position_only_all":
                    entry[2] = 0

    if variant == "no_height_no_table1_arm_rot":
        for entry in config["ik_match_table1"].values():
            if entry[0] in ARM_BODIES:
                entry[2] = 0

    if variant == "no_height_zero_upper_offsets":
        for table_name in ["ik_match_table1", "ik_match_table2"]:
            for entry in config[table_name].values():
                if entry[0] in ARM_BODIES:
                    entry[2] = 0
                    entry[3] = [0.0, 0.0, 0.0]

    if variant == "no_height_arm_chain_priority":
        for entry in config["ik_match_table2"].values():
            if entry[0] in {"LeftArm", "RightArm"}:
                entry[1] = 20
            if entry[0] in {"LeftForeArm", "RightForeArm"}:
                entry[1] = 20
            if entry[0] in {"LeftHand", "RightHand"}:
                entry[1] = 3

    return config


def summarize_first_record(record: dict) -> dict:
    task_table = record["task_table2"]
    bodies = ["Hips", "Head", "LeftFootMod", "RightFootMod", "LeftHand", "RightHand"]
    return {
        body_name: {
            "target_pos": task_table[body_name]["target_pos"],
            "current_pos": task_table[body_name]["current_pos"],
            "world_position_error_norm_m": task_table[body_name]["world_position_error_norm_m"],
            "world_orientation_error_deg": task_table[body_name]["world_orientation_error_deg"],
        }
        for body_name in bodies
        if body_name in task_table
    }


def run_case(
    *,
    case_name: str,
    config: dict,
    actual_human_height: float | None,
    frames: list[dict],
    robot: str,
    motion_fps: int,
    output_dir: Path,
) -> dict:
    config_path = output_dir / f"{case_name}.json"
    debug_log_path = output_dir / f"{case_name}.jsonl"
    pkl_path = output_dir / f"{case_name}.pkl"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    original_config_path = IK_CONFIG_DICT["bvh_human_robot_hit"][robot]
    IK_CONFIG_DICT["bvh_human_robot_hit"][robot] = config_path

    try:
        retargeter = GMR(
            src_human="bvh_human_robot_hit",
            tgt_robot=robot,
            actual_human_height=actual_human_height,
            debug_log_path=str(debug_log_path),
            debug_log_every_n=1,
            verbose=False,
            ik_safety_break=False,
        )

        qpos_list = []
        for frame_index, frame in enumerate(frames):
            qpos_list.append(retargeter.retarget(frame, frame_index=frame_index))

        pkl_path.write_bytes(pickle.dumps(build_motion_data_from_qpos_list(qpos_list, motion_fps)))
    finally:
        IK_CONFIG_DICT["bvh_human_robot_hit"][robot] = original_config_path

    qpos = np.asarray(qpos_list, dtype=np.float64)
    summary = summarize_debug_records(read_debug_records(debug_log_path), top_n=8)
    first_record = read_debug_records(debug_log_path)[0]
    return {
        "actual_human_height": actual_human_height,
        "root_z_minmax": [
            round(float(np.min(qpos[:, 2])), 6),
            round(float(np.max(qpos[:, 2])), 6),
        ],
        "dof_mean_first12": np.round(np.mean(qpos[:, 7:19], axis=0), 4).tolist(),
        "final_error1": summary["final_error1"],
        "final_error2": summary["final_error2"],
        "top_position_errors": summary["top_position_errors"][:8],
        "top_orientation_errors": summary["top_orientation_errors"][:8],
        "first_record_focus": summarize_first_record(first_record),
        "config_path": str(config_path),
        "debug_log_path": str(debug_log_path),
        "pkl_path": str(pkl_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe official BVH T800 retargeting frame/height variants without editing the checked-in IK config."
    )
    parser.add_argument("--bvh_file", required=True, type=Path)
    parser.add_argument("--base_config", required=True, type=Path)
    parser.add_argument("--robot", default="t800")
    parser.add_argument("--frame_count", default=20, type=int)
    parser.add_argument("--motion_fps", default=120, type=int)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--summary_json", required=True, type=Path)
    args = parser.parse_args()

    frames, default_actual_human_height, loader_height = prepare_official_probe_frames(
        args.bvh_file,
        args.frame_count,
    )
    base_config = json.loads(args.base_config.read_text(encoding="utf-8"))

    cases = [
        ("current_loader_height", base_config, loader_height),
        ("no_height_current_config", base_config, default_actual_human_height),
        ("height175_current_config", base_config, 1.75),
        ("loader_height_no_arm_rot", build_variant_config(base_config, "loader_height_no_arm_rot"), loader_height),
        ("no_height_no_arm_rot", build_variant_config(base_config, "no_height_no_arm_rot"), default_actual_human_height),
        (
            "no_height_zero_upper_offsets",
            build_variant_config(base_config, "no_height_zero_upper_offsets"),
            default_actual_human_height,
        ),
        (
            "no_height_arm_chain_priority",
            build_variant_config(base_config, "no_height_arm_chain_priority"),
            default_actual_human_height,
        ),
        ("height175_no_arm_rot", build_variant_config(base_config, "height175_no_arm_rot"), 1.75),
        (
            "no_height_no_table1_arm_rot",
            build_variant_config(base_config, "no_height_no_table1_arm_rot"),
            default_actual_human_height,
        ),
        (
            "no_height_position_only_all",
            build_variant_config(base_config, "no_height_position_only_all"),
            default_actual_human_height,
        ),
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        case_name: run_case(
            case_name=case_name,
            config=config,
            actual_human_height=actual_human_height,
            frames=frames,
            robot=args.robot,
            motion_fps=args.motion_fps,
            output_dir=args.output_dir,
        )
        for case_name, config, actual_human_height in cases
    }

    output = {
        "bvh_file": str(args.bvh_file),
        "base_config": str(args.base_config),
        "frame_count": len(frames),
        "loader_height": loader_height,
        "cases": results,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
