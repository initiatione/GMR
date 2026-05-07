from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.params import IK_CONFIG_DICT
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from scripts.summarize_gmr_debug_log import read_debug_records, summarize_debug_records


LOWER_BODIES = {
    "Hips",
    "Spine2",
    "Head",
    "LeftUpLeg",
    "RightUpLeg",
    "LeftLeg",
    "RightLeg",
    "LeftFootMod",
    "RightFootMod",
}

ARM_BODIES = {
    "LeftArm",
    "RightArm",
    "LeftForeArm",
    "RightForeArm",
    "LeftHand",
    "RightHand",
}

def build_variant_config(base_config: dict, variant: str) -> dict:
    config = copy.deepcopy(base_config)

    for table_name in ["ik_match_table1", "ik_match_table2"]:
        for entry in config[table_name].values():
            human_body = entry[0]

            if variant == "position_only_all":
                entry[2] = 0
                continue

            if variant == "no_lower_orientation" and human_body in LOWER_BODIES:
                entry[2] = 0
                continue

            if variant == "low_hand_position_no_lower_orientation":
                if human_body in LOWER_BODIES:
                    entry[2] = 0
                if human_body in {"LeftHand", "RightHand"}:
                    entry[1] = min(entry[1], 2)
                if human_body in {"LeftForeArm", "RightForeArm"}:
                    entry[1] = min(entry[1], 3)
                continue

            if variant == "lower_position_only_no_arms":
                if human_body in LOWER_BODIES:
                    entry[2] = 0
                if human_body in ARM_BODIES:
                    entry[1] = 0
                    entry[2] = 0
                continue

            if variant not in {
                "position_only_all",
                "no_lower_orientation",
                "low_hand_position_no_lower_orientation",
                "lower_position_only_no_arms",
            }:
                raise ValueError(f"Unknown variant: {variant}")

    return config


def run_variant(
    bvh_file: Path,
    base_config_path: Path,
    variant: str,
    frame_count: int,
    debug_dir: Path,
) -> dict:
    frames, human_height = load_bvh_file(str(bvh_file), format="lafan1")
    frames = frames[:frame_count]
    base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
    variant_config_path = debug_dir / f"temp_ik_{variant}.json"
    variant_config_path.write_text(
        json.dumps(build_variant_config(base_config, variant), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    original_config_path = IK_CONFIG_DICT["bvh_human_robot_hit"]["t800"]
    IK_CONFIG_DICT["bvh_human_robot_hit"]["t800"] = variant_config_path
    debug_log = debug_dir / f"probe_{variant}_{bvh_file.stem}_{frame_count}f.jsonl"

    try:
        retargeter = GMR(
            src_human="bvh_human_robot_hit",
            tgt_robot="t800",
            actual_human_height=human_height,
            debug_log_path=str(debug_log),
            debug_log_every_n=1,
            verbose=False,
        )

        qpos_list = []
        for frame_index, frame in enumerate(frames):
            qpos_list.append(retargeter.retarget(frame, frame_index=frame_index))
    finally:
        IK_CONFIG_DICT["bvh_human_robot_hit"]["t800"] = original_config_path

    summary = summarize_debug_records(read_debug_records(debug_log), top_n=8)
    summary["qpos_root_z_minmax"] = [
        round(float(min(qpos[2] for qpos in qpos_list)), 6),
        round(float(max(qpos[2] for qpos in qpos_list)), 6),
    ]
    summary["qpos_abs_max"] = round(float(np.max(np.abs(np.asarray(qpos_list)))), 6)
    summary["variant_config_path"] = str(variant_config_path)
    summary["debug_log"] = str(debug_log)
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Probe official BVH retargeting with temporary IK weight variants.")
    parser.add_argument("--bvh_file", required=True, type=Path)
    parser.add_argument("--base_config", required=True, type=Path)
    parser.add_argument("--frame_count", default=20, type=int)
    parser.add_argument("--debug_dir", required=True, type=Path)
    parser.add_argument("--output_json", required=True, type=Path)
    args = parser.parse_args()

    args.debug_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for variant in [
        "position_only_all",
        "no_lower_orientation",
        "low_hand_position_no_lower_orientation",
        "lower_position_only_no_arms",
    ]:
        results[variant] = run_variant(
            bvh_file=args.bvh_file,
            base_config_path=args.base_config,
            variant=variant,
            frame_count=args.frame_count,
            debug_dir=args.debug_dir,
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
