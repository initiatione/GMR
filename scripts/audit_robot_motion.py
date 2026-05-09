from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from general_motion_retargeting.motion_quality import (
    MotionQualityConfig,
    audit_motion_quality,
    build_review_commands,
    load_motion_pkl,
    write_json_report,
)
from general_motion_retargeting.params import ROBOT_XML_DICT


def _actuated_joint_info(robot: str) -> tuple[list[str], np.ndarray, np.ndarray, Path | None]:
    xml_path = ROBOT_XML_DICT.get(robot)
    if xml_path is None:
        return [], np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64), None

    import mujoco as mj

    model = mj.MjModel.from_xml_path(str(xml_path))
    names: list[str] = []
    lower: list[float] = []
    upper: list[float] = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id][0])
        joint_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id) or f"joint_{joint_id}"
        names.append(joint_name)
        if int(model.jnt_limited[joint_id]):
            lo, hi = model.jnt_range[joint_id]
        else:
            lo, hi = -np.inf, np.inf
        lower.append(float(lo))
        upper.append(float(hi))
    return names, np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64), Path(xml_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit source-side GMR robot motion PKL quality.")
    parser.add_argument("--robot", default="t800", help="Robot key registered in general_motion_retargeting.params.")
    parser.add_argument("--input", required=True, help="Input GMR robot motion PKL.")
    parser.add_argument("--output_json", default=None, help="Optional path to write the full JSON report.")
    parser.add_argument("--jump_threshold_rad", type=float, default=0.7)
    parser.add_argument("--velocity_threshold_rad_s", type=float, default=12.0)
    parser.add_argument("--limit_margin_rad", type=float, default=0.03)
    parser.add_argument("--floor_clearance", type=float, default=0.0)
    parser.add_argument("--review_padding_frames", type=int, default=12)
    parser.add_argument("--skip_model_checks", action="store_true", help="Skip MuJoCo model-based floor/collision checks.")
    parser.add_argument("--collision_stride", type=int, default=10)
    parser.add_argument("--max_collision_pairs", type=int, default=30)
    parser.add_argument("--print_review_commands", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --output_json file.")
    args = parser.parse_args()

    motion_path = Path(args.input)
    motion_data = load_motion_pkl(motion_path)
    joint_names, lower, upper, xml_path = _actuated_joint_info(args.robot)
    if args.skip_model_checks:
        xml_path = None

    config = MotionQualityConfig(
        robot=args.robot,
        joint_names=joint_names or None,
        joint_lower_limits=lower if lower.size else None,
        joint_upper_limits=upper if upper.size else None,
        jump_threshold_rad=args.jump_threshold_rad,
        velocity_threshold_rad_s=args.velocity_threshold_rad_s,
        limit_margin_rad=args.limit_margin_rad,
        floor_clearance=args.floor_clearance,
        review_padding_frames=args.review_padding_frames,
        model_path=xml_path,
        collision_stride=args.collision_stride,
        max_collision_pairs=args.max_collision_pairs,
    )
    report = audit_motion_quality(motion_data, config=config, motion_path=motion_path)
    report["review_commands"] = build_review_commands(report)

    print(
        "[motion_quality] "
        f"frames={report['schema']['frame_count']} "
        f"fps={report['schema']['fps']} "
        f"dof={report['schema']['dof_count']} "
        f"qpos_jumps={report['summary']['qpos_jump_count']} "
        f"velocity_spikes={report['summary']['velocity_spike_count']} "
        f"limit_pressure={report['summary']['limit_pressure_count']} "
        f"floor_anomalies={report['summary']['floor_anomaly_count']} "
        f"candidate_collisions={report['summary']['candidate_collision_count']}"
    )
    if report["review_windows"]:
        print("[motion_quality] review windows:")
        for window in report["review_windows"][:20]:
            print(
                "  "
                f"{window['start_frame']}:{window['end_frame']} "
                f"({window['start_sec']:.3f}s-{window['end_sec']:.3f}s) "
                f"reasons={','.join(window['reasons'])}"
            )
    if args.print_review_commands:
        for command in report["review_commands"]:
            print(command)
    if args.output_json:
        write_json_report(report, args.output_json, overwrite=args.overwrite)
        print(f"[motion_quality] wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
