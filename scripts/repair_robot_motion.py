from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from general_motion_retargeting.motion_quality import (
    apply_repair_specs,
    load_motion_pkl,
    load_repair_specs,
    save_motion_pkl,
    write_json_report,
)
from general_motion_retargeting.params import ROBOT_XML_DICT


def _joint_names_from_robot(robot: str) -> list[str]:
    xml_path = ROBOT_XML_DICT.get(robot)
    if xml_path is None:
        return []
    import mujoco as mj

    model = mj.MjModel.from_xml_path(str(xml_path))
    names = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id][0])
        names.append(mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id) or f"joint_{joint_id}")
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply bounded source-side repairs to a GMR robot motion PKL.")
    parser.add_argument("--input", required=True, help="Input source motion PKL.")
    parser.add_argument("--output", required=True, help="Output repaired motion PKL.")
    parser.add_argument("--repair_spec", required=True, help="JSON file containing a top-level repairs list.")
    parser.add_argument("--robot", default="t800", help="Robot key used to infer joint order when --joint_names is absent.")
    parser.add_argument(
        "--joint_names",
        nargs="*",
        default=None,
        help="Explicit joint names matching dof_pos columns. Overrides --robot inference.",
    )
    parser.add_argument("--report_json", default=None, help="Optional path to write before/after repair metrics.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing output/report files.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Refusing to overwrite input file. Choose a different --output path.")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}. Pass --overwrite to replace it.")

    motion = load_motion_pkl(input_path)
    repair_specs = load_repair_specs(args.repair_spec)
    joint_names = list(args.joint_names) if args.joint_names else _joint_names_from_robot(args.robot)
    if not joint_names:
        raise ValueError("Could not infer joint names. Pass --joint_names explicitly.")

    repaired, report = apply_repair_specs(motion, repair_specs, joint_names=joint_names, inplace=False)
    save_motion_pkl(repaired, output_path, overwrite=args.overwrite)
    print(
        "[motion_repair] "
        f"repairs={len(report['repairs'])} "
        f"frames={report['frame_count']} "
        f"output={output_path}"
    )
    if args.report_json:
        write_json_report(report, args.report_json, overwrite=args.overwrite)
        print(f"[motion_repair] wrote {args.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
