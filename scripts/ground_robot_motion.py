import argparse
import os
import pathlib
import pickle
import sys

HERE = pathlib.Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    # 允许从仓库根目录直接运行脚本，而不要求先安装成 site-package。
    sys.path.insert(0, str(HERE))

from general_motion_retargeting.motion_grounding import (
    GROUNDING_MODES,
    align_motion_root_to_ground,
    save_grounding_diagnostics_plot,
)
from general_motion_retargeting.params import ROBOT_XML_DICT


def main() -> None:
    # 这个脚本是“离线修复现有 pkl”的入口。
    # 它不重新做 retarget，只在已有 root/dof 轨迹之上做 grounding 后处理。
    parser = argparse.ArgumentParser(description="Ground an exported robot motion PKL using actual support geoms.")
    parser.add_argument("--input", required=True, help="Input PKL path.")
    parser.add_argument("--output", default=None, help="Output PKL path. Defaults to <input>_grounded.pkl.")
    parser.add_argument("--robot", default=None, choices=sorted(ROBOT_XML_DICT.keys()), help="Robot name from ROBOT_XML_DICT.")
    parser.add_argument("--robot_xml", default=None, help="Explicit MuJoCo XML path. Overrides --robot.")
    parser.add_argument("--clearance", type=float, default=0.002, help="Target minimum support clearance above ground in meters.")
    parser.add_argument(
        "--mode",
        choices=list(GROUNDING_MODES),
        default="per_frame",
        help="How to apply vertical correction.",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=9,
        help="Moving-average window for --mode smooth_per_frame.",
    )
    parser.add_argument(
        "--smooth_contact_threshold",
        type=float,
        default=0.04,
        help="Only smooth frames whose support min-z is within this height above ground.",
    )
    parser.add_argument(
        "--max_shift_step",
        type=float,
        default=None,
        help="Maximum adjacent-frame root-z correction step in meters for --mode contact_lowfreq.",
    )
    parser.add_argument(
        "--plot_path",
        default=None,
        help="Optional PNG path for before/after root_z, support_min_z, and applied_shift curves.",
    )
    args = parser.parse_args()

    if args.robot_xml is None and args.robot is None:
        raise ValueError("Either --robot_xml or --robot must be provided.")

    # 优先允许用户显式指定 xml，这样可以拿它去测自定义模型或实验分支里的 MJCF。
    model_path = args.robot_xml or str(ROBOT_XML_DICT[args.robot])
    output_path = args.output
    if output_path is None:
        input_path = pathlib.Path(args.input)
        output_path = str(input_path.with_name(f"{input_path.stem}_grounded{input_path.suffix}"))

    with open(args.input, "rb") as f:
        motion_data = pickle.load(f)

    # 这里不会原地覆盖输入文件，除非用户自己把 output 指到原路径。
    grounded_motion, stats = align_motion_root_to_ground(
        motion_data=motion_data,
        model_or_path=model_path,
        clearance=args.clearance,
        mode=args.mode,
        inplace=False,
        smooth_window=args.smooth_window,
        smooth_contact_threshold=args.smooth_contact_threshold,
        max_shift_step=args.max_shift_step,
        return_diagnostics=args.plot_path is not None,
    )
    diagnostics = stats.pop("diagnostics", None)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(grounded_motion, f)

    print(f"Saved grounded motion to {output_path}")
    if args.plot_path is not None:
        if diagnostics is None:
            raise RuntimeError("Grounding diagnostics were not returned; cannot write --plot_path.")
        save_grounding_diagnostics_plot(
            plot_path=args.plot_path,
            diagnostics=diagnostics,
            title=f"{pathlib.Path(args.input).name} -> {args.mode}",
        )
        print(f"Saved grounding plot to {args.plot_path}")

    # 把关键统计直接打印出来，方便判断这次修复是不是“轻微对齐”还是“重度纠偏”。
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
