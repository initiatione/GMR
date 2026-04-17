import argparse
import os
import pathlib
import pickle
import sys

HERE = pathlib.Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    # 允许从仓库根目录直接运行脚本，而不要求先安装成 site-package。
    sys.path.insert(0, str(HERE))

from general_motion_retargeting.motion_grounding import align_motion_root_to_ground
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
    parser.add_argument("--mode", choices=["per_frame", "global"], default="per_frame", help="How to apply vertical correction.")
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
    )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(grounded_motion, f)

    print(f"Saved grounded motion to {output_path}")
    # 把关键统计直接打印出来，方便判断这次修复是不是“轻微对齐”还是“重度纠偏”。
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
