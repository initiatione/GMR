import argparse
import json
from pathlib import Path
import sys

# 这里把仓库根目录显式塞进 `sys.path`，
# 这样脚本既可以在“已 pip install -e .”的环境里跑，也可以直接从源码目录裸跑。
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from general_motion_retargeting.utils.bvh_profile_adapter import build_bvh_comparison_report


def main():
    # 这个脚本的目标很明确：
    # 1. 把“官方标准 BVH”和“当前项目 BVH”的结构差异显式打印出来；
    # 2. 给出 GMR 适配层真正需要关注的兼容点，而不是只盯着表面上的关节名称。
    parser = argparse.ArgumentParser(description="Compare an official BVH profile against a project BVH profile.")
    parser.add_argument(
        "--reference_bvh",
        type=str,
        required=True,
        help="官方标准 BVH 文件路径，例如官方 LAFAN1 的 dance1_subject2.bvh。",
    )
    parser.add_argument(
        "--target_bvh",
        type=str,
        required=True,
        help="当前项目里待适配的 BVH 文件路径，例如 hit_data/540huixuantitui_001.bvh。",
    )
    parser.add_argument(
        "--json_out",
        type=str,
        default=None,
        help="可选：把完整对比报告额外保存成 JSON，方便后续复盘或自动化分析。",
    )
    args = parser.parse_args()

    # 生成结构化差异报告。
    report = build_bvh_comparison_report(args.reference_bvh, args.target_bvh)

    # 先打印 profile 概览，快速判断双方是否属于同一类骨架。
    print("[INFO] Reference profile:")
    print(json.dumps(report["reference"], ensure_ascii=False, indent=2))
    print()

    print("[INFO] Target profile:")
    print(json.dumps(report["target"], ensure_ascii=False, indent=2))
    print()

    # 再打印最关键的结构差异：共享关节里的旋转顺序是否一致。
    print("[INFO] Rotation order differences on shared joints:")
    print(json.dumps(report["rotation_order_differences"], ensure_ascii=False, indent=2))
    print()

    # 最后直接打印适配建议，方便把报告和 GMR 改造动作对上。
    print("[INFO] Suggested GMR adapter hints:")
    print(json.dumps(report["gmr_adapter_hints"], ensure_ascii=False, indent=2))

    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] Report saved to {output_path}")


if __name__ == "__main__":
    main()
