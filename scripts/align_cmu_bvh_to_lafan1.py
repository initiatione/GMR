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

import general_motion_retargeting.utils.lafan_vendor.utils as utils
from general_motion_retargeting.utils.bvh_profile_adapter import _parse_bvh_layout, inspect_bvh_profile, read_bvh_with_joint_orders
from general_motion_retargeting.utils.lafan1 import load_bvh_file


# 目标是把 CMU/ASF-AMC 导出的 BVH 重建到“标准 LAFAN1 模板骨架”上，
# 这样下游 `bvh_lafan1 -> t800` 的 IK 配置就能继续复用既有语义：
# - 相同的关键关节命名；
# - 相同的模板 offset/坐标系约定；
# - 仅保留动作本身，而不是沿用 CMU 导出 BVH 的骨架局部坐标定义。
SOURCE_TO_TARGET_JOINT_MAP = {
    'Hips': 'Hips',
    'LeftUpLeg': 'LeftUpLeg',
    'LeftLeg': 'LeftLeg',
    'LeftFoot': 'LeftFoot',
    'LeftToe': 'LeftToeBase',
    'RightUpLeg': 'RightUpLeg',
    'RightLeg': 'RightLeg',
    'RightFoot': 'RightFoot',
    'RightToe': 'RightToeBase',
    'Spine': 'LowerBack',
    'Spine1': 'Spine',
    'Spine2': 'Spine1',
    'Neck': 'Neck1',
    'Head': 'Head',
    'LeftShoulder': 'LeftShoulder',
    'LeftArm': 'LeftArm',
    'LeftForeArm': 'LeftForeArm',
    'LeftHand': 'LeftHand',
    'RightShoulder': 'RightShoulder',
    'RightArm': 'RightArm',
    'RightForeArm': 'RightForeArm',
    'RightHand': 'RightHand',
}


def wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    return quat_wxyz[[1, 2, 3, 0]]



def xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
    return quat_xyzw[[3, 0, 1, 2]]



def quaternion_multiply_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lhs_rot = R.from_quat(wxyz_to_xyzw(lhs))
    rhs_rot = R.from_quat(wxyz_to_xyzw(rhs))
    return xyzw_to_wxyz((lhs_rot * rhs_rot).as_quat())



def quaternion_inverse_wxyz(quat: np.ndarray) -> np.ndarray:
    quat_rot = R.from_quat(wxyz_to_xyzw(quat))
    return xyzw_to_wxyz(quat_rot.inv().as_quat())



def compute_global_transforms(anim):
    global_quats, global_positions = utils.quat_fk(anim.quats, anim.pos, anim.parents)
    return np.asarray(global_quats, dtype=np.float64), np.asarray(global_positions, dtype=np.float64)



def estimate_height_from_global_positions(global_positions, joint_name_to_index, head_name, left_foot_name, right_foot_name):
    head_index = joint_name_to_index[head_name]
    left_foot_index = joint_name_to_index[left_foot_name]
    right_foot_index = joint_name_to_index[right_foot_name]

    head_heights = global_positions[:, head_index, 1]
    foot_heights = np.minimum(global_positions[:, left_foot_index, 1], global_positions[:, right_foot_index, 1])
    return float(np.percentile(head_heights, 95) - np.percentile(foot_heights, 5))



def extract_hierarchy_text(reference_bvh_path: Path) -> str:
    text = reference_bvh_path.read_text(encoding='utf-8', errors='ignore')
    motion_index = text.find('MOTION')
    if motion_index < 0:
        raise ValueError(f'Invalid reference BVH without MOTION section: {reference_bvh_path}')
    hierarchy_text = text[:motion_index].rstrip() + '\n'
    return hierarchy_text



def build_template_channel_orders(layout: dict) -> dict[str, str]:
    joint_orders = {}
    for joint_name, joint_channels in zip(layout['joint_names'], layout['joint_channels']):
        rotation_order = ''.join(channel[0].lower() for channel in joint_channels if channel.endswith('rotation'))
        joint_orders[joint_name] = rotation_order
    return joint_orders



def compute_reference_alignment_rotation(reference_global_quats, reference_joint_indices, source_global_quats, source_joint_indices):
    reference_root_quat = reference_global_quats[0, reference_joint_indices['Hips']]
    source_root_quat = source_global_quats[0, source_joint_indices['Hips']]
    alignment_rot = R.from_quat(wxyz_to_xyzw(reference_root_quat)) * R.from_quat(wxyz_to_xyzw(source_root_quat)).inv()
    return alignment_rot



def reconstruct_template_motion(
    reference_layout,
    reference_global_positions,
    source_layout,
    source_global_positions,
    source_global_quats,
    alignment_rotation,
    displacement_scale,
):
    target_joint_names = reference_layout['joint_names']
    target_parents = reference_layout['joint_parents']
    target_joint_orders = build_template_channel_orders(reference_layout)

    reference_joint_indices = {name: idx for idx, name in enumerate(reference_layout['joint_names'])}
    source_joint_indices = {name: idx for idx, name in enumerate(source_layout['joint_names'])}

    source_root_initial = source_global_positions[0, source_joint_indices['Hips']]
    target_root_initial = reference_global_positions[0, reference_joint_indices['Hips']]

    frame_lines = []

    for frame_index in range(source_global_positions.shape[0]):
        target_global_quats = {}
        for target_joint_name in target_joint_names:
            source_joint_name = SOURCE_TO_TARGET_JOINT_MAP.get(target_joint_name)
            if source_joint_name is None:
                raise KeyError(f'Missing source mapping for target joint: {target_joint_name}')
            source_joint_index = source_joint_indices[source_joint_name]
            source_global_quat = source_global_quats[frame_index, source_joint_index]
            target_global_quat = xyzw_to_wxyz(
                (alignment_rotation * R.from_quat(wxyz_to_xyzw(source_global_quat))).as_quat()
            )
            target_global_quats[target_joint_name] = target_global_quat

        source_root_current = source_global_positions[frame_index, source_joint_indices['Hips']]
        root_displacement = source_root_current - source_root_initial
        target_root_position = target_root_initial + displacement_scale * alignment_rotation.apply(root_displacement)

        frame_values = []
        for joint_index, joint_name in enumerate(target_joint_names):
            joint_order = target_joint_orders[joint_name]
            global_quat = target_global_quats[joint_name]

            if joint_index == 0:
                local_quat = global_quat
                frame_values.extend(float(value) for value in target_root_position)
            else:
                parent_name = target_joint_names[target_parents[joint_index]]
                parent_global_quat = target_global_quats[parent_name]
                local_quat = quaternion_multiply_wxyz(
                    quaternion_inverse_wxyz(parent_global_quat),
                    global_quat,
                )

            # `lafan_vendor.utils.euler_to_quat()` 的 `order='zyx'` 实际对应的是按外旋顺序写回的 BVH 欧拉角。
            # 这里必须使用 SciPy 的大写外旋约定，否则 round-trip 后 root/躯干会整体塌掉。
            euler_deg = R.from_quat(wxyz_to_xyzw(local_quat)).as_euler(joint_order.upper(), degrees=True)
            frame_values.extend(float(value) for value in euler_deg)

        frame_line = ' '.join(f'{value:.6f}' for value in frame_values)
        frame_lines.append(frame_line)

    return frame_lines



def summarize_loaded_bvh(bvh_path: Path) -> dict:
    frames, estimated_height = load_bvh_file(str(bvh_path), format='lafan1')
    if not frames:
        raise ValueError(f'Empty converted BVH: {bvh_path}')

    first_frame = frames[0]
    foot_z = min(first_frame['LeftFootMod'][0][2], first_frame['RightFootMod'][0][2])
    return {
        'estimated_height': estimated_height,
        'first_frame_hips_z': float(first_frame['Hips'][0][2]),
        'first_frame_head_to_foot': float(first_frame['Head'][0][2] - foot_z),
        'first_frame_left_hand': [float(v) for v in first_frame['LeftHand'][0]],
        'first_frame_right_hand': [float(v) for v in first_frame['RightHand'][0]],
    }



def main():
    parser = argparse.ArgumentParser(
        description='Retarget CMU-style BVH onto a standard LAFAN1 BVH template so GMR t800 can consume it.'
    )
    parser.add_argument('--input_bvh', type=str, required=True, help='输入 CMU BVH 文件路径。')
    parser.add_argument('--output_bvh', type=str, required=True, help='输出 LAFAN1 模板化后的 BVH 文件路径。')
    parser.add_argument(
        '--reference_bvh',
        type=str,
        default=str((REPO_ROOT.parent / 'tmp_compare' / 'lafan1' / 'aiming1_subject1.bvh').resolve()),
        help='作为 LAFAN1 模板的参考 BVH。默认使用 tmp_compare/lafan1/aiming1_subject1.bvh。',
    )
    parser.add_argument(
        '--report_json',
        type=str,
        default=None,
        help='可选：输出一份转换报告 JSON。',
    )
    args = parser.parse_args()

    input_path = Path(args.input_bvh)
    output_path = Path(args.output_bvh)
    reference_path = Path(args.reference_bvh)

    input_layout = _parse_bvh_layout(input_path)
    reference_layout = _parse_bvh_layout(reference_path)

    source_anim = read_bvh_with_joint_orders(input_path)
    reference_anim = read_bvh_with_joint_orders(reference_path)

    source_global_quats, source_global_positions = compute_global_transforms(source_anim)
    reference_global_quats, reference_global_positions = compute_global_transforms(reference_anim)

    source_joint_indices = {name: idx for idx, name in enumerate(input_layout['joint_names'])}
    reference_joint_indices = {name: idx for idx, name in enumerate(reference_layout['joint_names'])}

    required_source_joints = set(SOURCE_TO_TARGET_JOINT_MAP.values())
    missing_source_joints = sorted(required_source_joints - set(source_joint_indices.keys()))
    if missing_source_joints:
        raise KeyError(f'Source BVH missing required joints for LAFAN1 conversion: {missing_source_joints}')

    reference_height_raw = estimate_height_from_global_positions(
        reference_global_positions,
        reference_joint_indices,
        'Head',
        'LeftFoot',
        'RightFoot',
    )
    source_height_raw = estimate_height_from_global_positions(
        source_global_positions,
        source_joint_indices,
        'Head',
        'LeftFoot',
        'RightFoot',
    )
    if source_height_raw <= 1e-9:
        raise ValueError(f'Invalid source height estimated from BVH: {source_height_raw}')

    displacement_scale = reference_height_raw / source_height_raw
    alignment_rotation = compute_reference_alignment_rotation(
        reference_global_quats,
        reference_joint_indices,
        source_global_quats,
        source_joint_indices,
    )

    frame_lines = reconstruct_template_motion(
        reference_layout=reference_layout,
        reference_global_positions=reference_global_positions,
        source_layout=input_layout,
        source_global_positions=source_global_positions,
        source_global_quats=source_global_quats,
        alignment_rotation=alignment_rotation,
        displacement_scale=displacement_scale,
    )

    hierarchy_text = extract_hierarchy_text(reference_path)
    output_text = (
        hierarchy_text
        + 'MOTION\n'
        + f'Frames: {len(frame_lines)}\n'
        + f'Frame Time: {input_layout["frame_time"]:.7f}\n'
        + '\n'.join(frame_lines)
        + '\n'
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output_text, encoding='utf-8')

    input_profile = inspect_bvh_profile(input_path)
    output_profile = inspect_bvh_profile(output_path)
    output_summary = summarize_loaded_bvh(output_path)
    reference_summary = summarize_loaded_bvh(reference_path)

    alignment_euler_deg = alignment_rotation.as_euler('xyz', degrees=True)

    report = {
        'input_bvh': str(input_path),
        'output_bvh': str(output_path),
        'reference_bvh': str(reference_path),
        'frame_count': len(frame_lines),
        'source_frame_time': input_layout['frame_time'],
        'reference_height_raw': reference_height_raw,
        'source_height_raw': source_height_raw,
        'applied_displacement_scale': displacement_scale,
        'alignment_rotation_euler_xyz_deg': [float(v) for v in alignment_euler_deg],
        'input_profile': {
            'detected_profile': input_profile['detected_profile'],
            'missing_official_core_joints': input_profile['missing_official_core_joints'],
            'rotation_orders': input_profile['rotation_orders'],
        },
        'output_profile': {
            'detected_profile': output_profile['detected_profile'],
            'missing_official_core_joints': output_profile['missing_official_core_joints'],
            'rotation_orders': output_profile['rotation_orders'],
        },
        'reference_loaded_summary': reference_summary,
        'output_loaded_summary': output_summary,
        'source_to_target_joint_map': SOURCE_TO_TARGET_JOINT_MAP,
    }

    print('[INFO] Input BVH:', input_path)
    print('[INFO] Reference BVH:', reference_path)
    print('[INFO] Output BVH:', output_path)
    print('[INFO] Applied displacement scale:', round(displacement_scale, 6))
    print('[INFO] Alignment rotation xyz deg:', [round(float(v), 4) for v in alignment_euler_deg])
    print('[INFO] Output loaded summary:')
    print(json.dumps(output_summary, ensure_ascii=False, indent=2))
    print('[INFO] Output missing official core joints:')
    print(json.dumps(output_profile['missing_official_core_joints'], ensure_ascii=False, indent=2))

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'[INFO] Report saved to {report_path}')


if __name__ == '__main__':
    main()
