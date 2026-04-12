import numpy as np
from scipy.spatial.transform import Rotation as R

import general_motion_retargeting.utils.lafan_vendor.utils as utils
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh
from general_motion_retargeting.utils.bvh_profile_adapter import (
    adapt_frame_for_gmr,
    estimate_human_height_from_frames,
    inspect_bvh_profile,
    read_bvh_with_joint_orders,
)


def load_bvh_file(bvh_file, format="lafan1"):
    """
    Must return a dictionary with the following structure:
    {
        "Hips": (position, orientation),
        "Spine": (position, orientation),
        ...
    }
    """
    # 先检查 BVH 结构画像。
    # 这样做有两个目的：
    # 1. 识别当前输入是否是官方标准 LAFAN1，还是当前项目 `hit_data` 这种扩展骨架；
    # 2. 判断是否需要启用“按关节真实 CHANNELS 顺序”解析的适配层。
    bvh_profile = inspect_bvh_profile(bvh_file)

    # 官方 LAFAN1 大多可以继续沿用 vendor 版读取器；
    # 但当前项目 `hit_data` 存在混合旋转顺序，如果仍然使用单一全局顺序解析，局部四元数会系统性错误。
    if bvh_profile["has_mixed_rotation_orders"] or bvh_profile["detected_profile"] == "human_robot_hit":
        data = read_bvh_with_joint_orders(bvh_file)
    else:
        data = read_bvh(bvh_file)
    global_data = utils.quat_fk(data.quats, data.pos, data.parents)

    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)

    frames = []
    for frame in range(data.pos.shape[0]):
        result = {}
        for i, bone in enumerate(data.bones):
            orientation = utils.quat_mul(rotation_quat, global_data[0][frame, i])
            position = global_data[1][frame, i] @ rotation_matrix.T / 100  # cm to m
            result[bone] = [position, orientation]

        # 这里把“项目 BVH 的骨架语义”适配成 GMR 更熟悉的 LAFAN1 主体骨架语义。
        # 当前主要做两类最小改动：
        # 1. `LeftToeBase/RightToeBase` -> `LeftToe/RightToe`；
        # 2. 用 `Spine3` 覆盖 `Spine2`，让 torso 目标更接近官方 LAFAN1 的上躯干含义。
        result = adapt_frame_for_gmr(result, bvh_profile["detected_profile"])

        if format == "lafan1":
            left_toe_name = "LeftToe" if "LeftToe" in result else "LeftToeBase"
            right_toe_name = "RightToe" if "RightToe" in result else "RightToeBase"
            if left_toe_name not in result or right_toe_name not in result:
                missing = [name for name in [left_toe_name, right_toe_name] if name not in result]
                raise KeyError(
                    f"Missing toe joints for BVH foot alignment: {missing}. "
                    "Expected LeftToe/RightToe or LeftToeBase/RightToeBase."
                )
            result["LeftFootMod"] = [result["LeftFoot"][0], result[left_toe_name][1]]
            result["RightFootMod"] = [result["RightFoot"][0], result[right_toe_name][1]]
        elif format == "nokov":
            result["LeftFootMod"] = [result["LeftFoot"][0], result["LeftToeBase"][1]]
            result["RightFootMod"] = [result["RightFoot"][0], result["RightToeBase"][1]]
        else:
            raise ValueError(f"Invalid format: {format}")
            
        frames.append(result)

    # 对官方标准 LAFAN1 继续保持原来的保守默认值，避免无关动作的缩放行为发生明显漂移；
    # 对当前项目 `hit_data` 这类扩展骨架，则优先尝试从动作本身估计身高，减少比例误差。
    human_height = 1.75  # cm to m
    if bvh_profile["detected_profile"] == "human_robot_hit":
        estimated_human_height = estimate_human_height_from_frames(frames)
        if estimated_human_height is not None:
            human_height = estimated_human_height

    return frames, human_height


