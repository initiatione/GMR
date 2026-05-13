from __future__ import annotations

"""Utilities for grounding exported robot motions using actual MuJoCo support geoms.

这个模块解决的是“导出的 qpos 能回放，但脚底和地面接触不物理”的问题。

核心思路不是直接检查 `root_pos[:, 2]`，而是：
1. 用 MuJoCo 模型把每一帧真正 forward 一次；
2. 找到机器人脚/踝/趾等承重碰撞体；
3. 计算这些碰撞体在世界坐标系下的最低 z；
4. 再决定是否需要把 root 整体上抬。

这样做的好处是：校正依据来自“真实碰撞几何”，而不是来自 base 高度的代理量。
"""

from pathlib import Path
from typing import Iterable, Sequence

import mujoco as mj
import numpy as np


SUPPORT_KEYWORDS: tuple[str, ...] = ("foot", "ankle", "toe", "sole")
T800_TRAINING_FOOT_SUPPORT_BODIES: tuple[str, ...] = ("LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R")
T800_MUJOCO_GROUP_COLLISION_URDF = 3
GROUNDING_MODES: tuple[str, ...] = ("per_frame", "global", "smooth_per_frame")


def _as_model(model_or_path: mj.MjModel | str | Path) -> mj.MjModel:
    """Allow callers to pass either a loaded model or an xml path."""
    if isinstance(model_or_path, mj.MjModel):
        return model_or_path
    return mj.MjModel.from_xml_path(str(model_or_path))


def _xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    """Exported PKL stores xyzw, while MuJoCo qpos expects scalar-first wxyz."""
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
    if quat_xyzw.shape != (4,):
        raise ValueError(f"Expected quaternion with shape (4,), got {quat_xyzw.shape}")
    return quat_xyzw[[3, 0, 1, 2]]


def find_support_geom_ids(
    model: mj.MjModel,
    keywords: Sequence[str] = SUPPORT_KEYWORDS,
) -> list[int]:
    """Find contact-enabled support geoms by fuzzy matching body/geom names.

    这里故意只扫描开启接触的 geom：
    - visual mesh / decoration 不应参与 grounding；
    - 真正影响训练接触的是碰撞体，而不是外观网格。

    对 T800 这类从训练侧 URDF 转出的 MJCF，优先使用 `collision_urdf`
    脚底盒，而不是 GMR 为 IK/debug 额外补的 `collision_fallback`。
    这样 grounding 的脚底定义和 Isaac/训练侧保持一致。
    """
    normalized_keywords = tuple(str(keyword).lower() for keyword in keywords)
    t800_training_foot_bodies = set(T800_TRAINING_FOOT_SUPPORT_BODIES)
    model_body_names = {
        name
        for body_id in range(model.nbody)
        if (name := mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id))
    }
    t800_like_model = t800_training_foot_bodies.issubset(model_body_names)

    training_foot_support_ids: list[int] = []
    training_foot_support_bodies: set[str] = set()
    for geom_id in range(model.ngeom):
        if model.geom_contype[geom_id] == 0 or model.geom_conaffinity[geom_id] == 0:
            continue
        if int(model.geom_group[geom_id]) != T800_MUJOCO_GROUP_COLLISION_URDF:
            continue

        body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom_id]) or ""
        if body_name in t800_training_foot_bodies:
            training_foot_support_ids.append(geom_id)
            training_foot_support_bodies.add(body_name)

    if t800_like_model:
        if training_foot_support_bodies == t800_training_foot_bodies:
            return training_foot_support_ids
        missing_bodies = sorted(t800_training_foot_bodies - training_foot_support_bodies)
        raise ValueError(
            "Failed to find T800 training-side foot support geoms. "
            f"Missing collision_urdf group={T800_MUJOCO_GROUP_COLLISION_URDF} "
            f"support bodies={missing_bodies}."
        )

    if training_foot_support_ids:
        return training_foot_support_ids

    support_geom_ids: list[int] = []
    for geom_id in range(model.ngeom):
        # 没有碰撞属性的 geom 即使名字像脚，也不会参与地面接触，不该被拿来做校正基准。
        if model.geom_contype[geom_id] == 0 or model.geom_conaffinity[geom_id] == 0:
            continue

        geom_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geom_id) or ""
        body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom_id]) or ""
        haystack = f"{geom_name} {body_name}".lower()
        if any(keyword in haystack for keyword in normalized_keywords):
            support_geom_ids.append(geom_id)

    if not support_geom_ids:
        raise ValueError(
            "Failed to find support geoms. "
            f"Searched keywords={list(normalized_keywords)} among contact-enabled geoms."
        )

    return support_geom_ids


def geom_lowest_z(model: mj.MjModel, data: mj.MjData, geom_id: int) -> float:
    """Compute the lowest world-space z of one geom.

    我们不依赖 MuJoCo 的接触求解结果，而是直接基于 geom 的位姿和尺寸做几何下界估计。
    这样即便当前帧还没真的发生 contact，也能判断“这一帧离地还有多远/已经穿地多少”。
    """
    geom_type = int(model.geom_type[geom_id])
    pos = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
    mat = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    size = np.asarray(model.geom_size[geom_id], dtype=np.float64)

    if geom_type == int(mj.mjtGeom.mjGEOM_SPHERE):
        return float(pos[2] - size[0])

    if geom_type == int(mj.mjtGeom.mjGEOM_BOX):
        # box 旋转后，世界 z 方向的投影长度等于各局部轴长度在 z 上投影的绝对值之和。
        z_extent = np.sum(np.abs(mat[2, :]) * size[:3])
        return float(pos[2] - z_extent)

    if geom_type == int(mj.mjtGeom.mjGEOM_CAPSULE):
        # capsule = 沿局部 z 轴的线段 + 半径。最低点由轴向投影和半径共同决定。
        axis_world = mat[:, 2]
        half_length = size[1]
        radius = size[0]
        return float(pos[2] - abs(axis_world[2]) * half_length - radius)

    if geom_type == int(mj.mjtGeom.mjGEOM_CYLINDER):
        # cylinder 比 capsule 少了球帽，因此径向外扩要单独投影到世界 z。
        axis_world = mat[:, 2]
        half_length = size[1]
        radius = size[0]
        radial_extent = radius * np.sqrt(max(0.0, 1.0 - axis_world[2] ** 2))
        return float(pos[2] - abs(axis_world[2]) * half_length - radial_extent)

    if geom_type == int(mj.mjtGeom.mjGEOM_ELLIPSOID):
        z_extent = np.sqrt(np.sum((mat[2, :] * size[:3]) ** 2))
        return float(pos[2] - z_extent)

    # 兜底逻辑：对不常见 geom 类型给一个保守的近似下界，避免工具直接失效。
    return float(pos[2] - np.max(size))


def compute_support_min_z(
    model_or_path: mj.MjModel | str | Path,
    root_pos: np.ndarray,
    root_rot: np.ndarray,
    dof_pos: np.ndarray,
    support_geom_ids: Iterable[int] | None = None,
) -> np.ndarray:
    """Replay the full motion and return per-frame minimum support height.

    返回值的语义：
    - `< 0` 说明穿地；
    - `= 0` 说明最低支撑几何刚好接地；
    - `> 0` 说明整只脚仍有悬空余量。
    """
    model = _as_model(model_or_path)
    data = mj.MjData(model)

    root_pos = np.asarray(root_pos, dtype=np.float64)
    root_rot = np.asarray(root_rot, dtype=np.float64)
    dof_pos = np.asarray(dof_pos, dtype=np.float64)

    if root_pos.ndim != 2 or root_pos.shape[1] < 3:
        raise ValueError("Expected root_pos with shape (N, >=3)")
    if root_rot.ndim != 2 or root_rot.shape[1] != 4:
        raise ValueError("Expected root_rot with shape (N, 4) in xyzw order")
    if dof_pos.ndim != 2:
        raise ValueError("Expected dof_pos with shape (N, D)")
    if not (root_pos.shape[0] == root_rot.shape[0] == dof_pos.shape[0]):
        raise ValueError("root_pos, root_rot, and dof_pos must have the same frame count")

    support_ids = list(support_geom_ids) if support_geom_ids is not None else find_support_geom_ids(model)
    support_min_z = np.empty(root_pos.shape[0], dtype=np.float64)

    for frame_idx in range(root_pos.shape[0]):
        # 这里直接回放导出的 floating-base 状态，再由 MuJoCo 计算各个碰撞体的世界位姿。
        data.qpos[:3] = root_pos[frame_idx, :3]
        data.qpos[3:7] = _xyzw_to_wxyz(root_rot[frame_idx])
        data.qpos[7:] = dof_pos[frame_idx]
        mj.mj_forward(model, data)

        frame_min_z = np.inf
        for geom_id in support_ids:
            frame_min_z = min(frame_min_z, geom_lowest_z(model, data, geom_id))
        support_min_z[frame_idx] = frame_min_z

    return support_min_z


def summarize_support_min_z(support_min_z: np.ndarray, clearance: float = 0.0) -> dict[str, float]:
    """Summarize how serious the current grounding problem is."""
    support_min_z = np.asarray(support_min_z, dtype=np.float64)
    shift = np.maximum(0.0, float(clearance) - support_min_z)
    return {
        "min_support_z": float(np.min(support_min_z)),
        "median_support_z": float(np.median(support_min_z)),
        "max_support_z": float(np.max(support_min_z)),
        "penetrating_frames": int(np.count_nonzero(support_min_z < 0.0)),
        "frames_below_clearance": int(np.count_nonzero(support_min_z < float(clearance))),
        "max_required_shift": float(np.max(shift)),
        "median_required_shift": float(np.median(shift)),
    }


def _smooth_signal_segmentwise(values: np.ndarray, mask: np.ndarray, window: int) -> np.ndarray:
    """Smooth only inside contiguous true segments, leaving non-candidate frames unchanged."""
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    window = int(window)
    smoothed = values.copy()
    if window <= 1 or values.shape[0] <= 2:
        return smoothed

    start = None
    for idx, active in enumerate(mask):
        if active and start is None:
            start = idx
        if not active and start is not None:
            _smooth_one_segment(smoothed, values, start, idx, window)
            start = None
    if start is not None:
        _smooth_one_segment(smoothed, values, start, values.shape[0], window)
    return smoothed


def _smooth_one_segment(output: np.ndarray, values: np.ndarray, start: int, end: int, window: int) -> None:
    segment = values[start:end]
    if segment.shape[0] <= 1:
        return

    effective_window = min(int(window), int(segment.shape[0]))
    if effective_window <= 1:
        return

    pad_left = effective_window // 2
    pad_right = effective_window - 1 - pad_left
    kernel = np.ones(effective_window, dtype=np.float64) / float(effective_window)
    padded = np.pad(segment, (pad_left, pad_right), mode="edge")
    output[start:end] = np.convolve(padded, kernel, mode="valid")


def align_motion_root_to_ground(
    motion_data: dict,
    model_or_path: mj.MjModel | str | Path,
    clearance: float = 0.002,
    mode: str = "per_frame",
    inplace: bool = False,
    support_geom_ids: Iterable[int] | None = None,
    smooth_window: int = 9,
    smooth_contact_threshold: float = 0.04,
) -> tuple[dict, dict[str, float]]:
    """Raise root z so support geoms stay above the ground plane.

    mode 的区别：
    - per_frame: 每一帧独立补偿。修得最干净，但会更强地改变竖直轨迹；
    - global:    整段动作统一上抬一个常数。对训练分布更保守，但不能消除“相对抖动”。
    - smooth_per_frame:
                 只在接近地面的候选支撑段里做带符号、分段平滑的 root-z 修正。
                 它不会把明显腾空帧硬拉回地面，适合生成需要人工 replay 验收的候选。

    这个函数只改 root z，不改关节角。
    所以它适合作为“数据修复/导出后处理”，而不是完整的接触一致性求解器。
    """
    if mode not in set(GROUNDING_MODES):
        raise ValueError(f"Unsupported grounding mode: {mode}")

    if inplace:
        grounded_motion = motion_data
    else:
        grounded_motion = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in motion_data.items()
        }

    root_pos = grounded_motion["root_pos"]
    root_rot = grounded_motion["root_rot"]
    dof_pos = grounded_motion["dof_pos"]

    support_ids = list(support_geom_ids) if support_geom_ids is not None else find_support_geom_ids(_as_model(model_or_path))

    before_min_z = compute_support_min_z(
        model_or_path=model_or_path,
        root_pos=root_pos,
        root_rot=root_rot,
        dof_pos=dof_pos,
        support_geom_ids=support_ids,
    )

    raw_shift = np.maximum(0.0, float(clearance) - before_min_z)
    if mode == "global":
        # 用全局最大需求量把整段轨迹一起抬高。
        # 这样不会引入逐帧 root z 变化，但会保留原始动作里“脚相对地面忽上忽下”的内部误差。
        applied_shift = np.full_like(raw_shift, np.max(raw_shift))
    elif mode == "per_frame":
        # 逐帧单独修正，能把每一帧都拉回到期望 clearance 上方。
        # 代价是对 base 高度、竖直速度、COM 轨迹的修改更强。
        applied_shift = raw_shift
    else:
        # smooth_per_frame 是给“疑似接地但整体偏高”的 PKL 做候选修复，不是强制接触求解。
        # 只有 support min-z 接近地面时才允许向下贴合；明显腾空帧保持原样，避免抹掉踮脚、
        # 摆腿、起跳等原动作语义。穿地帧仍然保留向上修正，防止生成物理上更差的候选。
        contact_threshold = max(float(smooth_contact_threshold), float(clearance))
        contact_candidate = before_min_z <= contact_threshold
        signed_target_shift = np.zeros_like(before_min_z)
        signed_target_shift[contact_candidate] = float(clearance) - before_min_z[contact_candidate]
        applied_shift = _smooth_signal_segmentwise(
            values=signed_target_shift,
            mask=contact_candidate,
            window=int(smooth_window),
        )
        applied_shift[~contact_candidate] = 0.0

        # 平滑可能把局部穿地帧的上抬量摊薄。这里只夹到“不低于 clearance”，
        # 不把所有负向修正都清掉，否则就无法修复“接近地面但整体偏高”的悬空。
        minimum_nonpenetration_shift = float(clearance) - before_min_z
        applied_shift = np.maximum(applied_shift, minimum_nonpenetration_shift)

    root_pos[:, 2] += applied_shift

    after_min_z = compute_support_min_z(
        model_or_path=model_or_path,
        root_pos=root_pos,
        root_rot=root_rot,
        dof_pos=dof_pos,
        support_geom_ids=support_ids,
    )

    stats = {
        "mode": mode,
        "clearance": float(clearance),
        "frames": int(root_pos.shape[0]),
        "support_geom_count": int(len(support_ids)),
        "before_min_support_z": float(np.min(before_min_z)),
        "before_median_support_z": float(np.median(before_min_z)),
        "before_penetrating_frames": int(np.count_nonzero(before_min_z < 0.0)),
        "applied_shift_max": float(np.max(applied_shift)),
        "applied_shift_min": float(np.min(applied_shift)),
        "applied_shift_median": float(np.median(applied_shift)),
        "applied_shift_max_step": float(np.max(np.abs(np.diff(applied_shift)))) if applied_shift.shape[0] > 1 else 0.0,
        "after_min_support_z": float(np.min(after_min_z)),
        "after_median_support_z": float(np.median(after_min_z)),
        "after_penetrating_frames": int(np.count_nonzero(after_min_z < -1e-6)),
    }
    if mode == "smooth_per_frame":
        stats.update(
            {
                "smooth_window": int(smooth_window),
                "smooth_contact_threshold": float(smooth_contact_threshold),
                "smooth_contact_candidate_frames": int(np.count_nonzero(contact_candidate)),
            }
        )
    return grounded_motion, stats
