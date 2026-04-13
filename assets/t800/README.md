# T800 Assets For GMR

这里存放 `GMR` 专用的 `T800` 资产。

## 说明

- `mujoco/t800_from_urdf.xml`
  是当前默认给 `GMR` retargeting / auto-ik 流程使用的 `T800` floating-base 模型。
- `mujoco/t800_full_gmr.xml`
  是在 `t800_from_urdf.xml` 基础上继续补齐 base / foot / wrist 惯量与末端几何后的更完整候选版本。
- `mujoco/t800_gmr.xml`
  是更早期的手工整理轻量版模型，仍可用于对照和回退。

## 当前默认模型

- `general_motion_retargeting/params.py` 中，`t800` 目前默认注册到 `assets/t800/mujoco/t800_from_urdf.xml`。
- 这份默认模型的目标是：
  - 满足 `GMR` 对 `qpos = root(7) + joints` 的结构假设。
  - 保持与训练侧 `T800` 一致的主链 body / joint 语义。
  - 尽量保留训练侧 URDF 的惯量与 link 信息。
  - 使用训练侧导出的 `STL` visual mesh，并保留适合 GMR 的 collision proxy / actuator 结构。

## URDF / Mesh 对应关系

- 运动学主链来自 `whole_body_tracking_engineai/source/whole_body_tracking/whole_body_tracking/assets/t800/urdf/serial_t800.urdf`。
- 当前仓库已经从训练侧 `.dae` 批量导出了一份 `assets/t800/meshes/*.stl`，并在 `mujoco/*.xml` 里作为 visual mesh 使用。
- `t800_from_urdf.xml`：以工具从 URDF 导出的基线 MJCF 为底稿，再补上 `LINK_BASE/freejoint`、visual mesh、末端 body 和 actuator。
- `t800_full_gmr.xml`：在 `t800_from_urdf.xml` 上进一步补齐 `LINK_BASE`、`LINK_FOOT_*`、`LINK_WRIST_END_*` 的惯量/末端几何，让模型更接近训练侧完整结构。
- `t800_gmr.xml`：保留了相同 body / joint 命名，但更偏轻量化和历史兼容用途。

## 边界

- 这里的模型不是训练时的动力学资产。
- 训练仍然应使用 `whole_body_tracking_engineai` 仓库中的原始 `T800` 资产。
- `GMR` 这些模型主要用于：
  - BVH 到 T800 的重定向求解
  - auto-ik 参数生成与 FK 对齐
  - 可视化调试
  - 导出中间动作结果
