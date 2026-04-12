# T800 Assets For GMR

这里存放 `GMR` 专用的 `T800` 资产。

## 说明

- `mujoco/t800_gmr.xml`
  是专门给 `GMR` 的 Mink + MuJoCo IK 流程准备的简化 floating-base 模型。
- 这份模型的目标是：
  - 满足 `GMR` 对 `qpos = root(7) + joints` 的结构假设。
  - 保持与训练侧 `T800` 一致的主链 body / joint 语义。
  - 不直接修改 `whole_body_tracking_engineai` 里的训练资产。
  - 先用 proxy geoms 跑通重定向链路，避免直接把训练侧 `.dae` mesh 硬塞进 MuJoCo 造成兼容问题。

## URDF / Mesh 对应关系

- 运动学主链来自 `whole_body_tracking_engineai/source/whole_body_tracking/whole_body_tracking/assets/t800/urdf/serial_t800.urdf`。
- 当前 `GMR` 资产保留了同一套 `LINK_* / J**_*` 命名，方便把 `qpos` 回对到训练链。
- 当前仓库已经从训练侧 `.dae` 批量导出了一份 `assets/t800/meshes/*.stl`，并在 `mujoco/t800_gmr.xml` 里作为 visual mesh 使用。
- 如果后续需要真实外观 mesh：
  - 当前仓库优先使用从训练侧 `.dae` 导出的 `.stl` 作为 MuJoCo visual mesh。
  - 然后在 `mujoco/t800_gmr.xml` 的 `<asset>` 中声明 mesh，并保持现有 body / joint 树不变。

## 边界

- 这里的模型不是训练时的动力学资产。
- 训练仍然应使用 `whole_body_tracking_engineai` 仓库中的原始 `T800` 资产。
- `GMR` 这份模型只用于：
  - BVH 到 T800 的重定向求解
  - 可视化调试
  - 导出中间动作结果
