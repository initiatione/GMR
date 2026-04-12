# T800 Assets For GMR

这里存放 `GMR` 专用的 `T800` 资产。

## 说明

- `mujoco/t800_gmr.xml`
  是专门给 `GMR` 的 Mink + MuJoCo IK 流程准备的简化 floating-base 模型。
- 这份模型的目标是：
  - 满足 `GMR` 对 `qpos = root(7) + joints` 的结构假设。
  - 保持与训练侧 `T800` 一致的主链 body / joint 语义。
  - 不直接修改 `whole_body_tracking_engineai` 里的训练资产。

## 边界

- 这里的模型不是训练时的动力学资产。
- 训练仍然应使用 `whole_body_tracking_engineai` 仓库中的原始 `T800` 资产。
- `GMR` 这份模型只用于：
  - BVH 到 T800 的重定向求解
  - 可视化调试
  - 导出中间动作结果
