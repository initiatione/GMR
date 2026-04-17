# 更新日志

## 2026-04-13

- 目标环境：`conda robot`
- 为当前本地 GMR 调参与运行安装了运行时依赖：`mink`、`rich`、`imageio`、`torch`、`smplx`
- 以可编辑模式安装项目：`python -m pip install -e . --no-deps`
- 在 `proxsuite` 轮子包发生 CRC 校验失败后，清理了一次异常的 pip 缓存并重新尝试安装
- 已在 `robot` 环境中验证 `GMR` 可以正常导入
- 已使用 `ik_config_manager/TPOSE.bvh` 验证 `bvh_lafan1 -> t800` 的冒烟测试
- 冒烟测试结果：成功创建 `GeneralMotionRetargeting(src_human='bvh_lafan1', tgt_robot='t800')`，并产出长度为 `32` 的首帧 `qpos`
- 可选组件 `xrobotoolkit_sdk` 仍未安装（当前本地 BVH/T800 的 ik_config 调参与验证不依赖该组件）
- 在 `assets/t800/mujoco/t800_full_gmr.xml` 中新增了可见且可碰撞的地面平面，用于本地 MuJoCo 调试与 IK 调参
- 在 `assets/t800/mujoco/t800_full_gmr.xml` 中新增了 `contact exclude` 规则，用来抑制初始帧中 `LINK_BASE` 与 `LINK_HIP_ROLL_{L,R}` 的自碰撞，避免首帧关节状态爆炸
- 在 `assets/t800/mujoco/t800_full_gmr.xml` 中新增渐变天空盒、提高头灯亮度并补充顶部场景光，以提升本地 IK 调参时的可视化效果
- 已将本地 LAFAN1 动作文件 `aiming1_subject1.bvh` 复制到工作区根目录，路径为 `D:\human_robot\aiming1_subject1.bvh`，用于 T800 的 ik_config 调参
- 新增了 `scripts/adjust_xml_transparency.py`，这是一个通用 XML 透明度调整工具，可读取已有 XML 并输出新的透明版 XML；已验证可基于 `assets/t800/mujoco/t800_full_gmr.xml` 生成透明度 alpha 为 `0.22` 的 `D:\human_robot\t800_full_gmr_transparent.xml`
- 已将 `t800_transparent` 加入 `bvh_to_robot.py` 的机器人选项，并将透明版 MuJoCo 模型标题改名为 `t800_full_gmr_transparent`，便于调试窗口识别

## 2026-04-14

- 新增 `scripts/align_cmu_bvh_to_lafan1.py`，这是一个最小化的 CMU-BVH 语义对齐脚本，用于把层级中的关节名称改写为更接近 GMR 所使用的 LAFAN1 核心骨架语义
- 当前 CMU boxing 样例 `D:\human_robot\hit_data\cmu\14_01.bvh` 对现有 `bvh_lafan1 -> t800` IK 配置缺少 `Spine2`、`LeftToe` 和 `RightToe`；新脚本会执行 `LowerBack -> Spine`、`Spine -> Spine1`、`Spine1 -> Spine2`、`Left/RightToeBase -> Left/RightToe` 的层级关节名改写
- 已通过 `python -m py_compile GMR/scripts/align_cmu_bvh_to_lafan1.py` 验证脚本语法
- 已在 `D:\human_robot\hit_data\cmu\14_01_lafan1_aligned.bvh` 上验证语义对齐输出；生成的报告确认改写后已不存在缺失的官方 LAFAN1 核心关节
- 当前 shell 会话中由于缺少运行时依赖 `mink`，未能完成首帧 IK 的冒烟测试；应在用户的 `conda robot` 环境中完成端到端重定向验证
- 已增强 `scripts/align_cmu_bvh_to_lafan1.py`，新增基于解析后 `Head -> Foot` 高度的自动尺度修复：在完成关节语义改写后，会自动缩放整份 BVH 的 `OFFSET` 与 root 平移通道，使输出尺度接近现有可运行的 LAFAN1/T800 链路
- 已重新处理 `D:\human_robot\hit_data\cmu\14_01.bvh`，生成 `D:\human_robot\hit_data\cmu\14_01_lafan1_aligned_scaled.bvh` 与对应报告；本次自动应用的缩放因子为 `6.364293`，估计的人体 `head-foot` 高度从 `0.243546m` 修复到 `1.55m`
- 已验证修复后的 CMU 样例关键高度恢复到正常量级：首帧 `hips_z` 约为 `1.1041m`，`head-foot` 高度约为 `1.55m`


## 2026-04-15


- 基于 `fight1_subject2.bvh` 的全帧调试日志（7347 帧）完成一轮 `t800` IK 小步调参：
  - 修改文件：`general_motion_retargeting/ik_configs/bvh_lafan1_to_t800_origin_manual.json`
  - 同步调整 `ik_match_table1/2` 中以下 link 的 `pos_offset.z`：
    - `LINK_HIP_YAW_L`: `0.00 -> -0.02`
    - `LINK_HIP_YAW_R`: `0.00 -> -0.02`
    - `LINK_SHOULDER_YAW_L`: `-0.01 -> -0.03`
    - `LINK_SHOULDER_YAW_R`: `-0.01 -> -0.03`
    - `LINK_WRIST_END_L`: `-0.12 -> -0.10`
    - `LINK_WRIST_END_R`: `-0.12 -> -0.10`
- A/B 对比（原配置 vs 调参后，统一使用 `--auto_ground --auto_ground_margin 0.01`）：
  - `final_error1 mean`: `1.0177 -> 0.9584`（`-5.83%`）
  - `final_error1 p95`: `1.8178 -> 1.4712`（`-19.07%`）
  - `final_error2 mean`: `1.0285 -> 0.9668`（`-6.00%`）
  - `LeftUpLeg pos_mean`: `0.1214 -> 0.1067`（`-12.06%`）
  - `RightUpLeg pos_mean`: `0.1313 -> 0.1079`（`-17.78%`）
  - `LeftArm pos_mean`: `0.1342 -> 0.1220`（`-9.10%`）
  - `RightArm pos_mean`: `0.1480 -> 0.1343`（`-9.27%`）

- 在 `scripts/bvh_to_robot.py` 新增 `--auto_ground` 开关（默认关闭），用于 BVH 重定向前自动估计并应用全局贴地偏移
- 在 `scripts/bvh_to_robot.py` 新增 `--auto_ground_margin`（默认 `0.0m`），用于控制自动贴地时的地面净空
- 自动贴地流程会打印 `min_z`、`margin`、`applied_ground_offset`，便于复现实验与排查脚部悬空/穿地问题
- 已通过 `python -m py_compile scripts/bvh_to_robot.py` 语法校验

## 2026-04-17

- 新增 `general_motion_retargeting/motion_grounding.py`，提供基于 MuJoCo 真实支撑碰撞几何的离线 grounding 工具；不再仅凭 `root_pos[:, 2]` 判断是否贴地，而是逐帧 forward 后计算脚/踝/趾等支撑 geom 的世界坐标最低点
- 新增 `scripts/ground_robot_motion.py`，可对已导出的机器人动作 `pkl` 做离线贴地修复；支持：
  - `--robot` / `--robot_xml` 指定 MuJoCo 模型
  - `--mode per_frame|global` 选择逐帧修复或整段统一上抬
  - `--clearance` 指定目标最小离地净空
- 在 `scripts/bvh_to_robot.py`、`scripts/smplx_to_robot.py`、`scripts/gvhmr_to_robot.py` 中新增显式导出后处理开关：
  - `--foot-ground-align`
  - `--foot-ground-mode`
  - `--foot-ground-clearance`
- 上述 grounding 开关默认关闭，不会无提示地强制修改导出轨迹；只有用户显式传入 `--foot-ground-align` 时，才会对保存的 `pkl` 做贴地修复
- grounding 后处理当前只修改 `root_pos[:, 2]`，不修改关节角；适合作为数据清洗/导出修复步骤，而不是完整的接触一致性求解器
- 已为 grounding 相关脚本补充详细中文注释，重点说明：
  - 为什么要按真实支撑碰撞几何而不是按 `root_z` 做贴地判断
  - `sphere / box / capsule / cylinder / ellipsoid` 的最低点近似是如何计算的
  - `per_frame` 与 `global` 两种模式对训练分布和竖直轨迹的影响差异
- 新增回归测试 `tests/test_motion_grounding.py`，覆盖：
  - `compute_support_min_z` 的 box 几何最低点计算
  - `per_frame` 模式消除穿地
  - `global` 模式使用统一上抬量
- 已通过 `python -m pytest tests/test_motion_grounding.py --basetemp D:\human_robot\.pytest_tmp` 验证测试通过
- 已通过 `python -m py_compile general_motion_retargeting/motion_grounding.py scripts/ground_robot_motion.py scripts/bvh_to_robot.py scripts/smplx_to_robot.py scripts/gvhmr_to_robot.py tests/test_motion_grounding.py` 语法校验
- 已在 `D:\human_robot\tmp_compare\lafan1\retarget_t800\fight1_subject2.pkl` 上验证离线 grounding：
  - 输出文件：`D:\human_robot\tmp_compare\lafan1\retarget_t800\fight1_subject2_grounded.pkl`
  - 支撑碰撞体数量：`4`
  - 修复前 `before_min_support_z = -0.218195m`
  - 修复前穿地帧数：`7114 / 7346`
  - 使用 `per_frame` 模式与 `0.002m` 净空后，修复后 `after_min_support_z = 0.002000m`
  - 修复后穿地帧数：`0`
