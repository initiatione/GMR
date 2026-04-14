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
