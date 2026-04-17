# 离线修复说明

## 这份文档是干什么的

这份说明只描述当前工作区里新增的本地离线修复能力，不试图替代上游项目的主 README。

它解决的是这类问题：

- 导出的机器人 `pkl` 存在脚部穿地
- 脚底有轻微悬空
- 支撑脚在地面上明显滑动

需要先明确一点：

- **GMR 主流程仍然是运动学重定向**
- 当前离线修复也是**运动学后处理**
- 它的作用是改善训练数据质量
- 它**不是**完整的动力学优化器，也**不会**自动保证轨迹严格动力学可执行

## 当前有两条离线修复链

### 1. 基础 grounding

入口脚本：

- `scripts/ground_robot_motion.py`

作用：

- 基于 MuJoCo 真实支撑碰撞体计算每一帧最低点
- 修正 `root_pos[:, 2]`
- 解决明显的穿地或悬空

适合：

- 只想做最小修复
- 不需要 stance 检测和支撑脚锁地

### 2. Contact-aware 导出后处理

接入位置：

- `scripts/bvh_to_robot.py`
- `scripts/smplx_to_robot.py`
- `scripts/gvhmr_to_robot.py`

作用：

- 估计左右脚 `stance phase`
- 在支撑段做支撑脚 `XY` 锁地
- 基于真实支撑碰撞体做 grounding
- 对 `root_z` 修正量做轻量平滑

设计原则：

- 不改 GMR 主求解流程
- 默认关闭
- 只在导出 `pkl` 时显式开启
- 保留旧接口兼容

## 推荐用法

### 最推荐：导出时直接启用 Contact-aware 后处理

```bash
python scripts/bvh_to_robot.py \
  --bvh_file /path/to/your_motion.bvh \
  --robot t800 \
  --save_path /path/to/output.pkl \
  --contact-aware-postprocess \
  --contact-profile balanced
```

### 更保守的修复

```bash
python scripts/bvh_to_robot.py \
  --bvh_file /path/to/your_motion.bvh \
  --robot t800 \
  --save_path /path/to/output.pkl \
  --contact-aware-postprocess \
  --contact-profile conservative
```

### 坏数据更明显时

```bash
python scripts/bvh_to_robot.py \
  --bvh_file /path/to/your_motion.bvh \
  --robot t800 \
  --save_path /path/to/output.pkl \
  --contact-aware-postprocess \
  --contact-profile aggressive
```

## 三档 profile 的含义

| 档位 | 含义 | 适用场景 |
| --- | --- | --- |
| `conservative` | 改动更少，更保守 | 先做轻量清洗，尽量少动原轨迹 |
| `balanced` | 当前推荐默认 | 大多数导出训练数据场景 |
| `aggressive` | 修正更积极 | 穿地、悬空、滑脚更明显的坏数据 |

## Expert Override

如果三档 profile 不够用，仍然可以覆写内部参数：

- `--contact-stance-height-threshold`
- `--contact-stance-speed-threshold`
- `--contact-stance-min-frames`
- `--contact-ground-mode`
- `--contact-ground-clearance`
- `--contact-root-z-smoothing-window`

建议顺序是：

1. 先用 `balanced`
2. 不够再试 `conservative / aggressive`
3. 只有 profile 仍不够时，再动 expert override

## 旧接口：仅做 grounding

如果你只想做最小修复，不想启用整条 contact-aware 链，也可以继续使用旧接口：

```bash
python scripts/bvh_to_robot.py \
  --bvh_file /path/to/your_motion.bvh \
  --robot t800 \
  --save_path /path/to/output.pkl \
  --foot-ground-align \
  --foot-ground-mode per_frame \
  --foot-ground-clearance 0.002
```

或者对已有 `pkl` 单独离线修复：

```bash
python scripts/ground_robot_motion.py \
  --input /path/to/input.pkl \
  --output /path/to/output_grounded.pkl \
  --robot t800 \
  --mode per_frame \
  --clearance 0.002
```

## 适配性说明

### 结论

- 对当前 GMR 已注册的**大多数双足 humanoid**，当前离线修复**可以直接适配**
- 但它**还不能自动适配全部机器人资产**

### 当前静态盘点结果

下面这个结论基于当前注册机器人 XML 实际跑过：

- `find_support_geom_ids(...)`
- `group_support_geom_ids_by_side(...)`

#### 当前可直接适配

| 机器人 | 状态 |
| --- | --- |
| `unitree_g1` | 可直接适配 |
| `unitree_g1_with_hands` | 可直接适配 |
| `unitree_h1` | 可直接适配 |
| `unitree_h1_2` | 可直接适配 |
| `booster_t1` | 可直接适配 |
| `booster_t1_29dof` | 可直接适配 |
| `fourier_n1` | 可直接适配 |
| `engineai_pm01` | 可直接适配 |
| `hightorque_hi` | 可直接适配 |
| `berkeley_humanoid_lite` | 可直接适配 |
| `booster_k1` | 可直接适配 |
| `pnd_adam_lite` | 可直接适配 |
| `tienkung` | 可直接适配 |
| `fourier_gr3` | 可直接适配 |
| `t800` | 可直接适配 |
| `t800_transparent` | 可直接适配 |

#### 当前不直接适配

| 机器人 | 原因 |
| --- | --- |
| `stanford_toddy` | 足部接触体命名更像 `ank_pitch / ank_roll`，当前关键词不完整 |
| `kuavo_s45` | 接触体命名更像 `leg_l6 / leg_r6`，没有 `foot/ankle/toe/sole` 语义 |
| `pal_talos` | 足部末端命名更像 `leg_left_6 / leg_right_6`，当前自动搜索抓不到 |
| `galaxea_r1pro` | 轮式平台，不属于当前“双足 stance + 锁脚 + grounding”假设 |

### 当前自动适配依赖的前提

当前离线修复默认假设：

1. 机器人是双足 humanoid
2. 模型里存在明确的脚部 / 踝部 / 趾部支撑碰撞体
3. 支撑碰撞体的命名能被当前规则识别，或者至少能在默认站姿下按左右 `y` 分开

## 当前方法的边界

需要再次强调：

- 当前这套离线修复仍然是**运动学后处理**
- 它考虑了几何接触一致性
- 但没有显式考虑：
  - 接触力
  - 摩擦锥
  - ZMP / CoM 动力学
  - 力矩极限
  - 多接触互补约束

所以它更适合：

- 清洗训练前的参考轨迹
- 减少明显的脚部穿地、悬空和滑脚

但它不能保证：

- 轨迹严格动力学可执行
- 控制器一定能无误跟踪
- 所有高动态动作都自动物理一致

## 验证入口

当前本地已经覆盖的测试：

- `tests/test_motion_grounding.py`
- `tests/test_motion_contact_postprocess.py`

如果你想追溯更细的本地变更，请看：

- `UPDATE_LOG.md`
