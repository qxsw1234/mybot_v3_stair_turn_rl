# MyBot V3 - 2027 ROBOCON 足式机器人障碍赛训练与验收计划

> 版本：v0.2（2026-10-11）
> 规则基线：《2027 年足式机器人挑战赛 V1.0》
> 范围：只覆盖障碍赛，不包含物资搬运任务赛
> 当前工程：`/home/ldl/mybot_v3_stair_turn_rl`
> v0.2 修订原则：从“训练八种能力”改为“尽快建立 180 s 内可稳定得分的系统”。

## 1. 目标与基本判断

### 1.1 比赛目标

- 在 180 s 内最大化有效得分，不把“全部障碍都会”作为前期唯一目标；
- 首先形成稳定的遥控高层指令 + 自主低层步态方案，保证可得分；
- 随后升级为全自主导航和技能调度，争取自主控制的更高分值；
- 全部 8 障碍完成后，再把“返回启动区”作为额外目标；
- 比赛过程中不能切换“遥控/自主”的控制方式，但同一种控制方式内部允许切换专项策略。

规则计分基线：7 个普通障碍每项自主/遥控为 150/100 分；沟壑组合最高为 300/200 分；全部完成后返回原启动区为 150/100 分；总满分为 1500/1000 分。

### 1.2 总体技术路线

采用“专项教师策略 + 统一部署接口 + 后期蒸馏/调度”的路线，而不是立刻训练一个包打天下的大策略。

1. 复用当前 `059950/063296` 等通用运动和限高 checkpoint；
2. 每个障碍训练一个可独立验收的 expert；
3. 统一 expert 的 observation/action contract、导出格式和安全状态机；
4. 遥控版本由操作手发速度、方向和技能选择；
5. 自主版本由感知/导航模块自动选择 skill 和目标方向；
6. 第一个专项技能可用后，立即打通 `locomotion -> expert -> recovery -> locomotion` 切换；
7. 多个 expert 稳定后，再考虑用 teacher-student/DAgger 蒸馏成统一 student。

不建议一开始把 8 个障碍混在一个 PPO 环境里。这样会导致奖励冲突、课程互相干扰，且失败时无法归因。

全场目标函数使用比赛口径：

```text
Score = 有效越障分 + 返回启动区额外分 - 罚分
```

技能研发优先级不只看单项分值，使用下列指标每周重排：

```text
ExpectedScoreRate = 预期有效得分 / (P95 完成时间 + 障碍间移动时间 + 预期恢复时间)
RDP = 预期有效得分 × 成功概率 / (预估研发人日 × 实机风险系数)
```

`ExpectedScoreRate` 用于全场选路，`RDP` 用于决定研发资源先投哪一项。

## 2. 当前可复用资产

| 资产 | 可复用方式 | 限制 |
|---|---|---|
| 平地、横移、转向、站立策略 | 所有 expert 的初始化 checkpoint | 不是规则级障碍能力 |
| 8/10/12 cm 台阶能力 | 沟壑组合台阶段 warm start | 尚未包含 36 cm 沟壑和逐级接触判据 |
| 斜坡能力 | 沟壑组合 20°斜坡段 warm start | 需要按新版尺寸复测 |
| Low Bar `063296` | 新版限高框架 expert 初始化 | 当前是单横杆、宽 1 m、目标 300 mm；新版为 290×800×800 mm 框架 |
| PPO-CSE/RMA、history、height scan | 继续作为低层算法主干 | 离散落足目标需要新增 obstacle/foothold observation |
| MuJoCo、TorchScript、ROS2/C++ | 所有专项策略的 sim2sim/sim2real 骨架 | 最新 7 维限高观测尚未贯通部署端 |
| 旧 RoboCon 全场 XML | 复用地图加载、传送点、键盘和高度查询代码 | 障碍几何和新版规则不一致，不能直接验收 |

## 3. 统一训练架构

### 3.1 算法选择

主算法继续使用 PPO，不在当前阶段切换 SAC/TD3。

- **低层 locomotion expert**：当前 PPO-CSE/RMA；
- **特权教师**：可读取精确障碍几何、下一落足块、阶段状态；
- **部署 student**：仅使用 IMU、关节状态、动作历史、高程图和可从真实传感器产生的障碍状态；
- **蒸馏**：DAgger/teacher-student，避免 student 仅模仿教师访问过的窄状态分布；
- **离散落足障碍**：采用“启发式/搜索生成 foothold target + RL 跟踪”的混合方案，优先于纯端到端探索；
- **绕桩、单木桥**：先测试外部 waypoint/pure-pursuit 给现有步态策略发命令，若已达标，不额外训练低层策略。

### 3.2 统一观测

保留当前本体感受历史和 11×7 高程点，新增版本化 `robocon_obstacle_state`：

```text
skill_id / phase_id
目标点 body-frame dx, dy
目标航向 sin(yaw_error), cos(yaw_error)
障碍前向距离、横向偏差
有效宽度/高度/间隙（归一化）
下一 foothold 的 body-frame xy 和有效标记（离散落足任务）
阶段完成标记（逐级台阶、梅花桩、绕桩必达区）
```

教师可以额外读取真实几何和接触标签；student 不得读取真机无法产生的 ground truth。

11×7 高程点只作为现有基线，不默认它足以支持精确落足。在梅花桩、轮胎阵开训前必须做空间分辨率实验：

- 将 200×200 mm 方块、轮胎内孔和足端安全边界投影到当前采样网格；
- 若任一可落足区在网格中不能稳定区分，则增加局部高分辨率 crop，或输入 foothold mask/next-foot target；
- student 输入增加可落足区置信度，低置信度时降速或停止；
- 高程图、foothold mask 和真机深度管线必须共享同一坐标系、尺度和延迟模型。

### 3.3 统一动作与频率

- Policy 输出 12 维关节目标偏移；
- `control_dt = 0.02 s`，action scale 初始沿用 `0.25`；
- 除高墙/跨沟 expert 外，不主动扩大 action scale；
- 高墙/跨沟如确需扩大，先检查关节限位、峰值扭矩、落地冲击和真机安全；
- 所有 expert 支持 zero-command 安全站立和观测异常降速/站立。

### 3.4 训练前硬件能力门槛

高墙、跨沟和限高不允许直接用 PPO 试错验证机械可行性。先使用 URDF/MuJoCo 运动学扫描、准静态轨迹和低速吊架实验建立能力表：

| 评估项 | 对应障碍 | 必须记录 | Go/No-Go |
|---|---|---|---|
| 最大稳定抬腿高度 | 高墙、台阶 | 足端工作空间、关节余量、机身姿态 | 有安全余量后才允许墙高 curriculum |
| 最大可控跨步/飞跃距离 | 沟壑、梅花桩 | 起跳速度、落足余量、落地冲击 | 360 mm 加落地安全余量可达 |
| 最低安全全机高度 | 限高框架 | 背部、腿部、线缆和传感器最高点 | 可在 290 mm 内连续移动 0.8 m |
| 单腿支撑与窄支撑面 | 梅花桩、轮胎阵 | 峰值/持续扭矩、侧向稳定时间 | 不超过电机和减速器安全包络 |
| 连续负载与落地冲击 | 全部 | 温度、电流/扭矩、关节峰值加速度 | 完成保守轨迹后无过温/过流/撞限位 |

结果写入 `hardware_capability.yaml`，作为动作限位、curriculum 上限和实机安全监视的唯一参数源。

### 3.5 通用奖励起点

以下系数适用于当前工程“普通 dense reward 乘以 `dt`，event reward 内部除以 `dt`”的约定。event 系数代表一次事件的实际分值。

| 奖励项 | 初始 scale | 目的 |
|---|---:|---|
| `task_progress` | `+2.0` | 沿当前阶段目标方向产生真实进度 |
| `tracking_lin_vel` | `+1.0` | 保持可控速度，不鼓励原地刷奖励 |
| `tracking_ang_vel` | `+0.5` | 跟踪规划器或遥控航向 |
| `orientation` | `-0.25` | 抑制非任务所需的滚转/俯仰 |
| `lin_vel_z` | `-2.0` | 普通技能抑制垂向振荡；墙/跨沟阶段动态调整 |
| `ang_vel_xy` | `-0.05` | 抑制翻滚 |
| `torques` | `-2e-4` | 限制持续大扭矩 |
| `dof_acc` | `-2.5e-7` | 限制冲击性关节变化 |
| `action_rate` | `-0.01` | 平滑动作 |
| `feet_slip` | `-0.025` | 减少打滑 |
| `dof_pos_limits` | `-2.0` | 避免撞限位 |
| `fall_event` | `-20` | 摔倒事件 |
| `task_fail_event` | `-10` | 规则级失败，如碰桩、两足落到禁区 |
| `task_success_event` | `+20` | 完整通过当前障碍 |

调参原则：先确保 success rate 与 reward 同向，再调整效率和动作质量。禁止仅因为总 reward 上升就晋级课程。

### 3.6 每个 expert 的统一八阶段流程

1. **G-1 物理可行性**：硬件能力表、准静态轨迹、扭矩/冲击上限；
2. **G0 几何测试**：不训练，验证尺寸、碰撞、出生点、接触标签和 success/fail 判据；
3. **G0.5 零训练基线**：当前 checkpoint 同时跑 Isaac Gym 与 MuJoCo，先分类为可直用、需 fine-tune、需新 expert；
4. **G1 行为发现**：固定动力学、中心出生、简单几何、特权观测，训练出动作；
5. **G2 比赛尺寸**：成功率驱动 curriculum，逐步达到规则精确尺寸；
6. **G3 跨引擎鲁棒化**：每个主要 curriculum 台阶都在 MuJoCo 回归，同时加入几何误差、摩擦、质量、延迟和观测噪声；
7. **G4 Student/部署观测**：移除 ground truth，用 DAgger/蒸馏或直接 fine-tune；
8. **G5 切换与实机**：吊架、低速放开，并验收 locomotion/expert/recovery 切换。

每个 run：25 iter smoke → 200 iter 短跑 → 正式训练；每 25 iter 存 checkpoint，每 100 iter 自动小样本评估。一旦首个 expert 达到 MuJoCo 门槛，即开始切换集成，不等其余障碍完成。

## 4. 八个障碍专项计划

## 4.1 绕桩

### 动作分解

1. 进入起点必达区；
2. 对准第一段桩列；
3. 左右交替绕过每根桩；
4. 进入拐角必达区并完成约 90°转向；
5. 沿第二段继续 S 形绕桩；
6. 进入终点必达区并停稳/退出。

### 训练方法

先不训练低层 RL：使用 pure-pursuit/waypoint planner 给现有策略发送 `(vx, vy, yaw_rate)`。如果 nominal 通过率 ≥90%，只训练/优化高层 planner。

若现有低层转弯半径或侧向跟踪不足，再训练 waypoint-conditioned expert。

课程：

- 杆间距 `1.2 → 1.0 → 0.9 m`；
- 杆半径 `0.03 → 0.05 → 0.0525 m`；
- 必达区命中半径 `0.25 → 0.15 m`；
- 速度 `0.25 → 0.50 m/s`；
- 起点横向偏差 `0 → ±0.10 m`，航向偏差 `0 → ±10°`。

### 专项奖励

- `waypoint_progress +2.0`：到当前 waypoint 的距离下降；
- `waypoint_reached +3`：首次进入当前必达区；
- `heading_alignment -1.0`：接近 waypoint 时惩罚航向误差；
- `pole_clearance -2.0`：距离桩小于安全半径时连续惩罚；
- `pole_contact -10`：任意碰桩立即失败；
- `skip_waypoint -10`：未经过拐角必达区直接到终点；
- `complete +20`。

### 验收

- 三个必达区按顺序命中；
- 不碰杆、不从杆投影上方跨过；
- nominal 1000 episodes ≥95%；robust ≥90%；
- MuJoCo 3×30 次 ≥85%；
- 实机连续 20 次至少 18 次成功。

## 4.2 限高框架

### 动作分解

1. 入口居中和对正；
2. 框架前 0.5–0.8 m 开始降低身体；
3. 前腿进入框架；
4. 在整个 0.8 m 长度内保持全身最高点低于横梁；
5. 后腿完全离开框架；
6. 恢复正常高度和速度。

### 训练方法

从当前 `063296` 恢复，但必须把单横杆改为完整框架：内高 0.29 m、内宽 0.8 m、长度 0.8 m。增加 phase：`approach/crouch/inside/recover`。

课程：

- 高度 `0.36 → 0.32 → 0.29 → 0.2755 m`；
- 宽度 `1.10 → 0.90 → 0.80 → 0.76 m`；
- 长度 `0.40 → 0.80 → 0.84 m`；
- 横向偏差 `0 → ±0.08 m`，航向 `0 → ±8°`。

### 专项奖励

- `approach_alignment -1.0`；
- `crouch_target +1.0`：只在接近框架后和框内阶段启用，框架外不惩罚正常站立高度；
- `body_top_clearance -100`：只在结构将进入/已进入框内时，对超出可通过包络的程度做 hinge penalty；
- `inside_progress +3.0`：身体在框内的前向进度；
- `frame_contact`：只记录接触次数、部位、冲量，轻微接触不罚、不终止；
- `excessive_collision -5`：当峰值力/接触冲量超过经实物标定的安全阈值时惩罚；
- `frame_displacement -10`：框架位移超过裁判可见阈值时判规则级失败；
- `early_recover -5`：后腿未离开框架就抬高；
- `recover_height +1.5`；
- `complete +20`。

### 验收

- 全身完整穿过，不能暴力撞移框架；
- 报告最大接触力、最小 base height、最大 body-top height；
- 0.29 m nominal ≥95%，±5%尺寸/位姿 robust ≥90%；
- MuJoCo ≥85%；实机 20 次至少 18 次成功。

## 4.3 高墙

### 动作分解

1. 对正 0.8 m 宽墙体；
2. 形成适当接近速度；
3. 前腿抬高/搭墙或直接跃过；
4. 后腿蹬地，重心越过墙面；
5. 四足全部越过墙线；
6. 另一侧稳定落地并继续行走。

### 训练方法

使用 privileged teacher PPO。前期允许读取墙高、距离和阶段；后期 student 只用高程/深度提取的墙状态。若 PPO 难以发现动作，可用手工关键姿态或优化出的短 motion 做 reference，但最终仍由 RL residual 控制。

课程：

- 墙高 `0.10 → 0.15 → 0.20 → 0.25 → 0.30 → 0.315 m`；
- 墙宽 `1.2 → 0.8 → 0.76 m`；
- 厚度 `0.15 → 0.08 m`；
- 起始距离和接近速度随机化；
- 最后加入落地侧摩擦、质量和延迟随机化。

### 专项奖励

- `approach_progress +2.0`；
- `front_feet_clearance +1.0`；
- `front_feet_on_top +2`（允许攀爬路线）；
- `com_crossed +8`；
- `all_feet_crossed +12`；
- `stable_landing +5`；
- `wall_side_stuck -5`；
- `hard_impact -2.0`；
- `fall -20`，`complete +20`。

墙顶接触不能一概惩罚，因为规则允许攀爬通过。

### 验收

- base/躯干和四足均越过墙线；
- 落地后稳定 1 s；
- 无绕墙、无从侧面作弊；
- Isaac nominal ≥90%、robust ≥85%；MuJoCo ≥80%；
- 实机必须先吊架，从 0.15 m 逐级增加。

## 4.4 沟壑组合

### 动作分解

1. 对准台阶或斜坡入口；
2. 逐级上三级 120 mm 台阶，或沿 20°斜坡登顶；
3. 在 800×800 mm 平台调整站位；
4. 跨越 360 mm 沟壑；
5. 在对侧 800×800 mm 平台稳定；
6. 从另一方向的斜坡/台阶返回地面。

建议先分别训练三个 expert/subskill：`stairs_120`、`gap_360`、`ramp_20deg`，再训练带 phase 的组合 teacher，最后蒸馏为一个沟壑组合 expert。

### 训练课程

- 台阶高度 `0.06 → 0.09 → 0.12 → 0.126 m`；
- 踏面深度 `0.40 → 0.32 m`；
- 沟宽 `0.10 → 0.20 → 0.28 → 0.36 → 0.378 m`；
- 斜坡 `10° → 15° → 20° → 21°`；
- 平台宽度 `1.2 → 0.8 → 0.76 m`；
- 分阶段先无 gap，再逐步打开 gap collision。

### 专项奖励

- `stage_progress +2.0`；
- `new_stair_surface_contact +2/级`：每级首次有效足底接触；
- `platform_reached +4`；
- `gap_takeoff +2`：从平台有效起跳；
- `gap_com_crossed +6`；
- `gap_all_feet_landed +8`；
- `ramp_complete +4`；
- `skipped_stair -10`；
- `>=2 feet on forbidden floor -10` 并终止；
- `fall -20`；完整完成 `+20`。

跨沟阶段可以降低 `lin_vel_z` 惩罚，增加 landing impact 惩罚，避免为追求距离形成不可控跳跃。

### 验收

分开报告：上台阶/斜坡、跨沟、下台阶/斜坡三部分，匹配比赛的分段计分。自主口径分别是 75/150/75 分，遥控口径分别是 50/100/50 分。规则表 4 已将三部分分开计分，但“重新越障时先前子项分是否保留”仍需向裁判组获得书面口径。

- 每级台阶顶面至少一次有效足底接触；
- 360 mm gap 完整跨越；
- 跨越过程中不得出现两足及以上落到障碍外地面；
- Isaac 各子项 ≥95%，完整组合 ≥90%；
- MuJoCo 完整组合 ≥85%；
- 实机先做上下台阶，再做带安全网的 gap。

## 4.5 砖木坑

### 动作分解

1. 从 1 m 短边对正进入；
2. 在碎砖区保持连续落足和前进；
3. 完成 L 形转向；
4. 在松树皮区控制下陷和打滑；
5. 从另一短边离开。

### 训练方法

第一版不做昂贵颗粒仿真，采用参数化近似：随机高度块、局部下陷偏置、低/非均匀摩擦、足端柔顺、随机滑移脉冲。实机采集足端电流、base 速度和下陷数据后再校准。

课程：

- 平整刚性坑 → 10 mm 噪声 → 20–40 mm 随机块；
- 摩擦 `0.8 → U(0.25, 0.9)`；
- 足端等效下陷 `0 → 30 mm`；
- 直线坑 → L 形坑；
- 速度 `0.25 → 0.45 m/s`。

### 专项奖励

- `inside_progress +2.0`；
- `inside_corridor +1.0`；
- `yaw_to_L_path -1.0`；
- `feet_slip -0.05`（强于普通地形）；
- `stuck -5`；
- `>=2 feet outside pit -10`；
- `fall -20`；`complete +20`。

### 验收

- 只从短边进入/离开；
- 完成 L 形全程；
- 任意时刻两足及以上在坑外地面即失败；
- 仿真 robust ≥85%；
- 必须在规则尺寸实物砖/松树皮上完成实机 20 次测试，仿真成绩不能替代实物验收。

## 4.6 轮胎阵

### 动作分解

1. 从短边对正；
2. 前足踩到第一排轮胎胎面；
3. 依次在 2×4 轮胎上选择稳定胎面落足；
4. 避免足端落入轮胎内孔或两胎缝隙；
5. 从另一短边离开。

### 训练方法

必须用真实轮胎 mesh 或带内孔的复合碰撞体分别表现环形胎面、内孔和胎间地面，不能只用 2.5D 高度场或实心圆盘做最终验收。规则中的 165 mm 是胎面宽度，不直接等于足端可用接触宽度，必须由实物断面或 mesh 实测安全环形区域。采用 privileged foothold teacher：teacher 知道胎面有效区域并给出下一 foothold，student 从局部高程/深度恢复。

课程：

- 先用实心圆盘；
- 逐渐打开中心孔；
- 胎面宽从大到规则 165 mm；
- 两胎间距从紧密到真实布局；
- 最后随机轮胎转角、轻微高度差和摩擦。

### 专项奖励

- `valid_tread_contact +1/次`；
- `next_cell_reached +2`；
- `foot_in_hole -5`：训练引导项，不直接等同规则失败；
- `foot_between_tires -3`；
- `body_tire_collision -2`；
- `skip/side_exit -10`；
- `>=2 feet on outside rectangle floor -10`：规则级失败并终止；
- `fall -20`；`complete +20`。

### 验收

- 从短边进入并从另一短边离开；
- 不得绕行；
- 只有两足及以上接触轮胎阵外矩形地面时才按规则失败，踩入内孔单独记录；
- 报告孔洞踩入次数、侧向退出次数和完成时间；
- Isaac nominal ≥90%、robust ≥85%；MuJoCo ≥80%；
- 实机必须配置侧面保护绳。

## 4.7 梅花桩

### 动作分解

1. 从 800×800×200 mm 平台起步；
2. 按两排交错顺序踩到所有 200 mm 方块；
3. 在窄支撑面上调整身体姿态；
4. 跨越 200/400 mm 间距；
5. 到达对侧平台并稳定离开。

### 训练方法

从现有 stepping-stones 策略 warm start。使用 foothold target + RL 跟踪，teacher 读取下一方块中心，student 使用高程图/深度。

课程：

- 方块宽 `0.40 → 0.30 → 0.20 → 0.19 m`；
- 间距 `0.10 → 0.20/0.40 → 0.42 m`；
- 高度 `0.05 → 0.20 → 0.21 m`；
- 固定直线排列 → 新版交错排列；
- 出生偏差 `0 → ±0.05 m/±5°`。

### 专项奖励

- `next_block_contact +2/块`：仅该顶面首次有效接触时发放，用 episode bitset 防止重复刷奖励；
- `all_required_blocks +10`：仅当 bitset 全部置位后一次性发放；
- `foothold_center +1.0`；
- `edge_contact -1.0`；
- `skip_block -10`；
- `>=2 feet on floor -10`；
- `side_fall -20`；`complete +20`。

### 验收

- 每个梅花桩顶面至少一次有效足底接触；
- 不从侧面落下；
- 两足及以上接触外部地面判失败；
- Isaac nominal ≥95%、robust ≥90%；MuJoCo ≥85%；实机 ≥18/20。

## 4.8 单木桥

### 动作分解

1. 从 400 mm 短边对准；
2. 前足登上 100 mm 高桥面；
3. 身体中心和航向锁定桥中心线；
4. 走完 3 m；
5. 从另一短边下桥并稳定。

### 训练方法

第一步先用现有通用策略 + pure-pursuit/横向稳定器测试，不急于新训练。达不到目标再训练 bridge expert。

课程：

- 桥宽 `0.80 → 0.60 → 0.40 → 0.38 m`；
- 桥长 `1.5 → 2.0 → 3.0 → 3.15 m`；
- 高度 `0.03 → 0.10 → 0.105 m`；
- 起点横向 `0 → ±0.10 m`、航向 `0 → ±8°`；
- 摩擦和轻微侧向扰动逐步加入。

### 专项奖励

- `bridge_progress +2.0`；
- `centerline -2.0*|y_error|`；
- `heading -1.0*|yaw_error|`；
- `foot_edge_clearance +1.0`：足端距桥沿大于 50 mm；
- `edge_contact -1.0`；
- `>=2 feet on outside floor -10`；
- `side_fall -20`；`complete +20`。

### 验收

- 从短边进入并从另一短边离开；
- 不得侧面上下桥；
- 两足及以上接触桥外地面判失败；
- Isaac nominal ≥98%、robust ≥95%；MuJoCo ≥90%；实机 ≥19/20。

## 5. 可复用开源项目

| 项目 | 建议用途 | 复用级别 |
|---|---|---|
| [RSL-RL](https://github.com/leggedrobotics/rsl_rl) | PPO、teacher-student distillation、训练 runner | **直接沿用**；当前工程已有同源实现，不建议大版本迁移 |
| [Legged Gym](https://github.com/leggedrobotics/legged_gym) | reward/terrain/curriculum 基础结构 | 当前工程已经派生自此，只取思路和小模块 |
| [Extreme Parkour](https://github.com/chengxuxin/extreme-parkour) | 高墙、跨沟、离散地形、scandot/camera student、terrain curriculum | **重点参考/移植模块**；与当前 Isaac Gym/Python 版本接近 |
| [Robot Parkour Learning](https://github.com/ZiwenZhuang/parkour) | climb/leap/crawl 专项教师、软到硬课程、视觉 student、实机部署 | **优先借鉴**高墙、沟壑、限高和 teacher-student；MIT license |
| [Walk These Ways](https://github.com/Improbable-AI/walk-these-ways) | body-height、步频、步态相位等行为参数化；sim2real 部署结构 | 借鉴限高姿态命令、慢速桥面步态和多行为控制，不整体替换当前策略 |
| [UW Lab locomotion environments](https://uw-lab.github.io/UWLab/main/source/overview/uw_environments.html) | Spot 的 gap、pit、stepping-stone、obstacle 位置跟踪环境 | 作为任务拆分和 evaluator 参考；它们基于 Isaac Lab，不直接替换当前 Isaac Gym 训练栈 |
| [Elevation Mapping CuPy](https://github.com/leggedrobotics/elevation_mapping_cupy) | 真机 GPU 高程图，为离散落足和桥面提供观测 | 部署阶段接入；需要与当前 11×7 observation contract 做适配 |
| [PACE Sim2Real](https://github.com/leggedrobotics/pace-sim2real) | 用编码器数据辨识关节阻尼、摩擦和执行器参数 | 当前版本基于 Isaac Lab，**不直接迁移训练栈**；借鉴数据采集和参数拟合流程 |

直接复用优先级：

1. 从 Robot Parkour/Extreme Parkour 移植 terrain curriculum、obstacle state 和 teacher-student 结构；
2. 保留当前 PPO-CSE、history、checkpoint 和部署接口；
3. 离散落足任务优先移植 foothold/terrain 逻辑，不复制其整套机器人配置；
4. 所有复制代码先核对 LICENSE，并在本仓库保留来源说明。

## 6. 统一验收体系

### 6.1 几何、规则和硬件门禁

每个障碍必须有自动测试：尺寸与规则一致、接触类型可分辨、scripted trajectory 能触发 success，明确违规轨迹能触发 fail，重置后一次性事件和 bitset 归零。高墙、跨沟、限高还必须先通过 `hardware_capability.yaml` 门禁。

### 6.2 分级仿真评估量

| 阶段 | Episode 数 | 用途 | 输出 |
|---|---:|---|---|
| Smoke Test | 20-50 | 验证代码、奖励、判据和无作弊路径 | 事件计数、失败录像 |
| 阶段评估 | 100-200 | 筛选 checkpoint、决定 curriculum 是否升级 | 成功率、耗时、失败类型 |
| 正式验收 | ≥1000 | held-out seeds 和扰动组合下的最终结论 | Wilson 95% CI、paired delta CI |

Isaac Gym 至少覆盖 nominal、初始位姿、几何±5%、动力学、延迟和传感器 dropout。每个 curriculum 主要台阶同步跑 MuJoCo 100-200 次回归，不允许等策略完成后才第一次 sim-to-sim。

### 6.3 每个策略必报指标

- 规则级成功率与置信区间；
- 平均、中位数和 P95 完成时间；
- 失败类型及占比；
- 峰值关节扭矩/电流、落地冲击、最高电机温度；
- Isaac Gym 与 MuJoCo 的成功率差、耗时差和峰值扭矩差；
- 失败后恢复成功率和平均恢复时间；
- 对 flat/stairs 基础能力的回归幅度。

Isaac 到 MuJoCo 的成功率下降超过 10 个百分点，或 12 cm 台阶的跨引擎偏差未解决时，禁止扩大高墙/跨沟 curriculum。

### 6.4 实机验收

实机不再只是最后一阶段。在训练前先完成能力表；每个技能达到首个 MuJoCo 门槛后，依次执行吊架/安全绳 10 次、低速 20 次、规则速度 30 次。每次记录视频、observation、action、IMU、关节状态、电流/扭矩、温度、接触和失败原因。规则速度下目标为 ≥27/30；没有实机成功率时不标记为完成。

### 6.5 切换和全场验收

首个 expert 完成后就验收：通用运动→入口对齐→专项策略→出口稳定→通用运动，以及任一阶段进入 recovery 后的安全恢复。要求不重启进程、不人工清 hidden state，切换过程不摔倒。

全场对 20 个随机布局、两个启动区和多个行走顺序进行 180 s 评测。报告有效得分、用时、罚分、返回分、风险暴露和放弃障碍的决策。选路器根据当前技能成功率、P95 耗时和障碍间距离最大化预期最终得分，不固定照研发优先级走。分别维护遥控和全自主两套成绩。

## 7. 得分优先级与里程碑

### 7.1 基线验证/研发资源顺序

| 优先级 | 障碍/子项 | 路线 | 原因 |
|---:|---|---|---|
| 1 | 绕桩 | 现有策略 + waypoint/pure-pursuit 零训练测试 | 可能不需要新 RL，最快形成规则级得分 |
| 2 | 单木桥 | 现有策略 + 中心线跟踪，失败再 fine-tune | 训练成本相对低 |
| 3 | 限高框架 | `063296` 零训练测试后 fine-tune | 已有相关基础 |
| 4 | 沟壑的台阶/斜坡子项 | `059950`/台阶 checkpoint 在新版地图复测 | 可尽快把已有能力变成 75/50 分子项 |
| 5 | 360 mm 跨沟 | 独立 gap expert，长周期实验可并行 | 高分但高成本，必须先过硬件/Sim-to-Sim 门禁 |
| 6 | 梅花桩 | Foothold Planner → next target → PPO → contact bitset | 需要精确落足 |
| 7 | 高墙 | climb/jump expert | 动力学和实机风险高 |
| 8 | 砖木坑/轮胎阵 | 复杂地形适应 + 落足引导 + 实物标定 | 仿真与实物差异最难建模 |

这是基线验证和研发资源顺序，不是全场固定行走顺序，也不要求前一项完全结束才开始下一项。

### 7.2 里程碑与粗略时间

以 1 名主要开发者、1 台可用训练工作站、实机每周至少 2 个半天为估算前提；时间是决策用范围，不是交付承诺。

| 里程碑 | 预估 | 核心工作 | 交付/Go-No-Go |
|---|---:|---|---|
| M0 | 3-5 天 | 规则建模、硬件能力表、12 cm 台阶 Isaac/MuJoCo 偏差定位 | 可信评测器；Sim-to-Sim 差距降到 ≤10 个百分点，否则不开始高风险 curriculum |
| M1 | 3-5 天 | 绕桩、400 mm 单木桥、290 mm 完整限高框架零训练测试 | 第一批“可直用/需微调/需重训”结论 |
| M2 | 1-2 周 | 限高 fine-tune、台阶/斜坡新版复测、第一个 expert 切换 | 多技能初步集成，完成至少两个可得分项 |
| M3 | 2-4 周 | 360 mm 跨沟与 300 mm 高墙长周期并行实验 | 两项均达 MuJoCo 门槛，否则保留台阶/斜坡子项得分路线 |
| M4 | 3-5 周 | 梅花桩、轮胎阵、砖木坑，实物数据驱动校准 | 离散/复杂地形技能和真实失败分布 |
| M5 | 2-4 周 | 180 s 全场计分、两启动区选路、自动规划、实机优化 | 比赛级系统；先稳定获得高期望分，再冲击 8 障碍+返回 |

## 8. 建议新增代码结构

```text
docs/2027_ROBOCON_OBSTACLE_TRAINING_PLAN.md
robodog_gym/envs/robocon_2027/
  task_base.py
  slalom.py
  low_frame.py
  wall.py
  gully.py
  brick_wood_pit.py
  tire_array.py
  plum_posts.py
  beam.py
robodog_gym/utils/robocon_2027_terrain.py
resources/objects/robocon_2027/
scripts/train_robocon_2027.py
scripts/eval_robocon_2027.py
scripts/eval_robocon_2027_mujoco.py
evaluations/robocon_2027/
```

不要继续在 `train_mybot_v3_stair_turn.py` 里堆所有障碍分支。共享逻辑放 `task_base.py`，每个障碍只实现 geometry、phase、reward、success/fail 和 curriculum。

## 9. 第一批具体任务

第一轮不启动八个 expert，只交付四个基线结果：

1. `059950`/通用运动策略能否通过规则版绕桩；
2. 现有运动策略能否通过 400 mm 宽、3 m 长、100 mm 高单木桥；
3. `063296` 能否通过 290×800×800 mm 完整框架，包括轻微接触和框架位移记录；
4. 同一台阶策略在 Isaac Gym 和 MuJoCo 中分别可稳定通过多高，并定位 12 cm/8 cm 差距的首个主要来源。

为获得这四个结果，先完成规则常量、几何资产、统一 evaluator、`hardware_capability.yaml`、自动时间/扭矩/失败类型记录。结果决定第二轮：可直用的不训练，只有接近门槛的做 fine-tune，明确不具备能力的才新建 expert。

## 10. 风险与回退规则

出现以下情况立即停止 curriculum：

- reward 持续升高但 success rate 不升；
- 策略绕过障碍、跳过必达区/踏面、通过站着不动刷奖励；
- Actor 使用了真机无法生成的 privileged observation；
- action saturation、落地冲击或峰值扭矩持续增大；
- Isaac 成功但 MuJoCo 下降超过 10 个百分点；
- 基础 flat/stairs 能力下降超过 5 个百分点；
- 导出 JIT 与训练模型相同输入下输出不一致。

排查顺序固定为：几何 → success/fail 判据 → observation → checkpoint/导出 → 动力学 → reward → curriculum → PPO 超参数。
