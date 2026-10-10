# MyBot V3 RoboCon 障碍续训计划

## 1. 目标、基线与总体原则

目标是在保留 `059950` 现有平地、转向、停止和台阶能力的前提下，让一个最终策略可靠完成 RoboCon 实际地图中的四类障碍：

- 木桥 A：1000 mm 宽、400 mm 分段桥面、150 mm 间隙、14° 上下桥坡；
- 木桥 B：约 200 mm 窄条板与约 100 mm 槽/间隙；
- 限高杆：300 mm 净空、1000 mm 横杆跨度；
- 高墙：300 mm 高、1000 mm 长、50 mm 厚（口语里的“矮墙”）。

本轮不更换整套强化学习框架，继续使用当前 PPO/CSE 与 Isaac Gym 训练链路，吸收 Extreme Parkour 的课程学习、特权 teacher 和 teacher-student/DAgger 思路。

执行时遵守以下原则：

1. `059950` 永久冻结，只读保存，不覆盖、不原地续写；
2. 第一阶段不同时启动三个专家，先用 Low Bar 打通端到端训练链路；
3. 训练顺序固定为 `Low Bar → Bridge A → Bridge B → Wall → Student`；
4. Actor 只能使用真机能够稳定产生的观测，仿真 ground truth 只能给 Critic、标签和评测器使用；
5. 用可量化 milestone 决定晋级，不以“训练了几天”或“reward 看起来上升”作为完成标准；
6. 每个新技能训练期间持续回放平地和台阶，任何障碍能力提升都不能以明显损伤原能力为代价；
7. 所有候选 checkpoint 必须经过自动评测，禁止只凭训练曲线或单次可视化挑选模型。

## 2. 为什么先做 Low Bar

当前仓库已经具有：

- `low_bar` 刚体障碍 actor；
- `obstacle_ahead` 三维观测；
- `low_bar_crouch` 奖励；
- Isaac Gym 训练入口；
- 现有 MuJoCo 导出和运行链路。

因此 Low Bar 是验证整套框架成本最低的任务。第一条必须证明成立的链路是：

```text
059950 冻结基线
    ↓
观测扩维兼容迁移
    ↓
Low Bar 单障碍续训
    ↓
独立 success rate 上升
    ↓
checkpoint 自动筛选
    ↓
策略导出
    ↓
MuJoCo 使用同一观测契约通过
    ↓
真机来源格式的观测回放/吊架验证
```

在这条链路完成前，不启动 Bridge 和 Wall 的正式长训练。可以提前写环境与测试，但不能同时消耗训练资源做三个专家调参，否则很难判断失败来自算法、观测、地形、导出还是评测链路。

## 3. 可借鉴的开源方案

### 3.1 Extreme Parkour（主要参考）

- 代码：[chengxuxin/extreme-parkour](https://github.com/chengxuxin/extreme-parkour)
- 论文：[Extreme Parkour with Legged Robots](https://arxiv.org/abs/2309.14341)
- 与本项目同属 Isaac Gym、PPO、teacher-student 技术路线；
- 可借用 step、gap、hurdle、tilted ramp 的课程设计；
- 可借用障碍方向输入、边缘接触惩罚、特权 teacher、BC + DAgger；
- 不直接照搬机器人模型、观测维度、动作尺度或网络权重。

### 3.2 RSL-RL（继续沿用）

- 代码：[leggedrobotics/rsl_rl](https://github.com/leggedrobotics/rsl_rl)
- 继续作为 PPO 和后续 student-teacher 训练后端；
- 本项目现有 checkpoint 与部署链路已经建立在相近体系上，整体替换框架的收益小于风险。

### 3.3 RoboCon 26 / Isaac Lab 参考工程

- 代码：[Taojunfeng123/quadruped_robot_lab](https://github.com/Taojunfeng123/quadruped_robot_lab)
- 参考其地形课程、动作课程、周期停止和奖励配置；
- 当前不整体迁移到 Isaac Lab，避免打断已验证的 Isaac Gym → MuJoCo 链路。

### 3.4 仓库内 RC_WheelLeg 参考工程

- 路径：`robocon_reference/RC_WheelLeg/05_software/train/rc_mjlab/`
- 其比赛方案将 Rough、Wall、Crawl 分开训练和控制，说明不同障碍的动作分布存在明显冲突；
- 这支持“先顺序训练专家、再蒸馏合并”，而不是一开始把所有障碍放进同一个 PPO run。

## 4. M0：训练前必须完成的基础设施

### 4.1 冻结并记录 `059950`

需要保存：

- 原始 checkpoint 和 SHA256；
- 对应训练配置；
- Actor/Estimator/Critic 的输入输出维度；
- 归一化参数和动作缩放；
- 导出模型；
- flat、turn、stairs、zero-command 的基线评测结果；
- 一批可重复使用的观测回放样本和基线动作输出。

之后所有新模型都使用新的 run 名称和目录，禁止覆盖基线。

### 4.2 解决 checkpoint 观测扩维兼容

当前 Low Bar 配置会给 policy 和 estimator 都追加 3 维 `obstacle_ahead`。RSL-RL 默认严格加载 `state_dict`，因此新的网络输入层不能直接加载旧维度的 `059950`。

迁移方法：

1. 先冻结最终 Actor observation contract，后续障碍不得再次随意改变输入长度；
2. 创建新维度网络；
3. 对相同形状的参数正常复制；
4. 对 Actor、Estimator、history/adaptation module 的第一层扩大输入列数；
5. 旧输入对应列完整复制，新观测对应列初始化为 0；
6. Critic 按新的特权观测单独迁移，不能把 Critic 的 ground truth 输入泄漏给 Actor；
7. 不恢复旧 optimizer state，使用较小学习率重新建立 optimizer；
8. 在新观测全为 0 时，用固定观测回放验证新旧模型动作输出。

验收要求：

- 固定回放 batch 上，新旧动作最大绝对误差建议 `≤ 1e-5`；
- flat、turn、stairs 成功率下降不超过统计误差；
- zero-command 站立没有重新出现四肢抽动；
- 模型可以完成训练加载、保存、导出和 MuJoCo 加载。

### 4.3 建立统一的 observation contract

#### 坐标和语义

障碍相对状态统一使用机器人 base yaw frame：

- `+x`：机器人前方；
- `+y`：机器人左侧；
- `+z`：向上；
- 距离和高度单位：米；
- 角度内部建议使用 `sin/cos` 表示，避免 `-π/π` 跳变。

建议的逻辑字段为：

```text
skill_id = one_hot(normal, bridge, low_bar, wall)

obstacle_state = [
    distance_x,
    offset_y,
    sin(heading_error),
    cos(heading_error),
    obstacle_height,
    clearance,
    valid_or_confidence,
]
```

桥面局部支撑扫描和上方净空扫描使用固定长度数组，具体点数必须根据真机已有高度传感器或准备新增的传感器确定。M0 完成后，不允许在不同任务中通过 append/remove observation 改变 Actor 输入长度；不适用的字段填 0，并将 `valid_or_confidence` 置为 0。

#### 仿真与真机来源

| Observation | Isaac Gym | MuJoCo | 真机建议来源 |
|---|---|---|---|
| `skill_id` | 场景状态机 | 路线状态机 | 上层路线状态机 |
| `distance_x` | 障碍位姿经传感器模型计算 | 同一计算模块 | 定位结果 + 已知地图，或视觉/雷达 |
| `offset_y` | 障碍位姿经传感器模型计算 | 同一计算模块 | 定位结果 + 已知地图，或视觉/雷达 |
| `heading_error` | base 与障碍方向差 | 同一计算模块 | IMU/里程计定位 + 地图障碍方向 |
| `obstacle_height` | GT 经过噪声和量化 | 同一传感器模型 | 已知比赛地图或视觉估计 |
| `clearance` | GT 经过噪声和量化 | 同一传感器模型 | 已知地图、姿态和高度估计，必要时专用传感器 |
| `surface_scan` | ray cast/height query | ray cast | 真实高度扫描 |
| `overhead_scan` | upward/forward ray | 同等射线 | 地图参数或真实上方测距 |
| `valid/confidence` | 模拟丢帧和异常 | 模拟丢帧和异常 | 时间戳、新鲜度和感知置信度 |

真机路线固定时，优先使用“定位 + 已知地图 + 路线状态机”，这是第一版风险最低的实现。视觉或激光识别可以作为后续校验和地图修正，不作为第一条训练链路的前置依赖。

#### 数据质量要求

每个字段还必须在单独的 contract 文档或配置中写清：

- normalization、clip 范围和无效值；
- 更新频率、时间戳和最大允许 age；
- 通信延迟与零阶保持策略；
- 丢帧、跳变和超时处理；
- 训练时对应的 noise、bias、latency、dropout 范围；
- `valid=0` 时的安全行为。

建议安全降级逻辑：观测短时失效时保持最近有效值并限制速度；超过超时阈值后退出障碍技能，进入站立或人工接管，不能继续使用陈旧障碍状态盲走。

### 4.4 自动评测 harness

每个 checkpoint 自动输出：

- 成功率和 95% 置信区间；
- 摔倒率、超时率和碰撞率；
- 通过时间；
- 横向误差和航向误差；
- 最小净空；
- 有效落足率和边缘/桥外接触次数；
- 最大力矩、动作饱和率、action rate；
- base 高度、roll/pitch 峰值；
- 通过障碍后的恢复时间；
- flat/stairs/zero-command 回归结果。

Checkpoint 排名顺序：

1. held-out success rate；
2. 碰撞、摔倒和踏空；
3. 原能力回归；
4. 动作安全性和平滑度；
5. 通过时间；
6. training reward 仅作诊断，不作为最终选择标准。

建议样本规模：

- Isaac Gym：每个候选至少 1000 episodes；
- MuJoCo：每个正式候选至少 200 episodes；
- 真机：先吊架和低速安全测试，再做每种条件 20～30 次可统计试验。

评测集必须包含训练中未出现的随机种子和障碍尺寸组合，避免只验证记住的 curriculum level。

### 4.5 M1.1 已实现的 Low Bar 评测协议（2026-10-10）

评测器 `scripts/eval_low_bar_isaac.py` 已按下述四条完成加固，配套脚本
`scripts/run_low_bar_gate.sh`（跑门禁）和 `scripts/report_low_bar_gate.py`
（出报告并给出判定）：

1. **置信区间**：所有比率同时输出计数和 Wilson 95% 区间
   （`*_ci95_percent`），不再只给一个点估计；
2. **配对比较**：同一次调用内的所有候选共用同一随机序列，按 `(seed, env)`
   逐 episode 配对，输出配对差值、其 95% 区间、精确 McNemar p 值和四格计数；
3. **分组报告**：同时给出 `competition_spec`（恰好 0.30 m）、
   `easier_than_spec`、`harder_than_spec` 和 `full_range`（0.25～0.35 m）；
4. **可追溯性**：protocol 记录 git commit、`git_dirty`、评测器 SHA256、种子
   列表、出生扰动范围、观测噪声配置和每个 clearance 档位。

#### 为什么必须配对

2026-10-10 的重复实验（同一 checkpoint `060922`、同一 seed `20261017`、
`train-matched`，只换进程）：

- 单候选重复三次的绝对成功率：`71.6% / 70.3% / 70.6%`；
- 多候选同进程跑时，同一 checkpoint 出现过 `68.0%` 和 `73.2%`；
- 而 `060922` 相对 `060898` 的**配对差值**三次为
  `+0.00 / -0.26 / +0.26` 个百分点。

结论：绝对成功率有约 1.3 个百分点以上的进程级抖动（GPU 求解与 cuBLAS 的非
确定性；`384` 与 `500` 环境还会消费不同的随机序列），**不能跨调用比较数字**；
配对差值稳定在 ±0.3 个百分点，是可用的判据。

#### 判定规则

- 候选只有在**配对差值 95% 区间下界大于 0** 时才记为提升；
- 区间包含 0 一律记为"无显著差异"，不得用点估计的 2～3 个百分点晋级；
- 参照基线必须在同一次调用内重新评测（`--baseline-run`），不复用历史数字；
- 384 环境只做筛选；正式候选使用 2 个 held-out seed × 500 环境
  （= 1000 episodes），此时配对差值区间半宽约 2 个百分点。

#### 已知限制

- `--num-envs` 是协议的一部分：不同批量大小对应不同随机序列，结果不可互比；
- 每个 clearance 档位在 1000 episodes 下只有约 84 个样本，档位级结论仍有
  ±9 个百分点量级的不确定度，不能凭单档位差异选模型。

## 5. Phase 1：Low Bar 专家——先打通完整链路

### 5.1 课程顺序

1. 固定位置、固定航向、0.35 m 净空；
2. 0.33 m 净空；
3. 0.30 m 比赛净空；
4. 0.25～0.35 m 随机净空；
5. 加入横向偏移、进场航向、横杆位置和控制延迟；
6. 加入观测噪声、bias、低频更新和短时 dropout；
7. 加入穿过后的站姿恢复和继续行走。

前四个阶段只训练 Low Bar，先证明单障碍链路可收敛。连续两个 checkpoint 在 Isaac Gym 达到 90% 后，再加入约 10%～20% 的 flat、turn 和 stairs replay 做防遗忘回归；如回归性能下降超过 5 个百分点，不允许进入鲁棒化阶段。

### 5.2 奖励和终止

正向奖励：

- 朝障碍方向的有效进度；
- 横杆前平滑降低 base；
- 保持全身最高点低于横杆；
- 全身完全穿过；
- 穿过后恢复正常高度和步态；
- 跟踪目标速度和航向。

惩罚或失败：

- 横杆或立柱接触；
- 腹部、机身和非允许部位碰撞；
- 摔倒；
- 在横杆前长期停滞；
- 全程趴低或通过后不恢复；
- 高频关节动作和力矩饱和。

`low_bar_crouch` 只能在横杆前的有效距离窗口开启。Reward 上升必须同时对应 success rate 上升，否则视为 reward hacking，不允许晋级。

### 5.3 Low Bar 通过定义

一次成功必须同时满足：

- base 和四足全部越过横杆平面；
- 横杆与两侧立柱零碰撞；
- 机器人未摔倒；
- 越过后至少稳定行走或站立 1 s；
- 未超过最大时间。

### 5.4 Low Bar 晋级门槛

- Isaac Gym held-out success rate `≥ 90%`，连续两个 checkpoint 达标；
- 摔倒率 `≤ 5%`，横杆/立柱接触率满足比赛安全要求；
- MuJoCo 200 episodes success rate `≥ 85%`；
- flat/stairs 相比 `059950` 下降不超过 5 个百分点；
- 使用真机来源格式的 obstacle observation 完成日志回放或吊架测试；
- 自动导出和 MuJoCo 评测不需要手工修改模型输入。

只有以上全部完成，才启动 Bridge A 的正式训练。

### 5.5 当前执行记录（2026-10-10）

- 已完成 checkpoint 观测扩维迁移、Low Bar 碰撞/穿越/恢复事件、专用 curriculum 和确定性 Isaac Gym 评测器；
- 已核验 Isaac Gym 横杆刚体索引，contact tensor 使用的索引与 simulator-domain 真实索引一致；
- Low Bar Actor observation 已从旧 3 维扩展为 7 维：`forward`、`lateral`、`bar_bottom_above_base`、`sin/cos(heading_error)`、`ground_clearance`、`valid`；旧字段顺序不变，新字段权重从 0 开始；
- 为防止障碍续训破坏成熟步态，已增加 `--observation-adapter-only`：只训练 10 帧历史中新增障碍输入列，原 actor、adaptation module 和动作噪声可冻结；
- 参数级 smoke test 确认：新增输入列发生更新，legacy actor 和 adaptation module 的最大参数变化为 0；
- 评测器现支持 `nominal`/`train-matched`、固定 held-out seed、横向/航向出生扰动和 action-noise 诊断，并把任务碰撞/越界与真实跌倒分开统计；
- 已修复训练监控的两个统计错误：episode batch 按实际完成数量加权，且每个 physics step 清空旧 `extras`，不再重复累计上一次 reset 的指标；Low Bar 首轮训练也已关闭随机 episode age，避免截断“接近→穿越→恢复”序列；
- M1.1 已完成：评测器加入 Wilson 95% 区间、配对比较（精确 McNemar + 配对差值
  区间）、分组报告和 git/评测器哈希溯源；新增 `scripts/run_low_bar_gate.sh` 与
  `scripts/report_low_bar_gate.py`；协议细节见 §4.5；
- 旧口径（把障碍碰撞/出界计入 `fell`）的 15 个评测日志已归档到
  `logs/archive_pre_metric_fix/`，含 `README.md` 说明；不要与当前数字比较；
- 用正式协议（2 个 held-out seed × 500 环境 = 1000 episodes）重测 `060898`
  家族，横向 ±0.05 m、航向 ±0.05 rad 的 `train-matched` 条件下：
  `060898` ≈ 73.1%、`060920` ≈ 73.1%、`060922` ≈ 73.4%；三者两两配对差值的
  95% 区间均包含 0，即**当前无可分辨的提升**；
- 因此此前"`060898` 约 70.1%"是 384 环境协议下的偏低读数。改用 1000 episodes
  后基线约 73%，主要失败仍是门框外穿越约 21%，横杆/立柱碰撞约 5.8%，真实
  跌倒 ≈ 0%；
- 碰杆率也从 384 环境的约 9.9% 修正为 1000 episodes 的约 5.8%，说明小样本
  曾高估碰杆；`0.30 m` 档位在 1000 episodes 下只有约 84 个样本，暂不能据此排序；
- 两轮 7 维 observation adapter-only，以及全 actor 小学习率 + policy anchor
  配置，在正式协议下都没有超过 `060898`；96 环境出现的 76%～79.2% 确认是小
  样本峰值，已按门槛停止；
- 下一轮不继续堆同配置 iteration。先做 M1.3（单独解决碰杆，含 base 高度与
  crouch shaping 诊断），再做 M1.2 分阶段位姿 curriculum；每阶段用
  `scripts/run_low_bar_gate.sh` 做配对判定，区间下界必须大于 0；
- M1 状态：**未完成**。在连续两个 checkpoint 用正式协议达到 90% 前，不启动
  Bridge A 正式训练。

## 6. Phase 2：Bridge A 专家

Bridge A 先建立“正常速度下稳定居中通过宽桥”的能力，不提前混入 20 cm 窄板。

### 6.1 课程顺序

1. 1.2 m 宽连续平桥；
2. 1.0 m 比赛宽度连续桥；
3. 1.0 m 宽分段桥，间隙 0.05 m；
4. 间隙逐步增加到 0.15 m；
5. 加入 14° 上下桥坡；
6. 起点横向偏差从小到大；
7. 进场航向误差逐步增加到约 ±5°～10°；
8. 加入摩擦、质量、动作延迟和小幅外力扰动。

### 6.2 奖励和指标

- 沿桥方向的前向进度；
- 中心线和航向对齐；
- 足端落在有效支撑面；
- 完整到达另一平台；
- 惩罚踏空、桥外接触、边缘危险落足、机身/大腿碰桥和横向漂移；
- 保留速度、base 高度和动作平滑度约束，避免策略通过极慢爬行刷成功率。

边缘判定应使用足端接触点与真实 bridge collision geometry，不能只看 base 是否在桥面投影内。

### 6.3 晋级门槛

- Isaac Gym held-out success rate `≥ 90%`；
- MuJoCo success rate `≥ 85%`；
- flat/stairs/Low Bar 回归下降不超过 5 个百分点；
- 完成时间、base 高度和步幅没有出现明显保守退化。

## 7. Phase 3：Bridge B 专家/子技能

Bridge B 的 20 cm 条板与 Bridge A 的 1 m 宽桥属于明显不同的 locomotion problem。它可以共享 Bridge expert 的网络和数据，但必须使用独立 curriculum、独立评测和独立 checkpoint 标记；必要时保留独立 `skill_id` 或子技能状态。

### 7.1 课程顺序

1. 使用 Bridge A 最佳 checkpoint 初始化；
2. 条板宽度 0.50 m、较小间隙；
3. 条板宽度 0.40 m；
4. 条板宽度 0.30 m；
5. 条板宽度逐步收窄至约 0.20 m；
6. 间隙逐步增加至约 0.10 m；
7. 最后加入横向偏差、航向误差和参数随机化。

训练 Bridge B 时，建议 30%～50% minibatch 来自 Bridge A、flat 和 stairs replay/curriculum，防止策略为了窄桥而全面退化。

### 7.2 防保守退化指标

除成功率外，必须监控：

- 平均步幅；
- 平均前进速度和完成时间；
- base 平均高度；
- 中心线误差；
- 足端有效支撑比例；
- 边缘接触和踏空次数；
- Bridge A 和 flat 上的回归性能。

如果 Bridge B 成功率上升但步幅持续缩小、base 长期降低或 Bridge A 明显变差，应回退 checkpoint，增加 replay 比例或拆成单独技能，不能继续强行联合优化。

### 7.3 晋级门槛

- Bridge B Isaac Gym `≥ 90%`；
- Bridge B MuJoCo `≥ 85%`；
- Bridge A、flat、stairs 相比各自最佳 checkpoint 下降不超过 5 个百分点；
- 没有依靠超时边缘附近的极慢移动获得伪成功。

## 8. Phase 4：Wall 专家

### 8.1 课程顺序

1. 0.10 m 墙；
2. 0.15 m；
3. 0.20 m；
4. 0.25 m；
5. 0.30 m 比赛高度；
6. 0.35 m 鲁棒性测试；
7. 加入起点距离、横向偏移、航向、摩擦和延迟随机化。

每个环境可重复布置 3～5 道墙，增加有效学习信号，但评测必须使用独立单墙和连续墙场景，避免策略只记住固定节奏。

### 8.2 奖励和接触规则

- 奖励正对墙体、接近、越过墙顶、base 与四足完全落到墙后以及稳定恢复；
- 规则允许跃过或攀爬时，前足在墙顶的可控低速接触不能一律处罚；
- 惩罚腹部、膝部高速撞击、后退卡住、翻倒和危险落地；
- 墙前跳跃准备奖励必须有距离窗口，避免在普通地面反复跳跃。

### 8.3 晋级门槛

- Wall Isaac Gym `≥ 90%`；
- Wall MuJoCo `≥ 85%`；
- 全身越过后稳定至少 1 s；
- flat、stairs 和已完成障碍回归下降不超过 5 个百分点。

## 9. Phase 5：专家蒸馏为一个 Student

专家全部通过单项门槛后再开始蒸馏：

1. 从 `059950`、Low Bar、Bridge A、Bridge B、Wall 专家采集 2M～5M transitions；
2. 数据必须包含成功轨迹、失败前状态、恢复轨迹、观测失效和障碍切换状态；
3. Student 输入固定的 `skill_id + 可部署 observation contract`；
4. 先行为克隆初始化，再用 RSL-RL student-teacher 或 Extreme Parkour 风格 DAgger 纠正分布外状态；
5. 最后进行多任务 PPO 联合微调；
6. 每次联合微调都跑完整回归矩阵，避免某一技能覆盖其他技能。

建议初始采样比例：

- 25% flat / turn / zero-command / stairs；
- 20% Bridge A；
- 15% Bridge B；
- 20% Low Bar；
- 15% Wall；
- 5% gravel / slope / disturbance。

比例不是固定超参数，应根据最低成功率技能和遗忘情况动态调整。零命令样本至少占全部正常地形样本的 10%，零命令时关闭会诱导抬腿的 foot-clearance 奖励，并启用站立姿态、关节速度、接触保持和 action-rate 约束。

## 10. Phase 6：Sim2Sim、Sim2Real 与完整地图

### 10.1 随机化开启顺序

只在单障碍无随机化成功率达到约 80% 后逐步加入：

1. 障碍尺寸与起点轻微随机；
2. 摩擦 0.5～1.2；
3. base 质量约 ±10%，质心偏移 ±2～3 cm；
4. motor strength 0.9～1.1，Kp 0.9～1.1，Kd 0.75～1.5；
5. 0～2 个控制周期动作延迟；
6. IMU、编码器和高度扫描噪声；
7. obstacle state 的低频、延迟、bias、dropout；
8. 最后加入小幅外力扰动。

不要在课程早期同时放大全部随机化，否则难以区分失败原因。

### 10.2 MuJoCo 要求

- 使用与部署一致的 observation builder，不写第二套仅供演示的计算逻辑；
- 障碍尺寸、碰撞体、关节限制、控制频率和 action scaling 与训练配置自动比对；
- 支持自动 episode 评测和键盘人工操控；
- 每个 checkpoint 产生可追溯的模型、配置、Git commit、评测 JSON/CSV 和视频；
- 人工操控只用于观察问题，正式成功率来自可重复的自动路线和初始条件。

### 10.3 真机安全顺序

1. 离线回放真实传感器日志，验证字段、单位、方向和时间戳；
2. 吊架下测试站立、模式切换和观测丢失；
3. 低速接近障碍但不通过；
4. 使用更高净空/更宽桥/更低墙；
5. 逐步进入比赛尺寸；
6. 单障碍稳定后再跑完整地图。

真机测试必须保留急停、速度限制、关节限位和观测超时保护。

## 11. 统一成功定义

- **Bridge A/B**：base 与四足到达另一平台；过程中无踏空、桥外接触、摔倒或超时；
- **Low Bar**：全身穿过；横杆和立柱零接触；通过后恢复稳定；
- **Wall**：base 与四足完全越过墙体并连续稳定至少 1 s；
- **Zero command**：站立 10 s 不摔倒，平均关节波动 `< 1°`，无持续交替抬腿；
- **Full Map**：从统一起点按比赛路线连续完成全部障碍，中途不重置状态。

不能把四个独立障碍的 success rate 直接当作整场成功率。作为风险估算，若四项各自只有 85%，独立近似下连续成功率约为：

```text
0.85^4 ≈ 52%
```

若目标是完整地图约 80% 连续成功率，各单项最终通常需要接近 95%，因为 `0.95^4 ≈ 81%`，实际还要考虑障碍间状态耦合。

## 12. Milestone 与 Go/No-Go 门槛

### M0：基础设施完成

- 固定 observation contract；
- 完成 `059950` 扩维迁移；
- 自动评测、导出和 MuJoCo 加载打通；
- flat/stairs/zero-command 基线无明显回归。

### M1：Low Bar Isaac Gym

- held-out success rate `≥ 90%`，由 §4.5 的正式协议测得（2 个 held-out seed，
  每个候选 ≥ 1000 episodes）；
- 连续两个 checkpoint 达标；
- 候选相对**同一次调用内重新评测**的基线，配对差值 95% 区间下界大于 0；
- reward 与 success rate 同步上升。

### M2：Low Bar MuJoCo/观测链路

- MuJoCo success rate `≥ 85%`；
- 使用部署格式 obstacle state；
- 完成真实来源格式日志回放或吊架验证。

### M3：Bridge A Isaac Gym `≥ 90%`

### M4：Bridge A MuJoCo `≥ 85%`

### M5：Bridge B Isaac Gym `≥ 90%`

### M6：Bridge B MuJoCo `≥ 85%`

### M7：Wall Isaac Gym `≥ 90%`

### M8：Wall MuJoCo `≥ 85%`

### M9：Student 单项验收

- 每个目标障碍至少 `≥ 85%`，之后继续向比赛目标 `≥ 95%` 优化；
- flat/stairs 相比 `059950` 下降不超过 5 个百分点；
- zero-command 满足站立指标；
- 观测短时失效时能安全减速或站立。

### M10：MuJoCo Full Map

- 连续完成完整路线；
- 初期目标 `≥ 70%`，最终目标 `≥ 80%`；
- 不允许在障碍间人工重置 policy hidden state 或机器人状态。

### M11：真机 Full Map

- 按吊架、低速、单障碍、连续障碍的顺序逐级放开；
- 统计成功率、失败模式和硬件安全数据；
- 未满足安全条件时回退 checkpoint 或 curriculum，不通过增加动作幅度硬冲。

## 13. 失败诊断与回退规则

每次失败先按下面顺序定位，不能直接通过加 reward 系数掩盖问题：

1. terrain/collision geometry 是否正确；
2. 成功和 termination 判据是否正确；
3. observation 坐标、单位、归一化和时间同步是否一致；
4. checkpoint 迁移和模型导出是否一致；
5. MuJoCo 与 Isaac 的控制频率、PD、动作尺度是否一致；
6. reward 是否存在捷径或互相冲突；
7. curriculum 是否跨级过大；
8. 最后才调整 PPO 超参数或网络规模。

出现下列任一情况应停止当前 curriculum 并回退：

- success rate 连续多个 checkpoint 不升但 reward 持续上升；
- flat/stairs 下降超过 5 个百分点；
- 动作饱和、关节抖动或 base 高度持续恶化；
- Isaac 成功但 MuJoCo 大幅失败；
- Actor 使用了真机无法产生的 ground truth；
- 导出模型与训练模型在相同 observation 上输出不一致。

## 14. 第一轮代码任务清单

按顺序执行：

1. 冻结并校验 `059950` 基线资产；
2. 新增版本化 observation contract 与配置校验；
3. 实现 `059950` 的 Actor/Estimator/Critic 输入扩维迁移工具；
4. 新旧模型零障碍观测等价性测试；
5. 为 Low Bar 补齐成功、碰撞、超时、恢复判据；
6. 实现统一 checkpoint 自动评测与排名；
7. 把同一 obstacle-state builder 接入 Isaac、MuJoCo 和 C++ 部署接口；
8. 启动一个 Low Bar 短 run，验证加载、反向传播、保存和导出；
9. 启动 Low Bar 正式 curriculum，并完成 M1、M2；
10. M2 达标后再实现和训练 Bridge A；
11. Bridge A 达标后再推进 Bridge B；
12. Bridge B 达标后推进 Wall；
13. 专家全部达标后再开始 student 蒸馏和 Full Map 联合训练。

项目进度由 M0～M11 的完成状态管理。天数只用于资源排期，不能替代 success rate、回归性能、部署观测可用性和真机安全验收。
