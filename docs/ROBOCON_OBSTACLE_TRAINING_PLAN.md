# MyBot V3 RoboCon 障碍续训计划

## 目标与结论

目标是在保留 `059950` 现有平地、转向和台阶能力的前提下，让一个最终策略可靠完成：

- 木桥 A：1000 mm 宽、400 mm 分段桥面、150 mm 间隙、14° 上下桥坡；
- 木桥 B：约 200 mm 窄条板与约 100 mm 槽/间隙；
- 限高杆：300 mm 净空、1000 mm 横杆跨度；
- 高墙：300 mm 高、1000 mm 长、50 mm 厚（口语里的“矮墙”）。

推荐主线不是更换成全新 RL 算法，而是继续使用当前 PPO/CSE 框架，吸收 **Extreme Parkour 的障碍课程与 teacher-student 方法**，训练三个障碍专家后蒸馏成一个 task-conditioned 最终策略。

## 可直接借鉴的开源方案

### 1. Extreme Parkour（首选）

- 代码：[chengxuxin/extreme-parkour](https://github.com/chengxuxin/extreme-parkour)
- 论文：[Extreme Parkour with Legged Robots](https://arxiv.org/abs/2309.14341)
- 与本项目同为 Isaac Gym、Python 3.8、Torch 1.10、CUDA 11.3 技术栈，迁移成本最低。
- 已实现 step、gap、hurdle、tilted ramp 等障碍课程。
- 先用精确地形/waypoint 训练 teacher，再用 BC + DAgger 将动作和方向能力蒸馏给可部署 student。
- 官方给出的 3090 参考训练量为 base policy 10k～15k iterations、distillation 5k～10k iterations。

适合借用：障碍生成、边缘踩踏惩罚、渐进难度、障碍方向输入、teacher-student/DAgger。机器人模型和观测契约不能直接照搬。

### 2. RSL-RL（继续沿用）

- 代码：[leggedrobotics/rsl_rl](https://github.com/leggedrobotics/rsl_rl)
- 提供 GPU PPO 和 Student-Teacher Distillation，并支持 Legged Gym、Isaac Lab、mjlab 与 MuJoCo Playground。
- 当前项目已经属于同类 PPO/Legged Gym 体系，最适合作为最终单模型的蒸馏和继续训练后端。

### 3. RoboCon 26 / Isaac Lab 参考工程

- 代码：[Taojunfeng123/quadruped_robot_lab](https://github.com/Taojunfeng123/quadruped_robot_lab)
- 面向 RC26 四足，包含地形课程、动作课程、周期停止与 RSL-RL 配置。
- 更适合作为奖励与课程设计参考；现在整体迁移到 Isaac Lab 会打断现有稳定链路，因此不作为本轮主线。

### 4. 本仓库内 RC_WheelLeg 参考工程

- `robocon_reference/RC_WheelLeg/05_software/train/rc_mjlab/`
- 其比赛方案将 Rough、Wall、Crawl 分开训练/控制；高墙使用 0.10～0.35 m 课程，Crawl 使用独立任务。
- 这证明三个障碍的动作分布存在明显冲突，因此应先训练专家，再合并，而不是直接把所有障碍一次性加入同一个 PPO run。

## 最终模型结构

Actor 输入在现有 119 维历史观测基础上增加：

1. `skill_id`（4 维 one-hot）：normal / bridge / low_bar / wall；
2. 障碍相对状态：前向距离、横向偏差、相对航向、障碍高度/净空；
3. 桥面局部精细高度/支撑扫描：建议 5 cm 分辨率，覆盖四足和前方 0.8～1.0 m；
4. 上方净空 ray/专用 low-bar observation；仅向下的 2.5D 高度图看不到横杆。

Critic 训练时额外获得精确地形、接触力、摩擦、质量、延迟和障碍尺寸。最终 Actor 只保留真机可以取得的 IMU、编码器、上一动作和传感器/任务输入。

如果比赛路线和障碍顺序固定，`skill_id` 可由上层路线状态机提供，比要求一个端到端网络自动识别全部障碍更稳定。

## 分阶段训练

### Phase 0：冻结基线与自动验收

1. 永久保留 `059950`，禁止覆盖。
2. 为 flat、stairs、bridge A/B、low bar、wall 建立独立评测场景。
3. 每个 checkpoint 输出：成功率、摔倒率、通过时间、横向偏差、最小净空、最大力矩、动作饱和率、接触违规。
4. Isaac Gym 先做每障碍 1000 episode；候选 checkpoint 再进入 MuJoCo 200 episode。

### Phase 1：三个专家策略

三个专家都从 `059950` 恢复，不从零开始。每轮仍混入约 20%～30% 平地/台阶样本，防止遗忘。

#### A. Bridge expert

课程顺序：

1. 1.2 m 宽连续平桥；
2. 1.0 m 比赛宽度连续桥；
3. 木桥 A：间隙从 0.05 m 增到 0.15 m；
4. 加入 14° 上下坡、起点横向误差和 ±5° 航向误差；
5. 木桥 B：条板从 0.30 m 收窄到约 0.20 m、间隙增到约 0.10 m。

关键奖励：前向进度、中心线/航向对齐、足端落在有效支撑面、过桥完成；惩罚踏空、桥外接触、机身/大腿碰桥、横向漂移和高频动作。边缘接触判定应使用足端接触点，而不是只看 base 位置。

#### B. Low-bar expert

现有 `low_bar` actor、`obstacle_ahead` 观测和 `low_bar_crouch` 奖励可以直接复用，但需要补完整通过判据。

课程顺序：净空 0.40 → 0.35 → 0.30 → 0.25 m，同时随机横向偏移、横杆位置与进场航向。

关键奖励：接近前保持正常步态、横杆前平滑降高、全身最高点低于横杆、穿过后恢复站姿、成功越过；横杆/立柱碰撞直接失败。蹲伏奖励只应在横杆前的有效窗口开启，不能全程鼓励趴低。

#### C. Wall expert

课程顺序：0.10 → 0.15 → 0.20 → 0.25 → 0.30 → 0.35 m；每个环境重复布置 3～5 道墙，减少偶然过墙得到的稀疏样本。

关键奖励：正对墙、起跳前进度、base/足端越过墙顶、四足和机身全部到达墙后、稳定落地。规则允许跃过或攀爬，因此前足在墙顶的可控接触不应一律处罚；应处罚腹部、膝部的高速撞击和翻倒。

### Phase 2：专家蒸馏为一个模型

1. 在三个专家加原 `059950` 上采集 2M～5M transition，覆盖成功轨迹和恢复轨迹。
2. Student 输入 `skill_id + 可部署观测`，用行为克隆初始化。
3. 使用 RSL-RL Student-Teacher 或 Extreme Parkour 风格 DAgger 在线纠正 student 偏离后的状态。
4. 再用多任务 PPO 联合微调，建议初始采样比例：
   - 30% flat / turn / stairs；
   - 25% bridge A/B；
   - 20% low bar；
   - 20% wall；
   - 5% gravel / slope / disturbance。
5. 精确零命令至少占 10%，零命令时关闭 foot-clearance 奖励，并启用站立姿态、关节速度和 action-rate 约束，避免再次出现原地抽腿。

### Phase 3：后期 Sim2Real 随机化

只在单障碍成功率达到约 80% 后逐步开启：

- 摩擦 0.5～1.2；
- base 质量约 ±10%，质心偏移 ±2～3 cm；
- motor strength 0.9～1.1，Kp 0.9～1.1，Kd 0.75～1.5；
- 0～2 个控制周期动作延迟；
- IMU、关节编码器、高度扫描噪声和延迟；
- 障碍尺寸 ±10～20 mm，进场横移 ±0.10 m、航向 ±5°～10°；
- 小幅外力扰动，最后再加入，不要在课程早期同时放大全部随机化。

## 晋级门槛

每一级课程连续两个 checkpoint 满足门槛后才升级：

- 专家策略：目标障碍成功率 ≥90%，摔倒率 ≤5%；
- 最终单模型：每个目标障碍成功率 ≥85%；
- flat/stairs 相比 `059950` 成功率下降不超过 5 个百分点；
- 零命令站立 10 s：不摔倒、平均关节波动 <1°；
- MuJoCo 随机参数评测通过后才进入真机。

通过定义：

- Bridge：四足和 base 到达另一平台，过程无踏空/桥外接触；
- Low bar：全身穿过，横杆与立柱零碰撞；
- Wall：base 与四足完全越过墙体并连续稳定站立 1 s；
- Full map：按比赛路线连续完成，不能只把单障碍成功率相乘估计整场成功率。

## 实施顺序与预计周期

1. 1～2 天：补 bridge/wall 环境、精细观测与统一评测 harness；
2. 2～4 天：三专家并行/串行续训与奖励消融；
3. 1～2 天：teacher-student 蒸馏和多任务 PPO；
4. 2～4 天：MuJoCo 参数扫描、全场连跑和真机吊架测试。

RTX 2080 Ti 22 GB 可从 2048 environments 起步，确认显存和吞吐后再升到 4096。实际训练停止点以成功率平台和评测门槛为准，不以固定迭代数为准。

## 第一轮应做的代码任务

1. 将 `bridge`、`wall` 加入 `--robocon-obstacle`，建立真实尺寸 3D collision actor；
2. 扩展统一 `obstacle_ahead` 为 `obstacle_state`，支持桥边缘、上方横杆和墙高；
3. 为三种障碍写成功/失败终止器和指标；
4. 修正零命令奖励：gate `feet_clearance_ji22`，启用静止速度/姿态约束；
5. 从 `059950` 各启动一个短专家 run，先验证 reward 是否真的随通过率增长；
6. 自动把候选 checkpoint 导出到现有 MuJoCo 单障碍与整场评测。
