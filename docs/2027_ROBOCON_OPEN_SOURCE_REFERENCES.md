# 2027 ROBOCON 可参考开源案例与采用决策

核查日期：2026-10-15。仅依据作者仓库、项目页与论文；实机演示证明作者平台上的可行性，不代表 MyBot 已复现。这里的“适合”是针对本工程的判断。

## 1. 建议顺序

| 优先级 | 项目与成功证据 | 与本工程的关系 | 本轮采用范围 |
|---|---|---|---|
| P0 | [PACE](https://github.com/leggedrobotics/pace-sim2real)：提供执行器/关节动力学辨识代码和数据采集、参数拟合流程 | 当前限高对阻尼极敏感，最需要辨识而非继续盲调 | 借鉴测量轨迹、拟合参数、留出验证的方法 |
| P0 | [Robot Parkour Learning](https://github.com/ZiwenZhuang/parkour)：CoRL 2023，作者项目展示实机 crawl/climb/leap 等技能 | 与限高、跨沟、高墙的动作类型最接近；使用 Isaac Gym | 参考 crawl 课程和真实碰撞微调，后续再考虑 DAgger |
| P1 | [ABS](https://github.com/LeCAR-Lab/ABS)：RSS 2024，提供 Go1 训练和部署路径 | 适合 locomotion/expert/recovery 状态切换 | 先实现可解释的恢复状态机；学习式切换作为后续实验 |
| P1 | [Extreme Parkour](https://github.com/chengxuxin/extreme-parkour)：ICRA 2024，实机跨沟、上障碍和斜坡演示 | 可参考足端避边与方向控制，对桥面、跨沟有用 | 选取奖励和地形课程进行小规模消融 |
| P2 | [HIMLoco](https://github.com/InternRobotics/HIMLoco)：ICLR 2024，作者展示 Unitree 上受扰动台阶行走 | 适合基础步态的历史状态估计与抗扰动对照 | 建立独立实验，待物理基线可信后比较 |

本轮只整理参考和计划，没有复制这些项目的算法代码，也没有安装或替换训练环境。

## 2. PACE：先让物理参数可解释

[官方说明](https://github.com/leggedrobotics/pace-sim2real)采用 CMA-ES，从测量数据估计执行器和关节参数；示例分为采集激励和拟合两步。[安装文档](https://pace.filipbjelonic.com/installation/)面向 Isaac Lab / Isaac Sim 5.0 及以上，不能直接视为当前 Isaac Gym 的插件。仓库为 Apache-2.0。

**本项目应用设计：**先在现有 MuJoCo 中重放相同关节目标，以 q/dq 为基本观测，力矩/电流可用时辅助约束。拟合正值阻尼、干摩擦、动作时延、执行器增益等少量参数；固定训练/验证轨迹划分。标定目标必须来自真实机器人，Isaac→MuJoCo 的拟合只能叫 sim-to-sim 对照。脚地接触和负载变化另做留出验证，不能仅靠悬空关节轨迹决定所有接触参数。

## 3. Robot Parkour Learning：最值得细读的限高案例

[作者项目页](https://robot-parkour.github.io/)展示爬越、跨沟、低姿穿越及失败案例；方法先用软约束发现行为，再恢复硬碰撞微调，最后用 DAgger 蒸馏。[训练教程](https://github.com/ZiwenZhuang/parkour/blob/main/legged_gym/README.md)有 `a1_crawl` 入口；[crawl 配置](https://github.com/ZiwenZhuang/parkour/blob/main/legged_gym/legged_gym/envs/a1/a1_crawl_config.py)可作为阅读起点。仓库为 [MIT](https://github.com/ZiwenZhuang/parkour/blob/main/LICENSE)。

教程中的 Go2 简化例子移除了 crawl，不能把它当作可直接复现限高的配置；完整视觉蒸馏的 GPU 要求也高于本机单张 22 GB 卡的已验证能力。

**本项目应用设计：**已有 `064341` 应先作为硬框架阶段的起点，不必重做软碰撞预训练。全身碰撞包络、四角立柱、整机离框和恢复判据始终作为验收条件。只有行为探索完全失败且硬件可行性已确认，才另开软约束研究分支；软碰撞结果不计入阶段完成度。

## 4. ABS：参考恢复架构，区分方法与当前实现

[官方项目页](https://agile-but-safe.github.io/)使用 agile policy、recovery policy 和学习得到的 reach-avoid 值网络决定切换；[仓库](https://github.com/LeCAR-Lab/ABS)提供训练与 Go1 部署入口，许可证为 CC BY-NC 4.0。

**本项目应用设计：**先把“整机离框→短时保持→关节目标插值→通用步态”变成可记录、可测试的状态机。当前 `059950` 是通用策略，不是专门训练的 ABS recovery policy；0.5 s 插值也不是完整 ABS 算法。若失败状态覆盖不足，再收集失败前状态训练恢复策略。用同一初态对照硬切换与平滑切换，记录最大目标跳变、姿态、恢复时间和接触峰值。

## 5. Extreme Parkour：足端避边与方向控制

[作者项目页](https://extreme-parkour.github.io/)有真实连续越障演示以及移除 clearance reward、方向蒸馏后的对照。[代码](https://github.com/chengxuxin/extreme-parkour)使用 Isaac Gym，安装栈与本项目较接近；当前根 [LICENSE](https://github.com/chengxuxin/extreme-parkour/blob/main/LICENSE)为 CC BY-NC 4.0。

**本项目应用设计：**先在桥面实验中引入“足端接触点到桥沿的余量”，与单纯基座中心线奖励配对比较。跨沟阶段再评估落地边缘余量；训练奖励不能替代“整机通过、禁区落足失败”的独立评测。作者的训练耗时不能直接作为 MyBot 的工期。

## 6. HIMLoco：适合作为后续算法对照

[作者页面](https://junfenglong.com/HIMLoco/)说明利用 IMU、关节历史估计速度和隐状态，并展示实机受扰台阶运动。[仓库](https://github.com/InternRobotics/HIMLoco)提供 [HIM PPO](https://github.com/InternRobotics/HIMLoco/blob/main/rsl_rl/rsl_rl/algorithms/him_ppo.py)；README 中未勾选的 TODO 不能据此解读为“没有训练代码”。许可证为 CC BY-NC-SA 4.0。

**本项目应用设计：**冻结 MyBot 模型和评测初态，用相同环境步数、至少三个训练 seed，对比现有 PPO-CSE 与 HIM 估计器。比较台阶/扰动/延迟/dropout 成功率及推理延迟。HIM 的观测和网络接口不同，已有 checkpoint 不能直接加载；本体历史也不能提前知道尚未接触的顶梁净空或离散落足区，限高/轮胎/梅花桩仍需要可信的障碍信息。

## 7. 引入前的最小验证

每次只改变一个主要因素：先验证数学定义、单位、关节映射、输入输出维度，再跑 20–50 回合 smoke。保留相同种子的原方法对照。只有独立回归通过、基础能力下降不超过 5 个百分点时才扩大训练。

引用代码时记录上游 commit、文件路径及修改说明，保留各项目许可；不要把不同许可代码统一标成仓库原有 MIT。GitHub 的演示或论文成绩都不替代本仓库的完成判据。

与比赛同名的场景仓库不等于已完成该赛项的策略。例如 [discoverse-dev/ROBOCON2026](https://github.com/discoverse-dev/ROBOCON2026)面向“武林探秘”场景，可参考地图/ROS 接口，不能拿它的场地替代本项目 2027 足式障碍规则。

下一步实际任务和验收见 [执行计划](2027_ROBOCON_NEXT_STEPS_20261015.md)。
