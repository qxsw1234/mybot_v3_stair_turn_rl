# 2027 足式机器人障碍赛执行状态

更新日期：2026-10-15（代码与证据收尾；训练/评测最新数据仍为2026-10-13）
工程：`/home/ldl/mybot_v3_stair_turn_rl`
计划基线：`docs/2027_ROBOCON_OBSTACLE_TRAINING_PLAN.md` v0.2

## 2026-10-15 收尾核查与后续入口

SSH已恢复。当前没有训练/评测进程；保留权重059950/064242/064341/064400均能加载，参数张量有限且原配置存在；MuJoCo 3.2.3、Torch 1.10.0+cu113和CUDA可用，13项规则测试通过。今天整理代码、评测证据和文档，没有新增训练成绩。用户确认GitHub仅发布代码和文档；201份原始评测JSON/NPZ保留desktop，公开结果摘要和文件校验清单。

- 当前为M2进行中，M0物理标定/实机能力表仍有遗留项；Stage 2关闭。
- M1基线确实已做：绕桩19/20、桥14/20、规则限高旧策略0/20。绕桩还需审查短接触采样与完整路径判据，不能凭小样本95%直接认定实机通过。
- 研究候选064341、备选064400、原候选064242、通用基线059950均保留；064540不晋级。
- [详细下一步计划](2027_ROBOCON_NEXT_STEPS_20261015.md)明确每个工作包的代码/数据交付、测试数量、完成门槛与回退条件。
- [开源调研](2027_ROBOCON_OPEN_SOURCE_REFERENCES.md)记录作者代码、实机证据、许可证及本工程采用范围；本次未引入外部算法代码。
- [文件清单](2027_ROBOCON_ARTIFACT_MANIFEST.json)记录完整run、checkpoint/配置/JIT与公开评测的SHA-256。权重本体仍在desktop，GitHub不包含`runs/`或2027原始评测文件。
- 部署C++依赖链接问题未修复；硬件限值和实机数据仍待测。诊断阻尼下的成功率不改变此状态。

以下10月13日实验详情保持原始记录；10月11日M1章节中的“下一步”仅代表当时安排，现在按上述新计划执行。

## 最新进展：第二轮 Stage 1 续训（2026-10-13）

从 `064341` 独立续训 200 次迭代，运行目录 `runs/mybot_v3_low_bar_expert_resume_064341/2026-10-12_03-45-02.595899`，正常保存 `064400/064500/064540` 等权重。末点 `064540` 在 MuJoCo 退化，已排除；`064242` 仍是原候选，`064341` 是研究候选。Stage 2 继续关闭。

| 同批协议 | `064341` | `064400` | `064500` | `064540` |
| --- | ---: | ---: | ---: | ---: |
| Isaac train-matched，四留出 seed，共 192 回合 | 129/192 | 133/192 | 133/192 | **137/192** |
| Isaac nominal，seed 20261127，48 回合 | 48/48 | 未测 | 未测 | 48/48 |
| MuJoCo 修正判据、诊断阻尼 ×4、无目标裁剪，seed 2027，24 回合 | 23/24 | 23/24 | **12/24** | **0/24** |
| 同上，独立 seed 2028，120 回合 | **117/120** | 113/120 | 未测 | 未测 |
| MuJoCo 默认阻尼、无目标裁剪，seed 2027，24 回合 | **2/24** | 未测 | 未测 | **1/24** |

Isaac 同批结果在 `evaluations/robocon_2027/low_frame_stage1_hard_ft2_trainmatched_4x48_seeds20261127_30.json`，nominal 为 `low_frame_stage1_hard_ft2_nominal_screen48_seed20261127.json`。MuJoCo 回归为同目录的 `low_frame_064400_stage1_mujoco_corrected_damping4_unclipped_regression120_seed2028.json`；其 7 次失败包括 4 次门外通行、3 次剧烈撞梁，5 ms 接触力峰值 406.54 N。`064500` 有 12 次门外通行，`064540` 在诊断阻尼 ×4 下 24 次全部门外通行。按训练末点自动选择权重会掩盖跨引擎退化。

当前优先辨识动力学。Isaac 保存的 hip/thigh/calf friction 数值为 0.1/0.1/0.3；MuJoCo 评测器将同数值直接填入 `dof_frictionloss`，两者摩擦参数的物理语义可能不等价。固定初态和相同关节目标，比较两端关节位置/速度/力矩、机身高度和首次顶梁接触时刻，再逐项辨识摩擦、阻尼、动作延迟与碰撞参数。诊断阻尼 ×4 可能在代偿其他失配，不能作为实测物理参数。若对照能重现 MuJoCo 默认失效，再从 `064341` 做短程混合动力学续训；单纯延长训练已出现反例。

开源参考：[PACE](https://github.com/leggedrobotics/pace-sim2real)（Apache-2.0）提供系统辨识流程，官方实现面向 Isaac Sim 5/Isaac Lab，先借鉴方法；[ABS](https://github.com/LeCAR-Lab/ABS)（CC BY-NC 4.0）可借鉴按状态切换恢复策略与平滑交接，完整方法需重训；[HIMLoco](https://github.com/InternRobotics/HIMLoco)（CC BY-NC-SA 4.0）适合后续历史本体感知消融，网络接口与现有 checkpoint 不兼容，不能直接补入。复用代码须遵守各仓库许可证。


### 动力学对照与新增 Stage 1 回归（2026-10-13）

`064341` 的 MuJoCo 5 ms 子步接触判据、无专家目标裁剪、离框后切换 `059950`，
在 25 s 上限和同一组初态下，关节被动阻尼倍率的剂量结果为：

| MuJoCo 被动阻尼倍率 | seed 2027 / 24 | seed 2028 / 120 | 最大顶梁接触力 |
| --- | ---: | ---: | ---: |
| 1×（原设 hip=1、thigh/calf=2 N·m·s/rad） | **2/24** | 未测 | 24 回合最大 546.14 N |
| 2× | **23/24** | **104/120** | 276.63 N |
| 3× | **24/24** | **120/120** | 218.49 N |
| 4× | **23/24** | **117/120** | 272.40 N |

同一 seed 2028 的 `064242` 在 3× 阻尼下仅 **41/120**（76 次 `outside_gate`、3 次剧烈撞梁），
因此 `064341` 比原权重在这一诊断模型上显著改善，但 **3× 未经实机标定，不能当作正式跨引擎验收**。
对应结果为 `evaluations/robocon_2027/low_frame_{064242,064341}_stage1_mujoco_corrected_damping3_unclipped_regression120_seed2028.json`；
`064341` 的 2× 回归为同目录 `low_frame_064341_stage1_mujoco_corrected_damping2_unclipped_regression120_seed2028.json`。
把大腿中心改到 1.55 的默认阻尼 24 回合，`064242` 和 `064341` 都仅 **2/24**，
所以此前单次安全离框探针不能推广为可用参数。

Isaac 对被动关节阻尼同样敏感：`064341`、nominal、seed 20261127、24 回合、
相同 25 s 时限下，hip/thigh/calf 固定阻尼 0/0/0 与 0.5/1/1 均为 **24/24**，
1/2/2 为 **0/24** 且全部从门外绕行。对应文件为
`evaluations/robocon_2027/low_frame_064341_stage1_isaac_passive_damping_*_nominal24_duration25_seed20261127.json`。
评测脚本新增可选 `--joint-passive-damping HIP THIGH CALF`，默认不改变旧协议。
12 s 的 0.5/1/1 失败由较短回合协议造成，不能混作 25 s 的失败证据。

固定 Isaac PD 目标的 MuJoCo 开环重放显示首步实际关节力矩接近（RMSE 0.325 N·m），
但 0.1 s 后关节轨迹和机身高度开始分离；默认阻尼下前 1 s MuJoCo 机身 z=0.148–0.272 m，
Isaac z=0.206–0.223 m，且 MuJoCo 后脚接地占空低。提高 `dof_frictionloss`
虽降低关节角误差，却使前向轨迹更差，不能单独解释差距。
3× 阻尼在前 2 s 的 base-x RMSE 比4×低（0.0697 vs 0.1215 m），
但关节角/速度和高度误差仍是4×更低；仅凭成功率不能定物理参数。

已按用户要求只读搜索 desktop 的本工程、robot_ws、ROS 日志、常用文档和挂载目录。
找到的关节 NPZ 均来自 Isaac 仿真，ROS 中相关日志来自 MuJoCo 节点；
**未找到实机关节目标、角度/速度、电流/力矩和框架接触力遥测文件**。
`robodog_gym/envs/robocon_2027/config/hardware_capability.yaml` 的实测日期、限值仍为空，故无法完成 PACE 式实机参数辨识及实机阶段验收。


### 对齐初态后的结论与候选选择

第二轮中间权重 `064400` 在诊断性 MuJoCo 阻尼 3×、无目标裁剪时，seed 2027 为
**24/24**，独立 seed 2028 为 **120/120**；平均完成 8.85 s、P95 8.96 s、
顶梁接触峰值 236.97 N。`064341` 同协议也是 **120/120**，平均 9.34 s、峰值
218.49 N；4× 时 `064341` 117/120、`064400` 113/120。
`064400` 的 Isaac 四种子 train-matched 为 133/192，`064341` 为 129/192，
这个差异不足以抵消未标定动力学造成的模型选择不确定性。
**保留 `064341` 为当前 Stage 1 研究候选，`064400` 为备选，`064242` 原件不覆盖；
不使用末点 `064540`，不进入 Stage 2。**
新增回归：`evaluations/robocon_2027/low_frame_064400_stage1_mujoco_corrected_damping3_unclipped_regression120_seed2028.json`。

已在 Isaac 评测器中记录 reset 后机身四元数/速度、12 关节 q/dq、四脚法向力与足端高度。
Isaac/MuJoCo reset 后关节位置及脚底几何高度接近，但速度和足地承重已不同；
首动作后 MuJoCo 足端载荷及机身弹跳明显偏高。
把 Isaac 精确 reset 物理状态覆写到 MuJoCo 后，默认阻尼下 `064242` 仍 **0/24**、
`064341` **2/24**；再覆写首帧策略历史，`064341` **1/24**。
只增大足地 `solref` 的四档单因素扫描未改善，首步足端力差仍约 22 N RMSE。
因此初态/历史/单一接触软化均不是足够的修复。
条件、文件校验和及轨迹记录见
`evaluations/robocon_2027/low_frame_stage1_dynamics_replay_audit.json`。

下一步需要从安全吊架的实机无框架激励开始采集带时间戳的关节目标、q/dq、
IMU、驱动电流或力矩，以及足端/框架接触；当前 desktop 搜索未发现这些日志。
在实测数据与硬件安全限值补齐前，不能确认哪组 MuJoCo 阻尼/接触参数代表真实机器人。
当前可继续做仿真诊断，但继续单一 Isaac PPO 迭代已不能作为跨引擎问题的替代修复。


### 实机遥测采集准备

部署端 `MotorDriver` 已有带方向和减速比换算的关节侧反馈力矩，但原 `/joint_states.effort` 固定为零。
现已在 `deploy_cpp` 加纯观测发布：实机模式填入反馈力矩；新话题 `/policy_joint_targets`
记录统一位置命令路径下的 12 关节目标、关节名与时间戳；fake/MuJoCo 模式不伪造力矩。
`deploy_cpp/README.md` 有 `ros2 bag record` 示例。**没有启动实机或电机。**
修改的 C++ 编译单元已编译、`git diff --check` 通过；Foxy colcon 全构建在
既有 Torch ABI/依赖组合的链接阶段失败：旧 CMake 缓存混用 Torch 1.10 头文件与 py3.7 Torch 1.13 库，并以 ABI=0 编译，而 Foxy rclcpp、yaml-cpp 与 Unitree SDK 为 ABI=1；尚未生成新版可执行文件。现成 ABI=1 Torch 2.10+cu128 的隔离构建又受系统 CMake 3.16 与 CUDA 11.8 工具链不兼容阻止。未安装依赖，也未启动节点/电机。
采集实机数据前需先修复该环境链接，并在安全吊架下验证目标/反馈时间戳及丢包行为。


## 0. 当前状态：M2 限高框架 Stage 1

**Stage 2 未开放。** 原候选 `064242` 完整保留在
`runs/mybot_v3_low_bar_expert_resume_064043/2026-10-11_14-06-04.613002`。
从它独立续训的硬框架 run 为
`runs/mybot_v3_low_bar_expert_resume_064242/2026-10-12_03-07-21.669914`，
产出 `064260/064280/064300/064320/064340/064341`。此 run 加载 Stage 1 的真实五碰撞体框架，
训练 100 次迭代，未覆盖 `064242`。进程在全部权重及 JIT 写出后以段错误退出；
`064341` 已被 Isaac Gym 和 MuJoCo 实际加载评测。**`064341` 是研究候选，尚不替换 `064242`。**

### Stage 1 阶段结果

框架净高 **0.36 m**、净宽 **1.10 m**、长 **0.40 m**；起点前距 **2.0 m**，
前进指令 **0.40 m/s**。MuJoCo 通过要求整机在门内离框，切换至 `059950`，
且连续保持姿态/高度稳定 1 s；交接参数为离框后继续保持专家 1 s，再混合 0.5 s。

| 引擎与协议 | `064242` | `064341` | 证据 |
| --- | ---: | ---: | --- |
| Isaac nominal，固定起点 | **192/192** | 未跑同批 192 | `evaluations/robocon_2027/low_frame_064242_x200_stage1_nominal_stage192.json` |
| Isaac nominal，横向 ±3 cm、航向 ±2°，留出 seed 20261127 | **48/48** | **48/48** | `evaluations/robocon_2027/low_frame_stage1_hard_ft_screen48_seed20261127.json` |
| Isaac train-matched，横向 ±3 cm、航向 ±2°，四个留出 seed | **119/192** | **126/192** | `evaluations/robocon_2027/low_frame_stage1_hard_ft_trainmatched_2x48_seeds20261127_28.json`、`low_frame_stage1_hard_ft_trainmatched_2x48_seeds20261129_30.json` |
| MuJoCo 修正评测器，默认阻尼、保留部署目标裁剪，seed 2027 | **0/24** | **0/24** | `evaluations/robocon_2027/low_frame_064242_stage1_mujoco_corrected_smoke24_seed2027.json`、`low_frame_064341_stage1_mujoco_corrected_smoke24_seed2027.json` |
| MuJoCo 诊断阻尼 ×4、保留目标裁剪，seed 2027 | **24/24** | **23/24** | `evaluations/robocon_2027/low_frame_064242_stage1_mujoco_corrected_damping4_smoke24_seed2027.json`、`low_frame_064341_stage1_mujoco_corrected_damping4_smoke24_seed2027.json` |
| MuJoCo 诊断阻尼 ×4、**去除独有的目标裁剪**，seed 2027 | **19/24** | **23/24** | `evaluations/robocon_2027/low_frame_064242_stage1_mujoco_corrected_damping4_unclipped_smoke24_seed2027.json`、`low_frame_064341_stage1_mujoco_corrected_damping4_unclipped_smoke24_seed2027.json` |
| 同上，独立 seed 2028 回归 | **109/120** | **117/120** | `evaluations/robocon_2027/low_frame_064242_stage1_mujoco_corrected_damping4_unclipped_regression120_seed2028.json`、`low_frame_064341_stage1_mujoco_corrected_damping4_unclipped_regression120_seed2028.json` |

Isaac train-matched 四个留出 seed 的配对提升为 **+3.65 个百分点**
（新策略独赢 18 次、独输 11 次，McNemar 精确检验 **p=0.265**），
不能确认提高。MuJoCo 阻尼 ×4、无目标裁剪的 120 组初态
逐回合相同；新策略 **117/120** 对旧策略 **109/120**，新策略独赢 11 次、
独输 3 次，McNemar 精确检验 **p=0.0574**。这是有价值的改进信号，仍未达到常用
0.05 显著性线。`064341` 的三次失败为两次顶梁冲击和一次门外通行，
5 ms 接触力峰值最大 **272.40 N**。旧策略 11 次失败均为顶梁冲击，
峰值最大 **363.89 N**。同样修正判据的阻尼 ×4、**有目标裁剪**旧策略回归为
**120/120**，最大 5 ms 接触力 **211.82 N**：
`evaluations/robocon_2027/low_frame_064242_stage1_mujoco_corrected_damping4_regression120_seed2028.json`。

### 评测器修复与差距解释

- MuJoCo 现按每个碰撞 geom 在框架 x 区间内的横向 AABB 判定门内通过；
  门外绕行或未进门直接越过均记 `outside_gate`。顶梁接触力改为每个 **0.005 s**
  物理子步采样峰值，并记录子步累计冲量。旧结果只在 **0.020 s** 控制步末采样，
  不能与新结果直接混用。新增 5 项定向测试以及远端 13 项规则测试均通过。
- Isaac 模型有 **19 个刚体、15.284568 kg**，原 MuJoCo 为 **17 个刚体、14.301965 kg**。
  当前评测开关补入固定载荷和 IMU、校正右后小腿与刚体惯量、消除错误的 DOF
  armature，并在改动模型后刷新 MuJoCo 常量。首个策略观测前还模拟 Isaac 的一次
  零动作 reset 步与历史预热；前向指令、动作中心和顶梁宽度也已对齐。
- **阻尼 ×4 是未物理验证的诊断参数**：在当前 MuJoCo 原设 1/2 N·m·s/rad 的
  被动关节阻尼上乘 4，而训练 URDF 关节阻尼为 0。它显著改变行进动力学；
  有目标裁剪时 120/120 还依赖 Isaac 没有的关节目标裁剪。
  因此不能把这些结果当作 `064242` 或 `064341` 已通过干净的跨引擎验收。
  旧策略默认阻尼仅 0/24，新策略也仅 0/24，仍是 Stage 1 的主要阻塞。
- Stage 1 的 nominal Isaac 成功率虽高，四个留出 seed 的 train-matched 仅约 62%–66%；
  `hardware_capability.yaml` 的最低安全高度、动态连续验证、扭矩/冲击上限仍未实测。
  计划要求 MuJoCo ≥85%、Isaac→MuJoCo 成功率下降 ≤10 个百分点，
  最终 0.29 m nominal ≥95%、robust ≥90%、实机 ≥18/20。当前均不能宣称完成。
  **不进入 Stage 2。**

### 下一步与开源方法

1. 依 [PACE](https://github.com/leggedrobotics/pace-sim2real) 的系统辨识思路，
   用相同无障碍 PD 目标比较 Isaac/MuJoCo 的关节位置、速度、扭矩和机身轨迹，
   拟合阻尼、摩擦与执行器响应；先解释为何阻尼 ×4 才接近 Isaac 轨迹，
   再确定有实物意义的部署模型。PACE 代码针对 Isaac Sim/Isaac Lab，
   应借鉴流程而非直接替换本项目 Isaac Gym。
2. 继续改进 Stage 1 随机初态鲁棒性，并用留出种子筛选 checkpoint；
   每次都以修正的门内判据和 5 ms 接触力回归验证。参考
   [ABS](https://github.com/LeCAR-Lab/ABS) 的恢复策略思路完善 `064242/064341 → 059950`
   切换条件和动作平滑；其完整方法需要单独训练恢复/价值网络。
3. [HIMLoco](https://github.com/InternRobotics/HIMLoco) 的历史本体估计适合作为后续
   鲁棒策略消融，但会改变 actor/估计器训练接口，不能直接补到现有 checkpoint，
   也不能解释当前的物理参数差距。desktop 已有其第三方 fork 用于 Isaac Gym 依赖，
   不代表当前策略用了 HIMLoco 算法。三者授权不同，若引入代码需按各仓库许可证处理。

使用现有 `/home/ldl/anaconda3/envs/robodog_gym/bin/python`，没有安装新依赖。
运行训练/Isaac 评测时需在 `PYTHONPATH` 前置本仓库并将该环境的 `bin` 加入 `PATH`，
否则可能误导入旧工程或找不到已有 `ninja`。当前 M2 训练日志为
`logs/low_frame_stage1_hard_resume_064242_x200_seed20261126_iter100.log`。

以下第 1–8 节保留 **2026-10-11 的 M1 历史记录和当时的下一阶段计划**；
若与本节冲突，以本节最新 M2 结果为准。

## 1. 本轮完成的工作

1. 按计划 §4.1 / §4.8 把绕桩和单木桥的“零训练基线”从逐点贪心控制换成
   真正的路径跟踪控制器，并加入可复现的初始位姿小扰动（±3 cm / ±2°）。
2. 用规则图和表 1 复核了绕桩几何：6 根 φ100×100 mm 竖桩、中心距 900 mm、
   L 形布局，起点圈中心距 1 号桩 400 mm，三处 φ300 mm 必达圈。
3. 定位并修复了两个控制器级缺陷（详见第 3 节）。
4. 完成三个障碍各 20 次的阶段评估，并补齐耗时、接触力、桩间距、
   桥面横向误差等比赛口径指标。
5. 完成限高框架的消融实验，定位了失败根因。

## 2. M1 阶段评估结论

| 障碍 | Checkpoint | 试验 | 成功率 | 判定 | 关键指标 |
| --- | --- | ---: | ---: | --- | --- |
| 绕桩（六桩 L 形） | `059950` + 路径跟踪 | 20 | **19/20 = 95%** | **可直接复用**（只需高层规划与提速） | 平均 72.6 s，P95 73.4 s，最小桩间距 −0.008 m |
| 单木桥 400×3000×100 mm | `059950` + 路径跟踪 | 20 | 14/20 = 70% | 需 fine-tune（桥面专家） | 平均 32.3 s，最大横向漂移 0.129 m |
| 限高框架 290×800×800 mm | `063296` | 20 | 0/20 | **暂不可判为需重训：先解决 Sim-to-Sim 差异** | 停在 x=−0.27，base_z 保持 0.33，峰值接触力 280 N |

结果文件：

- `evaluations/robocon_2027/slalom_pursuit_v7_smoke20.json`
- `evaluations/robocon_2027/beam_pursuit_v3_smoke20.json`
- `evaluations/robocon_2027/low_frame_pursuit_smoke20.json`

## 3. 两个被定位并修复的控制器缺陷

### 3.1 跟踪索引正反馈（导致“抄近路”）

第一版纯 pursuit 用 `[index−6, index+60]` 搜索最近点，同时又把 index 单调
前推到 look-ahead 点。两者叠加形成正反馈：index 每步前进约 12 个采样点，
最终冲到路径末端，机器人于是放弃 weave、直接朝终点走直线。

修复：对整条参考路径做全局最近点搜索，不再把 look-ahead 结果写回 index。

### 3.2 航向与横移耦合（导致摆动腿扫桩）

修复 3.1 后仍全部碰桩，而且接触部位是**小腿和脚**。实测：

- 机器人摆动腿前伸约 0.43 m，抬脚最高点只有约 0.11 m，而桩高 0.10 m；
- 因此只要 **脚在桩的投影范围内经过就会撞桩**；
- 原先控制器把横移修正混进航向，机器人在 weave 段航向滞后（实测 18°，
  路径切线要求约 45°），摆动腿正好扫过桩顶。

修复：让机体航向跟踪**路径切线**，横向误差只用 body-frame `vy` 修正。
修复后绕桩从 0/20 提升到 19/20。

实测“绕桩保持距离”门槛（直行通过 1 号桩，命令速度 0.30 m/s）：

| 基座横向偏移 | 是否碰桩 | 最近几体距离 |
| ---: | --- | ---: |
| 0.24 m | 碰桩 | 0.057 m |
| 0.28 m | 碰桩 | 0.036 m |
| 0.32 m | 不碰 | 0.046 m |
| 0.36 m | 不碰 | 0.083 m |
| 0.40 m | 不碰 | 0.118 m |

参考轨迹因此按“绕每根桩半径 0.37 m 的禁入圆”重新设计。

## 4. 限高框架的根因分析（当前最重要的发现）

20 次全部 `not_crossed`：机器人走到 x=−0.27（框架入口）后停住，
`base_z` 全程停在 0.33 左右，**完全没有下蹲**；FR 髋部在 z=0.311
直接撞上 0.29 m 横梁前沿，峰值接触力 280 N。

为区分“策略能力不足”和“评测环境不对”，做了单横杆消融实验
（完全复刻训练用的 low_bar.urdf：0.05×1.0×0.05 横杆 + 两根立柱）：

| 横杆净空 | 结果 |
| ---: | --- |
| 0.45 m | 通过（基座 0.333，机身不需要下蹲） |
| 0.38 m | 卡住，x=0.319 |
| 0.34 m | 卡住，峰值接触力 2238 N |
| 0.30 m（比赛值/训练值） | 卡住，x=0.206 |
| 0.27 m | 卡住，x=0.207 |

对照训练 run 自带的门禁评估
（`runs/mybot_v3_low_bar_expert_resume_062097/.../eval_low_bar_gate_m1_3b_final_*.json`）：

- Isaac Gym 侧：净空 0.30 m 时 **pass rate 77.3%**，碰撞率 5.8%，
  `mean_min_base_height = 0.3085 m`；
- MuJoCo 侧：同样净空 **0/5 通过**。

也就是说，**同一个 checkpoint 在 Isaac 能过、在 MuJoCo 完全过不去**，
跨引擎成功率差超过 70 个百分点。按计划 §6.3 的红线（差超过 10 个百分点
就禁止扩大课程），限高框架此时**不允许进入 fine-tune**，必须先修 Sim-to-Sim。

已排除的因素：

- 7 维障碍观测契约一致（训练侧 `low_bar_relative_state` 与
  `sim2sim_mujoco.obstacle_ahead_state` 逐字段对齐，净空/厚度/航向一致）；
- 关节阻尼/摩擦/电枢用的是本工程既有的 `aligned_damped` profile
  （1.0/2.0、0.3/0.1、armature 0.01），与 `sim2sim_mujoco_3stairs.py` 一致；
- 不是障碍观测导致停车（把 `obstacle_ahead_x` 设为 0.0 或 0.5 结果相同）。

### 训练侧奖励只约束“刚体原点”，不约束真实碰撞包络

`robodog_gym/envs/rewards/corl_rewards.py::_reward_low_bar_body_top`：

```python
z       = bodies[:, :, 2]                      # 刚体原点高度，不是外廓
ceiling = clearance - (leg_mask * 0.05 + (1-leg_mask) * (-0.10))
excess  = clamp(z - ceiling, min=0).amax()
```

也就是说：腿的**原点**上限是 `clearance − 0.05 = 0.25 m`，
躯干的**原点**上限是 `clearance + 0.10 = 0.40 m`。
奖励函数里没有任何一项直接约束“机体最高点 < 横梁底面”。

而 MuJoCo 里用精确网格顶点量出的碰撞包络是：

| 部位 | 相对基座的碰撞 z 范围 |
| --- | --- |
| `body`（躯干） | [−0.102, +0.102] |
| `FL_hip` / `FR_hip` | [−0.028, +0.028] |
| `FL_thigh` | [−0.137, +0.029] |

要在 0.29 m 横梁下无接触通过，躯干需要满足 `base_z + 0.102 < 0.29`，
即 `base_z < 0.188`；而 Isaac 评估记录的 `min_base_height = 0.3085`。
两者相差约 0.12 m，正好是“原点约束”和“外廓约束”的缺口。

最可疑的剩余差异（下一阶段优先排查）：

1. **机体碰撞包络不一致**。需要核对两端躯干/髋/大腿的碰撞体尺寸与形状
   （Isaac 用凸包 + `thickness=0.01`，MuJoCo 用 XML 里的 mesh/box），
   并解释为什么 Isaac 在 base 0.3085 时碰撞率只有 5.8%。
2. **奖励与验收口径不一致**：训练只压原点，而比赛判据是整机外廓，
   必须把奖励改成对真实碰撞包络（或显式机体最高点）施加约束。
3. 姿态/关节映射：确认 12 关节顺序、hip 符号约定、`clip_actions` 一致。
4. 观测噪声：训练侧 `observation_noise: true` 且高程噪声较大，
   需要确认评测端是否也要注入同类噪声才公平。

## 5. 台阶能力（M0 遗留项，MuJoCo 侧已完成）

| 高度 | robust 20 次 | 失败类型 |
| ---: | ---: | --- |
| 8 cm | 18/20 = 90% | 1 次未登顶、1 次前进不足 |
| 10 cm | 20/20 = 100% | — |
| 12 cm | 16/20 = 80% | 4 次跌倒 |

Isaac Gym 侧的同口径对比评测脚本**尚不存在**，属于下一阶段第一个要补的
工具（计划 §9 第 4 项）。在补齐之前，12 cm 台阶不允许作为沟壑组合的
warm start 进入正式训练。

## 6. 当前门禁状态（`scripts/validate_robocon_2027.py`）

- 高墙：`stable_foot_lift`、`repeated_landing` 仍为 `unmeasured` → 关闭
- 沟壑：`controlled_gap_crossing`、`repeated_landing` 仍为 `unmeasured` → 关闭
- 限高：`minimum_safe_robot_height` 模型运动学可达 0.163 m（几何 PASS），
  但动态连续通过未验证 → 训练门保持关闭

## 7. 下一阶段（M2）任务清单

按计划 §7.2，M2 的目标是“多技能初步集成，完成至少两个可得分项”。
结合本轮结论，执行顺序调整为：

### P0-1 修复限高框架 Sim-to-Sim（阻塞项）

1. 写 `scripts/eval_low_bar_isaac.py` 的对偶评测：同一 checkpoint、
   同一净空表、同一判据，在 Isaac 与 MuJoCo 各跑 100–200 次并输出差值。
2. 打印两端通过瞬间的机体最高点、髋部高度、关节角，找出第一个分歧点。
3. 对齐碰撞包络（凸包/`thickness` vs XML mesh）后复测。
4. 只有跨引擎差 ≤10 个百分点，才允许开始限高 fine-tune。

### P0-2 绕桩提速（已有 95% 成功率）

1. 当前 72.6 s 远超 180 s 全场预算，需要把速度提上去；
2. 先扫速度阶梯（0.28 → 0.34 → 0.40 命令值），记录成功率与 P95 用时；
3. 目标是规划器层面达成 40 s 内、成功率不下降；
4. 规划器方案已足够，不需要新建绕桩 expert。

### P0-3 单木桥 fine-tune

1. 实测站宽 0.405 m ≈ 桥宽 0.400 m，中心线误差预算只有 ±0.15 m，
   这是 70% 成功率的物理原因；
2. 训练目标：窄站宽步态 + 桥面中心线跟踪 + 桥沿接触惩罚；
3. 从 `059950` 初始化，按计划 §4.8 的奖励表实现；
4. 先把 0.80 m 宽桥面练到 ≥95%，再收窄到 0.40 m。

### P1 补 Isaac 侧台阶评测

1. 用 Isaac Gym 跑 8/10/12 cm 各 100 次，和 MuJoCo 的
   18/20、20/20、16/20 对比；
2. 只有跨引擎差 ≤10 个百分点，才把台阶/斜坡子项并入沟壑组合。

## 8. 常用命令

```bash
cd /home/ldl/mybot_v3_stair_turn_rl

# 规则与硬件门检查
/home/ldl/anaconda3/envs/robodog_gym/bin/python scripts/validate_robocon_2027.py

# 单元测试
PYTHONPATH=. /home/ldl/anaconda3/envs/robodog_gym/bin/python -m unittest discover -s tests -v

# 零训练基线（路径跟踪）
/home/ldl/anaconda3/envs/robodog_gym/bin/python scripts/eval_robocon_2027_baselines.py \
  --obstacle slalom --trials 20 --max-seconds 130 --seed 2027 --controller pursuit \
  --output evaluations/robocon_2027/slalom_pursuit_v7_smoke20.json
```
