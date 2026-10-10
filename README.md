# MyBot V3 RoboCon 足式机器人强化学习控制

本仓库是 **MyBot V3 四足机器人**（12 自由度）的强化学习运动控制工程，面向
ROBOCON 仿生足式机器人挑战赛。控制器在 **Isaac Gym** 中训练，在 **MuJoCo**
中做 sim2sim 验证，并通过 ROS2 / C++ 节点部署到实机。

当前覆盖的运动能力：

- 平地行走、前后/横向移动、原地与行进转向；
- 上行与下行台阶（训练 ~8–20 cm，鲁棒验收落到 10/12 cm）；
- 静止站立与零指令保持；
- RoboCon 障碍技能：限高杆（Low Bar）、木桥 A/B、高墙、T 字台阶、大斜坡、
  直角绕杆、砂砾碎木坑（障碍技能按里程碑逐个训练，进度见下文）。

---

## 1. 上游来源与许可

本工程派生自 ETH-PBL 的 ELMAP 项目，并沿用其技术栈：

| 项目 | 说明 |
|---|---|
| [ELMAP / elmap-rl-controller](https://github.com/ETH-PBL/elmap-rl-controller) | 本项目直接来源，外感受（高程图）鲁棒运动控制 |
| [walk-these-ways](https://github.com/Improbable-AI/walk-these-ways) | 步态条件化与部署链路参考 |
| [legged_gym](https://github.com/leggedrobotics/legged_gym) | Isaac Gym 环境框架 |
| [rsl_rl](https://github.com/leggedrobotics/rsl_rl) | PPO 训练框架 |

派生过程记录见 `PROJECT_ORIGIN.md`：本仓库基于
`elmap-rl-controller@7e90827` 复制而来，未携带上游 `.git`、`runs`、日志和
checkpoint。

许可：MIT（见 `LICENSE`）。第三方许可保留在 `LICENSES/`
（`legged_gym`、`rsl_rl`）。

---

## 2. 技术栈

| 部分 | 实现 |
|---|---|
| 训练仿真 | NVIDIA Isaac Gym Preview 4（GPU PhysX） |
| 训练算法 | PPO + CSE/RMA 式 student–estimator（privileged teacher） |
| 策略输入 | 本体感受历史 + 77 点局部高程图（11×7），10 帧观测历史 |
| 动作输出 | 12 个关节的目标角偏移，`action_scale = 0.25`，PD 位置控制 |
| sim2sim | MuJoCo + Python bridge（`scripts/sim2sim_mujoco*.py`） |
| 部署 | ROS2 / C++ 包（`deploy_cpp/`），TorchScript 导出 |

参考运行环境（本机已验证）：

```text
OS / GPU        Ubuntu + NVIDIA RTX 2080 Ti 22 GB
conda env       robodog_gym  (Python 3.8.16)
PyTorch         1.10.0+cu113
Isaac Gym       Preview 4
工程路径        /home/ldl/mybot_v3_stair_turn_rl
```

---

## 3. 目录结构

```text
mybot_v3_stair_turn_rl/
├── robodog_gym/                 # 环境、奖励、机器人配置
│   ├── envs/base/               # LeggedRobot 基类、观测、配置
│   ├── envs/rewards/            # 奖励项（含 low_bar 系列）
│   └── envs/robodog/            # mybot_v3 等机器人配置
├── robodog_gym_learn/           # 训练侧 PPO/CSE、runner、存储
│   └── ppo_cse/                 # 本项目实际使用的算法实现
├── scripts/                     # 训练 / 评测 / sim2sim 入口
├── deploy_cpp/                  # ROS2 + C++ 部署包
├── resources/                   # 机器人 URDF、障碍物模型（含 low_bar）
├── docs/                        # 训练计划与评测协议文档
├── evaluations/                 # 历史评测结果（JSON / CSV / 截图）
├── runs/                        # 训练输出与 checkpoint（不纳入版本控制）
└── logs/                        # 训练与评测日志（不纳入版本控制）
```

`runs/`、`logs/`、`*.jit`、`*.pdf`、`robocon_reference/` 等都在 `.gitignore`
中：**clone 本仓库不会得到 checkpoint，需要在本机训练或另行拷贝。**

---

## 4. 环境安装

```bash
conda create -n robodog_gym python=3.8.16
conda activate robodog_gym

# PyTorch（RTX 20/30 系；40 系请改用 cu121 版本）
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 \
  -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Isaac Gym Preview 4
cd isaacgym/python && pip install -e . && cd -

# 本工程
pip install -e .
```

注意事项：

- 必须 **先 import `isaacgym`，再 import `torch`**，否则会报
  `PyTorch was imported before isaacgym modules`。
- 如果出现 `libpython3.7m.so.1.0` 找不到，需要在 conda 激活脚本里把环境
  `lib` 目录加入 `LD_LIBRARY_PATH`。
- 训练默认 4096 环境需要约 10–12 GB 显存；显存不足时用 `--num-envs` 降低。

---

## 5. 快速开始：MuJoCo 交互仿真

项目提供一键启动脚本 `run_mujoco.sh`，加载**完整 RoboCon 场地**，默认进入
键盘遥控模式：

```bash
cd /home/ldl/mybot_v3_stair_turn_rl

./run_mujoco.sh                      # 默认从地面启动区开始
./run_mujoco.sh low_bar              # 从限高杆入口开始
./run_mujoco.sh --replace bridge     # 关掉旧窗口并重启到木桥入口
./run_mujoco.sh --stop               # 关闭当前仿真
```

可选出生位置：`start`、`t_stairs`、`wall`、`low_bar`、`slope`、`slalom`、
`gravel`、`bridge`。

键盘操作：

| 按键 | 功能 |
|---|---|
| `W` / `S` | 前进 / 后退 |
| `A` / `D` | 左右横移 |
| `Q` / `E` | 左转 / 右转 |
| `1`–`8` | 瞬移到 8 个障碍测位点 |
| `R` | 复位到当前测位点 |
| 方向键 / `U` / `O` | 施加水平 / 垂直扰动 |

脚本默认使用 `deploy_cpp/config/robots/mybot_v3_cse_sim.yaml` 配置和
`runs/mybot_v3_sim2sim_phase3b_resume_059500/.../checkpoints/` 下的
`body_best_sim2sim.jit` / `adaptation_module_best_sim2sim.jit`。这些文件是本地
训练产物；如果要换成自己的 checkpoint，用 `--body` / `--adaptation` 指定。

### 在 VS Code 中运行

仓库内置了 VS Code 任务（`.vscode/tasks.json`）：

- `Ctrl+Shift+B` 运行默认任务 **MuJoCo: 启动或重启 RoboCon**，会弹出下拉框
  选择出生位置；
- 另有 **MuJoCo: 停止 RoboCon** 任务用于关闭仿真。

这样不必每次手敲命令，也不用在终端里记参数。

---

## 6. 训练

训练入口统一为 `scripts/train_mybot_v3_stair_turn.py`，推荐通过包装脚本启动。

### 后台启动（推荐）

```bash
./scripts/start_mybot_v3_training.sh
./scripts/start_mybot_v3_training.sh --num-envs 2048 --iterations 30000 --seed 7
```

进程号写入 `logs/training.pid`，最新日志软链到 `logs/training_latest.log`：

```bash
cat logs/training.pid
tail -f logs/training_latest.log
nvidia-smi
```

### 前台直接运行

```bash
./scripts/run_mybot_v3_training.sh --num-envs 4096 --iterations 60000
```

### 从 checkpoint 续训

```bash
./scripts/run_mybot_v3_training.sh \
  --resume-run /绝对路径/runs/mybot_v3_stair_turn_from_scratch/2026-xx-xx_xx-xx-xx \
  --checkpoint 59000 \
  --iterations 2000
```

### 常用参数

| 参数 | 说明 |
|---|---|
| `--robocon-obstacle {mixed,stairs,low_bar}` | 选择任务集，默认 `mixed` |
| `--num-envs` | 并行环境数（默认 4096） |
| `--iterations` | 训练迭代数（默认 60000），每轮 `num_steps_per_env=24` |
| `--resume-run` / `--checkpoint` | 指定续训 run 目录与迭代号 |
| `--save-interval` | checkpoint 保存间隔 |
| `--allow-observation-expansion` | 观测扩维时只扩展已知输入层，新字段从 0 开始 |
| `--resume-action-std` / `--freeze-resume-action-std` | 重置并可选冻结动作噪声 |
| `--observation-adapter-only` | 冻结成熟 actor，仅训练新增障碍观测输入列 |
| `--freeze-adaptation-module` | 冻结 estimator，仅微调 actor |
| `--resume-learning-rate` / `--policy-anchor-coef` | 微调学习率与参考策略锚定系数 |
| `--low-bar-lateral-init-range` / `--low-bar-yaw-init-range` | 限高杆出生位姿扰动范围 |

训练输出目录形如 `runs/<task>_<from_scratch|resume_xxxxxx>/<timestamp>/`，包含
`checkpoints/ac_weights_*.pt`、JIT 导出、`parameters.yaml` 和 `metrics.pkl`。

---

## 7. 评测与验收

### 7.1 限高杆门禁评测（正式协议）

`scripts/run_low_bar_gate.sh` 在**同一次调用**里重测基线并评测所有候选，保证
逐 episode 配对可比：

```bash
./scripts/run_low_bar_gate.sh \
  --run runs/mybot_v3_low_bar_expert_resume_060898/2026-10-10_12-03-27.366847 \
  --iterations 60920 60922 \
  --baseline-run runs/mybot_v3_low_bar_expert_resume_060874/2026-10-10_11-31-45.883150 \
  --baseline-iteration 60898 \
  --tag m1_2_stage1
```

报告脚本 `scripts/report_low_bar_gate.py` 会输出 Wilson 95% 区间、配对差值
区间、精确 McNemar p 值和四格计数，并给出判定。底层评测器为
`scripts/eval_low_bar_isaac.py`。

**评测纪律（重要）**：绝对成功率存在约 1.3 个百分点的进程级抖动，不同
`--num-envs` 会消费不同随机序列，**不能跨调用比较数字**。只有配对差值 95%
区间下界大于 0 才算提升。384 环境只用于筛选，正式判定用 2 个 held-out seed ×
500 环境 = 1000 episodes。协议细节见 `docs/ROBOCON_OBSTACLE_TRAINING_PLAN.md`
第 4.5 节。

### 7.2 MuJoCo sim2sim 评测

```bash
# T 字台阶 / 大斜坡验收（nominal 或 robust）
/home/ldl/anaconda3/envs/robodog_gym/bin/python scripts/eval_mujoco_obstacles.py \
  --obstacle both --trials 30 --output-dir evaluations/m1_obstacles

# 三阶梯鲁棒性批量评测（10/12 cm 等）
/home/ldl/anaconda3/envs/robodog_gym/bin/python scripts/eval_mujoco_3stairs_robust.py

# 单次手动复测
/home/ldl/anaconda3/envs/robodog_gym/bin/python scripts/sim2sim_mujoco.py \
  --terrain-mode stairs --stairs-step-height 0.12 --vx 0.4
```

### 7.3 机器人合规与可视化

```bash
# 尺寸/质量/蹲姿合规审计（RoboCon 规则 V2.0）
python scripts/m0_robot_compliance.py

# 台阶可视化（固定地形等级，键盘遥控）
python -u scripts/view_stairs.py --iteration 37500 --level 8

# 限高杆离屏渲染（无 X display 也可用）
python -u scripts/view_low_bar.py --clearance 0.30
```

---

## 8. 当前进度（截至 2026-10-10）

以仓库中实际跑出的评测为准，不使用未复现的峰值数字。

| 能力 | 状态 |
|---|---|
| 平地行走 / 横移 / 转向 / 静止站立 | 可用，`059950` 为通用运动基线 |
| 上下台阶 | 可用；MuJoCo 12 cm 鲁棒压力测试通过 |
| T 字台阶、大斜坡 | 验收 harness 已就绪（`eval_mujoco_obstacles.py`） |
| 限高杆 Low Bar | 训练与评测链路已打通，尚未达到 M1 门槛 |
| 木桥 A / 木桥 B | 未开始正式训练 |
| 高墙 | 未开始正式训练 |
| 多障碍统一策略 | 未开始（计划用 teacher–student / DAgger 蒸馏） |

机器人合规（`M0_REPORT_20261005.md`）：质量 14.30 kg（≤35）、站立外廓
659×437×406 mm（≤800×600×600）、足端接地外接圆 56.5 mm（≤80）、蹲姿全身最高
点 162.7 mm（限高杆净空 300 mm，余量充足）。

限高杆 Low Bar 明细：

- Actor 观测已从 3 维扩展为 7 维，旧字段顺序保留，可安全扩维加载旧
  checkpoint：`forward`、`lateral`、`bar_bottom_above_base`、
  `sin(heading_error)`、`cos(heading_error)`、`ground_clearance`、`valid`；
- 横向 ±0.05 m、航向 ±0.05 rad 的 `train-matched` 条件下，用正式协议
  （1000 episodes）测得 `060898` ≈ 73.1%、`060920` ≈ 73.1%、`060922` ≈ 73.4%，
  三者配对差值区间均包含 0，即**当前无可分辨的提升**；
- 主要失败模式是门框外穿越（约 21%）；碰杆约 5.8%，真实跌倒约 0%；
- 因此 M1 判定为**未完成**：在连续两个 checkpoint 用正式协议达到 90% 前，
  不启动木桥 A 的正式训练。

---

## 9. 已知问题与注意事项

- **Isaac Gym 退出时可能 segfault**（core dumped）。这通常发生在评测结果和
  checkpoint 已写完之后，属于既存析构问题；门禁脚本已改为显式退出，避免把
  成功写成失败。
- **checkpoint 与 JIT 不在版本库中**：`runs/`、`logs/`、`*.jit` 均被忽略，
  换机器或重新 clone 后需要重新训练或手动拷贝。
- **MuJoCo bridge 当前按 119/116 单帧观测契约运行**（无 `obstacle_ahead`）。
  限高杆的 7 维障碍观测版本尚未接入 MuJoCo sim2sim；完整 RoboCon 地图仿真
  目前使用台阶专项 checkpoint。
- **脚本中存在本机绝对路径**（`/home/ldl/anaconda3/...`、
  `/home/ldl/mybot_v3_stair_turn_rl`）。迁移到其他机器时需要相应修改。
- **训练指标已修正**：episode 统计按实际完成数量加权，且每个 physics step
  清空旧 `extras`，不再重复累计上一次 reset 的指标。历史旧口径评测日志已归档
  到 `logs/archive_pre_metric_fix/`，不要与当前数字比较。

---

## 10. 部署

`deploy_cpp/` 是 ROS2 / C++ 部署包，同时支持：

- **Sim2Sim**：MuJoCo 物理由 Python bridge 运行，C++ 节点做策略推理与控制；
- **Sim2Real**：真机 mybot_v2_1，订阅 IMU 与高程信息，通过 Unitree
  GO-M8010-6 SDK 控制 12 个电机。

导出两个 TorchScript 模型：

```text
adaptation_module_latest.jit : estimator history -> latent
body_latest.jit              : policy history + latent -> 12 维动作
target_dof_pos = policy_dof_pos + action * 0.25
```

详细配置、话题和启动方式见 `deploy_cpp/README.md`。

---

## 11. 文档索引

| 文档 | 内容 |
|---|---|
| `docs/ROBOCON_OBSTACLE_TRAINING_PLAN.md` | 主训练计划：里程碑、课程、评测协议与判定规则 |
| `M0_REPORT_20261005.md` | 赛题尺寸核对、机器人合规、蹲姿可达性 |
| `PHASE3_SIM2SIM.md` | 台阶 sim2sim 动力学差距分析与第三阶段训练 |
| `PLAN_ROBOCON_OBSTACLE_20261005.md` | RoboCon 障碍训练总体规划（历史版本） |
| `PROJECT_ORIGIN.md` | 工程来源与派生说明 |
| `deploy_cpp/README.md` | 部署包结构与参数 |
| `DIAGNOSIS_2026-10-02.md` | 早期问题诊断记录 |

---

## 12. 引用

如果本工程对你有帮助，请同时引用上游工作：

```bibtex
@inproceedings{plozza2025robustRLlocomotion,
  author={Plozza, Davide and Apostol, Patricia and Joseph, Paul and Schl\"apfer, Simon and Magno, Michele},
  booktitle={2025 IEEE International Conference on Robotics and Automation (ICRA)},
  title={Robust Reinforcement Learning-Based Locomotion for Resource-Constrained Quadrupeds with Exteroceptive Sensing},
  year={2025},
  pages={8121-8127},
  doi={10.1109/ICRA55743.2025.11128474}}
```
