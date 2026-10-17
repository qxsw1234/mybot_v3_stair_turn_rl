# MyBot V3 RoboCon 足式机器人强化学习控制

本仓库是 **MyBot V3 四足机器人**（12 自由度）的强化学习运动控制工程，面向
ROBOCON 仿生足式机器人挑战赛。控制器在 **Isaac Gym** 中训练，在 **MuJoCo**
中做 sim2sim 验证，并通过 ROS2 / C++ 节点部署到实机。

当前已具备平地、转向、站立和部分台阶运动基础。2027 障碍赛已有绕桩、单木桥和限高框架的仿真基线，**当前处于 M2 限高 Stage 1，尚未完成跨引擎与实机验收，Stage 2 关闭**。
2026-10-16 更新：绕桩 MuJoCo 阶段回归 190/200、成功回合 P95 35.46 s；0.80 m 宽桥从 `059950` 完成 25+200 迭代续训，但留出 Isaac 成功率低于基线，台阶回归为 99/120 对基线 116/120，**未晋级窄桥**。desktop CPU ABI=1 部署链已构建，完成 10 分钟 fake 遥测；实机标定仍待完成。详见[执行状态](docs/2027_ROBOCON_EXECUTION_STATUS.md)。
2026-10-17 更新：新增桥/平地/转向/台阶回放课程和固定 `059950` 锚点的桥专家 `060148`；规则桥双模式 100 回合为 94/100，未达 ≥95/100 切换门槛，**不晋级 0.60 m 或实机**。视频与完整证据见[桥专家回放报告](docs/2027_ROBOCON_BEAM_REPLAY_20261017.md)。

优先阅读：

- [2027 当前进度快照](docs/2027_ROBOCON_CURRENT_PROGRESS_20261017.md)
- [下一步整体规划：专项重训与整场连续运行](docs/2027_ROBOCON_RETRAINING_MASTER_PLAN_20261017.md)
- [八项连续新地图与键盘入口](docs/2027_ROBOCON_FULL_MAP_PREVIEW_20261017.md)
- [2027 当前执行状态](docs/2027_ROBOCON_EXECUTION_STATUS.md)
- [当前 2027 规则场景现场键盘测试](docs/2027_ROBOCON_LIVE_TEST_20261017.md)
- [全场景键盘体验入口与场景版本说明](docs/2027_ROBOCON_INTERACTIVE_SCENES_20261017.md)
- [下一步详细计划与完成判据](docs/2027_ROBOCON_NEXT_STEPS_20261015.md)
- [八项目后续训练路线与开源算法](docs/2027_ROBOCON_ALL_EVENTS_ROADMAP_20261016.md)
- [单木桥回放课程与 100 回合双模式筛选](docs/2027_ROBOCON_BEAM_REPLAY_20261017.md)
- [单木桥双模式仿真试验](docs/2027_ROBOCON_MODE_SWITCH_PILOT_20261016.md)
- [开源案例调研与采用建议](docs/2027_ROBOCON_OPEN_SOURCE_REFERENCES.md)
- [checkpoint 与评测证据校验清单](docs/2027_ROBOCON_ARTIFACT_MANIFEST.json)

本次GitHub更新仅包含代码、文档、结果摘要与校验清单。原清单的201份原始评测JSON/NPZ、本轮新增原始结果以及checkpoint/JIT本体留在desktop，不随clone下载。

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

`runs/`、`logs/`、`*.jit`、`*.pdf`、`robocon_reference/`及2027原始评测JSON/NPZ等都在 `.gitignore`
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

要现场驾驶当前已实现的 **2027 规则场景**，在 `desktop` 图形桌面的终端运行：

```bash
cd /home/ldl/mybot_v3_stair_turn_rl
./run_robocon_2027_live.sh beam --duration 3600
```

窗口内按 `4/5/6` 分别切换到规则绕桩、单木桥、限高；其余键位和模型见[2027 现场测试说明](docs/2027_ROBOCON_LIVE_TEST_20261017.md)。2027 其余五项尚无可验收规则场景。下面的旧地图入口只用于历史场地观察。

### 2027 规则尺寸单木桥：当前双模式候选

在 `desktop` 的图形桌面或已连接到该桌面的远程桌面中，打开终端：

```bash
cd /home/ldl/mybot_v3_stair_turn_rl
./run_robocon_2027_beam.sh
```

默认显示 **2027 规则桥 0.40×3.00×0.10 m**，从桥前起步；加载冻结行走模型 `059950` 和研究桥专家 `060148`，位置到 `x≥−0.15 m` 时自动切入专家，出桥 `x≥3.30 m` 时切回行走。初次尝试按住 `W` 缓慢前进，观察桥边位置，必要时用 `A`/`D` 修正横向、`Q`/`E` 修正朝向。

| 按键 | 作用 |
| --- | --- |
| `W` / `S` | 前进 / 后退 |
| `A` / `D` | 左 / 右横移 |
| `Q` / `E` | 左 / 右转向 |
| `1` / `2` / `3` | 强制行走 / 强制桥专家 / 按位置自动切换 |
| `R` | 回到规则桥入口 |
| 关窗或终端 `Ctrl+C` | 退出 |

想先只看原始行走策略，可以用 `./run_robocon_2027_beam.sh --mode locomotion`；想减速，可加 `--vx 0.18`。按键由 `desktop` 图形会话接收，只有 SSH 终端而没有打开桌面画面时无法目视操控。若使用 `ssh desktop` 启动，窗口出现在 `desktop` 显示器上；请同时连接其远程桌面。终端会显示当前模型、切换位置、跌倒和桥外落足提示。这个入口用于交互观察，**不会给出正式规则得分或更新 100 回合通过率**。`060148` 尚未达到晋级门槛，不用于实机。

### 历史 RoboCon 地图

项目提供一键启动脚本 `run_mujoco.sh`，加载**历史 RoboCon 场地**（不能用于2027规则验收），默认进入
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

### 7.1 历史单横杆评测协议（2027 框架请使用下一步计划附录）

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

## 8. 当前进度（2026-10-16 更新）

采用 2027 计划的 M0–M5 编号；旧版木桥 A/B、T 字台阶等方案是历史资料。

| 能力 | 最近已核实结果 | 判断 |
|---|---|---|
| 通用运动 `059950` | 平地、转向、站立与台阶基线 | 保留，不覆盖 |
| 绕桩 | MuJoCo 留出 190/200，成功 P95 35.46 s | 阶段目标达到；待 Isaac 同规则与实机 |
| 单木桥 400×3000×100 mm | MuJoCo 14/20，6次桥外落足失败 | 需专项微调 |
| 台阶 8/10/12 cm | Isaac 历史直线台阶：`059950` 116/120，桥候选 `060173` 99/120 | 桥候选基础步态退化；待跨引擎配对 |
| 限高 Stage 1，.36×1.10×.40 m | `064242` Isaac nominal 192/192；`064341` MuJoCo原设阻尼2/24 | 未通过跨引擎门槛 |
| 限高诊断 | `064341` 在3×被动阻尼下120/120 | 未经实机标定，不能算正式通过 |
| 高墙/完整跨沟/砖木坑/轮胎阵/梅花桩 | 无完成验收证据 | 后续专项 |

已完成两轮限高续训和动力学对照。保留`064242`原候选，`064341`研究候选、`064400`备选；`064540`跨引擎退化，排除。完整run路径与SHA-256见清单，不能只靠数字编号或`latest`选择模型。

10月16日规则测试 19 项通过；desktop 的 CPU ABI=1 部署节点已干净构建和安装，fake StandUp 连续 600 s 遥测通过，详见[部署报告](docs/deployment_build_report.md)。`hardware_capability.yaml`实测字段未补齐，尚无真实电机反馈和实机标定。

### 复查 desktop 本地证据

以下校验需要desktop上的原始评测文件；仅clone公开仓库会报告缺少数据。

```bash
python scripts/audit_robocon_2027_artifacts.py --verify
# 仅在已保存对应 runs/ 的 desktop 或备份机执行：
python scripts/audit_robocon_2027_artifacts.py --verify --include-checkpoints
```

历史JSON使用不同判据与物理参数，不能直接混合比较；详见[执行状态](docs/2027_ROBOCON_EXECUTION_STATUS.md)与[证据说明](evaluations/robocon_2027/README.md)。

---

## 9. 已知问题与注意事项

- **Isaac Gym 退出时可能 segfault**（core dumped）。这通常发生在评测结果和
  checkpoint 已写完之后，属于既存析构问题；门禁脚本已改为显式退出，避免把
  成功写成失败。
- **checkpoint 与 JIT 不在版本库中**：`runs/`、`logs/`、`*.jit` 均被忽略，
  换机器或重新 clone 后需要从已保存的 run/备份拷贝并校验，无须因此从头训练。
- **Python MuJoCo bridge 已支持119/116及126/123单帧契约**，由JIT输入推断。
  7维限高观测已接入专项评测；这不代表C++部署观测和实机验收完成。
- **部署链仍需实机验证**：desktop CPU ABI=1 路径已成功构建并完成 fake 测试；
  机器人目标机依赖、硬件采样时间戳、数据年龄和动力学标定仍未验证。
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
| [2027 当前进度快照](docs/2027_ROBOCON_CURRENT_PROGRESS_20261017.md) | 八项证据层级、模型结果、整场地图与未完成项 |
| [下一步整体规划](docs/2027_ROBOCON_RETRAINING_MASTER_PLAN_20261017.md) | 课程重训、八项路线、阶段门槛、实机条件与整场验收 |
| [2027 八项连续地图](docs/2027_ROBOCON_FULL_MAP_PREVIEW_20261017.md) | 一张新地图、键盘驾驶、最新可用模型切换与预览边界 |
| [2027 下一步计划](docs/2027_ROBOCON_NEXT_STEPS_20261015.md) | 当前工作包、方法、样本数与完成判据 |
| [2027 执行状态](docs/2027_ROBOCON_EXECUTION_STATUS.md) | 原始结果、候选选择和阻塞项 |
| [2027 总计划](docs/2027_ROBOCON_OBSTACLE_TRAINING_PLAN.md) | 180秒、八障碍、M0–M5 |
| [开源案例调研](docs/2027_ROBOCON_OPEN_SOURCE_REFERENCES.md) | PACE、Robot Parkour、ABS、Extreme Parkour、HIMLoco |
| `docs/ROBOCON_OBSTACLE_TRAINING_PLAN.md` | 历史Low Bar→Bridge A/B续训方案；不是2027进度入口 |
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
