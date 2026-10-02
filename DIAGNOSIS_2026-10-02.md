# 线 B 训练诊断报告（能力停滞 + value loss 飙升）

> 时间：2026-10-02　｜　对象：`/home/ldl/mybot_v3_stair_turn_rl`（ELMAP/PPO-CSE，Mybot V3 楼梯+转向）
> 证据来源：`robodog_gym/envs/{base/legged_robot.py, rewards/corl_rewards.py}`、`robodog_gym_learn/ppo_cse/ppo.py`、
> 两个 run 的 `outputs.log` 全量窗口解析（3340 + 142 个窗口）、地形课程图像素量化、以及零显存 CPU 评估（各 216 回合）。

---

## 0. 一句话结论

**能力停滞和 value loss 飙升是同一个问题的两面**：奖励里"摔倒零惩罚 + 惩罚项权重碾压正向奖励"让策略收敛到"保守少动"；
而 critic 的回归目标漂移导致 value loss 高达 354，由于 `loss = surrogate + 1.0×value_loss`，
**共享 optimizer 的梯度几乎全被 critic 吃掉，把 actor 拖着走**——所以 3.4 万轮下来 actor 的 KL 只有 0.001（几乎没动）。

---

## 1. 现象（实测）

| 测试（同地图同协议，各 216 回合） | iter 5500 | iter 39500 |
|---|---|---|
| 平地 vx=0.5：8 秒净前进 | 1.39 m | 1.31 m |
| 平地：直立率 / 成功率(≥2 m) | 25% / 45% | 24% / 43% |
| 楼梯 vx=0.5：净前进 | 0.64 m | 0.53 m |
| 楼梯：直立率 / 成功率 | 20% / **0%** | 24% / **0%** |

同期**地形课程难度仍在上升**（高难度占比 16% → 37.5%），即"训练指标在变好、真实能力没变"。

**训练侧 loss 动态**（窗口解析）：
- 长跑 6000→39250：value loss 在 10170–35190 稳定在 **0.008**；**35790 起开始飙升**（6→15→…），末期 **354 → 194 → 158 → 134 → 240**。
- 当前重训 38750→now：重启即飙升（51.8 / 30.8 / 34.6），随后在 6–24 之间震荡，间歇回 0.2。
- **策略侧一直很健康**：`policy kl mean` 0.001–0.002、`kl max` 0.002–0.004（desired 0.008、硬限 0.02），
  `surrogate loss` ≈ 0，`invalid policy updates` = 0（无非有限梯度）。
- 地形难度在 35790 前后**没有跳变**（35000:5.89、35750:5.87、38750:5.94）→ 飙升不是地形突变引起的。

---

## 2. 根因一：奖励设计把策略推向"少动"

### 2.1 摔倒完全没有惩罚（已证实，不是配置问题）

- `legged_robot.py:1426-1428`：`_prepare_reward_function` 里 **`if name == "termination": continue`**（注释 `##why treated differenlty?`）。
- `rewards/corl_rewards.py` 里**根本不存在 `_reward_termination`**（全文件 grep 无命中）。
- 配置里 `termination: 0`（`parameters.yaml:360`）——即使非零也不会进 reward。

→ **摔倒在奖励上零后果**。只有"提前终止"会导致地形降级，而提前终止的判据是
`use_terminal_roll_pitch=True, terminal_body_ori=1.396 rad (80°)` 或身体冲击——
**倾覆 40°–80° 的翻倒既不扣分也不降级**。我的评估里 76–80% 的回合最大 |roll/pitch| > 0.7 rad，
正是这个盲区。

### 2.2 惩罚项权重是正向奖励的 10 倍以上

`legged_robot.py:1419-1422`：所有 scale 乘 `dt=0.02` 后每步生效：

| 项 | 配置 scale | ×dt 后每步 | 触发条件 |
|---|---:|---:|---|
| tracking_lin_vel | +1.0 | **+0.02（满额）** | 速度误差 ≈ 0 |
| tracking_ang_vel | +0.8 | +0.016 | |
| **dof_pos_limits** | **−10** | −0.2/单位超限 | 任一关节越软限 0.1 rad 即 −0.02 |
| **torque_limits** | **−10** | −0.2/单位超限 | 力矩超 `soft_torque_limit` 部分 |
| collision | −1.0 | −0.02 | 非足端触地 |
| lin_vel_z | −2.0 | −0.04 | 上下乱动 |
| dof_pos / feet_slip / action_rate | −0.025 / −0.025 / −0.01 | 小 | |

→ 一个关节越界 0.1 rad 的惩罚 = **满额速度跟踪奖励的 10 倍**。
策略的最优解是"待在限位内、少动"，而不是"走快走稳"。这直接解释了实测的 8 秒 0.5–1.4 m
（指令 0.5 m/s 应走 4 m）。

### 2.3 课程升级判据与真实能力脱钩

`legged_robot.py:653-705`（`legacy_curriculum=False` 分支）：
- **升级**要求：平均指令模长 > 0.5 m/s **且** vx/vy 平均残差 < 0.1、yaw 残差 < 0.5、**且未提前终止**；
- **降级**：任一残差 > 0.5/0.5/1.0，或提前终止。

问题：
1. 判据只看**速度跟踪误差**，不看"是否爬上台阶"。地形是 34% 上楼 / 34% 下楼 / 8% 平滑平地 / 6% 粗糙平地 / 8% 坡道 / 10% 离散障碍，
   **在平地 tile 上达标同样升级**，把整体难度均值推高（1.6 → 5.9），但楼梯能力一直是 0%。
2. 升级阈值 0.1、降级阈值 0.5——**五倍不对称**形成棘轮：难度只升不降（除非提前终止）。
3. `fast_envs` 门槛是"平均指令 >0.5 m/s"，而指令范围是 `vx∈[-0.38, 0.58]`——
   只有抽到接近上限的指令才参与升级，升级样本少且集中在"好走的地形"。

---

## 3. 根因二：critic 失稳 + 梯度配比把 actor 拖垮

`ppo_cse/ppo.py:159`：
```python
loss = surrogate_loss + value_loss_coef * value_loss - entropy_coef * entropy
```
- `value_loss_coef = 1.0`，而 value loss 一度是 surrogate 的 **4–5 个数量级**（240 vs 0.045）。
- 两者**共享同一个 optimizer**（`self.optimizer` 覆盖整个 `actor_critic`）→ critic 的误差主导每一步梯度，
  **actor 被 critic 的噪声拖着走**。这正是"KL 只有 0.001 却完全不涨能力"的机制：
  actor 的有效更新被 critic 的大梯度淹没/扭曲。

**为什么 critic 会失稳**：
- 不是策略突变（KL 很小）、不是地形跳变（难度恒定）、不是非有限梯度（invalid=0）；
- 最可能是**回报目标漂移**（地形难度持续上升改变回报分布，`feet_air_time_rsl` 还乘了一个
  `reward_curriculum_factor`：0.001 → 1.0 的指数收敛，`legged_robot.py:142,1139`），
  加上 **return 未做归一化**、`schedule: fixed`（学习率不随 KL 自适应，`parameters.yaml:508`）。
- ⚠️ **无法进一步定位**：训练**不记录回报/奖励的时间序列**——`outputs.log` 只有 loss，
  `metrics.pkl` 只有首个窗口的快照，奖励曲线只发到离线的 web dashboard。
  **这是当前最大的可观测性缺口**，导致奖励侧的变化无法事后追查（35790 的确切触发点无法证实）。

**现有保护基本没起作用**（`ppo_cse/ppo.py` + `__init__.py` 的 stability guard）：
- `hard_kl_limit = 0.02`，但实际 KL 只有 0.001–0.004 → 从未触发（skip 0.1–0.3 / 20 minibatch）；
- `divergence_value_loss_threshold = 25`、`patience = 3`：**长跑时这套代码还不存在**
  （`.before_stability_guard` 备份为证，是 38750 之后才加的），所以 354 也没有停机；
  现在虽已生效，但当前重训里 33 个窗口 value loss > 5（含 51.8、34.6）仍未停。

---

## 4. 建议（按性价比排序）

1. **改奖励**（改动最小、见效最大）：
   - 让摔倒真正进 reward：在 `corl_rewards.py` 加 `_reward_termination`，并把 `_prepare_reward_function`
     里的 `continue` 去掉（或单独处理）；scale 建议 −1 ~ −2；
   - `dof_pos_limits` / `torque_limits` 从 −10 降到 −1 ~ −2，或只罚大幅超出（如 20% 裕量外）；
   - `tracking_lin_vel` 从 1.0 提到 2 ~ 3，让"走得快"的收益压过"少动"的收益。
2. **解耦 actor / critic**：`value_loss_coef` 降到 0.5，或给 critic 单独更小学习率 / 加 return 归一化
   （简单标准化或 PopArt），避免 critic 失稳拖垮 actor。
3. **升级判据改硬指标**：加"本回合净前进 ≥ X 且未提前终止"，并对**楼梯类 tile 单独设门槛**
   （例如在上/下楼 tile 上必须爬升 N 级才算通过），不要让平地达标带飞整体难度。
4. **把回报/奖励分量/终止率写进本地 CSV**（每 10 轮一行），否则奖励侧问题永远无法追查。
5. **收紧保护**：`divergence_value_loss_threshold` 降到 5、`patience` 降到 1；或改成"连续 N 次
   value loss 环比翻倍即停"。
6. **重训起点**：当前 checkpoint 已落在"保守不动"的局部最优，且奖励要改。
   建议改完奖励后**从 6000–20000 区间的健康档重训**（那时 value loss 稳定 0.008、难度仍在合理区间），
   或直接从头跑；不要在 38750 之后继续磨。

## 5. 本次诊断的证据文件

- 评估脚本与快照：`scripts/eval_stairs_cpu.py`、`eval_results/{cmp,flat}_*.json`（commit `a69347c`）
- 清理后 checkpoint：每个 run 保留首/尾/隔一档，`stairs_mybot_v3*` 与 `selected_*` 全保留
- 训练日志：`runs/*/*/outputs.log`（loss 时间序列）、`figures/terrain_curriculum_*.jpg`（难度分布）
