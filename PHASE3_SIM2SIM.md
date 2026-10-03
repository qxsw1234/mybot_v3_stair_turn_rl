# MyBot V3 sim2sim 鲁棒训练（第三阶段）

## 为什么 8 cm 能过、12 cm 失败

59000/59250 在 Isaac Gym 的 11.7 cm 楼梯上可以连续爬升约 0.85 m，但在 MuJoCo 的 12 cm 楼梯前会卡住并产生明显横移。模型文件与观测排列已核对一致，改变起步距离、楼梯宽度或 PD 增益不能恢复，因此按跨引擎动力学差距处理。

## 第三阶段改动

- 训练集中在约 9.0–14.5 cm 台阶，并保留上下楼、平地与转向样本。
- 每个关节独立随机化 Kp 0.75–1.35 倍、Kd 0.5–5.5 倍，覆盖 MuJoCo 中额外的被动阻尼。
- 启用真实地形相对足端高度，并加入 13 cm 摆动足净空目标。
- 楼梯样本固定直行指令 0.25–0.50 m/s，避免策略通过侧移逃避台阶。
- 每 250 轮在 MuJoCo 完整测试 8/10/12 cm；只有登上平台、未摔倒且横向未绕行才算成功。

## 当前训练

- 入口：`scripts/train_mybot_v3_sim2sim_phase3.py`
- 起点：`ac_weights_059000.pt`
- 目标：61000（实际最后一轮约 60999）
- 运行目录：`runs/mybot_v3_sim2sim_phase3_resume_059000/2026-10-03_02-38-13.891327`
- MuJoCo 验收：`evaluations/sim2sim_stairs/summary.csv`

## 手动复测

```bash
cd /home/ldl/mybot_v3_stair_turn_rl
/home/ldl/anaconda3/envs/robodog_gym/bin/python scripts/sim2sim_mujoco.py \
  --terrain-mode stairs --stairs-step-height 0.12 --vx 0.4
```

交互模式按 `W` 时使用 0.4 m/s，避免旧版硬编码的 0.6 m/s 超出高台阶专项速度范围。
