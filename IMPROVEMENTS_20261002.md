# Mybot V3 楼梯与转向训练改进（2026-10-02）

## 当前训练

- 来源：上一轮健康检查点 `ac_weights_040500.pt`
- 新运行：`runs/mybot_v3_stair_turn_improved_resume_040500/2026-10-02_03-23-39.173775`
- 目标：从全局 40500 轮继续训练到 60000 轮
- 并行环境：5120

## 本轮修改

1. 动作限幅从 ±20 改为 ±3；配合 `action_scale=0.25`，关节目标偏移限制在约 ±0.75 rad。
2. 65° roll/pitch 触发终止；摔倒实际事件惩罚为 -1。
3. 线速度与角速度跟踪权重分别改为 2.0 和 1.0；关节与力矩软限位惩罚从 -10 降为 -2。
4. 地形课程升级同时要求速度跟踪、净移动距离和楼梯方向的高度变化。
5. actor 与 critic 使用独立优化器、学习率和梯度裁剪；critic 使用 Huber 目标抵抗异常回报。
6. 本地日志新增 actor/critic 梯度、回报分布、动作饱和率、楼梯达标率和课程升级率。
7. 修正 CPU 楼梯评估中的四元数顺序、yaw 指令覆盖、重复回合计数和成功判据。

## 启动命令

```bash
./scripts/start_mybot_v3_training.sh \
  --num-envs 5120 \
  --iterations 19500 \
  --resume-run /home/ldl/mybot_v3_stair_turn_rl/runs/mybot_v3_stair_turn_velocity_curriculum_resume_038750/2026-10-02_01-48-17.945737 \
  --checkpoint 40500
```

## 启动后检查

- 40750 轮已生成新检查点。
- 40770 轮：总奖励约 40.35，楼梯达标率约 57.2%，地形等级约 1.93。
- value loss 在完整回合边界出现短暂尖峰后回落到 0.077；KL 约 0.0015。
- actor 与 critic 均无无效更新。
