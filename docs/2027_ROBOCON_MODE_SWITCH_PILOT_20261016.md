# 2027 单木桥双模式仿真试验（2026-10-16）

## 结论

059950 负责普通行走；现有 060173 仅作试验桥专家。MuJoCo 按机器人基座位置在桥前 x≥−0.15 m 切入，在桥后 x≥3.30 m 切回；物理状态连续，切换时清空新策略历史并对关节目标做 0.3 s 过渡。完成点固定为 x≥3.45 m，确保切回行走后仍继续运动。这些数值是仿真场景的临时门槛，不是实机触发器。

20 个相同起点的配对试验，规则桥 0.40×3.00×0.10 m，seed 20261016：

| 策略 | 通过 | 成功回合 P95 | 最大中心线误差 |
| --- | ---: | ---: | ---: |
| 纯 059950 | 18/20 | 34.602 s | 0.1313 m |
| 059950→060173→059950 | 18/20 | 23.822 s | 0.1531 m |

双模式有 18/20 回合完成两次切换。第 2、17 号（零起算）在桥上出现两足或更多落到外侧地面，未到出口；纯 059950 通过了这两组。第 4、18 号则相反。当前观测是通过率持平、通行更快，样本量不足以证明成功率提升。宽桥课程的 Isaac 留出结果仍为 059950 197/200、060173 193/200，专家不得晋级实机。此前 12 cm 台阶退化发生在将 060173 用作通用策略时；只在桥上启用可隔离这类场景外退化，但不能免除桥上和切换安全的独立门槛。

结果文件：evaluations/robocon_2027/p2_20261016/beam_mode_switch_pilot20.json。单次视频：evaluations/robocon_2027/p2_20261016/beam_mode_switch_demo.mp4，旁边 JSON 记录权重 SHA-256、切换时间和判定。视频只是轨迹展示。

## 下一轮新模型

1. 先给桥训练器实现可核查的平地、转向、台阶回放，以及仅对桥环境生效的奖励和失败判据。将连续模式切换的对齐和出口窗口纳入评测，避免专家只会从固定初态上桥。
2. 从 059950 开独立桥专家 run，先在 0.80 m 宽桥做 25 iteration 烟雾训练，再做 200 iteration 短跑；每 100 iteration 检查桥与基础步态。先消融中心线和足端避边奖励，不接着用 060173 作为晋级起点。
3. 只有宽桥 Isaac/MuJoCo 和完整双切换同时过门槛后，按 0.80→0.60→0.40 m 收窄，最后规则桥 0.40×3.00×0.10 m。正式完整循环需至少 100 次中 95 次成功且切换致摔倒为 0；实机仍从吊架开始。

梅花桩是另一个专家模式，但现有仓库缺规则尺寸场景、逐块有效接触 bitset、落足观测可见性证明；先完成这三项再训练。该项目需要落足目标规划和策略跟踪，不能直接复用桥专家权重当已训练模型。

比赛配置中的 match.control_mode_switch_allowed: false 应与裁判确认其含义。当前试验在单一自主控制流程内切换内部策略，没有手动/自主控制权交接；本报告不把它视作裁判认可。

## 可重复命令

运行以下命令即可重新生成 20 组试验和视频；脚本默认桥专家路径指向 060173 的 JIT 文件：

    cd /home/ldl/mybot_v3_stair_turn_rl
    /home/ldl/anaconda3/envs/robodog_gym/bin/python scripts/eval_robocon_2027_mode_switch.py --trials 20 --seed 20261016 --output evaluations/robocon_2027/p2_20261016/beam_mode_switch_pilot20.json
    MUJOCO_GL=egl /home/ldl/anaconda3/envs/robodog_gym/bin/python scripts/render_robocon_2027_demo.py --obstacle beam --seed 20261016 --trial 0 --beam-expert-body runs/mybot_v3_beam_expert_resume_059974/2026-10-15_11-45-23.345648/checkpoints/body_latest.jit --beam-expert-adaptation runs/mybot_v3_beam_expert_resume_059974/2026-10-15_11-45-23.345648/checkpoints/adaptation_module_latest.jit --output evaluations/robocon_2027/p2_20261016/beam_mode_switch_demo.mp4
