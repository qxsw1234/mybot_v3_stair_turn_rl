# CSE Policy Sim2Sim & Sim2Real Deployment

## Background

Deploy the **exteroceptive CSE (Concurrent State Estimation)** policy trained via `train_exteroceptive_robust_icra_proposed.py` to:
1. **Sim2Sim**: MuJoCo simulation for validation
2. **Sim2Real**: Physical mybot_v2_1 robot

### Key Difference from Reference Code

The reference `code_reference/` deploys a simpler **HIM policy** (single 270-dim obs → 12 actions). The new CSE policy has:
- **Two separate JIT models**: `adaptation_module_latest.jit` + `body_latest.jit`
- **Exteroceptive observations** including 77-dim height measurements
- **10-frame observation history** (vs 6 in reference)
- Much larger observation space (1190 policy obs, 1160 estimator obs)

### Observation Dimension Analysis

| Component | Policy Obs | Estimator Obs | Notes |
|-----------|-----------|---------------|-------|
| commands × scale | 3 ✅ | 3 ❌ | `[vx*2.0, vy*2.0, yaw*0.25]` |
| projected_gravity | 3 ✅ | 3 ✅ | `quat_rotate_inverse(base_quat, gravity_vec)` |
| dof_positions | 12 ✅ | 12 ✅ | `(dof_pos - default_dof_pos) * 1.0` |
| dof_velocities | 12 ✅ | 12 ✅ | `dof_vel * 0.05` |
| last_actions | 12 ✅ | 12 ✅ | Raw actions from previous step |
| height_measurements | 77 ✅ | 77 ✅ | 11×7 grid, `clip(h - base_h + 0.3, -1, 1) * 5.0` |
| **Single-step total** | **119** | **116** | |
| **×10 history** | **1190** | **1160** | Dense history, 10 frames |

**Privileged obs (latent output)**: friction(1) + restitution(1) + body_velocity(3) + contact_states(4) + height_z_bias(1) = **10**

**Inference flow**:
```
estimator_obs_history (1160) → adaptation_module → latent (10)
cat(policy_obs_history (1190), latent (10)) = 1200 → actor_body → actions (12)
target_dof_pos = policy_dof_pos + actions * 0.25
```

### Height Measurement Grid
```
measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]  # 11 pts
measured_points_y = [-0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3]                           # 7 pts
```

---

## User Review Required

> [!NOTE]
> **Height measurements**: Via ROS2 topic subscription (`/height_measurements`). The height sensor node publishes `Float32MultiArray` (77 floats). This keeps the interface flexible for LiDAR, depth camera, or MuJoCo raycasting.

> [!NOTE]
> **Deployment language**: **C++** (LibTorch JIT), extending `code_reference/`. Python tools retained only for model export and MuJoCo sim2sim prototyping.

> [!WARNING]
> **Motor PD gains**: The training uses per-joint-type stiffness (hip=40, thigh=40, calf=80) and damping (hip=1, thigh=1, calf=2). The reference config uses uniform kp=60, kd=1. The new config should match training. Please confirm which gains to use on the real robot.

---

## Proposed Changes (C++ Deployment)

### Architecture: Extend `code_reference/` for CSE

The existing C++ deployment in `code_reference/` handles a simpler HIM policy (single model, 270-dim obs).
We modify it **in-place** to support the CSE two-model architecture with larger observation spaces.

---

### C++ Header Changes

#### [MODIFY] [robot_config.h](file:///home/getting/humble/Quadruped/elmap-rl-controller/deploy/code_reference/include/robot_config.h)
- Add CSE-specific dimensions as a second set of constants
- `NUM_CSE_POLICY_OBS_PER_STEP = 119`, `NUM_CSE_ESTIMATOR_OBS_PER_STEP = 116`
- `NUM_CSE_HISTORY = 10`, `NUM_HEIGHT_POINTS = 77`
- `NUM_CSE_PRIVILEGED_OBS = 10`

#### [NEW] [cse_policy_runner.h](file:///home/getting/humble/Quadruped/elmap-rl-controller/deploy/code_reference/include/cse_policy_runner.h)
- Loads two JIT models: `adaptation_module.jit` + `body.jit`
- Manages observation history (10-frame sliding window) as torch tensors
- `compute_policy_obs()` / `compute_estimator_obs()` — builds 119/116-dim vectors
- `step()` — full inference: obs → history → adapt → body → target_dof_pos

#### [NEW] [height_subscriber.h](file:///home/getting/humble/Quadruped/elmap-rl-controller/deploy/code_reference/include/height_subscriber.h)
- ROS2 subscriber for `/height_measurements` topic (Float32MultiArray, 77 floats)
- Thread-safe cached height data
- `get_measurements()` returns latest height array
- Flat-terrain fallback when no messages received

#### [MODIFY] [robot_runtime_config.h](file:///home/getting/humble/Quadruped/elmap-rl-controller/deploy/code_reference/include/robot_runtime_config.h)
- Add CSE-specific fields: obs_scales, obs_bias, height grid, action_scale, etc.
- Add `adaptation_module_path`, `body_path`
- Add `height_topic` field

---

### C++ Source Changes

#### [NEW] [cse_policy_runner.cpp](file:///home/getting/humble/Quadruped/elmap-rl-controller/deploy/code_reference/src/cse_policy_runner.cpp)
- CSE policy inference implementation
- Observation construction matching training pipeline exactly
- History management with sliding window

#### [NEW] [height_subscriber.cpp](file:///home/getting/humble/Quadruped/elmap-rl-controller/deploy/code_reference/src/height_subscriber.cpp)
- ROS2 height measurement subscriber implementation

#### [MODIFY] [deploy_node.cpp](file:///home/getting/humble/Quadruped/elmap-rl-controller/deploy/code_reference/src/deploy_node.cpp)
- Add `cse_mode` parameter to select CSE policy runner
- Wire up height subscriber
- Use CSE policy runner in `handle_rl()` when CSE mode is active

#### [MODIFY] [robot_runtime_config.cpp](file:///home/getting/humble/Quadruped/elmap-rl-controller/deploy/code_reference/src/robot_runtime_config.cpp)
- Parse new CSE-specific YAML fields

---

### Config

#### [NEW] [mybot_v2_1_cse.yaml](file:///home/getting/humble/Quadruped/elmap-rl-controller/deploy/code_reference/config/robots/mybot_v2_1_cse.yaml)
Extends reference YAML with CSE-specific parameters (obs scales, model paths, height topic, etc.)

---

### Python Tools (retained)

#### [KEEP] [export_policy.py](file:///home/getting/humble/Quadruped/elmap-rl-controller/deploy/scripts/export_policy.py)
- Export training checkpoint → JIT models

#### [KEEP] Python sim2sim (`deploy/src/`, `deploy/scripts/`)
- Standalone MuJoCo sim2sim for quick prototyping (no ROS2 needed)

#### [MODIFY] [mujoco_sim_node.py](file:///home/getting/humble/Quadruped/elmap-rl-controller/deploy/code_reference/sim/mujoco_sim_node.py)
- Add height measurement raycasting and publish to `/height_measurements` topic

---

## Motor Mapping (from user specification)

```
Policy DOF order:  FL(0-2), FR(3-5), RL(6-8), RR(9-11)
Motor CAN IDs:     FR(1-3)→port0, FL(4-6)→port0, RL(7-9)→port1, RR(10-12)→port1

joint_mapping:      [4, 5, 6,  1, 2, 3,  7, 8, 9,  10, 11, 12]
motor_is_reversed:  [F, F, T,  F, T, F,  T, F, T,  T,  T,  F]
transmission_ratio: [6.33,6.33,14.77, 6.33,6.33,14.77, 6.33,6.33,14.77, 6.33,6.33,14.77]
```

---

## Open Questions

> [!IMPORTANT]
> 1. **Height sensor for real robot**: Do you have a depth camera/LiDAR? If not, initial testing will use flat terrain assumption.
> 2. **PD gains for real robot**: Use training gains (hip kp=40, calf kp=80) or reference gains (uniform kp=60)?
> 3. **IMU topic**: Still using `/fast_livo2/state6_imu_prop`?
> 4. **Serial ports**: Still `/dev/ttyUSB0` and `/dev/ttyUSB1`?

---

## Verification Plan

### Automated Tests
1. **Observation dimension test**: Build observations from dummy data, verify shapes match training (119, 116, 1190, 1160)
2. **Policy export test**: Export model, load in deployment, verify output shape (12 actions)
3. **Motor mapping test**: Verify joint_mapping bijection and direction consistency

### Sim2Sim Validation
1. Run in MuJoCo with `run_sim2sim.py`
2. Verify robot stands up cleanly from default pose
3. Send velocity commands and verify tracking behavior
4. Compare qualitatively with IsaacGym training behavior

### Sim2Real Validation
1. Test motor communication with zero-torque mode
2. Test stand-up sequence with low gains
3. Test RL policy with small velocity commands
4. Gradually increase command range




@beautifulMention @beautifulMention@beautifulMention@beautifulMention@beautifulMention @beautifulMention @beautifulMention @beautifulMention@beautifulMention@beautifulMention@beautifulMention@beautifulMention 我要针对这个强化学习算法写一个sim2sim以及sim2real的部署代码，首先你要仔细阅读所有相关的代码文件，确定其输入输出维度，我的sim2sim要在mujoco上，然后我的sim2real可以参考我给你的代码@beautifulMention  你参考里面的实现，我所使用的机器人和这个reference中的mybot一致，然后我要求你实现时要考虑架构的合理性，便于调参数调试等，电机底层驱动在lib中@beautifulMention @beautifulMention ，实现在@beautifulMention @beautifulMention @beautifulMention ，然后你的配置文件要合理，注释清晰，我的urdf的关节顺序与我实际机器人的电机id不对应：
#  ┌────────────────────────────────────────────────────────────┐
#  │         关节-电机映射说明 (Joint-Motor Mapping)              │
#  ├────────────────────────────────────────────────────────────┤
#  │                                                            │
#  │  所有数组统一按 Policy DOF 顺序索引:                          │
#  │    DOF  0- 2: FL (Front-Left)  hip, thigh, calf            │
#  │    DOF  3- 5: FR (Front-Right) hip, thigh, calf            │
#  │    DOF  6- 8: RL (Rear-Left)   hip, thigh, calf            │
#  │    DOF  9-11: RR (Rear-Right)  hip, thigh, calf            │
#  │                                                            │
#  │  实机电机物理编组 (按 CAN ID):                                │
#  │    电机  1- 3: FR hip, thigh, calf → 串口 port0            │
#  │    电机  4- 6: FL hip, thigh, calf → 串口 port0            │
#  │    电机  7- 9: RL hip, thigh, calf → 串口 port1            │
#  │    电机 10-12: RR hip, thigh, calf → 串口 port1            │
#  │                                                            │
#  │  joint_mapping[policy_dof] = 电机 CAN ID:                  │
#  │    只需按 policy 顺序填写每个关节对应的电机 ID 即可            │
#  │    motor_is_reversed 也按相同的 policy 顺序填写               │
#  │    其余属性 (default_dof_pos 等) 全部按 policy 顺序       

电机传动比注意：joint_transmission_ratio: [6.33, 6.33, 14.77, 6.33, 6.33, 14.77, 6.33, 6.33, 14.77, 6.33, 6.33, 14.77]
你的配置文件参考我的@beautifulMention 实现，但是其中现在有一些重复的参数，比较冗余。你实现的时候记得避免
@beautifulMention 请你按照这个这个计划进行执行，然后我要使用cpp来完成，你在deploy_cpp文件中实现这一功能，高程点通过一个ros2话题的订阅来实现获取