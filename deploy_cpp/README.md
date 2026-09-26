# mybot_v2_1 CSE Sim2Sim / Sim2Real 部署说明

这个 ROS2/C++ 包用于把本仓库训练得到的 **exteroceptive CSE 强化学习策略** 部署到：

- **Sim2Sim**：MuJoCo 仿真。MuJoCo 物理仿真由 Python bridge 运行，C++ 节点负责策略推理和控制。
- **Sim2Real**：真实 mybot_v2_1 机器人。C++ 节点订阅 IMU 和高程信息，通过 Unitree GO-M8010-6 SDK 控制 12 个电机。

新的部署代码位于 `deploy_cpp` 根目录。`deploy_cpp/reference` 是之前 HIMLoco 的参考实现，只作为对照保留。

## 1. 部署策略结构

本部署使用两个 TorchScript JIT 模型：

```text
adaptation_module_latest.jit  : estimator history 1160 -> latent 10
body_latest.jit               : policy history 1190 + latent 10 -> actions 12
```

观测维度和训练代码保持一致：

```text
Policy 单帧观测:
  commands(3)
  projected_gravity(3)
  dof_positions(12)
  dof_velocities(12)
  last_actions(12)
  height_measurements(77)
  = 119

Estimator 单帧观测:
  projected_gravity(3)
  dof_positions(12)
  dof_velocities(12)
  last_actions(12)
  height_measurements(77)
  = 116

History:
  10 帧，按 oldest -> newest 展平
```

动作到关节目标角的转换：

```text
target_dof_pos = policy_dof_pos + action * 0.25
```

发送到电机或 MuJoCo 前，会根据 YAML 中的 `joint_pos_lower` / `joint_pos_upper` 对目标角做 clamp。

## 2. 目录结构

```text
deploy_cpp/
  CMakeLists.txt
  package.xml

  config/
    robots/mybot_v2_1_cse.yaml      主配置文件

  include/
    cse_policy_runner.h             CSE 双 JIT 推理
    height_subscriber.h             /height_measurements 订阅
    imu_subscriber.h                /fast_livo2/state6_imu_prop 订阅
    motor_driver.h                  实机电机驱动封装
    state_machine.h                 状态机
    keyboard_controller.h           键盘控制

  src/
    deploy_node.cpp                 主部署节点
    cse_policy_runner.cpp
    height_subscriber.cpp
    imu_subscriber.cpp
    motor_driver.cpp
    motor_debug_node.cpp

  sim/
    mujoco_sim_node.py              Python MuJoCo bridge

  scripts/
    check_motor_direction_rviz.sh   实机零阻尼电机方向和 RViz 一致性检查

  policy/
    adaptation_module_latest.jit     放置导出的 estimator JIT
    body_latest.jit                  放置导出的 actor body JIT

  robot/
    mybot_v2_1/                     URDF、MuJoCo XML、mesh

  launch/
    deploy.launch.py                实机 / fake motor 部署
    sim.launch.py                   MuJoCo topic 模式部署
    motor_debug.launch.py           电机零力矩读角 / RViz 方向调试
    visualize.launch.py             RViz 可视化
```

## 3. 模型文件放置

把训练导出的两个 JIT 文件放到：

```text
deploy_cpp/policy/adaptation_module_latest.jit
deploy_cpp/policy/body_latest.jit
```

默认配置文件中对应字段是：

```yaml
adaptation_module_path: policy/adaptation_module_latest.jit
body_path: policy/body_latest.jit
```

如果模型放在其他地方，可以直接在 `config/robots/mybot_v2_1_cse.yaml` 中改成绝对路径。

## 4. 主配置文件

主配置文件：

```text
deploy_cpp/config/robots/mybot_v2_1_cse.yaml
```

关键字段：

```yaml
dt: 0.002
decimation: 10
control_dt: 0.02

kp_joint: [40,40,80, 40,40,80, 40,40,80, 40,40,80]
kd_joint: [1,1,2, 1,1,2, 1,1,2, 1,1,2]
action_scale: 0.25

imu_topic: /fast_livo2/state6_imu_prop
height_topic: /height_measurements

port0: /dev/ttyUSB0
port1: /dev/ttyUSB1
```

所有 12 维关节数组都统一按 **Policy DOF 顺序**：

```text
DOF 0-2:   FL hip, thigh, calf
DOF 3-5:   FR hip, thigh, calf
DOF 6-8:   RL hip, thigh, calf
DOF 9-11:  RR hip, thigh, calf
```

实机电机映射：

```yaml
joint_mapping: [4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 11, 12]
motor_is_reversed: [false, false, true, false, true, false, true, false, true, true, true, false]
joint_transmission_ratio: [6.33, 6.33, 14.77, 6.33, 6.33, 14.77, 6.33, 6.33, 14.77, 6.33, 6.33, 14.77]
```

默认 PD 增益使用训练配置：

```text
hip/thigh: kp=40, kd=1
calf:      kp=80, kd=2
```

## 5. 高程观测说明

C++ 节点订阅：

```text
/height_measurements
std_msgs/msg/Float32MultiArray
长度: 77
layout.dim[0].label = "x", size = 11
layout.dim[1].label = "y", size = 7
```

策略内部使用的是正的“地面到机体高度距离”：

```text
distance = base_z - ground_z
```

如果高度 topic 发布的不是这个约定，可以在 YAML 中转换：

```yaml
height_measurement_scale: 1.0
height_measurement_offset: 0.0
```

实际进入策略的值是：

```text
distance = raw_topic_value * height_measurement_scale + height_measurement_offset
```

例如实机高度源发布 `ground_z - base_z`，平地约 `-0.34m`，则应设置：

```yaml
height_measurement_scale: -1.0
height_measurement_offset: 0.0
```

部署端会把它转换成训练时的 height observation：

```text
height_obs = clip(0.3 - distance, -1, 1) * 5.0
```

这里的 `0.3` 是训练配置中的 `obs_bias.height_measurements`，部署时不随实时机体高度动态变化。

例如平地上机体高度约 `0.34m`：

```text
distance = 0.34
height_obs = (0.3 - 0.34) * 5 = -0.2
```

如果 `/height_measurements` 还没有收到数据，部署端会使用 YAML 中的：

```yaml
nominal_base_height: 0.34
```

作为 flat fallback。

## 6. 编译方法

本仓库根目录同时也是 Python package，因此从仓库根目录直接 `colcon build` 不会自动发现 `deploy_cpp`。需要加 `--base-paths deploy_cpp`。

推荐从仓库根目录构建：

```bash
cd /home/getting/humble/Quadruped/elmap-rl-controller
source /opt/ros/humble/setup.bash
colcon build --base-paths deploy_cpp --packages-select deploy_cpp \
  --cmake-args -DTorch_DIR=/opt/libtorch/share/cmake/Torch
source install/setup.bash
```

也可以进入 `deploy_cpp` 目录构建：

```bash
cd /home/getting/humble/Quadruped/elmap-rl-controller/deploy_cpp
source /opt/ros/humble/setup.bash
colcon build --packages-select deploy_cpp \
  --cmake-args -DTorch_DIR=/opt/libtorch/share/cmake/Torch
source install/setup.bash
```

如果你的 LibTorch 不在 `/opt/libtorch`，可以查找 `TorchConfig.cmake`：

```bash
find /opt /home/getting -path '*TorchConfig.cmake' 2>/dev/null
```

然后把 `Torch_DIR` 改成对应目录，例如：

```bash
--cmake-args -DTorch_DIR=/path/to/torch/share/cmake/Torch
```

## 7. Sim2Sim: MuJoCo 仿真

MuJoCo 物理仿真由 Python 节点运行；C++ `deploy_node` 负责订阅仿真状态、运行 CSE 策略、发布关节目标。

### 终端 1: 启动 MuJoCo bridge

```bash
cd /home/getting/humble/Quadruped/elmap-rl-controller/deploy_cpp
conda activate mujoco_sim
source /opt/ros/humble/setup.bash
python3 sim/mujoco_sim_node.py --robot-config config/robots/mybot_v2_1_cse.yaml
```

### 终端 2: 启动 C++ 控制节点

```bash
cd /home/getting/humble/Quadruped/elmap-rl-controller
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run deploy_cpp deploy_node --ros-args \
  -p robot_config_file:=/home/getting/humble/Quadruped/elmap-rl-controller/deploy_cpp/config/robots/mybot_v2_1_cse.yaml \
  -p sim_mode:=true \
  -p debug_no_motor:=false \
  -p sim_pingpong_mode:=false

```

若使用手柄进行控制，使用时开一个终端跑：

```bash
ros2 run joy joy_node
```

如果想使用“仿真状态到来后再计算下一步控制”的 ping-pong 模式：

```bash
ros2 launch deploy_cpp sim.launch.py sim_pingpong_mode:=true
```

Sim2Sim 相关 topic：

```text
/mujoco/joint_cmd             deploy_node -> MuJoCo
/mujoco/joint_state           MuJoCo -> deploy_node
/fast_livo2/state6_imu_prop   MuJoCo -> deploy_node
/height_measurements          MuJoCo -> deploy_node
/joint_states                 RViz 可视化
```

## 8. Sim2Real: 实机部署

实机运行前，建议按下面顺序验证。

### 8.1 电机方向 / RViz 一致性检查

用于确认串口、电机通信、编码器读数，以及代码中的电机方向是否和实机一致。

这个测试会对所有电机循环发送：

```text
q=0, dq=0, kp=0, kd=0, tau=0
```

也就是电机不抱死、阻尼为 0。请先把机器人托起或可靠支撑，避免腿部掉落受力。

推荐使用脚本启动：

```bash
cd /home/getting/humble/Quadruped/elmap-rl-controller
./deploy_cpp/scripts/check_motor_direction_rviz.sh
```

如果串口不同：

```bash
./deploy_cpp/scripts/check_motor_direction_rviz.sh \
  --port0 /dev/ttyUSB0 \
  --port1 /dev/ttyUSB1
```

脚本会自动尝试 source `/opt/ros/humble/setup.bash` 和当前工作区的 `install/setup.bash`，然后打开 RViz。确认安全后输入 `YES` 才会真正连接电机；如果已经确认环境安全，可以加 `--yes` 跳过提示。

等价的 ROS2 launch 命令是：

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch deploy_cpp motor_debug.launch.py
```

检查方法：

```text
1. RViz 打开后，轻轻转动某一个实机关节。
2. 看 RViz 中同名关节是否跟着动，并且方向是否一致。
3. 同时看终端表格里该 joint 的 Angle(rad/deg) 是否变化合理。
4. 每次只动一个关节，按 FL/FR/RL/RR 的 hip、thigh、calf 逐个检查。
```

判断和修改：

```text
实机转动 A 关节，但 RViz 动的是 B 关节:
  修改 YAML 中 A 对应 DOF 的 joint_mapping；如果串口分组也不一致，再显式加 motor_port_idx。

实机和 RViz 是同一个关节，但方向相反:
  修改 YAML 中该 DOF 对应的 motor_is_reversed，true/false 对调。

RViz 方向一致，但角度比例明显不对:
  检查 joint_transmission_ratio。
```

默认配置文件是：

```text
deploy_cpp/config/robots/mybot_v2_1_cse.yaml
```

修改 YAML 后重新启动这个脚本即可，不需要重新编译。

### 8.2 fake motor 干跑

不会控制真实电机，只验证节点、模型加载、键盘状态机等逻辑：

```bash
ros2 launch deploy_cpp deploy.launch.py debug_no_motor:=true
```

### 8.3 实机运行

先确保 IMU 和高程 topic 已经有数据：

```bash
ros2 topic echo /fast_livo2/state6_imu_prop
ros2 topic echo /height_measurements
```

启动部署节点：

```bash
ros2 launch deploy_cpp deploy.launch.py
```

如果需要手动指定参数：

```bash
ros2 run deploy_cpp deploy_node --ros-args \
  -p robot_config_file:=/home/getting/humble/Quadruped/elmap-rl-controller/deploy_cpp/config/robots/mybot_v2_1_cse.yaml \
  -p sim_mode:=false \
  -p debug_no_motor:=false \
  -p sim_pingpong_mode:=false
```

推荐第一次实机操作流程：

```text
1. 按 1: StandUp
2. 按 4: ReturnDefault，确认能平滑收回到上电 default 姿态
3. 按 6: JointSweep，逐关节确认电机 ID、方向、限位
4. 按 1: 回到 StandUp
5. 按 5: SingleStepRL，先看单步策略输出，再按 Enter 或手柄 Confirm 执行
6. 确认安全后按 2: RL 连续控制
```

## 9. 键盘控制

```text
0      Idle / 零力矩
1      StandUp / 站立
2      RL 策略控制
3      Joint damping / 关节阻尼
4      ReturnDefault / 回到上电 default 姿态
5      SingleStepRL / 单步策略调试
6      JointSweep / 单关节方向调试
Space  急停，强制回 Idle
Esc    退出程序

W/S    vx 增加 / 减小
Q/E    vy 增加 / 减小
A/D    yaw 增加 / 减小
R      速度命令清零

JointSweep 模式:
J/K    下一个 / 上一个关节
+/-    调整目标偏移
Enter  发送当前目标

SingleStepRL 模式:
Enter  执行当前 pending 的策略目标
手柄 Confirm 也可执行当前 pending 的策略目标
```

`ReturnDefault` 的行为就是 `StandUp` 的反向过程：从当前姿态按 `standup_duration` 插值回到 YAML 中的 `default_dof_pos`。

## 10. 手柄控制

如果你已经能通过下面命令看到手柄数据：

```bash
ros2 run joy joy_node
```

那么 `deploy_node` 会默认订阅 `/joy`，并把手柄作为一套额外输入源：

- 左摇杆 / 右摇杆轴控制 `vx / vy / yaw`
- 按钮可以切换状态
- 在 `SingleStepRL` 和 `JointSweep` 模式下，手柄也可以发送“确认执行”

默认输入优先级：

```text
/joy（若最近 0.5s 内有消息） > UDP teleop > 键盘
```

也就是说，只要手柄还在持续发消息，状态栏里会显示 `input=JOY`，此时速度命令以手柄为准；手柄超时后会自动退回到 UDP 或键盘。

### 10.1 默认手柄映射

默认 YAML 映射如下：

```yaml
joy_axis_vx: 1
joy_axis_vy: 0
joy_axis_yaw: 3
joy_invert_vx: true

joy_button_stand_up: 0
joy_button_return_default: 1
joy_button_rl: 2
joy_button_damping: 3
joy_button_single_step: 4
joy_button_joint_sweep: 5
joy_button_idle: 6
joy_button_confirm: 7
joy_button_emergency: 8
```

这套编号更接近常见 Xbox 风格控制器，但不同手柄的按钮编号不一定一致。最稳妥的做法是先运行：

```bash
ros2 topic echo /joy
```

按一下按钮，看看 `buttons[]` 里哪个索引从 `0` 变成 `1`，再去 YAML 改对应字段。

默认状态按钮含义：

```text
0  -> StandUp
1  -> ReturnDefault
2  -> RL
3  -> JointDamping
4  -> SingleStepRL
5  -> JointSweep
6  -> Idle
7  -> Confirm
8  -> Emergency -> Idle
```

### 10.2 手柄启动方法

先开 joy 节点：

```bash
source /opt/ros/humble/setup.bash
ros2 run joy joy_node
```

再启动部署：

```bash
source install/setup.bash
ros2 launch deploy_cpp deploy.launch.py
```

如果想禁用手柄，可以在 YAML 中改：

```yaml
joy_enable: false
```

### 10.3 手柄调参字段

以下字段都可以在 `config/robots/mybot_v2_1_cse.yaml` 中修改：

```yaml
joy_enable: true
joy_topic: /joy
joy_axis_deadzone: 0.15
joy_timeout_s: 0.5
joy_axis_vx: 1
joy_axis_vy: 0
joy_axis_yaw: 3
joy_invert_vx: true
joy_invert_vy: false
joy_invert_yaw: false
```

- `joy_axis_deadzone`：摇杆死区，小于这个值按 0 处理
- `joy_timeout_s`：超过这个时间没收到 `/joy`，就不再使用手柄命令
- `joy_invert_*`：如果方向反了，直接改这里
- `joy_axis_*` / `joy_button_*`：不同手柄最常调整的就是这些索引

## 11. 主要节点参数

### deploy_node

```text
robot_config_file   YAML 配置文件路径
debug_no_motor      true 时使用 fake motor，不控制真实电机
sim_mode            true 时使用 /mujoco/joint_cmd 和 /mujoco/joint_state
sim_pingpong_mode   true 时等待新的仿真状态再算下一步
```

### motor_debug_node

只发送零力矩并读取电机反馈。用于确认：

```text
电机是否在线
串口是否正确
编码器读数是否合理
motor_is_reversed 是否需要调整
joint_mapping 是否正确
```

## 12. 常见问题

### 11.1 找不到 Torch

错误类似：

```text
Could not find TorchConfig.cmake
```

解决方式：

```bash
colcon build --base-paths deploy_cpp --packages-select deploy_cpp \
  --cmake-args -DTorch_DIR=/opt/libtorch/share/cmake/Torch
```

如果 `/opt/libtorch` 不存在，用：

```bash
find /opt /home/getting -path '*TorchConfig.cmake' 2>/dev/null
```

找到本机可用的 `TorchConfig.cmake`。

### 11.2 从仓库根目录找不到 deploy_cpp 包

使用：

```bash
colcon list --base-paths deploy_cpp
colcon build --base-paths deploy_cpp --packages-select deploy_cpp
```

### 11.3 找不到 JIT 模型

确认文件存在：

```bash
ls deploy_cpp/policy/adaptation_module_latest.jit
ls deploy_cpp/policy/body_latest.jit
```

或者在 YAML 里改成绝对路径。

### 11.4 高程符号错误

部署端期望：

```text
distance = base_z - ground_z
```

平地、机体高度约 `0.34m` 时，topic 值应该接近 `0.34`。

如果你的发布端输出的是：

```text
ground_z - base_z
```

那么符号反了，策略看到的 height observation 会错。

### 11.5 IMU 一直 WAIT

节点状态栏中 `imu=WAIT` 表示没有收到：

```text
/fast_livo2/state6_imu_prop
```

消息布局必须是：

```text
[wx, wy, wz, projected_gravity_x, projected_gravity_y, projected_gravity_z]
```

### 11.6 电机方向不对

先进入 `JointSweep`：

```text
按 6
J/K 选择关节
+/- 调目标偏移
Enter 发送
```

如果关节正方向和预期相反，修改 YAML 中该 DOF 对应的：

```yaml
motor_is_reversed
```

### 12.7 手柄没反应

先确认 `/joy` 正常：

```bash
ros2 run joy joy_node
ros2 topic echo /joy
```

然后看部署节点状态栏是否显示：

```text
input=JOY
```

如果一直是 `KEY`：

1. 检查 YAML 里的 `joy_topic` 是否就是 `/joy`
2. 检查 `joy_enable` 是否为 `true`
3. 检查 `joy_timeout_s` 是否太小
4. 检查按钮/轴索引是否和你的手柄一致

### 11.7 串口顺序

默认：

```yaml
port0: /dev/ttyUSB0
port1: /dev/ttyUSB1
```

默认规则：

```text
CAN ID 1-6  -> port0
CAN ID 7-12 -> port1
```

如果你的硬件接线不同，可以在 YAML 里显式添加 `motor_port_idx`，按 policy DOF 顺序指定每个电机使用哪个串口。

### 11.8 RViz 只显示骨架，不显示模型

如果 RViz 左侧 `RobotModel -> URDF` 显示：

```text
Errors loading geometries
```

但各个 link 都是 `Transform OK`，说明 `/joint_states` 和 TF 正常，问题通常是 URDF 中 mesh 路径没有被 RViz 找到。`mybot_v2_1.urdf` 的 mesh 路径应使用：

```text
package://deploy_cpp/robot/mybot_v2_1/meshes/xxx.STL
```

改完 URDF 后需要重新安装资源：

```bash
colcon build --base-paths deploy_cpp --packages-select deploy_cpp \
  --cmake-args -DTorch_DIR=/opt/libtorch/share/cmake/Torch
source install/setup.bash
```

## 13. 验证命令

Python MuJoCo bridge 语法检查：

```bash
python3 -m py_compile deploy_cpp/sim/mujoco_sim_node.py
```

C++ 编译：

```bash
colcon build --base-paths deploy_cpp --packages-select deploy_cpp \
  --cmake-args -DTorch_DIR=/opt/libtorch/share/cmake/Torch
```

期望结果：

```text
Summary: 1 package finished
```

构建过程中 LibTorch 可能输出 CUDA/NVTX 相关 CMake warning。只要最终 build 成功，这些 warning 不是本部署源码错误。
