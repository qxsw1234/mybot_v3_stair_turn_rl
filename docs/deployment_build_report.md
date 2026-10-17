# 2027 ROBOCON 部署构建与 fake 遥测报告

日期：2026-10-16。工程：`/home/ldl/mybot_v3_stair_turn_rl`。本报告只验证 desktop 的 CPU 推理和 fake motor，不涉及实机电机。

## 构建链与修复

原缓存把 ABI=0 的训练 Torch 与 ABI=1 的 ROS Foxy、yaml-cpp 和 Unitree SDK 混用；训练环境不做升级。本机已有 `lerobot-pi05-sim` 中的 PyTorch **2.10.0+cu128，C++11 ABI=true**。其 `libtorch_cpu.so` 和 `libc10.so` 可通过 GCC 9.4.0、系统 CMake 3.16.3 直接链接；最小 `torch::ones()` 程序编译运行输出 `1`。即使安装的 CUDA 工具链是 11.8，此 CPU 推理路径不要求 CMake 配置 CUDA 12.8。

`deploy_cpp/CMakeLists.txt` 新增可选 `ROBOCON_TORCH_CPU_ABI1_ROOT`。指定已核实为 ABI=1 的 Torch 包目录时，直接引用 CPU 库和头文件，`deploy_node` 使用 ABI=1；未指定时保留现有 `find_package(Torch)` 行为。这个选项只验证 desktop x86_64 CPU 推理；安装目录仍通过 RPATH 引用该 Conda 环境的 Torch 动态库，移到另一台机器必须另行打包和校验依赖。

干净构建、安装命令：

```bash
cd /home/ldl/mybot_v3_stair_turn_rl
source /opt/ros/foxy/setup.bash
cmake -S deploy_cpp -B /tmp/robocon_deploy_abi1_20261015 \
  -DCMAKE_BUILD_TYPE=Release \
  -DROBOCON_TORCH_CPU_ABI1_ROOT=/home/ldl/anaconda3/envs/lerobot-pi05-sim/lib/python3.10/site-packages/torch
cmake --build /tmp/robocon_deploy_abi1_20261015 -j4
cmake --install /tmp/robocon_deploy_abi1_20261015 \
  --prefix /tmp/robocon_deploy_abi1_install
```

结果：配置、两个可执行目标 `deploy_node`/`motor_debug_node` 的编译与链接、隔离安装均成功。安装版 `ldd` 能解析 Unitree SDK、`librclcpp.so`、`libtorch_cpu.so`，没有 `not found`。构建日志：desktop 的 `/tmp/robocon_deploy_abi1_{configure,build,install}.log`。之前的训练环境 ABI=0 构建目录未被覆盖。

## fake motor 运行与遥测

以下命令明确选择 fake 驱动；不打开串口或控制真实电机：

```bash
source /opt/ros/foxy/setup.bash
export ROS_DOMAIN_ID=97
export LD_LIBRARY_PATH=/home/ldl/mybot_v3_stair_turn_rl/deploy_cpp/lib:$LD_LIBRARY_PATH
# 启动节点时必须显式传入这两项：
/tmp/robocon_deploy_abi1_20261015/deploy_node --ros-args \
  -p robot_config_file:=/home/ldl/mybot_v3_stair_turn_rl/deploy_cpp/config/robots/mybot_v3_cse_sim.yaml \
  -p debug_no_motor:=true -p sim_mode:=false
```

短时测试：模型从 `deploy_cpp/policy/` 加载，estimator `1160→10`、body `1200→12` 形状检查成功；15 s fake 空闲运行后正常关停。用普通 stdin 管道发送 `1` 进入 fake StandUp 后，`scripts/check_deploy_telemetry.py` 连续 10 s 收到 `/joint_states` **501** 条、`/policy_joint_targets` **500** 条。两话题关节名顺序正确、位置均 12 维且有限、时间戳严格递增；源时间戳最大间隔分别 **20.081/20.079 ms**。fake 模式的 `effort` 为空，表示未知力矩。采集 JSON 和日志在 desktop `/tmp/robocon_deploy_fake_telemetry10.json`、`/tmp/robocon_deploy_abi1_fakepipe25.log`。

10 分钟稳定性测试：独立 ROS domain 98、显式 `debug_no_motor:=true`，经普通 stdin 管道进入 fake StandUp；节点运行约 615 s 后按 SIGINT 正常关停，日志有 `shutdown complete`。`scripts/check_deploy_telemetry.py --duration 600` 连续采样 600 s：`/joint_states` **30001** 条，`/policy_joint_targets` **30000** 条；两话题 12 维位置均有限、关节名正确、时间戳严格递增、无缺失话题。最大发布时间戳间隔分别 **20.953/20.929 ms**；订阅端本机墙钟计算的最大接收年龄分别 **119.703/119.801 ms**。结果为 desktop 的 `/tmp/robocon_deploy_fake_telemetry600.json`，节点和采集日志分别为 `/tmp/robocon_deploy_abi1_fake10min.log`、`/tmp/robocon_deploy_fake_telemetry600.log`。此测试只覆盖 fake 驱动在 StandUp 状态的话题发布和进程稳定性，未驱动真实电机。

## 尚未满足的部署门槛

- 这不是实机标定。`/joint_states` 的发布时刻不是电机反馈采样时刻，仍需在驱动接收处保留硬件时间戳、数据年龄与丢包/重复包标志；当前解析器只能计算发布话题的间隔和接收年龄。
- 10 分钟 fake 采集中，按订阅端本机墙钟计算的最大接收年龄约 **120 ms**。需检查 DDS 排队、回调调度和时钟来源；不能把它当作硬件采样年龄。
- MuJoCo topic motor 模式尚未完成 10 分钟闭环验证；安全吊架、低速和实机规则速度证据仍缺。`064341` 限高策略继续停在动力学辨识前置门槛，不因 C++ 链接成功而晋级。
