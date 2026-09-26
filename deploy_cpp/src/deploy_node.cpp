/**
 * @file deploy_node.cpp
 * @brief CSE policy deployment node for sim2sim and sim2real.
 */

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <thread>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

#include "cse_policy_runner.h"
#include "height_subscriber.h"
#include "imu_subscriber.h"
#include "keyboard_controller.h"
#include "motor_driver.h"
#include "robot_runtime_config.h"
#include "robot_visualizer.h"
#include "state_machine.h"
#include "udp_controller.h"

namespace deploy {

namespace fs = std::filesystem;

namespace {

std::string resolve_package_path(const fs::path &package_root,
                                 const std::string &path) {
  if (path.empty() || fs::path(path).is_absolute()) {
    return path;
  }
  return (package_root / path).lexically_normal().string();
}

fs::path infer_package_root_from_config(const std::string &config_file) {
  fs::path p(config_file);
  if (p.has_parent_path() && p.parent_path().filename() == "robots") {
    return p.parent_path().parent_path().parent_path();
  }
  if (p.has_parent_path()) {
    return p.parent_path();
  }
  return fs::current_path();
}

} // namespace

class FakeMotorDriver {
public:
  explicit FakeMotorDriver(const RobotRuntimeConfig &config) : config_(config) {
    dof_pos_ = config_.default_dof_pos;
    dof_vel_.fill(0.0f);
  }

  void send_commands(const std::array<float, NUM_JOINTS> &target_dof_pos,
                     const std::array<float, NUM_JOINTS> &kp,
                     const std::array<float, NUM_JOINTS> &kd) {
    const float dt = config_.control_dt;
    for (int i = 0; i < NUM_JOINTS; ++i) {
      const float acc =
          std::clamp(kp[i] * (target_dof_pos[i] - dof_pos_[i]) -
                         kd[i] * dof_vel_[i],
                     -20.0f, 20.0f);
      dof_vel_[i] = (dof_vel_[i] + acc * dt) * 0.98f;
      dof_pos_[i] = std::clamp(dof_pos_[i] + dof_vel_[i] * dt,
                               config_.joint_pos_lower[i],
                               config_.joint_pos_upper[i]);
    }
  }

  void send_damping(float kd) {
    const float damping = std::clamp(kd * 0.6f, 0.0f, 0.9f);
    for (int i = 0; i < NUM_JOINTS; ++i) {
      dof_vel_[i] *= (1.0f - damping);
    }
  }

  void set_zero_torque() { send_damping(0.01f); }
  bool check_errors() const { return false; }
  const std::array<float, NUM_JOINTS> &dof_pos() const { return dof_pos_; }
  const std::array<float, NUM_JOINTS> &dof_vel() const { return dof_vel_; }

private:
  RobotRuntimeConfig config_;
  std::array<float, NUM_JOINTS> dof_pos_{};
  std::array<float, NUM_JOINTS> dof_vel_{};
};

class MujocoMotorDriver {
public:
  MujocoMotorDriver(rclcpp::Node *node, const RobotRuntimeConfig &config)
      : config_(config) {
    dof_pos_ = config_.default_dof_pos;
    dof_vel_.fill(0.0f);

    rclcpp::QoS qos(rclcpp::KeepLast(1));
    qos.best_effort();
    qos.durability_volatile();

    pub_ = node->create_publisher<std_msgs::msg::Float32MultiArray>(
        "/mujoco/joint_cmd", qos);
    sub_ = node->create_subscription<std_msgs::msg::Float32MultiArray>(
        "/mujoco/joint_state", qos,
        [this](const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
          if (msg->data.size() < NUM_JOINTS * 2) {
            return;
          }
          for (int i = 0; i < NUM_JOINTS; ++i) {
            dof_pos_[i] = msg->data[i];
            dof_vel_[i] = msg->data[NUM_JOINTS + i];
          }
          msg_count_.fetch_add(1, std::memory_order_relaxed);
        });
  }

  void send_commands(const std::array<float, NUM_JOINTS> &target_dof_pos,
                     const std::array<float, NUM_JOINTS> &kp,
                     const std::array<float, NUM_JOINTS> &kd) {
    std_msgs::msg::Float32MultiArray msg;
    msg.data.resize(NUM_JOINTS * 3);
    for (int i = 0; i < NUM_JOINTS; ++i) {
      msg.data[i] = target_dof_pos[i];
      msg.data[NUM_JOINTS + i] = kp[i];
      msg.data[2 * NUM_JOINTS + i] = kd[i];
    }
    pub_->publish(msg);
  }

  void send_damping(float kd) {
    std::array<float, NUM_JOINTS> kp{};
    std::array<float, NUM_JOINTS> kd_arr{};
    kd_arr.fill(kd);
    send_commands(dof_pos_, kp, kd_arr);
  }

  void set_zero_torque() {
    std::array<float, NUM_JOINTS> zeros{};
    send_commands(dof_pos_, zeros, zeros);
  }

  bool check_errors() const { return false; }
  uint64_t msg_count() const {
    return msg_count_.load(std::memory_order_relaxed);
  }
  const std::array<float, NUM_JOINTS> &dof_pos() const { return dof_pos_; }
  const std::array<float, NUM_JOINTS> &dof_vel() const { return dof_vel_; }

private:
  RobotRuntimeConfig config_;
  std::array<float, NUM_JOINTS> dof_pos_{};
  std::array<float, NUM_JOINTS> dof_vel_{};
  std::atomic<uint64_t> msg_count_{0};
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr sub_;
};

class DeployNode : public rclcpp::Node {
public:
  DeployNode() : Node("deploy_node") {
    declare_parameter<std::string>("robot_config_file", "");
    declare_parameter<bool>("debug_no_motor", false);
    declare_parameter<bool>("sim_mode", false);
    declare_parameter<bool>("sim_pingpong_mode", false);
  }

  void initialize() {
    const std::string config_file =
        get_parameter("robot_config_file").as_string();
    if (config_file.empty()) {
      throw std::runtime_error("robot_config_file is required");
    }
    package_root_ = infer_package_root_from_config(config_file);
    config_ = load_robot_runtime_config(config_file);
    config_.adaptation_module_path =
        resolve_package_path(package_root_, config_.adaptation_module_path);
    config_.body_path = resolve_package_path(package_root_, config_.body_path);
    config_.urdf_relpath = resolve_package_path(package_root_, config_.urdf_relpath);
    config_.mujoco_xml_relpath =
        resolve_package_path(package_root_, config_.mujoco_xml_relpath);
    config_.isaac_xml_relpath =
        resolve_package_path(package_root_, config_.isaac_xml_relpath);

    debug_no_motor_ = get_parameter("debug_no_motor").as_bool();
    sim_mode_ = get_parameter("sim_mode").as_bool();
    sim_pingpong_mode_ = get_parameter("sim_pingpong_mode").as_bool();

    print_banner();

    imu_ = std::make_shared<IMUSubscriber>(config_.imu_topic,
                                           config_.imu_yaw_correction_deg);
    height_ = std::make_shared<HeightSubscriber>(config_.height_topic,
                                                 config_.nominal_base_height,
                                                 config_.height_measurement_scale,
                                                 config_.height_measurement_offset);

    if (sim_mode_) {
      mujoco_motor_ = std::make_unique<MujocoMotorDriver>(this, config_);
      RCLCPP_INFO(get_logger(), "Using MuJoCo topic motor driver");
    } else if (debug_no_motor_) {
      fake_motor_ = std::make_unique<FakeMotorDriver>(config_);
      RCLCPP_INFO(get_logger(), "Using fake motor driver");
    } else {
      motor_ = std::make_unique<MotorDriver>(config_, config_.port0,
                                             config_.port1);
      RCLCPP_INFO(get_logger(), "Using real motor driver");
    }

    policy_ = std::make_unique<CSEPolicyRunner>(config_);
    keyboard_ = std::make_unique<KeyboardController>(config_);
    if (config_.teleop_udp_enable) {
      udp_ctrl_ = std::make_unique<UdpController>(config_.teleop_udp_port);
    }
    if (config_.joy_enable) {
      joy_sub_ = create_subscription<sensor_msgs::msg::Joy>(
          config_.joy_topic, rclcpp::SystemDefaultsQoS(),
          std::bind(&DeployNode::handle_joy, this, std::placeholders::_1));
    }
    sm_ = std::make_unique<StateMachine>(config_);
    visualizer_ = std::make_unique<RobotVisualizer>(this, config_.joint_names);
    last_safe_target_ = config_.standup_target_pos;
  }

  void run() {
    rclcpp::Rate rate(1.0 / config_.control_dt);
    uint64_t loop_count = 0;
    uint64_t last_joint_count = 0;
    uint64_t last_imu_count = 0;
    auto last_print = std::chrono::steady_clock::now();

    while (rclcpp::ok() && !keyboard_->is_exit()) {
      if (sim_mode_ && sim_pingpong_mode_ && mujoco_motor_) {
        while (rclcpp::ok() && !keyboard_->is_exit()) {
          rclcpp::spin_some(imu_);
          rclcpp::spin_some(height_);
          rclcpp::spin_some(shared_from_this());
          const bool fresh_joint = mujoco_motor_->msg_count() > last_joint_count;
          const bool fresh_imu = imu_->msg_count() > last_imu_count;
          if (fresh_joint && fresh_imu) {
            last_joint_count = mujoco_motor_->msg_count();
            last_imu_count = imu_->msg_count();
            break;
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
      } else {
        rclcpp::spin_some(imu_);
        rclcpp::spin_some(height_);
        if (sim_mode_) {
          rclcpp::spin_some(shared_from_this());
        }
      }

      handle_udp();
      if (auto req = consume_joy_state_request()) {
        handle_state_request(*req);
      }
      if (auto req = keyboard_->consume_state_request()) {
        handle_state_request(*req);
      }

      switch (sm_->state()) {
      case RobotState::IDLE:
        handle_idle();
        break;
      case RobotState::STAND_UP:
        handle_standup();
        break;
      case RobotState::RL:
        handle_rl();
        break;
      case RobotState::JOINT_DAMPING:
        handle_joint_damping();
        break;
      case RobotState::RETURN_DEFAULT:
        handle_return_default();
        break;
      case RobotState::JOINT_SWEEP:
        handle_joint_sweep();
        break;
      case RobotState::SINGLE_STEP_RL:
        handle_single_step_rl();
        break;
      }

      if (visualizer_) {
        visualizer_->publish_joint_states(get_dof_pos(), get_dof_vel());
      }

      ++loop_count;
      const auto now = std::chrono::steady_clock::now();
      if (std::chrono::duration<float>(now - last_print).count() >= 1.0f) {
        print_status(loop_count);
        last_print = now;
      }

      if (!(sim_mode_ && sim_pingpong_mode_)) {
        rate.sleep();
      }
    }

    shutdown();
  }

private:
  void print_banner() const {
    std::cout << "\n"
              << "============================================================\n"
              << " mybot_v2_1 CSE deployment\n"
              << " 0 Idle | 1 StandUp | 2 RL | 3 Damping | 4 ReturnDefault | 5 SingleStep | 6 JointSweep\n"
              << " W/S vx | Q/E vy | A/D yaw | R zero cmd | Space emergency | Esc exit\n"
              << " Joy: run `ros2 run joy joy_node`, then use /joy for cmd + state switch\n"
              << "============================================================\n"
              << std::endl;
  }

  static float apply_deadzone(float value, float deadzone) {
    if (std::fabs(value) < deadzone) {
      return 0.0f;
    }
    return std::clamp(value, -1.0f, 1.0f);
  }

  template <typename T>
  static T read_index_or(const std::vector<T> &values, int index, T fallback) {
    if (index < 0 || static_cast<size_t>(index) >= values.size()) {
      return fallback;
    }
    return values[static_cast<size_t>(index)];
  }

  void handle_joy(const sensor_msgs::msg::Joy::SharedPtr msg) {
    joy_received_ = true;
    last_joy_msg_time_ = std::chrono::steady_clock::now();

    auto axis_to_command = [this, &msg](int axis_idx, bool invert, float scale) {
      float value = read_index_or<float>(msg->axes, axis_idx, 0.0f);
      value = apply_deadzone(value, config_.joy_axis_deadzone);
      if (invert) {
        value = -value;
      }
      return std::clamp(value * scale, -scale, scale);
    };

    joy_commands_[0] =
        axis_to_command(config_.joy_axis_vx, config_.joy_invert_vx,
                        config_.cmd_vx_max);
    joy_commands_[1] =
        axis_to_command(config_.joy_axis_vy, config_.joy_invert_vy,
                        config_.cmd_vy_max);
    joy_commands_[2] =
        axis_to_command(config_.joy_axis_yaw, config_.joy_invert_yaw,
                        config_.cmd_yaw_max);

    if (last_joy_buttons_.size() != msg->buttons.size()) {
      last_joy_buttons_.assign(msg->buttons.size(), 0);
    }

    auto pressed_now = [&msg](int idx) {
      return read_index_or<int32_t>(msg->buttons, idx, 0) != 0;
    };
    auto rising_edge = [this, &msg](int idx) {
      const bool now = read_index_or<int32_t>(msg->buttons, idx, 0) != 0;
      const bool prev =
          idx >= 0 && static_cast<size_t>(idx) < last_joy_buttons_.size() &&
          last_joy_buttons_[static_cast<size_t>(idx)] != 0;
      return now && !prev;
    };

    if (rising_edge(config_.joy_button_stand_up)) {
      joy_state_request_ = StateRequest{RobotState::STAND_UP, false};
    } else if (rising_edge(config_.joy_button_return_default)) {
      joy_state_request_ = StateRequest{RobotState::RETURN_DEFAULT, false};
    } else if (rising_edge(config_.joy_button_rl)) {
      joy_state_request_ = StateRequest{RobotState::RL, false};
    } else if (rising_edge(config_.joy_button_damping)) {
      joy_state_request_ = StateRequest{RobotState::JOINT_DAMPING, false};
    } else if (rising_edge(config_.joy_button_single_step)) {
      joy_state_request_ = StateRequest{RobotState::SINGLE_STEP_RL, false};
    } else if (rising_edge(config_.joy_button_joint_sweep)) {
      joy_state_request_ = StateRequest{RobotState::JOINT_SWEEP, false};
    } else if (rising_edge(config_.joy_button_idle)) {
      joy_state_request_ = StateRequest{RobotState::IDLE, false};
    }

    if (rising_edge(config_.joy_button_confirm)) {
      joy_step_confirmed_ = true;
    }
    if (pressed_now(config_.joy_button_emergency)) {
      joy_state_request_ = StateRequest{RobotState::IDLE, true};
      joy_commands_ = {0.0f, 0.0f, 0.0f};
    }

    last_joy_buttons_.assign(msg->buttons.begin(), msg->buttons.end());
  }

  bool joy_active() const {
    if (!config_.joy_enable || !joy_received_) {
      return false;
    }
    const auto age =
        std::chrono::duration<float>(std::chrono::steady_clock::now() -
                                     last_joy_msg_time_)
            .count();
    return age <= config_.joy_timeout_s;
  }

  std::optional<StateRequest> consume_joy_state_request() {
    if (!joy_state_request_) {
      return std::nullopt;
    }
    const StateRequest out = *joy_state_request_;
    joy_state_request_.reset();
    return out;
  }

  bool consume_step_confirm_any() {
    if (keyboard_->consume_step_confirm()) {
      return true;
    }
    if (joy_step_confirmed_) {
      joy_step_confirmed_ = false;
      return true;
    }
    return false;
  }

  void handle_udp() {
    if (!udp_ctrl_ || !udp_ctrl_->has_data()) {
      return;
    }
    const auto cmd = udp_ctrl_->get_latest();
    if (cmd.e_stop) {
      handle_state_request({RobotState::IDLE, true});
      udp_vx_ = udp_vy_ = udp_yaw_ = 0.0f;
      return;
    }
    auto target = static_cast<RobotState>(cmd.mode);
    if (target != sm_->state()) {
      handle_state_request({target, false});
    }
    udp_vx_ = cmd.vx * config_.cmd_vx_max;
    udp_vy_ = cmd.vy * config_.cmd_vy_max;
    udp_yaw_ = cmd.yaw * config_.cmd_yaw_max;
  }

  void handle_state_request(const StateRequest &req) {
    const RobotState old_state = sm_->state();
    if (!sm_->request_transition(req.target, req.emergency)) {
      return;
    }
    if (old_state != req.target &&
        (req.target == RobotState::RL ||
         req.target == RobotState::SINGLE_STEP_RL)) {
      policy_->reset();
      single_step_pending_ = false;
      single_step_count_ = 0;
      last_safe_target_ = config_.standup_target_pos;
      RCLCPP_INFO(get_logger(), "CSE policy history reset");
    }
    if (old_state != req.target && req.target == RobotState::JOINT_SWEEP) {
      keyboard_->reset_sweep();
      sweep_last_sent_ = config_.standup_target_pos;
      sweep_has_sent_ = false;
    }
    if (old_state != req.target &&
        (req.target == RobotState::RETURN_DEFAULT ||
         req.target == RobotState::STAND_UP ||
         req.target == RobotState::IDLE ||
         req.target == RobotState::JOINT_DAMPING)) {
      single_step_pending_ = false;
    }
    if (req.target == RobotState::STAND_UP) {
      standup_printed_ = false;
    }
  }

  void handle_idle() { set_zero_torque(); }

  void handle_standup() {
    const auto target = sm_->get_standup_target(get_dof_pos());
    send_to_motors(target, config_.kp_joint, config_.kd_joint);
    if (sm_->standup_complete() && !standup_printed_) {
      standup_printed_ = true;
      RCLCPP_INFO(get_logger(), "Standup complete. Press 2 for RL.");
    }
  }

  void handle_return_default() {
    const auto target = sm_->get_return_default_target(get_dof_pos());
    send_to_motors(target, config_.kp_joint, config_.kd_joint);
  }

  static float vector_norm3(const std::array<float, 3> &v) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  }

  static void height_stats(
      const std::array<float, NUM_HEIGHT_POINTS> &height_distances,
      float &min_height, float &max_height, float &mean_height) {
    min_height = height_distances[0];
    max_height = height_distances[0];
    mean_height = 0.0f;
    for (float h : height_distances) {
      min_height = std::min(min_height, h);
      max_height = std::max(max_height, h);
      mean_height += h;
    }
    mean_height /= static_cast<float>(NUM_HEIGHT_POINTS);
  }

  bool policy_inputs_safe(
      const std::array<float, 3> &gravity,
      const std::array<float, NUM_HEIGHT_POINTS> &height_distances,
      std::string &reason) const {
    if (config_.require_imu_ready_for_rl && !imu_->is_ready()) {
      reason = "IMU not ready";
      return false;
    }

    const float gravity_norm = vector_norm3(gravity);
    if (!std::isfinite(gravity_norm) ||
        gravity_norm < config_.gravity_norm_min ||
        gravity_norm > config_.gravity_norm_max) {
      reason = "projected_gravity norm out of range";
      return false;
    }
    if (!std::isfinite(gravity[2]) || gravity[2] > config_.gravity_z_max) {
      reason = "projected_gravity z has wrong sign";
      return false;
    }

    if (config_.height_sanity_check_enable && height_->is_ready()) {
      for (float h : height_distances) {
        if (!std::isfinite(h) || h < config_.height_distance_min ||
            h > config_.height_distance_max) {
          reason = "height distance out of range";
          return false;
        }
      }
    }

    reason.clear();
    return true;
  }

  void print_policy_input_warning(
      const std::string &reason, const std::array<float, 3> &gravity,
      const std::array<float, NUM_HEIGHT_POINTS> &height_distances) {
    const auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration<float>(now - last_policy_input_warning_).count() <
        0.5f) {
      return;
    }
    last_policy_input_warning_ = now;

    float h_min = 0.0f;
    float h_max = 0.0f;
    float h_mean = 0.0f;
    height_stats(height_distances, h_min, h_max, h_mean);
    std::cout << "\n[RL_INPUT_BLOCKED] " << reason << " grav=[" << gravity[0]
              << ", " << gravity[1] << ", " << gravity[2]
              << "] |g|=" << vector_norm3(gravity) << " height_dist[min,max,mean]=["
              << h_min << ", " << h_max << ", " << h_mean << "]"
              << " imu=" << (imu_->is_ready() ? "OK" : "WAIT")
              << " height=" << (height_->is_ready() ? "OK" : "FALLBACK")
              << std::endl;
  }

  void handle_rl() {
    auto commands = get_commands();
    const auto gravity = imu_->get_projected_gravity();
    const auto height_distances = height_->get_distances();
    std::string unsafe_reason;
    if (!policy_inputs_safe(gravity, height_distances, unsafe_reason)) {
      print_policy_input_warning(unsafe_reason, gravity, height_distances);
      send_to_motors(last_safe_target_, config_.kp_joint, config_.kd_joint);
      return;
    }

    std::array<float, NUM_ACTIONS> actions{};
    std::array<float, NUM_JOINTS> target{};
    policy_->step(commands, gravity, get_dof_pos(), get_dof_vel(),
                  height_distances, target, actions);
    last_safe_target_ = target;
    send_to_motors(target, config_.kp_joint, config_.kd_joint);
  }

  void handle_joint_damping() { send_damping(config_.kd_damp_motor); }

  void handle_joint_sweep() {
    const int idx = keyboard_->get_sweep_joint_idx();
    const float offset = keyboard_->get_sweep_offset();
    std::array<float, NUM_JOINTS> target = config_.standup_target_pos;
    target[idx] = std::clamp(config_.standup_target_pos[idx] + offset,
                             config_.joint_pos_lower[idx],
                             config_.joint_pos_upper[idx]);

    static auto last_print = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration<float>(now - last_print).count() > 0.5f) {
      last_print = now;
      std::cout << "\n[JOINT_SWEEP] DOF " << idx << " "
                << config_.joint_names[idx] << " motor_id="
                << config_.motor_map[idx].motor_id
                << " reversed=" << (config_.motor_map[idx].is_reversed ? "Y" : "N")
                << " target=" << target[idx] << " offset=" << offset
                << "\nEnter sends, J/K joint, +/- offset, Space stop"
                << std::endl;
    }

    if (consume_step_confirm_any()) {
      sweep_last_sent_ = target;
      sweep_has_sent_ = true;
    }
    send_to_motors(sweep_has_sent_ ? sweep_last_sent_ : config_.standup_target_pos,
                   config_.kp_joint, config_.kd_joint);
  }

  void handle_single_step_rl() {
    if (single_step_pending_) {
      send_to_motors(last_safe_target_, config_.kp_joint, config_.kd_joint);
      if (consume_step_confirm_any()) {
        send_to_motors(pending_target_, config_.kp_joint, config_.kd_joint);
        last_safe_target_ = pending_target_;
        single_step_pending_ = false;
      }
      return;
    }

    std::array<float, NUM_ACTIONS> actions{};
    std::array<float, NUM_JOINTS> target{};
    const auto commands = get_commands();
    const auto gravity = imu_->get_projected_gravity();
    const auto height_distances = height_->get_distances();
    std::string unsafe_reason;
    if (!policy_inputs_safe(gravity, height_distances, unsafe_reason)) {
      print_policy_input_warning(unsafe_reason, gravity, height_distances);
      send_to_motors(last_safe_target_, config_.kp_joint, config_.kd_joint);
      return;
    }

    policy_->step(commands, gravity, get_dof_pos(), get_dof_vel(),
                  height_distances, target, actions);
    pending_target_ = target;
    ++single_step_count_;
    single_step_pending_ = true;

    float h_min = 0.0f;
    float h_max = 0.0f;
    float h_mean = 0.0f;
    height_stats(height_distances, h_min, h_max, h_mean);
    float max_abs_action = 0.0f;
    for (float action : actions) {
      max_abs_action = std::max(max_abs_action, std::fabs(action));
    }

    std::cout << "\n========== SINGLE STEP CSE " << single_step_count_
              << " ==========\n";
    std::cout << "cmd=[" << commands[0] << ", " << commands[1] << ", "
              << commands[2] << "] imu=" << (imu_->is_ready() ? "OK" : "WAIT")
              << " height=" << (height_->is_ready() ? "OK" : "FALLBACK")
              << " grav=[" << gravity[0] << ", " << gravity[1] << ", "
              << gravity[2] << "] |g|=" << vector_norm3(gravity)
              << " height_dist[min,max,mean]=[" << h_min << ", " << h_max
              << ", " << h_mean << "] max|action|=" << max_abs_action
              << "\n";
    std::cout << std::left << std::setw(18) << "Joint" << std::right
              << std::setw(10) << "q" << std::setw(10) << "dq"
              << std::setw(10) << "action" << std::setw(10) << "target"
              << std::setw(8) << "motor" << std::setw(6) << "rev" << "\n";
    const auto &q = get_dof_pos();
    const auto &dq = get_dof_vel();
    for (int i = 0; i < NUM_JOINTS; ++i) {
      std::cout << std::left << std::setw(18) << config_.joint_names[i]
                << std::right << std::fixed << std::setprecision(3)
                << std::setw(10) << q[i] << std::setw(10) << dq[i]
                << std::setw(10) << actions[i] << std::setw(10) << target[i]
                << std::setw(8) << config_.motor_map[i].motor_id
                << std::setw(6)
                << (config_.motor_map[i].is_reversed ? "Y" : "N") << "\n";
    }
    std::cout << "Press Enter to execute this target." << std::endl;
    send_to_motors(last_safe_target_, config_.kp_joint, config_.kd_joint);
  }

  std::array<float, 3> get_commands() const {
    if (joy_active()) {
      return joy_commands_;
    }
    if (udp_ctrl_ && udp_ctrl_->has_data()) {
      return {udp_vx_, udp_vy_, udp_yaw_};
    }
    return keyboard_->get_commands();
  }

  void send_to_motors(const std::array<float, NUM_JOINTS> &target,
                      const std::array<float, NUM_JOINTS> &kp,
                      const std::array<float, NUM_JOINTS> &kd) {
    if (motor_) {
      motor_->send_commands(target, kp, kd);
    } else if (mujoco_motor_) {
      mujoco_motor_->send_commands(target, kp, kd);
    } else if (fake_motor_) {
      fake_motor_->send_commands(target, kp, kd);
    }
  }

  void send_damping(float kd) {
    if (motor_) {
      motor_->send_damping(kd);
    } else if (mujoco_motor_) {
      mujoco_motor_->send_damping(kd);
    } else if (fake_motor_) {
      fake_motor_->send_damping(kd);
    }
  }

  void set_zero_torque() {
    if (motor_) {
      motor_->set_zero_torque();
    } else if (mujoco_motor_) {
      mujoco_motor_->set_zero_torque();
    } else if (fake_motor_) {
      fake_motor_->set_zero_torque();
    }
  }

  const std::array<float, NUM_JOINTS> &get_dof_pos() const {
    if (motor_) {
      return motor_->dof_pos();
    }
    if (mujoco_motor_) {
      return mujoco_motor_->dof_pos();
    }
    return fake_motor_->dof_pos();
  }

  const std::array<float, NUM_JOINTS> &get_dof_vel() const {
    if (motor_) {
      return motor_->dof_vel();
    }
    if (mujoco_motor_) {
      return mujoco_motor_->dof_vel();
    }
    return fake_motor_->dof_vel();
  }

  void print_status(uint64_t loop_count) const {
    const auto commands = get_commands();
    const auto gravity = imu_->get_projected_gravity();
    const char *input_source =
        joy_active() ? "JOY" : ((udp_ctrl_ && udp_ctrl_->has_data()) ? "UDP" : "KEY");
    std::cout << "\r[" << robot_state_name(sm_->state()) << "] loop="
              << loop_count << " imu=" << (imu_->is_ready() ? "OK" : "WAIT")
              << " height=" << (height_->is_ready() ? "OK" : "FALLBACK")
              << " input=" << input_source
              << " cmd=[" << std::fixed << std::setprecision(2)
              << commands[0] << "," << commands[1] << "," << commands[2]
              << "] grav=[" << gravity[0] << "," << gravity[1] << ","
              << gravity[2] << "]       " << std::flush;
  }

  void shutdown() {
    RCLCPP_INFO(get_logger(), "Shutting down...");
    set_zero_torque();
    set_zero_torque();
    if (keyboard_) {
      keyboard_->cleanup();
    }
    std::cout << "\n[deploy_node] shutdown complete" << std::endl;
  }

  fs::path package_root_;
  RobotRuntimeConfig config_;
  bool debug_no_motor_ = false;
  bool sim_mode_ = false;
  bool sim_pingpong_mode_ = false;
  bool standup_printed_ = false;

  std::shared_ptr<IMUSubscriber> imu_;
  std::shared_ptr<HeightSubscriber> height_;
  std::unique_ptr<MotorDriver> motor_;
  std::unique_ptr<FakeMotorDriver> fake_motor_;
  std::unique_ptr<MujocoMotorDriver> mujoco_motor_;
  std::unique_ptr<CSEPolicyRunner> policy_;
  std::unique_ptr<KeyboardController> keyboard_;
  std::unique_ptr<UdpController> udp_ctrl_;
  std::unique_ptr<StateMachine> sm_;
  std::unique_ptr<RobotVisualizer> visualizer_;
  rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_sub_;

  float udp_vx_ = 0.0f;
  float udp_vy_ = 0.0f;
  float udp_yaw_ = 0.0f;
  std::array<float, 3> joy_commands_{0.0f, 0.0f, 0.0f};
  std::optional<StateRequest> joy_state_request_;
  std::vector<int32_t> last_joy_buttons_{};
  std::chrono::steady_clock::time_point last_joy_msg_time_{};
  bool joy_received_ = false;
  bool joy_step_confirmed_ = false;

  std::array<float, NUM_JOINTS> sweep_last_sent_{};
  bool sweep_has_sent_ = false;
  std::array<float, NUM_JOINTS> last_safe_target_{};
  std::array<float, NUM_JOINTS> pending_target_{};
  bool single_step_pending_ = false;
  uint64_t single_step_count_ = 0;
  std::chrono::steady_clock::time_point last_policy_input_warning_{};
};

} // namespace deploy

void signal_handler(int) { rclcpp::shutdown(); }

int main(int argc, char **argv) {
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);
  rclcpp::init(argc, argv);

  auto node = std::make_shared<deploy::DeployNode>();
  try {
    node->initialize();
    node->run();
  } catch (const std::exception &e) {
    RCLCPP_ERROR(node->get_logger(), "Fatal error: %s", e.what());
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::shutdown();
  return 0;
}
