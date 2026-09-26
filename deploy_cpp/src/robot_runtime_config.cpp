/**
 * @file robot_runtime_config.cpp
 * @brief YAML-based runtime configuration loader.
 */

#include "robot_runtime_config.h"

#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace deploy {
namespace {

template <typename T, size_t N>
std::array<T, N> parse_required_array(const YAML::Node &root,
                                      const std::string &key) {
  const YAML::Node n = root[key];
  if (!n || !n.IsSequence() || n.size() != N) {
    throw std::runtime_error("Missing or invalid length for key: " + key);
  }

  std::array<T, N> out{};
  for (size_t i = 0; i < N; ++i) {
    out[i] = n[i].as<T>();
  }
  return out;
}

template <typename T>
T parse_optional(const YAML::Node &root, const std::string &key,
                 T default_value) {
  return root[key] ? root[key].as<T>() : default_value;
}

template <typename T, size_t N>
std::array<T, N> parse_optional_array(const YAML::Node &root,
                                      const std::string &key,
                                      const std::array<T, N> &default_value) {
  if (!root[key]) {
    return default_value;
  }
  return parse_required_array<T, N>(root, key);
}

void validate_positive(const std::array<float, NUM_JOINTS> &values,
                       const std::string &name) {
  for (size_t i = 0; i < values.size(); ++i) {
    if (values[i] <= 0.0f) {
      throw std::runtime_error(name + " must be positive at index " +
                               std::to_string(i));
    }
  }
}

void fill_names(std::array<std::string, NUM_JOINTS> &dst,
                const YAML::Node &node, const std::string &key) {
  if (!node || !node.IsSequence() || node.size() != NUM_JOINTS) {
    throw std::runtime_error(key + " must be an array of size 12");
  }
  for (int i = 0; i < NUM_JOINTS; ++i) {
    dst[i] = node[i].as<std::string>();
  }
}

} // namespace

RobotRuntimeConfig default_robot_runtime_config() {
  RobotRuntimeConfig cfg;

  const std::array<std::string, NUM_JOINTS> joint_names = {
      "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
      "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
      "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
      "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"};
  const std::array<std::string, NUM_JOINTS> actuator_names = {
      "FL_hip", "FL_thigh", "FL_calf", "FR_hip", "FR_thigh", "FR_calf",
      "RL_hip", "RL_thigh", "RL_calf", "RR_hip", "RR_thigh", "RR_calf"};

  cfg.joint_names = joint_names;
  cfg.joint_controller_names = actuator_names;
  cfg.default_dof_pos = {0.1f, 1.27f, -2.70f, -0.1f, 1.27f, -2.70f,
                         0.1f, 1.27f, -2.70f, -0.1f, 1.27f, -2.70f};
  cfg.standup_target_pos = {0.1f, 0.8f, -1.5f, -0.1f, 0.8f, -1.5f,
                            0.1f, 0.8f, -1.5f, -0.1f, 0.8f, -1.5f};
  cfg.policy_dof_pos = cfg.standup_target_pos;
  cfg.joint_pos_lower = {-0.8f, -1.04f, -2.69f, -0.8f, -1.04f, -2.69f,
                         -0.8f, -1.04f, -2.69f, -0.8f, -1.04f, -2.69f};
  cfg.joint_pos_upper = {0.8f, 4.18f, -0.91f, 0.8f, 4.18f, -0.91f,
                         0.8f, 4.18f, -0.91f, 0.8f, 4.18f, -0.91f};
  cfg.kp_joint = {40.0f, 40.0f, 80.0f, 40.0f, 40.0f, 80.0f,
                  40.0f, 40.0f, 80.0f, 40.0f, 40.0f, 80.0f};
  cfg.kd_joint = {1.0f, 1.0f, 2.0f, 1.0f, 1.0f, 2.0f,
                  1.0f, 1.0f, 2.0f, 1.0f, 1.0f, 2.0f};
  cfg.torque_limits.fill(23.5f);
  cfg.joint_mapping = {4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 11, 12};
  cfg.motor_is_reversed = {false, false, true,  false, true,  false,
                           true,  false, true,  true,  true,  false};
  cfg.joint_transmission_ratio = {6.33f, 6.33f, 14.77f, 6.33f, 6.33f,
                                  14.77f, 6.33f, 6.33f, 14.77f, 6.33f,
                                  6.33f, 14.77f};

  for (int i = 0; i < NUM_JOINTS; ++i) {
    cfg.motor_map[i].motor_id = cfg.joint_mapping[i];
    cfg.motor_map[i].port_idx = cfg.joint_mapping[i] >= 7 ? 1 : 0;
    cfg.motor_map[i].is_reversed = cfg.motor_is_reversed[i];
  }

  return cfg;
}

RobotRuntimeConfig load_robot_runtime_config(const std::string &yaml_file) {
  if (yaml_file.empty()) {
    throw std::runtime_error("robot_config_file is required");
  }

  const YAML::Node root = YAML::LoadFile(yaml_file);
  RobotRuntimeConfig cfg = default_robot_runtime_config();

  cfg.num_of_dofs = parse_optional<int>(root, "num_of_dofs", cfg.num_of_dofs);
  if (cfg.num_of_dofs != NUM_JOINTS) {
    throw std::runtime_error("num_of_dofs must be 12");
  }
  cfg.dt = parse_optional<float>(root, "dt", cfg.dt);
  cfg.decimation = parse_optional<int>(root, "decimation", cfg.decimation);

  cfg.kp_joint =
      parse_optional_array<float, NUM_JOINTS>(root, "kp_joint", cfg.kp_joint);
  cfg.kd_joint =
      parse_optional_array<float, NUM_JOINTS>(root, "kd_joint", cfg.kd_joint);
  cfg.torque_limits = parse_optional_array<float, NUM_JOINTS>(
      root, "torque_limits", cfg.torque_limits);
  validate_positive(cfg.torque_limits, "torque_limits");

  cfg.default_dof_pos = parse_required_array<float, NUM_JOINTS>(
      root, "default_dof_pos");
  cfg.standup_target_pos = parse_required_array<float, NUM_JOINTS>(
      root, "standup_target_pos");
  cfg.policy_dof_pos = parse_required_array<float, NUM_JOINTS>(
      root, "policy_dof_pos");
  cfg.joint_pos_lower = parse_required_array<float, NUM_JOINTS>(
      root, "joint_pos_lower");
  cfg.joint_pos_upper = parse_required_array<float, NUM_JOINTS>(
      root, "joint_pos_upper");

  fill_names(cfg.joint_names, root["joint_names"], "joint_names");
  fill_names(cfg.joint_controller_names, root["joint_controller_names"],
             "joint_controller_names");

  cfg.joint_mapping = parse_optional_array<int, NUM_JOINTS>(
      root, "joint_mapping", cfg.joint_mapping);
  cfg.motor_is_reversed = parse_optional_array<bool, NUM_JOINTS>(
      root, "motor_is_reversed", cfg.motor_is_reversed);
  cfg.joint_transmission_ratio = parse_required_array<float, NUM_JOINTS>(
      root, "joint_transmission_ratio");
  validate_positive(cfg.joint_transmission_ratio, "joint_transmission_ratio");

  cfg.wheel_indices.clear();
  if (root["wheel_indices"]) {
    for (const auto &w : root["wheel_indices"]) {
      cfg.wheel_indices.push_back(w.as<int>());
    }
  }

  cfg.adaptation_module_path = parse_optional<std::string>(
      root, "adaptation_module_path", cfg.adaptation_module_path);
  cfg.body_path =
      parse_optional<std::string>(root, "body_path", cfg.body_path);
  cfg.device = parse_optional<std::string>(root, "device", cfg.device);
  cfg.port0 = parse_optional<std::string>(root, "port0", cfg.port0);
  cfg.port1 = parse_optional<std::string>(root, "port1", cfg.port1);
  cfg.imu_topic =
      parse_optional<std::string>(root, "imu_topic", cfg.imu_topic);
  cfg.height_topic =
      parse_optional<std::string>(root, "height_topic", cfg.height_topic);
  cfg.imu_yaw_correction_deg = parse_optional<float>(
      root, "imu_yaw_correction_deg", cfg.imu_yaw_correction_deg);

  cfg.robot_name =
      parse_optional<std::string>(root, "robot_name", cfg.robot_name);
  cfg.urdf_relpath =
      parse_optional<std::string>(root, "urdf_relpath", cfg.urdf_relpath);
  cfg.mujoco_xml_relpath = parse_optional<std::string>(
      root, "mujoco_xml_relpath", cfg.mujoco_xml_relpath);
  cfg.isaac_xml_relpath = parse_optional<std::string>(
      root, "isaac_xml_relpath", cfg.isaac_xml_relpath);

  cfg.command_lin_vel_scale =
      parse_optional<float>(root, "command_lin_vel_scale",
                            cfg.command_lin_vel_scale);
  cfg.command_yaw_scale =
      parse_optional<float>(root, "command_yaw_scale", cfg.command_yaw_scale);
  cfg.projected_gravity_scale = parse_optional<float>(
      root, "projected_gravity_scale", cfg.projected_gravity_scale);
  cfg.dof_pos_scale =
      parse_optional<float>(root, "dof_pos_scale", cfg.dof_pos_scale);
  cfg.dof_vel_scale =
      parse_optional<float>(root, "dof_vel_scale", cfg.dof_vel_scale);
  cfg.height_bias =
      parse_optional<float>(root, "height_bias", cfg.height_bias);
  cfg.height_scale =
      parse_optional<float>(root, "height_scale", cfg.height_scale);
  cfg.nominal_base_height = parse_optional<float>(
      root, "nominal_base_height", cfg.nominal_base_height);
  cfg.height_measurement_scale = parse_optional<float>(
      root, "height_measurement_scale", cfg.height_measurement_scale);
  cfg.height_measurement_offset = parse_optional<float>(
      root, "height_measurement_offset", cfg.height_measurement_offset);
  cfg.action_scale =
      parse_optional<float>(root, "action_scale", cfg.action_scale);
  cfg.require_imu_ready_for_rl = parse_optional<bool>(
      root, "require_imu_ready_for_rl", cfg.require_imu_ready_for_rl);
  cfg.height_sanity_check_enable = parse_optional<bool>(
      root, "height_sanity_check_enable", cfg.height_sanity_check_enable);
  cfg.gravity_norm_min =
      parse_optional<float>(root, "gravity_norm_min", cfg.gravity_norm_min);
  cfg.gravity_norm_max =
      parse_optional<float>(root, "gravity_norm_max", cfg.gravity_norm_max);
  cfg.gravity_z_max =
      parse_optional<float>(root, "gravity_z_max", cfg.gravity_z_max);
  cfg.height_distance_min = parse_optional<float>(
      root, "height_distance_min", cfg.height_distance_min);
  cfg.height_distance_max = parse_optional<float>(
      root, "height_distance_max", cfg.height_distance_max);

  cfg.cmd_deadband =
      parse_optional<float>(root, "cmd_deadband", cfg.cmd_deadband);
  cfg.control_dt =
      parse_optional<float>(root, "control_dt", cfg.control_dt);
  cfg.standup_duration = parse_optional<float>(
      root, "standup_duration", cfg.standup_duration);
  cfg.cmd_vx_min = parse_optional<float>(root, "cmd_vx_min", cfg.cmd_vx_min);
  cfg.cmd_vx_max = parse_optional<float>(root, "cmd_vx_max", cfg.cmd_vx_max);
  cfg.cmd_vy_min = parse_optional<float>(root, "cmd_vy_min", cfg.cmd_vy_min);
  cfg.cmd_vy_max = parse_optional<float>(root, "cmd_vy_max", cfg.cmd_vy_max);
  cfg.cmd_yaw_min =
      parse_optional<float>(root, "cmd_yaw_min", cfg.cmd_yaw_min);
  cfg.cmd_yaw_max =
      parse_optional<float>(root, "cmd_yaw_max", cfg.cmd_yaw_max);
  cfg.cmd_vx_step =
      parse_optional<float>(root, "cmd_vx_step", cfg.cmd_vx_step);
  cfg.cmd_vy_step =
      parse_optional<float>(root, "cmd_vy_step", cfg.cmd_vy_step);
  cfg.cmd_yaw_step =
      parse_optional<float>(root, "cmd_yaw_step", cfg.cmd_yaw_step);
  cfg.clip_obs = parse_optional<float>(root, "clip_obs", cfg.clip_obs);
  cfg.clip_actions =
      parse_optional<float>(root, "clip_actions", cfg.clip_actions);
  cfg.kd_damp_motor =
      parse_optional<float>(root, "kd_damp_motor", cfg.kd_damp_motor);
  cfg.hip_indices =
      parse_optional_array<int, 4>(root, "hip_indices", cfg.hip_indices);

  cfg.teleop_udp_enable = parse_optional<bool>(
      root, "teleop_udp_enable", cfg.teleop_udp_enable);
  cfg.teleop_udp_port =
      parse_optional<int>(root, "teleop_udp_port", cfg.teleop_udp_port);
  cfg.joy_enable =
      parse_optional<bool>(root, "joy_enable", cfg.joy_enable);
  cfg.joy_topic =
      parse_optional<std::string>(root, "joy_topic", cfg.joy_topic);
  cfg.joy_axis_deadzone = parse_optional<float>(
      root, "joy_axis_deadzone", cfg.joy_axis_deadzone);
  cfg.joy_timeout_s =
      parse_optional<float>(root, "joy_timeout_s", cfg.joy_timeout_s);
  cfg.joy_axis_vx =
      parse_optional<int>(root, "joy_axis_vx", cfg.joy_axis_vx);
  cfg.joy_axis_vy =
      parse_optional<int>(root, "joy_axis_vy", cfg.joy_axis_vy);
  cfg.joy_axis_yaw =
      parse_optional<int>(root, "joy_axis_yaw", cfg.joy_axis_yaw);
  cfg.joy_invert_vx =
      parse_optional<bool>(root, "joy_invert_vx", cfg.joy_invert_vx);
  cfg.joy_invert_vy =
      parse_optional<bool>(root, "joy_invert_vy", cfg.joy_invert_vy);
  cfg.joy_invert_yaw = parse_optional<bool>(
      root, "joy_invert_yaw", cfg.joy_invert_yaw);
  cfg.joy_button_stand_up = parse_optional<int>(
      root, "joy_button_stand_up", cfg.joy_button_stand_up);
  cfg.joy_button_return_default = parse_optional<int>(
      root, "joy_button_return_default", cfg.joy_button_return_default);
  cfg.joy_button_rl =
      parse_optional<int>(root, "joy_button_rl", cfg.joy_button_rl);
  cfg.joy_button_damping = parse_optional<int>(
      root, "joy_button_damping", cfg.joy_button_damping);
  cfg.joy_button_single_step = parse_optional<int>(
      root, "joy_button_single_step", cfg.joy_button_single_step);
  cfg.joy_button_joint_sweep = parse_optional<int>(
      root, "joy_button_joint_sweep", cfg.joy_button_joint_sweep);
  cfg.joy_button_idle =
      parse_optional<int>(root, "joy_button_idle", cfg.joy_button_idle);
  cfg.joy_button_confirm = parse_optional<int>(
      root, "joy_button_confirm", cfg.joy_button_confirm);
  cfg.joy_button_emergency = parse_optional<int>(
      root, "joy_button_emergency", cfg.joy_button_emergency);
  cfg.debug_print_policy = parse_optional<bool>(
      root, "debug_print_policy", cfg.debug_print_policy);
  cfg.debug_print_interval = parse_optional<int>(
      root, "debug_print_interval", cfg.debug_print_interval);

  const YAML::Node motor_ports = root["motor_port_idx"];
  std::array<int, NUM_JOINTS> explicit_port{};
  explicit_port.fill(-1);
  if (motor_ports) {
    if (!motor_ports.IsSequence() || motor_ports.size() != NUM_JOINTS) {
      throw std::runtime_error("motor_port_idx must be size 12 if provided");
    }
    for (int i = 0; i < NUM_JOINTS; ++i) {
      explicit_port[i] = motor_ports[i].as<int>();
      if (explicit_port[i] != 0 && explicit_port[i] != 1) {
        throw std::runtime_error("motor_port_idx values must be 0 or 1");
      }
    }
  }

  std::array<bool, NUM_JOINTS + 1> seen{};
  seen.fill(false);
  for (int i = 0; i < NUM_JOINTS; ++i) {
    const int motor_id = cfg.joint_mapping[i];
    if (motor_id < 1 || motor_id > NUM_JOINTS) {
      throw std::runtime_error("joint_mapping motor_id out of range at index " +
                               std::to_string(i));
    }
    if (seen[motor_id]) {
      throw std::runtime_error("joint_mapping repeats motor_id " +
                               std::to_string(motor_id));
    }
    seen[motor_id] = true;

    cfg.motor_map[i].motor_id = motor_id;
    cfg.motor_map[i].port_idx =
        explicit_port[i] >= 0 ? explicit_port[i] : (motor_id >= 7 ? 1 : 0);
    cfg.motor_map[i].is_reversed = cfg.motor_is_reversed[i];
  }

  if (cfg.teleop_udp_port <= 0 || cfg.teleop_udp_port > 65535) {
    throw std::runtime_error("teleop_udp_port must be in [1, 65535]");
  }
  if (cfg.debug_print_interval <= 0) {
    cfg.debug_print_interval = 50;
  }

  return cfg;
}

} // namespace deploy
