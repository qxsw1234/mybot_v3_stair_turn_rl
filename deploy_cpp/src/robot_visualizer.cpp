/**
 * @file robot_visualizer.cpp
 * @brief RViz visualization implementation.
 */

#include "robot_visualizer.h"

namespace deploy {

RobotVisualizer::RobotVisualizer(
    rclcpp::Node *node,
    const std::array<std::string, NUM_JOINTS> &joint_names) {
  clock_ = node->get_clock();
  pub_ =
      node->create_publisher<sensor_msgs::msg::JointState>("/joint_states", 10);
  target_pub_ = node->create_publisher<sensor_msgs::msg::JointState>(
      "/policy_joint_targets", 10);

  // Build joint name list matching URDF joint names
  joint_names_.reserve(NUM_JOINTS);
  for (int i = 0; i < NUM_JOINTS; ++i) {
    joint_names_.emplace_back(joint_names[i]);
  }

  RCLCPP_INFO(node->get_logger(),
              "RobotVisualizer: publishing to /joint_states and /policy_joint_targets");
}

void RobotVisualizer::publish_joint_states(
    const std::array<float, NUM_JOINTS> &dof_pos,
    const std::array<float, NUM_JOINTS> &dof_vel,
    const std::array<float, NUM_JOINTS> *dof_tau) {
  sensor_msgs::msg::JointState msg;
  msg.header.stamp = clock_->now();

  msg.name = joint_names_;
  msg.position.resize(NUM_JOINTS);
  msg.velocity.resize(NUM_JOINTS);
  if (dof_tau != nullptr) {
    msg.effort.resize(NUM_JOINTS);
  }

  for (int i = 0; i < NUM_JOINTS; ++i) {
    msg.position[i] = static_cast<double>(dof_pos[i]);
    msg.velocity[i] = static_cast<double>(dof_vel[i]);
    if (dof_tau != nullptr) {
      msg.effort[i] = static_cast<double>((*dof_tau)[i]);
    }
  }

  pub_->publish(msg);
}

void RobotVisualizer::publish_joint_targets(
    const std::array<float, NUM_JOINTS> &targets) {
  sensor_msgs::msg::JointState msg;
  msg.header.stamp = clock_->now();
  msg.name = joint_names_;
  msg.position.assign(targets.begin(), targets.end());
  target_pub_->publish(msg);
}

} // namespace deploy
