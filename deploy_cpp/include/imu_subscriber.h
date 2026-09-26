/**
 * @file imu_subscriber.h
 * @brief Subscriber for angular velocity and projected gravity.
 */
#pragma once

#include <array>
#include <atomic>
#include <mutex>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

namespace deploy {

class IMUSubscriber : public rclcpp::Node {
public:
  explicit IMUSubscriber(const std::string &topic,
                         float yaw_correction_deg = 0.0f);

  std::array<float, 3> get_ang_vel() const;
  std::array<float, 3> get_projected_gravity() const;
  bool is_ready() const { return received_.load(std::memory_order_acquire); }
  uint64_t msg_count() const {
    return msg_count_.load(std::memory_order_relaxed);
  }

private:
  void callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);

  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr sub_;
  mutable std::mutex mutex_;
  std::array<float, 3> ang_vel_ = {0.0f, 0.0f, 0.0f};
  std::array<float, 3> projected_gravity_ = {0.0f, 0.0f, -1.0f};
  float yaw_cos_ = 1.0f;
  float yaw_sin_ = 0.0f;
  std::atomic<bool> received_{false};
  std::atomic<uint64_t> msg_count_{0};
};

} // namespace deploy
