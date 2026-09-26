/**
 * @file height_subscriber.h
 * @brief Subscriber for 11x7 terrain-to-body distance measurements.
 */
#pragma once

#include <array>
#include <atomic>
#include <mutex>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

#include "robot_config.h"

namespace deploy {

class HeightSubscriber : public rclcpp::Node {
public:
  HeightSubscriber(const std::string &topic, float nominal_base_height,
                   float measurement_scale, float measurement_offset);

  std::array<float, NUM_HEIGHT_POINTS> get_distances() const;
  bool is_ready() const { return received_.load(std::memory_order_acquire); }
  uint64_t msg_count() const {
    return msg_count_.load(std::memory_order_relaxed);
  }

private:
  void callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);

  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr sub_;
  mutable std::mutex mutex_;
  std::array<float, NUM_HEIGHT_POINTS> distances_{};
  float measurement_scale_ = 1.0f;
  float measurement_offset_ = 0.0f;
  std::atomic<bool> received_{false};
  std::atomic<uint64_t> msg_count_{0};
};

} // namespace deploy
