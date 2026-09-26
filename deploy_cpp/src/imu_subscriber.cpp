/**
 * @file imu_subscriber.cpp
 * @brief Subscriber for /fast_livo2/state6_imu_prop.
 */

#include "imu_subscriber.h"

#include <cmath>
#include <functional>

namespace deploy {
namespace {
constexpr float kPi = 3.14159265358979323846f;
}

IMUSubscriber::IMUSubscriber(const std::string &topic,
                             float yaw_correction_deg)
    : Node("imu_subscriber") {
  const float yaw = yaw_correction_deg * kPi / 180.0f;
  yaw_cos_ = std::cos(yaw);
  yaw_sin_ = std::sin(yaw);

  rclcpp::QoS qos(rclcpp::KeepLast(1));
  qos.best_effort();
  qos.durability_volatile();

  sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
      topic, qos, std::bind(&IMUSubscriber::callback, this,
                            std::placeholders::_1));

  RCLCPP_INFO(get_logger(), "Subscribing IMU state: %s", topic.c_str());
}

void IMUSubscriber::callback(
    const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
  if (msg->data.size() < 6) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "IMU msg has %zu floats, expected 6",
                         msg->data.size());
    return;
  }

  const float wx = msg->data[0];
  const float wy = msg->data[1];
  const float gx = msg->data[3];
  const float gy = msg->data[4];

  std::lock_guard<std::mutex> lock(mutex_);
  ang_vel_[0] = yaw_cos_ * wx - yaw_sin_ * wy;
  ang_vel_[1] = yaw_sin_ * wx + yaw_cos_ * wy;
  ang_vel_[2] = msg->data[2];
  projected_gravity_[0] = yaw_cos_ * gx - yaw_sin_ * gy;
  projected_gravity_[1] = yaw_sin_ * gx + yaw_cos_ * gy;
  projected_gravity_[2] = msg->data[5];

  received_.store(true, std::memory_order_release);
  msg_count_.fetch_add(1, std::memory_order_relaxed);
}

std::array<float, 3> IMUSubscriber::get_ang_vel() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return ang_vel_;
}

std::array<float, 3> IMUSubscriber::get_projected_gravity() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return projected_gravity_;
}

} // namespace deploy
