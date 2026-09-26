/**
 * @file height_subscriber.cpp
 * @brief Height measurement subscriber implementation.
 */

#include "height_subscriber.h"

#include <algorithm>
#include <functional>

namespace deploy {

HeightSubscriber::HeightSubscriber(const std::string &topic,
                                   float nominal_base_height,
                                   float measurement_scale,
                                   float measurement_offset)
    : Node("height_subscriber"), measurement_scale_(measurement_scale),
      measurement_offset_(measurement_offset) {
  distances_.fill(nominal_base_height);

  rclcpp::QoS qos(rclcpp::KeepLast(1));
  qos.best_effort();
  qos.durability_volatile();

  sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
      topic, qos, std::bind(&HeightSubscriber::callback, this,
                            std::placeholders::_1));

  RCLCPP_INFO(get_logger(),
              "Subscribing height measurements: %s (distance = raw * %.3f + %.3f)",
              topic.c_str(), measurement_scale_, measurement_offset_);
}

void HeightSubscriber::callback(
    const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
  if (msg->data.size() != NUM_HEIGHT_POINTS) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "Height msg has %zu floats, expected %d",
                         msg->data.size(), NUM_HEIGHT_POINTS);
    return;
  }

  if (msg->layout.dim.size() >= 2) {
    const auto &dx = msg->layout.dim[0];
    const auto &dy = msg->layout.dim[1];
    if (dx.size != NUM_HEIGHT_POINTS_X || dy.size != NUM_HEIGHT_POINTS_Y) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "Height layout is %u x %u, expected %d x %d",
                           dx.size, dy.size, NUM_HEIGHT_POINTS_X,
                           NUM_HEIGHT_POINTS_Y);
    }
  }

  std::lock_guard<std::mutex> lock(mutex_);
  for (int i = 0; i < NUM_HEIGHT_POINTS; ++i) {
    distances_[i] = msg->data[static_cast<size_t>(i)] * measurement_scale_ +
                    measurement_offset_;
  }
  received_.store(true, std::memory_order_release);
  msg_count_.fetch_add(1, std::memory_order_relaxed);
}

std::array<float, NUM_HEIGHT_POINTS> HeightSubscriber::get_distances() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return distances_;
}

} // namespace deploy
