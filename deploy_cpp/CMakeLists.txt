cmake_minimum_required(VERSION 3.14)
project(deploy_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Torch REQUIRED)

set(UNITREE_SDK_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/Unitree_Motor")
set(UNITREE_SDK_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib")

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(UNITREE_SDK_LIB "${UNITREE_SDK_LIB_DIR}/libUnitreeMotorSDK_Arm64.so")
else()
  set(UNITREE_SDK_LIB "${UNITREE_SDK_LIB_DIR}/libUnitreeMotorSDK_Linux64.so")
endif()

if(NOT EXISTS "${UNITREE_SDK_LIB}")
  message(FATAL_ERROR "Unitree SDK library not found: ${UNITREE_SDK_LIB}")
endif()

include_directories(
  include
  ${UNITREE_SDK_INCLUDE_DIR}
)

set(COMMON_SOURCES
  src/robot_runtime_config.cpp
  src/motor_driver.cpp
  src/imu_subscriber.cpp
  src/height_subscriber.cpp
  src/cse_policy_runner.cpp
  src/keyboard_controller.cpp
  src/udp_controller.cpp
  src/state_machine.cpp
  src/robot_visualizer.cpp
)

set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib;/opt/libtorch/lib")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

add_executable(deploy_node
  ${COMMON_SOURCES}
  src/deploy_node.cpp
)
target_link_libraries(deploy_node
  ${UNITREE_SDK_LIB}
  ${TORCH_LIBRARIES}
  yaml-cpp
)
ament_target_dependencies(deploy_node
  rclcpp
  std_msgs
  sensor_msgs
)

add_executable(motor_debug_node
  src/robot_runtime_config.cpp
  src/motor_driver.cpp
  src/robot_visualizer.cpp
  src/motor_debug_node.cpp
)
target_link_libraries(motor_debug_node
  ${UNITREE_SDK_LIB}
  yaml-cpp
)
ament_target_dependencies(motor_debug_node
  rclcpp
  std_msgs
  sensor_msgs
)

install(TARGETS deploy_node motor_debug_node
  DESTINATION lib/${PROJECT_NAME}
)
install(FILES ${UNITREE_SDK_LIB}
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY launch config robot policy sim scripts
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

ament_package()
