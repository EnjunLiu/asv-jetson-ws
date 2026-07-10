cmake_minimum_required(VERSION 3.8)
project(asv_control_manager)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(asv_interfaces REQUIRED)
find_package(asv_jetson_interfaces REQUIRED)

add_executable(control_input_mux_node src/control_input_mux_node.cpp)
ament_target_dependencies(
  control_input_mux_node rclcpp asv_interfaces asv_jetson_interfaces)

add_executable(safety_supervisor_node src/safety_supervisor_node.cpp)
ament_target_dependencies(safety_supervisor_node rclcpp std_msgs asv_interfaces)

add_executable(thruster_allocator_node src/thruster_allocator_node.cpp)
ament_target_dependencies(
  thruster_allocator_node rclcpp asv_interfaces asv_jetson_interfaces)

add_executable(esp32_param_manager_node src/esp32_param_manager_node.cpp)
ament_target_dependencies(esp32_param_manager_node rclcpp rcl_interfaces)

add_executable(system_monitor_node src/system_monitor_node.cpp)
ament_target_dependencies(system_monitor_node rclcpp std_msgs asv_interfaces asv_jetson_interfaces)

install(
  TARGETS
    control_input_mux_node
    safety_supervisor_node
    thruster_allocator_node
    esp32_param_manager_node
    system_monitor_node
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
ament_package()
