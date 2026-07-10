cmake_minimum_required(VERSION 3.8)
project(asv_planning)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(asv_jetson_interfaces REQUIRED)

add_executable(state_predictor_node src/state_predictor_node.cpp)
ament_target_dependencies(state_predictor_node rclcpp asv_jetson_interfaces)

add_executable(decision_node src/decision_node.cpp)
ament_target_dependencies(decision_node rclcpp asv_jetson_interfaces)

install(TARGETS state_predictor_node decision_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
ament_package()