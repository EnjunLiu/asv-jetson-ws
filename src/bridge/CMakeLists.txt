cmake_minimum_required(VERSION 3.8)
project(bridge)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_target_dependencies REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosgraph_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(interfaces REQUIRED)
find_package(nlohmann_json REQUIRED)

add_executable(
  bridge_node
  src/bridge_node.cpp
)
if(COMMAND ament_target_dependencies)
  ament_target_dependencies(
    bridge_node
    rclcpp
    rosgraph_msgs
    std_msgs
    interfaces
  )
else()
  # ROS distributions that removed the legacy helper still export the
  # package targets; linking those targets preserves include paths and ABI.
  target_link_libraries(
    bridge_node
    rclcpp::rclcpp
    rosgraph_msgs::rosgraph_msgs
    std_msgs::std_msgs
    interfaces::interfaces
  )
endif()
target_link_libraries(
  bridge_node
  nlohmann_json::nlohmann_json
  pthread
)

install(TARGETS bridge_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

ament_package()
