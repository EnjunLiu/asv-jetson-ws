cmake_minimum_required(VERSION 3.8)
project(asv_ue_bridge)

include(CTest)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosgraph_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(asv_jetson_interfaces REQUIRED)
find_package(nlohmann_json REQUIRED)

add_executable(
  ue_object_deliverer_bridge_node
  src/ue_object_deliverer_bridge_node.cpp
)
ament_target_dependencies(
  ue_object_deliverer_bridge_node
  rclcpp
  rosgraph_msgs
  std_msgs
  asv_jetson_interfaces
)
target_link_libraries(
  ue_object_deliverer_bridge_node
  nlohmann_json::nlohmann_json
  pthread
)

install(TARGETS ue_object_deliverer_bridge_node DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS scripts/validate_ue_packet.py DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(
    NAME ue_packet_validator
    COMMAND
      ${Python3_EXECUTABLE} -m unittest discover
      -s ${CMAKE_CURRENT_SOURCE_DIR}/test
      -p test_validate_ue_packet.py
  )
endif()

ament_package()
