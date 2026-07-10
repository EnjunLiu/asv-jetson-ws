cmake_minimum_required(VERSION 3.8)
project(asv_tools)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(asv_interfaces REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Threads REQUIRED)

add_executable(fake_esp32_wrench_node src/fake_esp32_wrench_node.cpp)
ament_target_dependencies(fake_esp32_wrench_node rclcpp asv_interfaces)

add_executable(fake_ue_client src/fake_ue_client.cpp)
target_link_libraries(fake_ue_client nlohmann_json::nlohmann_json Threads::Threads)

install(TARGETS fake_esp32_wrench_node fake_ue_client DESTINATION lib/${PROJECT_NAME})
ament_package()
