cmake_minimum_required(VERSION 3.8)
project(asv_perception)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(asv_jetson_interfaces REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs calib3d aruco)

add_executable(perception_node src/perception_node.cpp)
ament_target_dependencies(perception_node rclcpp asv_jetson_interfaces)
target_include_directories(perception_node PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(perception_node ${OpenCV_LIBS})

install(TARGETS perception_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
ament_package()
