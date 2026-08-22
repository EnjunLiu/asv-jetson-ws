cmake_minimum_required(VERSION 3.8)
project(interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  # UE5、感知和跟踪共用的结构化实体观测。
  "msg/ASVState.msg"
  "msg/Entity.msg"
  "msg/EntityArray.msg"
  "msg/CameraFrame.msg"
  "msg/DesiredDisplacement.msg"
  "msg/TaskEmbedding.msg"
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
