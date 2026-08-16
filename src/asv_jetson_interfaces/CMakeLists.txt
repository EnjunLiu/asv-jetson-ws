cmake_minimum_required(VERSION 3.8)
project(asv_jetson_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  # Structured entity observations shared by UE5, perception, and tracking.
  "msg/ASVState.msg"
  "msg/Entity.msg"
  "msg/EntityArray.msg"
  "msg/CameraFrame.msg"
  "msg/DesiredDisplacement.msg"
  # UE5-only kinematic execution bypasses the physical controller. The
  # ObjectDeliverer bridge sends one body-frame displacement per source frame.
  "msg/UESetpoint.msg"
  "msg/TaskEmbedding.msg"
  "msg/EntityFeatures.msg"
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
