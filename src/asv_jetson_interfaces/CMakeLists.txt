cmake_minimum_required(VERSION 3.8)
project(asv_jetson_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  # Existing UE5/perception/control path. PredictedWorldState remains because
  # the legacy state_predictor_node and decision_node still consume it.
  # UEEntity and UEEntityArray append provenance, instruction, bbox,
  # confidence, and velocity-validity fields with ROS-compatible defaults.
  "msg/UEASVState.msg"
  "msg/UEEntity.msg"
  "msg/UEEntityArray.msg"
  "msg/TargetGroundTruth.msg"
  "msg/CameraFrame.msg"
  "msg/WorldState.msg"
  "msg/PredictedWorldState.msg"
  "msg/DecisionOutput.msg"
  "msg/ThrusterCommand.msg"
  # UE5-only kinematic execution bypasses the physical controller. The
  # ObjectDeliverer bridge sends one body-frame displacement per source frame.
  "msg/UEKinematicSetpoint.msg"
  "msg/SystemStatus.msg"
  "msg/RunContext.msg"
  "msg/TaskEmbedding.msg"
  "msg/VisualFeatures.msg"
  "msg/TaskFeatures.msg"
  # Offline expert labels retain the full UE frame identity because adjacent
  # Frame_Index values may share a UE5 game-time timestamp.
  "msg/ExpertTrajectory.msg"
  # The VLA policy publishes exactly one body-frame trajectory. Candidate and
  # learned-world-model evaluation messages were intentionally removed.
  "msg/SelectedTrajectory.msg"
  "msg/ModuleStatus.msg"
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
