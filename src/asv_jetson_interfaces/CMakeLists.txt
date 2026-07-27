cmake_minimum_required(VERSION 3.8)
project(asv_jetson_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/UEASVState.msg"
  "msg/TargetGroundTruth.msg"
  "msg/CameraFrame.msg"
  "msg/WorldState.msg"
  "msg/PredictedWorldState.msg"
  "msg/DecisionOutput.msg"
  "msg/ThrusterCommand.msg"
  "msg/SystemStatus.msg"
  "msg/RunContext.msg"
  "msg/TaskEmbedding.msg"
  "msg/VisualFeatures.msg"
  "msg/TaskFeatures.msg"
  "msg/TrajectoryCandidates.msg"
  "msg/WorldModelEvaluation.msg"
  "msg/SelectedTrajectory.msg"
  "msg/ModuleStatus.msg"
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
