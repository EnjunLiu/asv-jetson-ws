#include <rclcpp/rclcpp.hpp>
#include <rosgraph_msgs/msg/clock.hpp>
#include <std_msgs/msg/bool.hpp>

#include <asv_jetson_interfaces/msg/camera_frame.hpp>
#include <asv_jetson_interfaces/msg/asv_state.hpp>
#include <asv_jetson_interfaces/msg/entity.hpp>
#include <asv_jetson_interfaces/msg/entity_array.hpp>
#include <asv_jetson_interfaces/msg/ue_setpoint.hpp>

#include <nlohmann/json.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>

using json = nlohmann::json;

namespace
{
constexpr double kPi = 3.14159265358979323846;

const json * find_member(
  const json & object,
  std::initializer_list<const char *> names)
{
  if (!object.is_object()) {
    return nullptr;
  }

  for (const char * name : names) {
    const auto iter = object.find(name);
    if (iter != object.end()) {
      return &(*iter);
    }
  }
  return nullptr;
}

bool read_number(
  const json & object,
  std::initializer_list<const char *> names,
  double & output)
{
  const json * value = find_member(object, names);
  if (value == nullptr || !value->is_number()) {
    return false;
  }

  output = value->get<double>();
  return std::isfinite(output);
}

bool read_string(
  const json & object,
  std::initializer_list<const char *> names,
  std::string & output)
{
  const json * value = find_member(object, names);
  if (value == nullptr || !value->is_string()) {
    return false;
  }

  output = value->get<std::string>();
  return !output.empty();
}

bool read_bool(
  const json & object,
  std::initializer_list<const char *> names,
  bool & output)
{
  const json * value = find_member(object, names);
  if (value == nullptr || !value->is_boolean()) {
    return false;
  }

  output = value->get<bool>();
  return true;
}

bool read_int64(
  const json & object,
  std::initializer_list<const char *> names,
  int64_t & output)
{
  const json * value = find_member(object, names);
  if (value == nullptr) {
    return false;
  }

  if (value->is_number_unsigned()) {
    const uint64_t unsigned_value = value->get<uint64_t>();
    if (unsigned_value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      return false;
    }
    output = static_cast<int64_t>(unsigned_value);
    return true;
  }
  if (!value->is_number_integer()) {
    return false;
  }

  output = value->get<int64_t>();
  return true;
}

bool read_uint64(
  const json & object,
  std::initializer_list<const char *> names,
  uint64_t & output)
{
  const json * value = find_member(object, names);
  if (value == nullptr) {
    return false;
  }

  if (value->is_number_unsigned()) {
    output = value->get<uint64_t>();
    return true;
  }
  if (!value->is_number_integer()) {
    return false;
  }

  const int64_t signed_value = value->get<int64_t>();
  if (signed_value < 0) {
    return false;
  }
  output = static_cast<uint64_t>(signed_value);
  return true;
}

bool is_frame_padding(char value)
{
  return value == '\0' || value == ' ' || value == '\t' ||
         value == '\r' || value == '\n';
}

std::string trim_frame_padding(const std::string & packet)
{
  std::size_t begin = 0;
  while (begin < packet.size() && is_frame_padding(packet[begin])) {
    ++begin;
  }

  std::size_t end = packet.size();
  while (end > begin && is_frame_padding(packet[end - 1])) {
    --end;
  }

  return packet.substr(begin, end - begin);
}

const json * read_object(
  const json & object,
  std::initializer_list<const char *> names)
{
  const json * value = find_member(object, names);
  return value != nullptr && value->is_object() ? value : nullptr;
}

struct FrameMetadata
{
  std::string run_id;
  int64_t scene_seed{0};
  uint64_t frame_index{0};
};

}  // namespace

class UeObjectDelivererBridgeNode final : public rclcpp::Node
{
public:
  UeObjectDelivererBridgeNode()
  : Node("ue_object_deliverer_bridge_node")
  {
    listen_address_ = declare_parameter<std::string>("listen_address", "0.0.0.0");
    port_ = declare_parameter<int>("port", 8080);
    terminator_ = declare_parameter<std::string>("terminator", "__OD_END__");

    position_scale_ = declare_parameter<double>("position_scale", 0.01);
    velocity_scale_ = declare_parameter<double>("velocity_scale", 0.01);
    yaw_rate_scale_ = declare_parameter<double>("yaw_rate_scale", 1.0);
    yaw_rate_sign_ = declare_parameter<double>("yaw_rate_sign", 1.0);
    entity_frame_id_ = declare_parameter<std::string>("entity_frame_id", "base_link");
    entity_lateral_sign_ = declare_parameter<double>("entity_lateral_sign", -1.0);
    entity_vertical_sign_ = declare_parameter<double>("entity_vertical_sign", 1.0);
    max_entities_ = declare_parameter<int>("max_entities", 64);

    camera_encoding_ = declare_parameter<std::string>("camera_encoding", "jpeg");
    max_camera_bytes_ = declare_parameter<int>("max_camera_bytes", 8 * 1024 * 1024);
    log_raw_json_ = declare_parameter<bool>("log_raw_json", false);
    publish_clock_ = declare_parameter<bool>("publish_clock", true);
    outbound_command_mode_ =
      declare_parameter<std::string>("outbound_command_mode", "kinematic");
    kinematic_position_scale_ =
      declare_parameter<double>("kinematic_position_scale", 100.0);
    kinematic_lateral_sign_ =
      declare_parameter<double>("kinematic_lateral_sign", -1.0);
    execution_address_ =
      declare_parameter<std::string>("execution_address", "");
    execution_port_ = declare_parameter<int>("execution_port", 8081);

    if (port_ <= 0 || port_ > 65535) {
      throw std::runtime_error("port must be in 1..65535");
    }
    if (terminator_.empty()) {
      throw std::runtime_error("terminator must not be empty");
    }
    if (entity_frame_id_.empty()) {
      throw std::runtime_error("entity_frame_id must not be empty");
    }
    if (!std::isfinite(entity_lateral_sign_) || entity_lateral_sign_ == 0.0 ||
      !std::isfinite(entity_vertical_sign_) || entity_vertical_sign_ == 0.0)
    {
      throw std::runtime_error("entity coordinate signs must be finite and non-zero");
    }
    if (max_entities_ <= 0) {
      throw std::runtime_error("max_entities must be positive");
    }
    if (
      outbound_command_mode_ != "kinematic" &&
      outbound_command_mode_ != "disabled")
    {
      throw std::runtime_error(
              "outbound_command_mode must be kinematic or disabled");
    }
    if (
      !std::isfinite(kinematic_position_scale_) ||
      kinematic_position_scale_ <= 0.0)
    {
      throw std::runtime_error(
              "kinematic_position_scale must be positive and finite");
    }
    if (
      !std::isfinite(kinematic_lateral_sign_) ||
      kinematic_lateral_sign_ == 0.0)
    {
      throw std::runtime_error(
              "kinematic_lateral_sign must be finite and non-zero");
    }

    asv_state_pub_ = create_publisher<asv_jetson_interfaces::msg::ASVState>(
      "/ue/asv_state", rclcpp::QoS(10).reliable());

    camera_pub_ = create_publisher<asv_jetson_interfaces::msg::CameraFrame>(
      "/ue/camera_frame", rclcpp::QoS(2).best_effort());

    entities_pub_ = create_publisher<asv_jetson_interfaces::msg::EntityArray>(
      "/ue/entities", rclcpp::QoS(10).reliable());

    connected_pub_ = create_publisher<std_msgs::msg::Bool>(
      "/ue/connected", rclcpp::QoS(1).reliable().transient_local());

    clock_pub_ = create_publisher<rosgraph_msgs::msg::Clock>(
      "/clock", rclcpp::QoS(1).best_effort());

    if (outbound_command_mode_ == "kinematic") {
      kinematic_setpoint_sub_ =
        create_subscription<asv_jetson_interfaces::msg::UESetpoint>(
          "/ue/kinematic_setpoint",
          rclcpp::QoS(10).reliable(),
          [this](
            const asv_jetson_interfaces::msg::UESetpoint::SharedPtr message)
          {
            RCLCPP_INFO(
              get_logger(),
              "kinematic setpoint received stamp=%ld valid=%d",
              static_cast<long>(message->stamp_us),
              static_cast<int>(message->valid));
            send_kinematic_setpoint(*message);
          });
    }

    publish_connected(false);
    running_.store(true);
    server_thread_ = std::thread(&UeObjectDelivererBridgeNode::server_loop, this);

    // Optional C++ kinematic executor connection (headless closed loop):
    // the UE5 Connection blueprint does not apply kinematic setpoints when
    // running headless, so the bridge can deliver setpoints to the EDGE
    // executor socket instead.  Enabled when execution_address is set.
    if (!execution_address_.empty()) {
      execution_thread_ =
        std::thread(&UeObjectDelivererBridgeNode::execution_loop, this);
    }

    RCLCPP_INFO(
      get_logger(),
      "Listening for UE5 ObjectDeliverer on %s:%d, terminator='%s', "
      "outbound_command_mode='%s'%s",
      listen_address_.c_str(), port_, terminator_.c_str(),
      outbound_command_mode_.c_str(),
      execution_address_.empty() ? "" :
        (std::string("; kinematic executor -> ") + execution_address_ +
         ":" + std::to_string(execution_port_)).c_str());
  }

  ~UeObjectDelivererBridgeNode() override
  {
    running_.store(false);

    {
      std::lock_guard<std::mutex> lock(socket_mutex_);
      close_socket(client_fd_);
      close_socket(server_fd_);
      close_socket(execution_fd_);
    }

    if (server_thread_.joinable()) {
      server_thread_.join();
    }
    if (execution_thread_.joinable()) {
      execution_thread_.join();
    }
  }

private:
  static void close_socket(int & file_descriptor)
  {
    if (file_descriptor >= 0) {
      ::shutdown(file_descriptor, SHUT_RDWR);
      ::close(file_descriptor);
      file_descriptor = -1;
    }
  }

  void execution_loop()
  {
    while (running_.load()) {
      const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
      if (fd < 0) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        continue;
      }
      sockaddr_in address{};
      address.sin_family = AF_INET;
      address.sin_port = htons(static_cast<uint16_t>(execution_port_));
      if (::inet_pton(AF_INET, execution_address_.c_str(),
                      &address.sin_addr) != 1) {
        RCLCPP_ERROR(
          get_logger(), "Invalid execution address: %s",
          execution_address_.c_str());
        ::close(fd);
        return;
      }
      if (::connect(fd, reinterpret_cast<sockaddr *>(&address),
                    sizeof(address)) < 0) {
        ::close(fd);
        std::this_thread::sleep_for(std::chrono::seconds(1));
        continue;
      }
      {
        std::lock_guard<std::mutex> lock(socket_mutex_);
        close_socket(execution_fd_);
        execution_fd_ = fd;
      }
      RCLCPP_INFO(
        get_logger(), "Connected to UE5 kinematic executor %s:%d",
        execution_address_.c_str(), execution_port_);

      // Keep the connection open; reconnect on close.
      while (running_.load()) {
        char probe;
        const ssize_t n = ::recv(fd, &probe, 1, MSG_PEEK);
        if (n == 0) {
          break;
        }
        if (n < 0 && errno != EINTR && errno != EAGAIN
            && errno != EWOULDBLOCK) {
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
      }
      {
        std::lock_guard<std::mutex> lock(socket_mutex_);
        close_socket(execution_fd_);
      }
    }
  }

  void publish_connected(bool connected)
  {
    std_msgs::msg::Bool message;
    message.data = connected;
    connected_pub_->publish(message);
  }

  void publish_simulation_clock(int64_t stamp_us)
  {
    if (!publish_clock_) {
      return;
    }

    rosgraph_msgs::msg::Clock message;
    message.clock.sec = static_cast<int32_t>(stamp_us / 1'000'000LL);
    message.clock.nanosec = static_cast<uint32_t>(
      (stamp_us % 1'000'000LL) * 1'000LL);
    clock_pub_->publish(message);
  }

  void server_loop()
  {
    const int local_server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (local_server_fd < 0) {
      RCLCPP_ERROR(get_logger(), "socket() failed: %s", std::strerror(errno));
      return;
    }

    {
      std::lock_guard<std::mutex> lock(socket_mutex_);
      server_fd_ = local_server_fd;
    }

    int reuse = 1;
    (void)::setsockopt(
      local_server_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<uint16_t>(port_));

    if (::inet_pton(AF_INET, listen_address_.c_str(), &address.sin_addr) != 1) {
      RCLCPP_ERROR(
        get_logger(), "Invalid IPv4 listen address: %s", listen_address_.c_str());
      return;
    }

    if (::bind(
        local_server_fd,
        reinterpret_cast<sockaddr *>(&address),
        sizeof(address)) < 0)
    {
      RCLCPP_ERROR(get_logger(), "bind() failed: %s", std::strerror(errno));
      return;
    }

    if (::listen(local_server_fd, 1) < 0) {
      RCLCPP_ERROR(get_logger(), "listen() failed: %s", std::strerror(errno));
      return;
    }

    while (running_.load()) {
      sockaddr_in client_address{};
      socklen_t client_address_size = sizeof(client_address);
      const int accepted_fd = ::accept(
        local_server_fd,
        reinterpret_cast<sockaddr *>(&client_address),
        &client_address_size);

      if (accepted_fd < 0) {
        if (running_.load()) {
          RCLCPP_WARN(get_logger(), "accept() failed: %s", std::strerror(errno));
        }
        continue;
      }

      int no_delay = 1;
      (void)::setsockopt(
        accepted_fd, IPPROTO_TCP, TCP_NODELAY, &no_delay, sizeof(no_delay));

      {
        std::lock_guard<std::mutex> lock(socket_mutex_);
        close_socket(client_fd_);
        client_fd_ = accepted_fd;
      }

      char client_ip[INET_ADDRSTRLEN]{};
      (void)::inet_ntop(
        AF_INET, &client_address.sin_addr, client_ip, sizeof(client_ip));

      RCLCPP_INFO(
        get_logger(), "UE5 connected from %s:%u",
        client_ip, static_cast<unsigned int>(ntohs(client_address.sin_port)));
      publish_connected(true);

      receive_loop(accepted_fd);

      {
        std::lock_guard<std::mutex> lock(socket_mutex_);
        if (client_fd_ == accepted_fd) {
          close_socket(client_fd_);
        }
      }

      publish_connected(false);
      RCLCPP_WARN(get_logger(), "UE5 disconnected");
    }
  }

  void receive_loop(int file_descriptor)
  {
    std::string buffer;
    buffer.reserve(64 * 1024);
    char chunk[8192];

    while (running_.load()) {
      const ssize_t received = ::recv(file_descriptor, chunk, sizeof(chunk), 0);
      if (received == 0) {
        return;
      }
      if (received < 0) {
        if (errno == EINTR) {
          continue;
        }
        return;
      }

      buffer.append(chunk, static_cast<std::size_t>(received));

      while (true) {
        const std::size_t end = buffer.find(terminator_);
        if (end == std::string::npos) {
          break;
        }

        const std::string packet = trim_frame_padding(buffer.substr(0, end));
        buffer.erase(0, end + terminator_.size());
        if (!packet.empty()) {
          process_json(packet);
        }
      }

      constexpr std::size_t kMaximumBuffer = 32U * 1024U * 1024U;
      if (buffer.size() > kMaximumBuffer) {
        RCLCPP_ERROR(
          get_logger(),
          "Receive buffer exceeded %zu bytes. Check the UE5 Terminate packet rule.",
          kMaximumBuffer);
        buffer.clear();
      }
    }
  }

  void process_json(const std::string & text)
  {
    try {
      if (log_raw_json_) {
        RCLCPP_INFO(get_logger(), "UE5 JSON: %s", text.c_str());
      }

      const json root = json::parse(text);
      const json & body =
        root.contains("Body") && root["Body"].is_object() ? root["Body"] : root;

      double simulation_time = 0.0;
      double surge_velocity = 0.0;
      double yaw_rate = 0.0;

      bool valid =
        read_number(body, {"Time", "time"}, simulation_time) &&
        read_number(
          body,
          {"Surge_Velocity", "SurgeVelocity", "surge_velocity"},
          surge_velocity) &&
        read_number(
          body,
          {"Angular_Velocity", "AngularVelocity", "angular_velocity", "YawRate"},
          yaw_rate);
      valid = valid && simulation_time >= 0.0;

      const json * asv_location = read_object(
        body, {"ASV_Location", "ASVLocation", "AsvLocation", "asv_location"});
      const json * asv_rotation = read_object(
        body, {"ASV_Rotation", "ASVRotation", "AsvRotation", "asv_rotation"});

      valid = valid && asv_location != nullptr && asv_rotation != nullptr;

      double asv_x = 0.0;
      double asv_y = 0.0;
      double asv_z = 0.0;
      double roll_degrees = 0.0;
      double pitch_degrees = 0.0;
      double yaw_degrees = 0.0;

      if (valid) {
        valid =
          read_number(*asv_location, {"X", "x"}, asv_x) &&
          read_number(*asv_location, {"Y", "y"}, asv_y) &&
          read_number(*asv_location, {"Z", "z"}, asv_z) &&
          read_number(*asv_rotation, {"Roll", "roll"}, roll_degrees) &&
          read_number(*asv_rotation, {"Pitch", "pitch"}, pitch_degrees) &&
          read_number(*asv_rotation, {"Yaw", "yaw"}, yaw_degrees);
      }

      if (!valid) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "UE5 JSON misses required fields. Set log_raw_json=true to inspect it.");
        return;
      }

      // UE5 sends Get Game Time in Seconds as a floating-point value.
      // Convert it to integer microseconds so every ROS message generated from
      // the same UE5 packet carries the same simulation-time timestamp.
      const int64_t stamp_us =
        static_cast<int64_t>(simulation_time * 1'000'000.0);

      FrameMetadata metadata;
      std::string metadata_detail;
      if (!validate_frame_metadata(body, metadata, metadata_detail)) {
        publish_invalid_entities(stamp_us, metadata, metadata_detail);
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "Rejected UE5 frame metadata: %s", metadata_detail.c_str());
        return;
      }

      // Make UE5 game time the ROS 2 time source before publishing any data
      // produced by this simulation step.
      publish_simulation_clock(stamp_us);

      asv_jetson_interfaces::msg::ASVState state;
      state.stamp_us = stamp_us;
      state.run_id = metadata.run_id;
      state.scene_seed = metadata.scene_seed;
      state.frame_index = metadata.frame_index;
      state.simulation_time = simulation_time;
      state.position_x = asv_x * position_scale_;
      state.position_y = asv_y * position_scale_;
      state.position_z = asv_z * position_scale_;
      state.roll = roll_degrees * kPi / 180.0;
      state.pitch = pitch_degrees * kPi / 180.0;
      state.yaw = yaw_degrees * kPi / 180.0;
      state.surge_velocity = surge_velocity * velocity_scale_;
      state.yaw_rate = yaw_rate_sign_ * yaw_rate * yaw_rate_scale_;
      state.valid = true;
      asv_state_pub_->publish(state);

      publish_optional_entities(body, stamp_us, metadata, metadata_detail);
      publish_optional_camera(body, stamp_us, metadata);
    } catch (const std::exception & exception) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Failed to parse UE5 JSON: %s", exception.what());
    }
  }

  bool validate_frame_metadata(
    const json & body,
    FrameMetadata & metadata,
    std::string & detail)
  {
    if (!read_string(body, {"Run_ID", "Run_Id", "RunId", "run_id"}, metadata.run_id)) {
      detail = "missing or empty Run_ID";
      return false;
    }
    if (!read_int64(
        body, {"Scene_Seed", "SceneSeed", "scene_seed"}, metadata.scene_seed))
    {
      detail = "Scene_Seed must be an integer";
      return false;
    }
    if (!read_uint64(
        body, {"Frame_Index", "FrameIndex", "frame_index"}, metadata.frame_index))
    {
      detail = "Frame_Index must be a non-negative integer";
      return false;
    }

    if (!have_last_frame_) {
      have_last_frame_ = true;
      last_run_id_ = metadata.run_id;
      last_scene_seed_ = metadata.scene_seed;
      last_frame_index_ = metadata.frame_index;
      detail = metadata.frame_index == 0 ?
        "ok" : "ok; joined run after frame 0";
      return true;
    }

    if (metadata.run_id != last_run_id_) {
      if (metadata.frame_index != 0) {
        detail = "new Run_ID must start at Frame_Index 0";
        return false;
      }
      last_run_id_ = metadata.run_id;
      last_scene_seed_ = metadata.scene_seed;
      last_frame_index_ = metadata.frame_index;
      detail = "ok";
      return true;
    }

    if (metadata.scene_seed != last_scene_seed_) {
      detail = "Scene_Seed changed within Run_ID";
      return false;
    }
    if (metadata.frame_index <= last_frame_index_) {
      detail = "duplicate or out-of-order Frame_Index";
      return false;
    }

    if (metadata.frame_index > last_frame_index_ + 1) {
      const uint64_t dropped = metadata.frame_index - last_frame_index_ - 1;
      detail = "ok; frame gap: " + std::to_string(dropped);
      RCLCPP_WARN(
        get_logger(),
        "UE5 frame gap: run_id=%s previous=%llu current=%llu dropped=%llu",
        metadata.run_id.c_str(),
        static_cast<unsigned long long>(last_frame_index_),
        static_cast<unsigned long long>(metadata.frame_index),
        static_cast<unsigned long long>(dropped));
    } else {
      detail = "ok";
    }

    last_frame_index_ = metadata.frame_index;
    return true;
  }

  void publish_invalid_entities(
    int64_t stamp_us,
    const FrameMetadata & metadata,
    const std::string & detail)
  {
    asv_jetson_interfaces::msg::EntityArray output;
    output.stamp_us = stamp_us;
    output.run_id = metadata.run_id;
    output.scene_seed = metadata.scene_seed;
    output.frame_index = metadata.frame_index;
    output.frame_id = entity_frame_id_;
    output.source = "ue_truth";
    output.valid = false;
    output.detail = detail;
    entities_pub_->publish(output);
  }

  void publish_optional_entities(
    const json & body,
    int64_t stamp_us,
    const FrameMetadata & metadata,
    const std::string & metadata_detail)
  {
    asv_jetson_interfaces::msg::EntityArray output;
    output.stamp_us = stamp_us;
    output.run_id = metadata.run_id;
    output.scene_seed = metadata.scene_seed;
    output.frame_index = metadata.frame_index;
    output.frame_id = entity_frame_id_;
    output.source = "ue_truth";

    const json * entities = find_member(body, {"Entities", "entities"});
    if (entities == nullptr) {
      output.detail = "missing Entities";
      entities_pub_->publish(output);
      return;
    }
    if (!entities->is_array()) {
      output.detail = "Entities is not an array";
      entities_pub_->publish(output);
      return;
    }
    if (entities->size() > static_cast<std::size_t>(max_entities_)) {
      output.detail = "Entities exceeds max_entities";
      entities_pub_->publish(output);
      return;
    }

    output.entities.reserve(entities->size());
    std::unordered_set<std::string> entity_ids;

    for (std::size_t index = 0; index < entities->size(); ++index) {
      const json & source = (*entities)[index];
      if (!source.is_object()) {
        output.entities.clear();
        output.detail = "entity[" + std::to_string(index) + "] is not an object";
        entities_pub_->publish(output);
        return;
      }

      asv_jetson_interfaces::msg::Entity entity;
      bool is_target = false;
      bool visible = false;
      const bool metadata_valid =
        read_string(source, {"Entity_Id", "EntityId", "entity_id"}, entity.entity_id) &&
        read_string(source, {"Class", "class"}, entity.class_name) &&
        read_string(source, {"Color", "color"}, entity.color) &&
        read_bool(source, {"Is_Target", "IsTarget", "is_target"}, is_target) &&
        read_bool(source, {"Visible", "visible"}, visible);

      const json * position = read_object(
        source, {"RelativePosition", "Relative_Position", "relative_position"});
      const json * velocity = read_object(
        source, {"RelativeVelocity", "Relative_Velocity", "relative_velocity"});

      double relative_x = 0.0;
      double relative_y = 0.0;
      double relative_z = 0.0;
      double velocity_x = 0.0;
      double velocity_y = 0.0;
      double velocity_z = 0.0;
      const bool vectors_valid =
        position != nullptr && velocity != nullptr &&
        read_number(*position, {"X", "x"}, relative_x) &&
        read_number(*position, {"Y", "y"}, relative_y) &&
        read_number(*position, {"Z", "z"}, relative_z) &&
        read_number(*velocity, {"X", "x"}, velocity_x) &&
        read_number(*velocity, {"Y", "y"}, velocity_y) &&
        read_number(*velocity, {"Z", "z"}, velocity_z);

      if (!metadata_valid || !vectors_valid) {
        output.entities.clear();
        output.detail = "entity[" + std::to_string(index) + "] has invalid fields";
        entities_pub_->publish(output);
        return;
      }
      if (!entity_ids.insert(entity.entity_id).second) {
        output.entities.clear();
        output.detail = "duplicate Entity_Id: " + entity.entity_id;
        entities_pub_->publish(output);
        return;
      }

      entity.is_target = is_target;
      entity.visible = visible;
      entity.relative_x = relative_x * position_scale_;
      entity.relative_y = entity_lateral_sign_ * relative_y * position_scale_;
      entity.relative_z = entity_vertical_sign_ * relative_z * position_scale_;
      entity.relative_velocity_x = velocity_x * velocity_scale_;
      entity.relative_velocity_y =
        entity_lateral_sign_ * velocity_y * velocity_scale_;
      entity.relative_velocity_z =
        entity_vertical_sign_ * velocity_z * velocity_scale_;
      entity.valid = true;
      entity.source = "ue_truth";
      entity.confidence = 1.0F;
      entity.velocity_valid = true;
      output.entities.push_back(std::move(entity));
    }

    output.valid = true;
    output.detail = metadata_detail;
    entities_pub_->publish(output);
  }

  void publish_optional_camera(
    const json & body,
    int64_t stamp_us,
    const FrameMetadata & metadata)
  {
    const json * camera = find_member(
      body, {"Camera_Capture", "CameraCapture", "camera_capture"});

    if (camera == nullptr || !camera->is_array() || camera->empty()) {
      return;
    }

    if (camera->size() > static_cast<std::size_t>(max_camera_bytes_)) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Camera payload has %zu bytes, limit=%d",
        camera->size(), max_camera_bytes_);
      return;
    }

    asv_jetson_interfaces::msg::CameraFrame frame;
    frame.stamp_us = stamp_us;
    frame.run_id = metadata.run_id;
    frame.scene_seed = metadata.scene_seed;
    frame.frame_index = metadata.frame_index;
    frame.encoding = camera_encoding_;
    frame.data.reserve(camera->size());

    for (const auto & value : *camera) {
      if (!value.is_number_integer()) {
        frame.valid = false;
        camera_pub_->publish(frame);
        return;
      }
      const int integer_value = value.get<int>();
      frame.data.push_back(static_cast<uint8_t>(std::clamp(integer_value, 0, 255)));
    }

    frame.valid = !frame.data.empty();
    camera_pub_->publish(frame);
  }

  void send_kinematic_setpoint(
    const asv_jetson_interfaces::msg::UESetpoint & command)
  {
    const bool finite_step =
      std::isfinite(command.step_dt) && command.step_dt > 0.0F &&
      std::isfinite(command.delta_x_m) &&
      std::isfinite(command.delta_y_m);
    const bool executable =
      command.valid && finite_step && command.frame_id == "base_link";
    const bool hold_position = command.hold_position || !executable;
    const double delta_x_cm = hold_position ? 0.0 :
      static_cast<double>(command.delta_x_m) * kinematic_position_scale_;
    const double delta_y_cm = hold_position ? 0.0 :
      static_cast<double>(command.delta_y_m) *
      kinematic_position_scale_ * kinematic_lateral_sign_;
    const std::string reason = finite_step ? command.reason :
      "BRIDGE_REJECTED_NONFINITE_OR_INVALID_STEP";

    const json response = {
      {"Command_Type", "Kinematic_Setpoint"},
      {"Stamp_Us", command.stamp_us},
      {"Source_Stamp_Us", command.source_stamp_us},
      {"Run_ID", command.run_id},
      {"Scene_Seed", command.scene_seed},
      {"Source_Frame_Index", command.source_frame_index},
      {"Sequence", command.sequence},
      {"Frame_ID", "ue_actor_local"},
      {"Source_Model_Version", command.source_model_version},
      {"Step_Dt", finite_step ? command.step_dt : 0.0F},
      {"Delta_X_Cm", delta_x_cm},
      {"Delta_Y_Cm", delta_y_cm},
      {"Hold_Position", hold_position},
      {"Valid", executable},
      {"Reason", reason}
    };

    if (!execution_address_.empty()) {
      // Headless closed loop: deliver to the UE5 C++ executor (the
      // blueprint does not apply kinematic setpoints headless).
      send_json_payload_to(response, execution_fd_, "kinematic setpoint");
      return;
    }
    send_json_payload(response, "kinematic setpoint");
  }

  void send_json_payload(const json & response, const char * label)
  {
    send_json_payload_to(response, client_fd_, label);
  }

  void send_json_payload_to(
    const json & response, int file_descriptor, const char * label)
  {
    std::string payload = response.dump() + terminator_;
    payload.push_back('\0');

    std::lock_guard<std::mutex> lock(socket_mutex_);
    if (file_descriptor < 0) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Dropping %s: UE5 client is not connected", label);
      return;
    }

    std::size_t sent_total = 0;
    while (sent_total < payload.size()) {
      const ssize_t sent = ::send(
        file_descriptor,
        payload.data() + sent_total,
        payload.size() - sent_total,
        MSG_NOSIGNAL);

      if (sent < 0 && errno == EINTR) {
        continue;
      }
      if (sent <= 0) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "Failed to send %s to UE5: %s", label, std::strerror(errno));
        return;
      }
      sent_total += static_cast<std::size_t>(sent);
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "Sent %s to UE5: bytes=%zu, payload=%s",
        label,
        sent_total,
        payload.c_str());
    }
  }

  std::string listen_address_;
  int port_{8080};
  std::string terminator_;
  double position_scale_{0.01};
  double velocity_scale_{0.01};
  double yaw_rate_scale_{1.0};
  double yaw_rate_sign_{1.0};
  std::string entity_frame_id_{"base_link"};
  double entity_lateral_sign_{-1.0};
  double entity_vertical_sign_{1.0};
  int max_entities_{64};
  std::string camera_encoding_;
  int max_camera_bytes_{8 * 1024 * 1024};
  bool log_raw_json_{false};
  bool publish_clock_{true};
  std::string outbound_command_mode_{"kinematic"};
  double kinematic_position_scale_{100.0};
  double kinematic_lateral_sign_{-1.0};

  // Optional UE5 C++ kinematic executor (headless closed loop).
  std::string execution_address_;
  int execution_port_{8081};
  int execution_fd_{-1};
  std::thread execution_thread_;

  std::atomic<bool> running_{false};
  std::thread server_thread_;
  std::mutex socket_mutex_;
  int server_fd_{-1};
  int client_fd_{-1};
  bool have_last_frame_{false};
  std::string last_run_id_;
  int64_t last_scene_seed_{0};
  uint64_t last_frame_index_{0};

  rclcpp::Publisher<asv_jetson_interfaces::msg::ASVState>::SharedPtr asv_state_pub_;
  rclcpp::Publisher<asv_jetson_interfaces::msg::CameraFrame>::SharedPtr camera_pub_;
  rclcpp::Publisher<asv_jetson_interfaces::msg::EntityArray>::SharedPtr entities_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr connected_pub_;
  rclcpp::Publisher<rosgraph_msgs::msg::Clock>::SharedPtr clock_pub_;
  rclcpp::Subscription<asv_jetson_interfaces::msg::UESetpoint>::SharedPtr
    kinematic_setpoint_sub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<UeObjectDelivererBridgeNode>());
  rclcpp::shutdown();
  return 0;
}
