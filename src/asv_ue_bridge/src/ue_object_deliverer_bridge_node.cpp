#include <rclcpp/rclcpp.hpp>
#include <rosgraph_msgs/msg/clock.hpp>
#include <std_msgs/msg/bool.hpp>

#include <asv_jetson_interfaces/msg/camera_frame.hpp>
#include <asv_jetson_interfaces/msg/target_ground_truth.hpp>
#include <asv_jetson_interfaces/msg/thruster_command.hpp>
#include <asv_jetson_interfaces/msg/ueasv_state.hpp>

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
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

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

    camera_encoding_ = declare_parameter<std::string>("camera_encoding", "jpeg");
    max_camera_bytes_ = declare_parameter<int>("max_camera_bytes", 8 * 1024 * 1024);
    log_raw_json_ = declare_parameter<bool>("log_raw_json", false);
    publish_clock_ = declare_parameter<bool>("publish_clock", true);

    if (port_ <= 0 || port_ > 65535) {
      throw std::runtime_error("port must be in 1..65535");
    }
    if (terminator_.empty()) {
      throw std::runtime_error("terminator must not be empty");
    }

    asv_state_pub_ = create_publisher<asv_jetson_interfaces::msg::UEASVState>(
      "/ue/asv_state", rclcpp::QoS(10).reliable());

    target_truth_pub_ =
      create_publisher<asv_jetson_interfaces::msg::TargetGroundTruth>(
      "/ue/target_ground_truth", rclcpp::QoS(10).reliable());

    camera_pub_ = create_publisher<asv_jetson_interfaces::msg::CameraFrame>(
      "/ue/camera_frame", rclcpp::QoS(2).best_effort());

    connected_pub_ = create_publisher<std_msgs::msg::Bool>(
      "/ue/connected", rclcpp::QoS(1).reliable().transient_local());

    clock_pub_ = create_publisher<rosgraph_msgs::msg::Clock>(
      "/clock", rclcpp::QoS(1).best_effort());

    command_sub_ =
      create_subscription<asv_jetson_interfaces::msg::ThrusterCommand>(
      "/ue/thruster_command",
      rclcpp::QoS(10).reliable(),
      [this](const asv_jetson_interfaces::msg::ThrusterCommand::SharedPtr message) {
        send_thruster_command(*message);
      });

    publish_connected(false);
    running_.store(true);
    server_thread_ = std::thread(&UeObjectDelivererBridgeNode::server_loop, this);

    RCLCPP_INFO(
      get_logger(),
      "Listening for UE5 ObjectDeliverer on %s:%d, terminator='%s'",
      listen_address_.c_str(), port_, terminator_.c_str());
  }

  ~UeObjectDelivererBridgeNode() override
  {
    running_.store(false);

    {
      std::lock_guard<std::mutex> lock(socket_mutex_);
      close_socket(client_fd_);
      close_socket(server_fd_);
    }

    if (server_thread_.joinable()) {
      server_thread_.join();
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
      const json * target_location = read_object(
        body, {"Target_Location", "TargetLocation", "target_location"});
      const json * asv_rotation = read_object(
        body, {"ASV_Rotation", "ASVRotation", "AsvRotation", "asv_rotation"});

      valid = valid && asv_location != nullptr && target_location != nullptr &&
        asv_rotation != nullptr;

      double asv_x = 0.0;
      double asv_y = 0.0;
      double asv_z = 0.0;
      double target_x = 0.0;
      double target_y = 0.0;
      double target_z = 0.0;
      double roll_degrees = 0.0;
      double pitch_degrees = 0.0;
      double yaw_degrees = 0.0;

      if (valid) {
        valid =
          read_number(*asv_location, {"X", "x"}, asv_x) &&
          read_number(*asv_location, {"Y", "y"}, asv_y) &&
          read_number(*asv_location, {"Z", "z"}, asv_z) &&
          read_number(*target_location, {"X", "x"}, target_x) &&
          read_number(*target_location, {"Y", "y"}, target_y) &&
          read_number(*target_location, {"Z", "z"}, target_z) &&
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

      // Make UE5 game time the ROS 2 time source before publishing any data
      // produced by this simulation step.
      publish_simulation_clock(stamp_us);

      asv_jetson_interfaces::msg::UEASVState state;
      state.stamp_us = stamp_us;
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

      asv_jetson_interfaces::msg::TargetGroundTruth target;
      target.stamp_us = stamp_us;
      target.position_x = target_x * position_scale_;
      target.position_y = target_y * position_scale_;
      target.position_z = target_z * position_scale_;
      target.valid = true;
      target_truth_pub_->publish(target);

      publish_optional_camera(body, stamp_us);
    } catch (const std::exception & exception) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Failed to parse UE5 JSON: %s", exception.what());
    }
  }

  void publish_optional_camera(const json & body, int64_t stamp_us)
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

  void send_thruster_command(
    const asv_jetson_interfaces::msg::ThrusterCommand & command)
  {
    const json response = {
      {"Stamp_Us", command.stamp_us},
      {"Left_Thruster", command.left_thruster},
      {"Right_Thruster", command.right_thruster},
      {"Valid", command.valid}
    };

    std::string payload = response.dump() + terminator_;
    payload.push_back('\0');

    std::lock_guard<std::mutex> lock(socket_mutex_);
    if (client_fd_ < 0) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Dropping thruster command: UE5 client is not connected");
      return;
    }

    std::size_t sent_total = 0;
    while (sent_total < payload.size()) {
      const ssize_t sent = ::send(
        client_fd_,
        payload.data() + sent_total,
        payload.size() - sent_total,
        MSG_NOSIGNAL);

      if (sent < 0 && errno == EINTR) {
        continue;
      }
      if (sent <= 0) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "Failed to send command to UE5: %s", std::strerror(errno));
        return;
      }
      sent_total += static_cast<std::size_t>(sent);
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "Sent to UE5: bytes=%zu, payload=%s",
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
  std::string camera_encoding_;
  int max_camera_bytes_{8 * 1024 * 1024};
  bool log_raw_json_{false};
  bool publish_clock_{true};

  std::atomic<bool> running_{false};
  std::thread server_thread_;
  std::mutex socket_mutex_;
  int server_fd_{-1};
  int client_fd_{-1};

  rclcpp::Publisher<asv_jetson_interfaces::msg::UEASVState>::SharedPtr asv_state_pub_;
  rclcpp::Publisher<asv_jetson_interfaces::msg::TargetGroundTruth>::SharedPtr
    target_truth_pub_;
  rclcpp::Publisher<asv_jetson_interfaces::msg::CameraFrame>::SharedPtr camera_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr connected_pub_;
  rclcpp::Publisher<rosgraph_msgs::msg::Clock>::SharedPtr clock_pub_;
  rclcpp::Subscription<asv_jetson_interfaces::msg::ThrusterCommand>::SharedPtr command_sub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<UeObjectDelivererBridgeNode>());
  rclcpp::shutdown();
  return 0;
}
