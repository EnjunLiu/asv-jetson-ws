#include <rclcpp/rclcpp.hpp>

#include <asv_interfaces/msg/asv_wrench.hpp>
#include <std_msgs/msg/bool.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

class SafetySupervisorNode final : public rclcpp::Node
{
public:
  SafetySupervisorNode()
  : Node("safety_supervisor_node")
  {
    publish_rate_hz_ = declare_parameter<double>("publish_rate_hz", 10.0);
    wrench_timeout_sec_ = declare_parameter<double>("wrench_timeout_sec", 0.5);
    max_force_ = declare_parameter<double>("max_force", 50);
    max_moment_ = declare_parameter<double>("max_moment", 50);
    emergency_stop_ = declare_parameter<bool>("emergency_stop", false);

    if (!std::isfinite(publish_rate_hz_) || publish_rate_hz_ <= 0.0) {
      throw std::invalid_argument("publish_rate_hz must be finite and positive");
    }
    if (!std::isfinite(wrench_timeout_sec_) || wrench_timeout_sec_ < 0.0) {
      throw std::invalid_argument("wrench_timeout_sec must be finite and non-negative");
    }
    if (!std::isfinite(max_force_) || max_force_ < 0.0) {
      throw std::invalid_argument("max_force must be finite and non-negative");
    }
    if (!std::isfinite(max_moment_) || max_moment_ < 0.0) {
      throw std::invalid_argument("max_moment must be finite and non-negative");
    }

    output_pub_ = create_publisher<asv_interfaces::msg::ASVWrench>(
      "/control/safe_wrench", rclcpp::QoS(10).reliable());
    emergency_stop_pub_ = create_publisher<std_msgs::msg::Bool>(
      "/safety/emergency_stop",
      rclcpp::QoS(1).reliable().transient_local());

    connected_sub_ = create_subscription<std_msgs::msg::Bool>(
      "/ue/connected",
      rclcpp::QoS(1).reliable().transient_local(),
      [this](const std_msgs::msg::Bool::SharedPtr message) {
        ue_connected_ = message->data;
      });

    input_sub_ = create_subscription<asv_interfaces::msg::ASVWrench>(
      "/control/asv_wrench",
      rclcpp::QoS(10).reliable(),
      [this](const asv_interfaces::msg::ASVWrench::SharedPtr message) {
        latest_wrench_ = *message;
        have_wrench_ = true;
        wrench_receive_time_ = std::chrono::steady_clock::now();
      });

    const auto period = std::chrono::duration<double>(1.0 / publish_rate_hz_);
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      [this]() { publish_safe_wrench(); });
  }

private:
  void publish_safe_wrench()
  {
    // Reading the parameter here makes `ros2 param set ... emergency_stop`
    // effective without restarting this safety-critical node.
    get_parameter("emergency_stop", emergency_stop_);

    std_msgs::msg::Bool emergency_stop_state;
    emergency_stop_state.data = emergency_stop_;
    emergency_stop_pub_->publish(emergency_stop_state);

    const auto steady_now = std::chrono::steady_clock::now();
    const double age = have_wrench_ ?
      std::chrono::duration<double>(steady_now - wrench_receive_time_).count() :
      std::numeric_limits<double>::infinity();

    asv_interfaces::msg::ASVWrench output;
    output.seq = latest_wrench_.seq;
    output.stamp_us = latest_wrench_.stamp_us;

    const bool valid =
      !emergency_stop_ &&
      ue_connected_ &&
      have_wrench_ &&
      latest_wrench_.valid &&
      age <= wrench_timeout_sec_ &&
      std::isfinite(latest_wrench_.force) &&
      std::isfinite(latest_wrench_.moment);

    if (valid) {
      output.force = static_cast<float>(std::clamp(
        static_cast<double>(latest_wrench_.force), -max_force_, max_force_));
      output.moment = static_cast<float>(std::clamp(
        static_cast<double>(latest_wrench_.moment), -max_moment_, max_moment_));
      output.valid = true;
    } else {
      output.force = 0.0F;
      output.moment = 0.0F;
      output.valid = false;
    }

    output_pub_->publish(output);
  }

  double publish_rate_hz_{10.0};
  double wrench_timeout_sec_{0.5};
  double max_force_{0.5};
  double max_moment_{0.1};
  bool emergency_stop_{false};

  bool ue_connected_{false};
  bool have_wrench_{false};
  asv_interfaces::msg::ASVWrench latest_wrench_;
  std::chrono::steady_clock::time_point wrench_receive_time_;

  rclcpp::Publisher<asv_interfaces::msg::ASVWrench>::SharedPtr output_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr emergency_stop_pub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr connected_sub_;
  rclcpp::Subscription<asv_interfaces::msg::ASVWrench>::SharedPtr input_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SafetySupervisorNode>());
  rclcpp::shutdown();
  return 0;
}
