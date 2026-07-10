#include <rclcpp/rclcpp.hpp>

#include <asv_interfaces/msg/control_input.hpp>
#include <asv_jetson_interfaces/msg/decision_output.hpp>
#include <asv_jetson_interfaces/msg/ueasv_state.hpp>

#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

class ControlInputMuxNode final : public rclcpp::Node
{
public:
  ControlInputMuxNode()
  : Node("control_input_mux_node")
  {
    publish_rate_hz_ = declare_parameter<double>("publish_rate_hz", 10.0);
    state_timeout_sec_ = declare_parameter<double>("state_timeout_sec", 0.5);
    decision_timeout_sec_ = declare_parameter<double>("decision_timeout_sec", 0.5);

    if (!std::isfinite(publish_rate_hz_) || publish_rate_hz_ <= 0.0) {
      throw std::invalid_argument("publish_rate_hz must be finite and positive");
    }
    if (!std::isfinite(state_timeout_sec_) || state_timeout_sec_ < 0.0) {
      throw std::invalid_argument("state_timeout_sec must be finite and non-negative");
    }
    if (!std::isfinite(decision_timeout_sec_) || decision_timeout_sec_ < 0.0) {
      throw std::invalid_argument("decision_timeout_sec must be finite and non-negative");
    }

    output_pub_ = create_publisher<asv_interfaces::msg::ControlInput>(
      "/control/control_input", rclcpp::QoS(10).reliable());

    state_sub_ = create_subscription<asv_jetson_interfaces::msg::UEASVState>(
      "/ue/asv_state",
      rclcpp::QoS(10).reliable(),
      [this](const asv_jetson_interfaces::msg::UEASVState::SharedPtr message) {
        latest_state_ = *message;
        have_state_ = true;
        state_receive_time_ = std::chrono::steady_clock::now();
      });

    decision_sub_ =
      create_subscription<asv_jetson_interfaces::msg::DecisionOutput>(
      "/decision/output",
      rclcpp::QoS(10).reliable(),
      [this](const asv_jetson_interfaces::msg::DecisionOutput::SharedPtr message) {
        latest_decision_ = *message;
        have_decision_ = true;
        decision_receive_time_ = std::chrono::steady_clock::now();
      });

    const auto period = std::chrono::duration<double>(1.0 / publish_rate_hz_);
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      [this]() { publish_control_input(); });
  }

private:
  void publish_control_input()
  {
    const auto steady_now = std::chrono::steady_clock::now();
    const double state_age = have_state_ ?
      std::chrono::duration<double>(steady_now - state_receive_time_).count() :
      std::numeric_limits<double>::infinity();
    const double decision_age = have_decision_ ?
      std::chrono::duration<double>(steady_now - decision_receive_time_).count() :
      std::numeric_limits<double>::infinity();

    asv_interfaces::msg::ControlInput output;
    output.seq = ++sequence_;
    output.stamp_us = now().nanoseconds() / 1000;

    const bool valid =
      have_state_ &&
      have_decision_ &&
      latest_state_.valid &&
      latest_decision_.valid &&
      state_age <= state_timeout_sec_ &&
      decision_age <= decision_timeout_sec_ &&
      std::isfinite(latest_decision_.desired_x) &&
      std::isfinite(latest_decision_.desired_y) &&
      std::isfinite(latest_state_.surge_velocity) &&
      std::isfinite(latest_state_.yaw_rate);

    if (valid) {
      output.desired_x =
        static_cast<float>(latest_decision_.desired_x);
      output.desired_y =
        static_cast<float>(latest_decision_.desired_y);
      output.surge_velocity =
        static_cast<float>(latest_state_.surge_velocity);
      output.yaw_rate = static_cast<float>(latest_state_.yaw_rate);
      output.valid = true;
    } else {
      output.desired_x = 0.0F;
      output.desired_y = 0.0F;
      output.surge_velocity = 0.0F;
      output.yaw_rate = 0.0F;
      output.valid = false;
    }

    output_pub_->publish(output);
  }

  double publish_rate_hz_{10.0};
  double state_timeout_sec_{0.5};
  double decision_timeout_sec_{0.5};
  uint32_t sequence_{0};

  bool have_state_{false};
  bool have_decision_{false};
  asv_jetson_interfaces::msg::UEASVState latest_state_;
  asv_jetson_interfaces::msg::DecisionOutput latest_decision_;
  std::chrono::steady_clock::time_point state_receive_time_;
  std::chrono::steady_clock::time_point decision_receive_time_;

  rclcpp::Publisher<asv_interfaces::msg::ControlInput>::SharedPtr output_pub_;
  rclcpp::Subscription<asv_jetson_interfaces::msg::UEASVState>::SharedPtr state_sub_;
  rclcpp::Subscription<asv_jetson_interfaces::msg::DecisionOutput>::SharedPtr decision_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ControlInputMuxNode>());
  rclcpp::shutdown();
  return 0;
}
