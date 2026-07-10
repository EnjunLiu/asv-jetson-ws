#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>

#include <asv_interfaces/msg/asv_wrench.hpp>
#include <asv_interfaces/msg/control_input.hpp>
#include <asv_jetson_interfaces/msg/decision_output.hpp>
#include <asv_jetson_interfaces/msg/predicted_world_state.hpp>
#include <asv_jetson_interfaces/msg/system_status.hpp>
#include <asv_jetson_interfaces/msg/world_state.hpp>

#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

class SystemMonitorNode final : public rclcpp::Node
{
public:
  SystemMonitorNode()
  : Node("system_monitor_node")
  {
    publish_rate_hz_ = declare_parameter<double>("publish_rate_hz", 2.0);
    perception_timeout_sec_ = declare_parameter<double>("perception_timeout_sec", 1.0);
    prediction_timeout_sec_ = declare_parameter<double>("prediction_timeout_sec", 0.5);
    decision_timeout_sec_ = declare_parameter<double>("decision_timeout_sec", 0.5);
    control_input_timeout_sec_ = declare_parameter<double>("control_input_timeout_sec", 0.5);
    esp32_wrench_timeout_sec_ = declare_parameter<double>("esp32_wrench_timeout_sec", 0.5);
    safe_wrench_timeout_sec_ = declare_parameter<double>("safe_wrench_timeout_sec", 0.5);
    validate_configuration();

    status_pub_ = create_publisher<asv_jetson_interfaces::msg::SystemStatus>(
      "/system/status", rclcpp::QoS(5).reliable());

    ue_sub_ = create_subscription<std_msgs::msg::Bool>(
      "/ue/connected",
      rclcpp::QoS(1).reliable().transient_local(),
      [this](const std_msgs::msg::Bool::SharedPtr message) {
        ue_connected_ = message->data;
      });

    emergency_stop_sub_ = create_subscription<std_msgs::msg::Bool>(
      "/safety/emergency_stop",
      rclcpp::QoS(1).reliable().transient_local(),
      [this](const std_msgs::msg::Bool::SharedPtr message) {
        emergency_stop_ = message->data;
      });

    perception_sub_ = create_subscription<asv_jetson_interfaces::msg::WorldState>(
      "/perception/world_state",
      rclcpp::QoS(10).reliable(),
      [this](const asv_jetson_interfaces::msg::WorldState::SharedPtr message) {
        update_state(perception_state_, message->valid);
      });

    prediction_sub_ =
      create_subscription<asv_jetson_interfaces::msg::PredictedWorldState>(
      "/prediction/world_state",
      rclcpp::QoS(10).reliable(),
      [this](const asv_jetson_interfaces::msg::PredictedWorldState::SharedPtr message) {
        update_state(prediction_state_, message->valid);
      });

    decision_sub_ = create_subscription<asv_jetson_interfaces::msg::DecisionOutput>(
      "/decision/output",
      rclcpp::QoS(10).reliable(),
      [this](const asv_jetson_interfaces::msg::DecisionOutput::SharedPtr message) {
        update_state(decision_state_, message->valid);
      });

    control_input_sub_ = create_subscription<asv_interfaces::msg::ControlInput>(
      "/control/control_input",
      rclcpp::QoS(10).reliable(),
      [this](const asv_interfaces::msg::ControlInput::SharedPtr message) {
        update_state(control_input_state_, message->valid);
      });

    esp32_wrench_sub_ = create_subscription<asv_interfaces::msg::ASVWrench>(
      "/control/asv_wrench",
      rclcpp::QoS(10).reliable(),
      [this](const asv_interfaces::msg::ASVWrench::SharedPtr message) {
        update_state(esp32_wrench_state_, message->valid);
      });

    safe_wrench_sub_ = create_subscription<asv_interfaces::msg::ASVWrench>(
      "/control/safe_wrench",
      rclcpp::QoS(10).reliable(),
      [this](const asv_interfaces::msg::ASVWrench::SharedPtr message) {
        update_state(safe_wrench_state_, message->valid);
      });

    const auto period = std::chrono::duration<double>(1.0 / publish_rate_hz_);
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      [this]() { publish_status(); });
  }

private:
  struct MonitoredState
  {
    bool received{false};
    bool message_valid{false};
    std::chrono::steady_clock::time_point receive_time;
  };

  void validate_configuration() const
  {
    if (!std::isfinite(publish_rate_hz_) || publish_rate_hz_ <= 0.0) {
      throw std::invalid_argument("publish_rate_hz must be finite and positive");
    }

    const double timeouts[] = {
      perception_timeout_sec_, prediction_timeout_sec_, decision_timeout_sec_,
      control_input_timeout_sec_, esp32_wrench_timeout_sec_, safe_wrench_timeout_sec_};
    for (const double timeout : timeouts) {
      if (!std::isfinite(timeout) || timeout < 0.0) {
        throw std::invalid_argument("monitor timeouts must be finite and non-negative");
      }
    }
  }

  static void update_state(MonitoredState & state, bool message_valid)
  {
    state.received = true;
    state.message_valid = message_valid;
    state.receive_time = std::chrono::steady_clock::now();
  }

  static bool is_fresh_and_valid(
    const MonitoredState & state,
    const std::chrono::steady_clock::time_point & now,
    double timeout_sec)
  {
    if (!state.received || !state.message_valid) {
      return false;
    }
    return std::chrono::duration<double>(now - state.receive_time).count() <= timeout_sec;
  }

  void publish_status()
  {
    const auto steady_now = std::chrono::steady_clock::now();
    const bool perception_valid = is_fresh_and_valid(
      perception_state_, steady_now, perception_timeout_sec_);
    const bool prediction_valid = is_fresh_and_valid(
      prediction_state_, steady_now, prediction_timeout_sec_);
    const bool decision_valid = is_fresh_and_valid(
      decision_state_, steady_now, decision_timeout_sec_);
    const bool control_input_valid = is_fresh_and_valid(
      control_input_state_, steady_now, control_input_timeout_sec_);
    const bool esp32_wrench_valid = is_fresh_and_valid(
      esp32_wrench_state_, steady_now, esp32_wrench_timeout_sec_);
    const bool safe_wrench_valid = is_fresh_and_valid(
      safe_wrench_state_, steady_now, safe_wrench_timeout_sec_);

    asv_jetson_interfaces::msg::SystemStatus status;
    status.stamp_us = now().nanoseconds() / 1000;
    status.ue_connected = ue_connected_;
    status.perception_valid = perception_valid;
    status.prediction_valid = prediction_valid;
    status.decision_valid = decision_valid;
    status.control_input_valid = control_input_valid;
    status.esp32_wrench_valid = esp32_wrench_valid;
    status.safe_wrench_valid = safe_wrench_valid;
    status.emergency_stop = emergency_stop_;

    if (emergency_stop_) {
      status.detail = "emergency stop active";
    } else if (!ue_connected_) {
      status.detail = "UE5 disconnected";
    } else if (!perception_valid) {
      status.detail = "perception invalid or timed out";
    } else if (!prediction_valid) {
      status.detail = "prediction invalid or timed out";
    } else if (!decision_valid) {
      status.detail = "decision invalid or timed out";
    } else if (!control_input_valid) {
      status.detail = "control input invalid or timed out";
    } else if (!esp32_wrench_valid) {
      status.detail = "ESP32 wrench invalid or timed out";
    } else if (!safe_wrench_valid) {
      status.detail = "safe wrench invalid or timed out";
    } else {
      status.detail = "system nominal";
    }

    status_pub_->publish(status);
  }

  double publish_rate_hz_{2.0};
  double perception_timeout_sec_{1.0};
  double prediction_timeout_sec_{0.5};
  double decision_timeout_sec_{0.5};
  double control_input_timeout_sec_{0.5};
  double esp32_wrench_timeout_sec_{0.5};
  double safe_wrench_timeout_sec_{0.5};

  bool ue_connected_{false};
  bool emergency_stop_{false};
  MonitoredState perception_state_;
  MonitoredState prediction_state_;
  MonitoredState decision_state_;
  MonitoredState control_input_state_;
  MonitoredState esp32_wrench_state_;
  MonitoredState safe_wrench_state_;

  rclcpp::Publisher<asv_jetson_interfaces::msg::SystemStatus>::SharedPtr status_pub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr ue_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr emergency_stop_sub_;
  rclcpp::Subscription<asv_jetson_interfaces::msg::WorldState>::SharedPtr perception_sub_;
  rclcpp::Subscription<asv_jetson_interfaces::msg::PredictedWorldState>::SharedPtr
    prediction_sub_;
  rclcpp::Subscription<asv_jetson_interfaces::msg::DecisionOutput>::SharedPtr
    decision_sub_;
  rclcpp::Subscription<asv_interfaces::msg::ControlInput>::SharedPtr control_input_sub_;
  rclcpp::Subscription<asv_interfaces::msg::ASVWrench>::SharedPtr esp32_wrench_sub_;
  rclcpp::Subscription<asv_interfaces::msg::ASVWrench>::SharedPtr safe_wrench_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SystemMonitorNode>());
  rclcpp::shutdown();
  return 0;
}
