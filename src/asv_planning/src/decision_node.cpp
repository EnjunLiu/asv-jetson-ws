#include <rclcpp/rclcpp.hpp>

#include <asv_jetson_interfaces/msg/decision_output.hpp>
#include <asv_jetson_interfaces/msg/predicted_world_state.hpp>

#include <cmath>

class DecisionNode final : public rclcpp::Node
{
public:
  DecisionNode()
  : Node("decision_node")
  {
    output_pub_ = create_publisher<asv_jetson_interfaces::msg::DecisionOutput>(
      "/decision/output", rclcpp::QoS(10).reliable());

    input_sub_ =
      create_subscription<asv_jetson_interfaces::msg::PredictedWorldState>(
      "/prediction/world_state",
      rclcpp::QoS(10).reliable(),
      [this](const asv_jetson_interfaces::msg::PredictedWorldState::SharedPtr message) {
        decide(*message);
      });
  }

private:
  void decide(const asv_jetson_interfaces::msg::PredictedWorldState & world_state)
  {
    asv_jetson_interfaces::msg::DecisionOutput output;
    output.stamp_us = world_state.stamp_us;

    // Minimal placeholder matching the original ESP32 project:
    // d_x = predicted_x, d_y = predicted_y.
    output.desired_x = world_state.relative_x;
    output.desired_y = world_state.relative_y;
    output.valid =
      world_state.valid &&
      std::isfinite(output.desired_x) &&
      std::isfinite(output.desired_y);

    if (!output.valid) {
      output.desired_x = 0.0;
      output.desired_y = 0.0;
    }
    output_pub_->publish(output);
  }

  rclcpp::Publisher<asv_jetson_interfaces::msg::DecisionOutput>::SharedPtr
    output_pub_;
  rclcpp::Subscription<asv_jetson_interfaces::msg::PredictedWorldState>::SharedPtr
    input_sub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DecisionNode>());
  rclcpp::shutdown();
  return 0;
}
