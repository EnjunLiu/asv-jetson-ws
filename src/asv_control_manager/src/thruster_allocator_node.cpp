#include <rclcpp/rclcpp.hpp>

#include <asv_interfaces/msg/asv_wrench.hpp>
#include <asv_jetson_interfaces/msg/thruster_command.hpp>

#include <algorithm>
#include <cmath>
#include <string>

class ThrusterAllocatorNode final : public rclcpp::Node
{
public:
  ThrusterAllocatorNode()
  : Node("thruster_allocator_node")
  {
    allocation_mode_ = declare_parameter<std::string>("allocation_mode", "physical");
    force_gain_ = declare_parameter<double>("force_gain", 0.5);
    moment_gain_ = declare_parameter<double>("moment_gain", 200.0);
    thruster_separation_ = declare_parameter<double>("thruster_separation", 1.0);
    left_sign_ = declare_parameter<double>("left_sign", 1.0);
    right_sign_ = declare_parameter<double>("right_sign", 1.0);
    thruster_limit_ = declare_parameter<double>("thruster_limit", 25.0);

    output_pub_ =
      create_publisher<asv_jetson_interfaces::msg::ThrusterCommand>(
      "/ue/thruster_command", rclcpp::QoS(10).reliable());

    input_sub_ = create_subscription<asv_interfaces::msg::ASVWrench>(
      "/control/safe_wrench",
      rclcpp::QoS(10).reliable(),
      [this](const asv_interfaces::msg::ASVWrench::SharedPtr message) {
        allocate(*message);
      });
  }

private:
  void allocate(const asv_interfaces::msg::ASVWrench & wrench)
  {
    asv_jetson_interfaces::msg::ThrusterCommand output;
    output.stamp_us = now().nanoseconds() / 1000;

    double left = 0.0;
    double right = 0.0;
    bool valid = wrench.valid;

    if (valid && allocation_mode_ == "physical") {
      if (thruster_separation_ <= 1.0e-6) {
        valid = false;
        RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "thruster_separation must be positive in physical mode");
      } else {
        // F = TL + TR, M = (TR - TL) * d / 2
        left =
          0.5 * static_cast<double>(wrench.force) +
          static_cast<double>(wrench.moment) / thruster_separation_;
        right =
          0.5 * static_cast<double>(wrench.force) -
          static_cast<double>(wrench.moment) / thruster_separation_;
      }
    } else if (valid) {
      // Compatibility with the original edge.py:
      // left = 0.5 * (F + 200 * M), right = 0.5 * (F - 200 * M)
      left = force_gain_ *
        (static_cast<double>(wrench.force) +
        moment_gain_ * static_cast<double>(wrench.moment));
      right = force_gain_ *
        (static_cast<double>(wrench.force) -
        moment_gain_ * static_cast<double>(wrench.moment));
    }

    left *= left_sign_;
    right *= right_sign_;
    left = std::clamp(left, -thruster_limit_, thruster_limit_);
    right = std::clamp(right, -thruster_limit_, thruster_limit_);

    output.left_thruster = valid ? left : 0.0;
    output.right_thruster = valid ? right : 0.0;
    output.valid = valid && std::isfinite(left) && std::isfinite(right);
    output_pub_->publish(output);
  }

  std::string allocation_mode_;
  double force_gain_{0.5};
  double moment_gain_{200.0};
  double thruster_separation_{1.0};
  double left_sign_{1.0};
  double right_sign_{1.0};
  double thruster_limit_{25.0};

  rclcpp::Publisher<asv_jetson_interfaces::msg::ThrusterCommand>::SharedPtr output_pub_;
  rclcpp::Subscription<asv_interfaces::msg::ASVWrench>::SharedPtr input_sub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ThrusterAllocatorNode>());
  rclcpp::shutdown();
  return 0;
}
