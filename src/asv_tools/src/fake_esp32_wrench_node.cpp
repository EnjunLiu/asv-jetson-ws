#include <rclcpp/rclcpp.hpp>
#include <asv_interfaces/msg/asv_wrench.hpp>

#include <chrono>
#include <cmath>

class FakeEsp32WrenchNode final : public rclcpp::Node
{
public:
  FakeEsp32WrenchNode()
  : Node("fake_esp32_wrench_node")
  {
    force_ = declare_parameter<double>("force", 0.2);
    moment_ = declare_parameter<double>("moment", 0.001);
    rate_hz_ = declare_parameter<double>("rate_hz", 10.0);
    valid_ = declare_parameter<bool>("valid", false);

    publisher_ = create_publisher<asv_interfaces::msg::ASVWrench>(
      "/control/asv_wrench", rclcpp::QoS(10).reliable());

    const auto period = std::chrono::duration<double>(1.0 / rate_hz_);
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      [this]() {
        asv_interfaces::msg::ASVWrench message;
        message.seq = ++sequence_;
        message.stamp_us = now().nanoseconds() / 1000;
        const bool output_valid =
          valid_ && std::isfinite(force_) && std::isfinite(moment_);
        message.force = output_valid ? static_cast<float>(force_) : 0.0F;
        message.moment = output_valid ? static_cast<float>(moment_) : 0.0F;
        message.valid = output_valid;
        publisher_->publish(message);
      });
  }

private:
  double force_{0.2};
  double moment_{0.001};
  double rate_hz_{10.0};
  bool valid_{false};
  uint32_t sequence_{0};
  rclcpp::Publisher<asv_interfaces::msg::ASVWrench>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FakeEsp32WrenchNode>());
  rclcpp::shutdown();
  return 0;
}
