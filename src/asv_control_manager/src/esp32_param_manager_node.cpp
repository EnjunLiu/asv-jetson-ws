#include <rclcpp/rclcpp.hpp>
#include <rclcpp/parameter.hpp>
#include <rcl_interfaces/srv/set_parameters.hpp>

#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using SetParameters = rcl_interfaces::srv::SetParameters;

class Esp32ParamManagerNode final : public rclcpp::Node
{
public:
  Esp32ParamManagerNode()
  : Node("esp32_param_manager_node")
  {
    target_node_ = declare_parameter<std::string>(
      "target_node", "/esp32_node");
    apply_on_start_ = declare_parameter<bool>("apply_on_start", true);

    time_constant_ = declare_parameter<double>("time_constant", 10.0);
    v_max_ = declare_parameter<double>("v_max", 0.5);
    e_max_ = declare_parameter<double>("e_max", 0.2);
    delta_t_ = declare_parameter<double>("delta_t", 0.1);
    gamma_rl_ = declare_parameter<double>("gamma_rl", 0.9);
    lambda_rls_ = declare_parameter<double>("lambda_rls", 0.99);
    max_force_ = declare_parameter<double>("max_force", 0.5);
    max_moment_ = declare_parameter<double>("max_moment", 0.1);
    i_bound_force_ = declare_parameter<double>("i_bound_force", 0.1);
    d_bound_force_ = declare_parameter<double>("d_bound_force", 0.1);
    i_bound_moment_ = declare_parameter<double>("i_bound_moment", 0.0075);
    d_bound_moment_ = declare_parameter<double>("d_bound_moment", 0.005);
    param_version_ = declare_parameter<int64_t>("param_version", 1);
    reset_controller_ = declare_parameter<bool>("reset_controller", true);

    validate_configuration();
    while (target_node_.size() > 1 && target_node_.back() == '/') {
      target_node_.pop_back();
    }

    const std::string service_name = target_node_ + "/set_parameters";
    client_ = create_client<SetParameters>(service_name);

    timer_ = create_wall_timer(
      std::chrono::seconds(1), [this]() { try_apply(); });
  }

private:
  void validate_configuration() const
  {
    if (target_node_.empty()) {
      throw std::invalid_argument("target_node must not be empty");
    }

    const auto require_positive = [](double value, const char * name) {
        if (!std::isfinite(value) || value <= 0.0) {
          throw std::invalid_argument(std::string(name) + " must be finite and positive");
        }
      };
    const auto require_non_negative = [](double value, const char * name) {
        if (!std::isfinite(value) || value < 0.0) {
          throw std::invalid_argument(
                  std::string(name) + " must be finite and non-negative");
        }
      };

    require_positive(time_constant_, "time_constant");
    require_non_negative(v_max_, "v_max");
    require_positive(e_max_, "e_max");
    require_positive(delta_t_, "delta_t");
    require_non_negative(max_force_, "max_force");
    require_non_negative(max_moment_, "max_moment");
    require_non_negative(i_bound_force_, "i_bound_force");
    require_non_negative(d_bound_force_, "d_bound_force");
    require_non_negative(i_bound_moment_, "i_bound_moment");
    require_non_negative(d_bound_moment_, "d_bound_moment");

    if (!std::isfinite(gamma_rl_) || gamma_rl_ < 0.0 || gamma_rl_ > 1.0) {
      throw std::invalid_argument("gamma_rl must be in [0, 1]");
    }
    if (!std::isfinite(lambda_rls_) || lambda_rls_ <= 0.0 || lambda_rls_ > 1.0) {
      throw std::invalid_argument("lambda_rls must be in (0, 1]");
    }
    if (param_version_ < 0) {
      throw std::invalid_argument("param_version must be non-negative");
    }
  }

  static rcl_interfaces::msg::Parameter to_message(const rclcpp::Parameter & parameter)
  {
    return parameter.to_parameter_msg();
  }

  void try_apply()
  {
    if (!apply_on_start_ || applying_ || completed_) {
      return;
    }

    if (!client_->service_is_ready()) {
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Waiting for %s parameter service", target_node_.c_str());
      return;
    }

    applying_ = true;
    auto request = std::make_shared<SetParameters::Request>();
    const std::vector<rclcpp::Parameter> parameters = {
      {"time_constant", time_constant_},
      {"v_max", v_max_},
      {"e_max", e_max_},
      {"delta_t", delta_t_},
      {"gamma_rl", gamma_rl_},
      {"lambda_rls", lambda_rls_},
      {"max_force", max_force_},
      {"max_moment", max_moment_},
      {"i_bound_force", i_bound_force_},
      {"d_bound_force", d_bound_force_},
      {"i_bound_moment", i_bound_moment_},
      {"d_bound_moment", d_bound_moment_}
    };

    for (const auto & parameter : parameters) {
      request->parameters.push_back(to_message(parameter));
    }

    client_->async_send_request(
      request,
      [this](rclcpp::Client<SetParameters>::SharedFuture future) {
        on_base_parameters_set(future);
      });
  }

  static bool all_successful(const SetParameters::Response & response)
  {
    if (response.results.empty()) {
      return false;
    }
    for (const auto & result : response.results) {
      if (!result.successful) {
        return false;
      }
    }
    return true;
  }

  void on_base_parameters_set(rclcpp::Client<SetParameters>::SharedFuture future)
  {
    SetParameters::Response::SharedPtr response;
    try {
      response = future.get();
    } catch (const std::exception & error) {
      applying_ = false;
      RCLCPP_ERROR(get_logger(), "ESP32 parameter service failed: %s", error.what());
      return;
    }
    if (!all_successful(*response)) {
      applying_ = false;
      RCLCPP_ERROR(get_logger(), "ESP32 rejected one or more controller parameters");
      return;
    }

    // Current ESP32 reads the coherent parameter set when param_version changes,
    // so version must be sent last.
    send_single_parameter(
      rclcpp::Parameter("param_version", param_version_),
      [this](bool success) {
        if (!success) {
          applying_ = false;
          RCLCPP_ERROR(get_logger(), "ESP32 rejected param_version");
          return;
        }

        if (reset_controller_) {
          send_single_parameter(
            rclcpp::Parameter("reset_controller", true),
            [this](bool reset_success) {
              finish(reset_success);
            });
        } else {
          finish(true);
        }
      });
  }

  template<typename CallbackT>
  void send_single_parameter(
    const rclcpp::Parameter & parameter,
    CallbackT callback)
  {
    auto request = std::make_shared<SetParameters::Request>();
    request->parameters.push_back(to_message(parameter));
    client_->async_send_request(
      request,
      [callback](rclcpp::Client<SetParameters>::SharedFuture future) {
        try {
          callback(Esp32ParamManagerNode::all_successful(*future.get()));
        } catch (const std::exception &) {
          callback(false);
        }
      });
  }

  void finish(bool success)
  {
    if (success) {
      completed_ = true;
      RCLCPP_INFO(
        get_logger(), "ESP32 parameters applied, param_version=%ld",
        static_cast<long>(param_version_));
    } else {
      applying_ = false;
      RCLCPP_ERROR(get_logger(), "ESP32 reset_controller request failed");
    }
  }

  std::string target_node_;
  bool apply_on_start_{true};
  double time_constant_{10.0};
  double v_max_{0.5};
  double e_max_{0.2};
  double delta_t_{0.1};
  double gamma_rl_{0.9};
  double lambda_rls_{0.99};
  double max_force_{0.5};
  double max_moment_{0.1};
  double i_bound_force_{0.1};
  double d_bound_force_{0.1};
  double i_bound_moment_{0.0075};
  double d_bound_moment_{0.005};
  int64_t param_version_{1};
  bool reset_controller_{true};

  bool applying_{false};
  bool completed_{false};
  rclcpp::Client<SetParameters>::SharedPtr client_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Esp32ParamManagerNode>());
  rclcpp::shutdown();
  return 0;
}
