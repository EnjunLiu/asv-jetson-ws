#include <rclcpp/rclcpp.hpp>
#include <rclcpp/create_timer.hpp>

#include <asv_jetson_interfaces/msg/predicted_world_state.hpp>
#include <asv_jetson_interfaces/msg/world_state.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <stdexcept>

class StatePredictorNode final : public rclcpp::Node
{
public:
  StatePredictorNode()
  : Node("state_predictor_node")
  {
    publish_rate_hz_ = declare_parameter<double>("publish_rate_hz", 10.0);
    history_window_size_ = declare_parameter<int>("history_window_size", 3);
    maximum_extrapolation_sec_ =
      declare_parameter<double>("maximum_extrapolation_sec", 1.0);
    recency_weight_ = declare_parameter<double>("recency_weight", 2.0);
    velocity_regularization_ =
      declare_parameter<double>("velocity_regularization", 1.0e-2);
    acceleration_regularization_ =
      declare_parameter<double>("acceleration_regularization", 1.0e-1);

    validate_parameters();

    output_pub_ =
      create_publisher<asv_jetson_interfaces::msg::PredictedWorldState>(
      "/prediction/world_state", rclcpp::QoS(10).reliable());

    input_sub_ = create_subscription<asv_jetson_interfaces::msg::WorldState>(
      "/perception/world_state",
      rclcpp::QoS(10).reliable(),
      [this](const asv_jetson_interfaces::msg::WorldState::SharedPtr message) {
        update_history(*message);
      });

    const auto period = std::chrono::duration<double>(1.0 / publish_rate_hz_);
    prediction_timer_ = rclcpp::create_timer(
      this,
      get_clock(),
      rclcpp::Duration(std::chrono::duration_cast<std::chrono::nanoseconds>(period)),
      [this]() {publish_prediction();});

    RCLCPP_INFO(
      get_logger(),
      "World-state predictor: input is measurement-driven, output=%.2f Hz, "
      "history=%d, polynomial order=2",
      publish_rate_hz_, history_window_size_);
  }

private:
  using Vector3 = std::array<double, 3>;
  using Matrix3 = std::array<std::array<double, 3>, 3>;

  struct Sample
  {
    int64_t stamp_us{0};
    double x{0.0};
    double y{0.0};
    double z{0.0};
    float confidence{0.0F};
    uint32_t tracking_id{0};
  };

  void validate_parameters() const
  {
    if (!positive_finite(publish_rate_hz_)) {
      throw std::runtime_error("publish_rate_hz must be finite and positive");
    }
    if (history_window_size_ < 3) {
      throw std::runtime_error("history_window_size must be at least 3");
    }
    if (!positive_finite(maximum_extrapolation_sec_)) {
      throw std::runtime_error(
              "maximum_extrapolation_sec must be finite and positive");
    }
    if (!std::isfinite(recency_weight_) || recency_weight_ < 0.0) {
      throw std::runtime_error("recency_weight must be finite and non-negative");
    }
    if (!nonnegative_finite(velocity_regularization_) ||
      !nonnegative_finite(acceleration_regularization_))
    {
      throw std::runtime_error("regularization values must be finite and non-negative");
    }
  }

  static bool positive_finite(double value)
  {
    return std::isfinite(value) && value > 0.0;
  }

  static bool nonnegative_finite(double value)
  {
    return std::isfinite(value) && value >= 0.0;
  }

  static bool finite_position(const asv_jetson_interfaces::msg::WorldState & state)
  {
    return
      std::isfinite(state.relative_x) &&
      std::isfinite(state.relative_y) &&
      std::isfinite(state.relative_z);
  }

  void update_history(const asv_jetson_interfaces::msg::WorldState & input)
  {
    if (!input.valid || !finite_position(input)) {
      reset_predictor();
      return;
    }

    if (!history_.empty()) {
      const Sample & latest = history_.back();
      if (input.tracking_id != latest.tracking_id || input.stamp_us < latest.stamp_us) {
        // A new tracked object or a restarted UE simulation starts a new model.
        reset_predictor();
      } else if (input.stamp_us == latest.stamp_us) {
        history_.back() = make_sample(input);
        fit_motion_model();
        return;
      }
    }

    history_.push_back(make_sample(input));
    while (history_.size() > static_cast<std::size_t>(history_window_size_)) {
      history_.pop_front();
    }

    have_measurement_ = true;
    fit_motion_model();
  }

  static Sample make_sample(const asv_jetson_interfaces::msg::WorldState & input)
  {
    Sample sample;
    sample.stamp_us = input.stamp_us;
    sample.x = input.relative_x;
    sample.y = input.relative_y;
    sample.z = input.relative_z;
    sample.confidence = input.confidence;
    sample.tracking_id = input.tracking_id;
    return sample;
  }

  void reset_predictor()
  {
    history_.clear();
    coefficients_x_ = {0.0, 0.0, 0.0};
    coefficients_y_ = {0.0, 0.0, 0.0};
    coefficients_z_ = {0.0, 0.0, 0.0};
    have_measurement_ = false;
    have_model_ = false;
  }

  void fit_motion_model()
  {
    if (history_.empty()) {
      have_model_ = false;
      return;
    }

    const int64_t reference_stamp_us = history_.back().stamp_us;
    Matrix3 normal{};
    Vector3 rhs_x{};
    Vector3 rhs_y{};
    Vector3 rhs_z{};

    const std::size_t sample_count = history_.size();
    for (std::size_t index = 0; index < sample_count; ++index) {
      const Sample & sample = history_[index];
      const double time_sec =
        static_cast<double>(sample.stamp_us - reference_stamp_us) * 1.0e-6;
      const double normalized_recency = sample_count > 1U ?
        static_cast<double>(index) / static_cast<double>(sample_count - 1U) : 1.0;
      const double weight = std::exp(recency_weight_ * normalized_recency);
      const Vector3 basis{1.0, time_sec, time_sec * time_sec};

      for (std::size_t row = 0; row < 3U; ++row) {
        rhs_x[row] += weight * basis[row] * sample.x;
        rhs_y[row] += weight * basis[row] * sample.y;
        rhs_z[row] += weight * basis[row] * sample.z;
        for (std::size_t column = 0; column < 3U; ++column) {
          normal[row][column] += weight * basis[row] * basis[column];
        }
      }
    }

    // Same ridge terms as the old ESP32 quadratic predictor. Position remains
    // unregularized; velocity and acceleration are progressively constrained.
    normal[1][1] += velocity_regularization_;
    normal[2][2] += acceleration_regularization_;

    Vector3 fitted_x{};
    Vector3 fitted_y{};
    Vector3 fitted_z{};
    have_model_ =
      solve_3x3(normal, rhs_x, fitted_x) &&
      solve_3x3(normal, rhs_y, fitted_y) &&
      solve_3x3(normal, rhs_z, fitted_z);

    if (have_model_) {
      coefficients_x_ = fitted_x;
      coefficients_y_ = fitted_y;
      coefficients_z_ = fitted_z;
    } else {
      // A constant model is a safe fallback for an unexpectedly singular fit.
      const Sample & latest = history_.back();
      coefficients_x_ = {latest.x, 0.0, 0.0};
      coefficients_y_ = {latest.y, 0.0, 0.0};
      coefficients_z_ = {latest.z, 0.0, 0.0};
      have_model_ = true;
    }
  }

  static bool solve_3x3(Matrix3 matrix, Vector3 right_hand_side, Vector3 & solution)
  {
    constexpr double kMinimumPivot = 1.0e-12;

    for (std::size_t pivot = 0; pivot < 3U; ++pivot) {
      std::size_t best_row = pivot;
      for (std::size_t row = pivot + 1U; row < 3U; ++row) {
        if (std::abs(matrix[row][pivot]) > std::abs(matrix[best_row][pivot])) {
          best_row = row;
        }
      }

      if (std::abs(matrix[best_row][pivot]) < kMinimumPivot) {
        return false;
      }
      if (best_row != pivot) {
        std::swap(matrix[best_row], matrix[pivot]);
        std::swap(right_hand_side[best_row], right_hand_side[pivot]);
      }

      const double divisor = matrix[pivot][pivot];
      for (std::size_t column = pivot; column < 3U; ++column) {
        matrix[pivot][column] /= divisor;
      }
      right_hand_side[pivot] /= divisor;

      for (std::size_t row = 0; row < 3U; ++row) {
        if (row == pivot) {
          continue;
        }
        const double factor = matrix[row][pivot];
        for (std::size_t column = pivot; column < 3U; ++column) {
          matrix[row][column] -= factor * matrix[pivot][column];
        }
        right_hand_side[row] -= factor * right_hand_side[pivot];
      }
    }

    solution = right_hand_side;
    return std::all_of(
      solution.begin(), solution.end(),
      [](double value) {return std::isfinite(value);});
  }

  void publish_prediction()
  {
    asv_jetson_interfaces::msg::PredictedWorldState output;
    if (!have_measurement_ || !have_model_ || history_.empty()) {
      output_pub_->publish(output);
      return;
    }

    const Sample & latest = history_.back();
    const int64_t current_simulation_stamp_us = now().nanoseconds() / 1'000LL;
    output.stamp_us = current_simulation_stamp_us;
    output.tracking_id = latest.tracking_id;

    if (current_simulation_stamp_us < latest.stamp_us) {
      // UE5 restarted or /clock jumped backwards. Wait for a new measured
      // WorldState before extrapolating in the new simulation timeline.
      output.valid = false;
      output_pub_->publish(output);
      return;
    }

    const double extrapolation_sec =
      static_cast<double>(current_simulation_stamp_us - latest.stamp_us) * 1.0e-6;

    if (extrapolation_sec > maximum_extrapolation_sec_) {
      output.valid = false;
      output_pub_->publish(output);
      return;
    }

    output.relative_x = evaluate(coefficients_x_, extrapolation_sec);
    output.relative_y = evaluate(coefficients_y_, extrapolation_sec);
    output.relative_z = evaluate(coefficients_z_, extrapolation_sec);
    output.confidence = latest.confidence;
    output.valid =
      std::isfinite(output.relative_x) &&
      std::isfinite(output.relative_y) &&
      std::isfinite(output.relative_z);
    if (!output.valid) {
      output.confidence = 0.0F;
    }
    output_pub_->publish(output);
  }

  static double evaluate(const Vector3 & coefficients, double time_sec)
  {
    return coefficients[0] +
           coefficients[1] * time_sec +
           coefficients[2] * time_sec * time_sec;
  }

  double publish_rate_hz_{10.0};
  int history_window_size_{3};
  double maximum_extrapolation_sec_{1.0};
  double recency_weight_{2.0};
  double velocity_regularization_{1.0e-2};
  double acceleration_regularization_{1.0e-1};

  std::deque<Sample> history_;
  Vector3 coefficients_x_{0.0, 0.0, 0.0};
  Vector3 coefficients_y_{0.0, 0.0, 0.0};
  Vector3 coefficients_z_{0.0, 0.0, 0.0};
  bool have_measurement_{false};
  bool have_model_{false};

  rclcpp::Publisher<asv_jetson_interfaces::msg::PredictedWorldState>::SharedPtr
    output_pub_;
  rclcpp::Subscription<asv_jetson_interfaces::msg::WorldState>::SharedPtr input_sub_;
  rclcpp::TimerBase::SharedPtr prediction_timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<StatePredictorNode>());
  rclcpp::shutdown();
  return 0;
}
