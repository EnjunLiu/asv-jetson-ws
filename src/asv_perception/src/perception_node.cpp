#include <rclcpp/rclcpp.hpp>
#include <rclcpp/create_timer.hpp>

#include <asv_jetson_interfaces/msg/camera_frame.hpp>
#include <asv_jetson_interfaces/msg/target_ground_truth.hpp>
#include <asv_jetson_interfaces/msg/ueasv_state.hpp>
#include <asv_jetson_interfaces/msg/world_state.hpp>

#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/version.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

class PerceptionNode final : public rclcpp::Node
{
public:
  PerceptionNode()
  : Node("perception_node")
  {
    mode_ = declare_parameter<std::string>("mode", "aruco");
    processing_rate_hz_ = declare_parameter<double>("processing_rate_hz", 2.0);
    input_timeout_sec_ = declare_parameter<double>("input_timeout_sec", 1.0);

    target_marker_id_ = declare_parameter<int>("target_marker_id", 0);
    marker_length_m_ = declare_parameter<double>("marker_length_m", 0.1);
    detection_confidence_ = declare_parameter<double>("detection_confidence", 1.0);

    calibration_width_ = declare_parameter<int>("calibration_width", 640);
    calibration_height_ = declare_parameter<int>("calibration_height", 320);
    camera_fx_ = declare_parameter<double>("camera_fx", 1194.2563);
    camera_fy_ = declare_parameter<double>("camera_fy", 1194.2563);
    camera_cx_ = declare_parameter<double>("camera_cx", 320.0);
    camera_cy_ = declare_parameter<double>("camera_cy", 160.0);
    distortion_coefficients_ = declare_parameter<std::vector<double>>(
      "distortion_coefficients", std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0});

    lateral_sign_ = declare_parameter<double>("lateral_sign", 1.0);
    vertical_sign_ = declare_parameter<double>("vertical_sign", -1.0);
    ground_truth_confidence_ =
      declare_parameter<double>("ground_truth_confidence", 1.0);

    validate_parameters();
    initialise_aruco();

    observation_pub_ =
      create_publisher<asv_jetson_interfaces::msg::WorldState>(
      "/perception/world_state", rclcpp::QoS(10).reliable());

    // UE5 may publish at 10 Hz or faster. The callbacks only cache the newest
    // sample; the 2 Hz timer below determines the actual perception rate.
    state_sub_ = create_subscription<asv_jetson_interfaces::msg::UEASVState>(
      "/ue/asv_state",
      rclcpp::QoS(1).reliable(),
      [this](const asv_jetson_interfaces::msg::UEASVState::SharedPtr message) {
        if (mode_ != "ground_truth_passthrough") {
          return;
        }
        latest_state_ = *message;
        have_state_ = true;
        update_ground_truth_candidate();
      });

    truth_sub_ =
      create_subscription<asv_jetson_interfaces::msg::TargetGroundTruth>(
      "/ue/target_ground_truth",
      rclcpp::QoS(1).reliable(),
      [this](const asv_jetson_interfaces::msg::TargetGroundTruth::SharedPtr message) {
        if (mode_ != "ground_truth_passthrough") {
          return;
        }
        latest_truth_ = *message;
        have_truth_ = true;
        update_ground_truth_candidate();
      });

    camera_sub_ = create_subscription<asv_jetson_interfaces::msg::CameraFrame>(
      "/ue/camera_frame",
      rclcpp::QoS(1).best_effort(),
      [this](const asv_jetson_interfaces::msg::CameraFrame::SharedPtr message) {
        if (mode_ != "aruco") {
          return;
        }
        latest_camera_ = message;
        have_camera_ = true;
        camera_receive_time_ = std::chrono::steady_clock::now();
      });

    const auto period = std::chrono::duration<double>(1.0 / processing_rate_hz_);
    processing_timer_ = rclcpp::create_timer(
      this,
      get_clock(),
      rclcpp::Duration(std::chrono::duration_cast<std::chrono::nanoseconds>(period)),
      [this]() { process_latest_input(); });

    RCLCPP_INFO(
      get_logger(),
      "Perception mode='%s', input is cached at its source rate and processed "
      "at %.2f Hz; target ArUco ID=%d, marker length=%.3f m",
      mode_.c_str(), processing_rate_hz_, target_marker_id_, marker_length_m_);
  }

private:
  void validate_parameters() const
  {
    if (mode_ != "aruco" && mode_ != "ground_truth_passthrough") {
      throw std::runtime_error(
              "mode must be 'aruco' or 'ground_truth_passthrough'");
    }
    if (!std::isfinite(processing_rate_hz_) || processing_rate_hz_ <= 0.0) {
      throw std::runtime_error("processing_rate_hz must be finite and positive");
    }
    if (!std::isfinite(input_timeout_sec_) || input_timeout_sec_ <= 0.0) {
      throw std::runtime_error("input_timeout_sec must be finite and positive");
    }
    if (target_marker_id_ < 0) {
      throw std::runtime_error("target_marker_id must be non-negative");
    }
    if (!std::isfinite(marker_length_m_) || marker_length_m_ <= 0.0) {
      throw std::runtime_error("marker_length_m must be finite and positive");
    }
    if (!valid_confidence(detection_confidence_) ||
      !valid_confidence(ground_truth_confidence_))
    {
      throw std::runtime_error("confidence parameters must be in [0, 1]");
    }
    if (calibration_width_ <= 0 || calibration_height_ <= 0) {
      throw std::runtime_error("calibration image dimensions must be positive");
    }
    if (!positive_finite(camera_fx_) || !positive_finite(camera_fy_) ||
      !std::isfinite(camera_cx_) || !std::isfinite(camera_cy_))
    {
      throw std::runtime_error("camera intrinsic parameters are invalid");
    }
    if (distortion_coefficients_.size() != 5U ||
      !std::all_of(
        distortion_coefficients_.begin(), distortion_coefficients_.end(),
        [](double value) {return std::isfinite(value);}))
    {
      throw std::runtime_error(
              "distortion_coefficients must contain five finite values");
    }
    if (!nonzero_finite(lateral_sign_) || !nonzero_finite(vertical_sign_)) {
      throw std::runtime_error("coordinate signs must be finite and non-zero");
    }
  }

  static bool positive_finite(double value)
  {
    return std::isfinite(value) && value > 0.0;
  }

  static bool nonzero_finite(double value)
  {
    return std::isfinite(value) && std::abs(value) > 1.0e-9;
  }

  static bool valid_confidence(double value)
  {
    return std::isfinite(value) && value >= 0.0 && value <= 1.0;
  }

  void initialise_aruco()
  {
#if CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 7)
    dictionary_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    detector_parameters_ = cv::aruco::DetectorParameters();

    detector_parameters_.adaptiveThreshWinSizeMin = 5;
    detector_parameters_.adaptiveThreshWinSizeMax = 15;
    detector_parameters_.adaptiveThreshWinSizeStep = 10;
    detector_parameters_.minMarkerPerimeterRate = 0.05;
    detector_parameters_.polygonalApproxAccuracyRate = 0.1;
    detector_parameters_.minOtsuStdDev = 10.0;
    detector_parameters_.maxErroneousBitsInBorderRate = 0.5;
    aruco_detector_ = std::make_unique<cv::aruco::ArucoDetector>(
      dictionary_, detector_parameters_);
#else
    dictionary_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    detector_parameters_ = cv::aruco::DetectorParameters::create();

    // Preserve the useful detector tuning from the original ESP32 project
    // while using APIs available in Ubuntu 22.04's OpenCV 4.5.4.
    detector_parameters_->adaptiveThreshWinSizeMin = 5;
    detector_parameters_->adaptiveThreshWinSizeMax = 15;
    detector_parameters_->adaptiveThreshWinSizeStep = 10;
    detector_parameters_->minMarkerPerimeterRate = 0.05;
    detector_parameters_->polygonalApproxAccuracyRate = 0.1;
    detector_parameters_->minOtsuStdDev = 10.0;
    detector_parameters_->maxErroneousBitsInBorderRate = 0.5;
#endif
  }

  void process_latest_input()
  {
    if (mode_ == "ground_truth_passthrough") {
      publish_ground_truth_observation();
    } else {
      process_latest_camera();
    }
  }

  void update_ground_truth_candidate()
  {
    if (!have_state_ || !have_truth_) {
      return;
    }

    // State and truth originating from one UE5 JSON packet share a timestamp.
    if (latest_state_.stamp_us != latest_truth_.stamp_us) {
      return;
    }

    asv_jetson_interfaces::msg::WorldState candidate;
    candidate.stamp_us = latest_truth_.stamp_us;
    candidate.tracking_id = static_cast<uint32_t>(target_marker_id_);

    if (latest_state_.valid && latest_truth_.valid) {
      const double delta_x = latest_truth_.position_x - latest_state_.position_x;
      const double delta_y = latest_truth_.position_y - latest_state_.position_y;
      const double delta_z = latest_truth_.position_z - latest_state_.position_z;
      const double cosine = std::cos(latest_state_.yaw);
      const double sine = std::sin(latest_state_.yaw);

      candidate.relative_x = cosine * delta_x + sine * delta_y;
      candidate.relative_y =
        lateral_sign_ * (-sine * delta_x + cosine * delta_y);
      candidate.relative_z = delta_z;
      candidate.confidence = static_cast<float>(ground_truth_confidence_);
      candidate.valid = finite_position(candidate);
    }

    ground_truth_candidate_ = candidate;
    have_ground_truth_candidate_ = true;
    ground_truth_receive_time_ = std::chrono::steady_clock::now();
  }

  void publish_ground_truth_observation()
  {
    asv_jetson_interfaces::msg::WorldState output;
    output.tracking_id = static_cast<uint32_t>(target_marker_id_);

    if (have_ground_truth_candidate_) {
      output = ground_truth_candidate_;
      const double age = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - ground_truth_receive_time_).count();
      if (age > input_timeout_sec_) {
        output.confidence = 0.0F;
        output.valid = false;
      }
    }

    observation_pub_->publish(output);
  }

  void process_latest_camera()
  {
    asv_jetson_interfaces::msg::WorldState output;
    output.tracking_id = static_cast<uint32_t>(target_marker_id_);

    if (!have_camera_ || !latest_camera_) {
      observation_pub_->publish(output);
      return;
    }

    output.stamp_us = latest_camera_->stamp_us;
    const double age = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - camera_receive_time_).count();
    if (age > input_timeout_sec_ || !latest_camera_->valid ||
      latest_camera_->data.empty())
    {
      observation_pub_->publish(output);
      return;
    }

    try {
      const cv::Mat encoded_image(
        1, static_cast<int>(latest_camera_->data.size()), CV_8UC1,
        latest_camera_->data.data());
      const cv::Mat image = cv::imdecode(encoded_image, cv::IMREAD_GRAYSCALE);
      if (image.empty()) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "OpenCV could not decode the latest camera frame (encoding='%s')",
          latest_camera_->encoding.c_str());
        observation_pub_->publish(output);
        return;
      }

      std::vector<std::vector<cv::Point2f>> marker_corners;
      std::vector<int> marker_ids;
#if CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 7)
      aruco_detector_->detectMarkers(image, marker_corners, marker_ids);
#else
      cv::aruco::detectMarkers(
        image, dictionary_, marker_corners, marker_ids, detector_parameters_);
#endif

      const int target_index = select_target_marker(marker_ids, marker_corners);
      if (target_index < 0) {
        observation_pub_->publish(output);
        return;
      }

      const cv::Mat camera_matrix = camera_matrix_for(image.cols, image.rows);
      const cv::Mat distortion(
        1, static_cast<int>(distortion_coefficients_.size()), CV_64F,
        distortion_coefficients_.data());

      const float half_length = static_cast<float>(marker_length_m_ * 0.5);
      const std::vector<cv::Point3f> object_points = {
        {-half_length, half_length, 0.0F},
        {half_length, half_length, 0.0F},
        {half_length, -half_length, 0.0F},
        {-half_length, -half_length, 0.0F}
      };

      cv::Vec3d rotation_vector;
      cv::Vec3d translation_vector;
      const bool pose_ok = cv::solvePnP(
        object_points,
        marker_corners[static_cast<std::size_t>(target_index)],
        camera_matrix,
        distortion,
        rotation_vector,
        translation_vector,
        false,
        cv::SOLVEPNP_IPPE_SQUARE);

      if (!pose_ok) {
        observation_pub_->publish(output);
        return;
      }

      // OpenCV camera frame: +X right, +Y down, +Z forward.
      // ASV observation frame: +X forward, +Y lateral, +Z vertical.
      // The camera is assumed to be aligned with the ASV body frame.
      output.relative_x = translation_vector[2];
      output.relative_y = lateral_sign_ * translation_vector[0];
      output.relative_z = vertical_sign_ * translation_vector[1];
      output.confidence = static_cast<float>(detection_confidence_);
      output.valid = finite_position(output) && output.relative_x > 0.0;
      if (!output.valid) {
        output.confidence = 0.0F;
      }
      observation_pub_->publish(output);
    } catch (const cv::Exception & exception) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "OpenCV ArUco processing failed: %s", exception.what());
      observation_pub_->publish(output);
    }
  }

  int select_target_marker(
    const std::vector<int> & marker_ids,
    const std::vector<std::vector<cv::Point2f>> & marker_corners) const
  {
    int selected_index = -1;
    double selected_area = -1.0;

    for (std::size_t index = 0; index < marker_ids.size(); ++index) {
      if (marker_ids[index] != target_marker_id_ || marker_corners[index].size() != 4U) {
        continue;
      }

      const double area = std::abs(cv::contourArea(marker_corners[index]));
      if (area > selected_area) {
        selected_area = area;
        selected_index = static_cast<int>(index);
      }
    }
    return selected_index;
  }

  cv::Mat camera_matrix_for(int image_width, int image_height) const
  {
    const double scale_x =
      static_cast<double>(image_width) / static_cast<double>(calibration_width_);
    const double scale_y =
      static_cast<double>(image_height) / static_cast<double>(calibration_height_);

    return (cv::Mat_<double>(3, 3) <<
      camera_fx_ * scale_x, 0.0, camera_cx_ * scale_x,
      0.0, camera_fy_ * scale_y, camera_cy_ * scale_y,
      0.0, 0.0, 1.0);
  }

  static bool finite_position(
    const asv_jetson_interfaces::msg::WorldState & observation)
  {
    return
      std::isfinite(observation.relative_x) &&
      std::isfinite(observation.relative_y) &&
      std::isfinite(observation.relative_z);
  }

  std::string mode_;
  double processing_rate_hz_{2.0};
  double input_timeout_sec_{1.0};

  int target_marker_id_{0};
  double marker_length_m_{0.1};
  double detection_confidence_{1.0};

  int calibration_width_{640};
  int calibration_height_{320};
  double camera_fx_{1194.2563};
  double camera_fy_{1194.2563};
  double camera_cx_{320.0};
  double camera_cy_{160.0};
  std::vector<double> distortion_coefficients_;

  double lateral_sign_{1.0};
  double vertical_sign_{-1.0};
  double ground_truth_confidence_{1.0};

#if CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 7)
  cv::aruco::Dictionary dictionary_;
  cv::aruco::DetectorParameters detector_parameters_;
  std::unique_ptr<cv::aruco::ArucoDetector> aruco_detector_;
#else
  cv::Ptr<cv::aruco::Dictionary> dictionary_;
  cv::Ptr<cv::aruco::DetectorParameters> detector_parameters_;
#endif

  bool have_state_{false};
  bool have_truth_{false};
  asv_jetson_interfaces::msg::UEASVState latest_state_;
  asv_jetson_interfaces::msg::TargetGroundTruth latest_truth_;

  bool have_ground_truth_candidate_{false};
  asv_jetson_interfaces::msg::WorldState ground_truth_candidate_;
  std::chrono::steady_clock::time_point ground_truth_receive_time_;

  bool have_camera_{false};
  asv_jetson_interfaces::msg::CameraFrame::SharedPtr latest_camera_;
  std::chrono::steady_clock::time_point camera_receive_time_;

  rclcpp::Publisher<asv_jetson_interfaces::msg::WorldState>::SharedPtr
    observation_pub_;
  rclcpp::Subscription<asv_jetson_interfaces::msg::UEASVState>::SharedPtr state_sub_;
  rclcpp::Subscription<asv_jetson_interfaces::msg::TargetGroundTruth>::SharedPtr
    truth_sub_;
  rclcpp::Subscription<asv_jetson_interfaces::msg::CameraFrame>::SharedPtr camera_sub_;
  rclcpp::TimerBase::SharedPtr processing_timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PerceptionNode>());
  rclcpp::shutdown();
  return 0;
}
