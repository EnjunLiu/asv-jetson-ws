string entity_id
string class_name
string color
bool is_target
bool visible
float64 relative_x
float64 relative_y
float64 relative_z
float64 relative_velocity_x
float64 relative_velocity_y
float64 relative_velocity_z
bool valid
# Optional provenance and observability fields.  Legacy publishers leave
# these at their ROS defaults (empty strings, false, and zero values).
# source is one of: ue_truth, image_perception, temporal_tracker.
string source
float32 bbox_x_min
float32 bbox_y_min
float32 bbox_x_max
float32 bbox_y_max
bool bbox_valid
float32 confidence
bool velocity_valid
