# One receding-horizon body-frame displacement command for UE5-only execution.
# This message bypasses the physical controller and must never be sent to ESP32.
int64 stamp_us
int64 source_stamp_us
string run_id
int64 scene_seed
uint64 source_frame_index
uint32 sequence
string frame_id
string source_model_version
float32 step_dt
float32 delta_x_m
float32 delta_y_m
bool hold_position
bool valid
string reason
