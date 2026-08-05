# Online VLA control contract: one body-frame displacement for one dt.
# +X is forward and +Y is left in base_link. This is not a trajectory.
int64 stamp_us
string run_id
int64 scene_seed
uint64 frame_index
string frame_id
string model_version
float32 dt
float32 desired_x
float32 desired_y
bool safe_stop
bool valid
string reason
