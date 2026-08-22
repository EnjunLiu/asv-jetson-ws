# 一个控制周期对应一条机体坐标系位移命令。
# base_link 中 +X 向前，+Y 向左。
int64 stamp_us
string run_id
int64 scene_seed
uint64 frame_index
string frame_id
string source
float32 step_dt
float32 desired_x
float32 desired_y
bool safe_stop
bool valid
string reason
