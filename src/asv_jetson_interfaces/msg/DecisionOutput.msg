# Minimal placeholder decision consumed by the control layer.
# In the ASV body frame, +X is longitudinal/forward and +Y is lateral.
int64 stamp_us
float64 desired_x
float64 desired_y
bool valid
# Source identity retained across the control boundary.  These fields are
# append-only so existing consumers of the legacy decision fields remain
# source-compatible while downstream execution can reject unidentified data.
string run_id
int64 scene_seed
uint64 source_frame_index
string source_model_version
