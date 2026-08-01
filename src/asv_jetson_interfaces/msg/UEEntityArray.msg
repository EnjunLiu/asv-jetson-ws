int64 stamp_us
string run_id
int64 scene_seed
uint64 frame_index
string frame_id
UEEntity[] entities
bool valid
string detail
# Optional frame-level provenance and task metadata.  Empty values preserve
# compatibility with legacy UE truth publishers.
# source is one of: ue_truth, image_perception, temporal_tracker.
string source
string instruction_id
string instruction
