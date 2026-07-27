uint8 STARTING=0
uint8 READY=1
uint8 DEGRADED=2
uint8 SAFE_STOP=3
uint8 ERROR=4

int64 stamp_us
string run_id
string module_name
uint8 state
bool alive
bool input_ready
bool output_valid
string detail
