# Day 11 handoff

Updated: 2026-07-29

## Repository state

- Repository: `EnjunLiu/asv-jetson-ws`
- Jetson checkout: `/home/jetson/jetson_asv_ws`
- Merged base: `ddc1489` (PR #13)
- Active branch: `feature/day11-kinematic-executor`
- Draft PR: `https://github.com/EnjunLiu/asv-jetson-ws/pull/14`
- Scope: Day 11A UE5-only expert execution and Day 12 recording-mode identity

Read these files first:

1. `TODO.md`, Day 11A and Day 11B
2. `docs/ue5_kinematic_command_v1.md`
3. `src/asv_vla/asv_vla/kinematic_executor.py`
4. `src/asv_vla/asv_vla/expert_kinematic_executor_node.py`
5. `src/asv_ue_bridge/src/ue_object_deliverer_bridge_node.cpp`
6. `src/asv_bringup/launch/day11_expert_kinematic.launch.py`

## Frozen decision

The physical controller is not required for data collection. There are two
separate execution families:

- UE5-only: expert trajectory -> first point -> kinematic JSON -> teleport;
- physical: selected safe trajectory -> `desired_x/y` -> controller -> ESP32.

They may share a high-level trajectory contract but must never run
simultaneously. The simulation path is not evidence of physical tracking.

The expert message contains 20 cumulative `base_link` waypoints at `dt=0.2 s`.
The executor does not walk that array. At 5 Hz it consumes waypoint 0 from the
newest source frame once, then waits for the next frame and replans.

## Implemented

- New message: `asv_jetson_interfaces/msg/UEKinematicSetpoint`
- New topic: `/ue/kinematic_setpoint`
- New executable: `expert_kinematic_executor`
- New launch: `day11_expert_kinematic.launch.py`
- Bridge outbound mode: `thruster`, `kinematic`, or `disabled`
- Kinematic JSON conversion:
  - ROS `delta_x_m` -> UE `Delta_X_Cm = 100*x`
  - ROS `delta_y_m` -> UE `Delta_Y_Cm = -100*y`
- Duplicate source-frame suppression and one stale invalid-hold output
- STOP -> valid hold; malformed/stale/oversized step -> invalid hold
- Day 8 recorder `execution_mode` field and manifest validation
- Day 8 recorder can reuse another bridge with `start_ue_bridge:=false`

## Verified on Jetson

- Four affected ROS packages built successfully.
- Final Jetson asv_vla pytest: `79 passed`.
- Synthetic ROS input:
  - source shape `[20,2]`
  - first point `(0.3, 0.0) m`
  - output valid single setpoint `(0.3, 0.0) m`
- Synthetic TCP output contained:
  - `"Command_Type":"Kinematic_Setpoint"`
  - `"Delta_X_Cm":30.000001...`
  - `"Delta_Y_Cm":-10.000000...`
  - `"Valid":true`
- Dedicated launch graph had one setpoint publisher and one bridge subscriber.
- `/ue/thruster_command` did not exist in kinematic mode.
- The first ROS run exposed an invalid multi-argument logger call. It was fixed
  with an f-string and the runtime probe then passed.

The final four-package build, complete pytest and UE packet validator all
passed. Re-run them after any additional edit. Do not turn this evidence into
UE5 acceptance: no Blueprint consumed the packet yet.

## User's UE5 task

Implement the exact JSON behavior in `docs/ue5_kinematic_command_v1.md`:

1. Parse only `Command_Type=Kinematic_Setpoint`.
2. Reject duplicate or decreasing `Sequence` within the Run ID.
3. On invalid or hold, do not move.
4. Convert actor-local `(Delta_X_Cm, Delta_Y_Cm, 0)` to world space.
5. Add that vector to current location.
6. Set yaw to the world-vector direction when non-zero.
7. Preserve Z, roll and pitch; use teleport semantics.
8. Clear physics velocity and apply no left/right force in this mode.
9. Continue emitting increasing observation `Frame_Index`.

## Runtime acceptance command

Jetson first:

```bash
cd /home/jetson/jetson_asv_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch asv_bringup day11_expert_kinematic.launch.py \
  action:=follow target_attribute:=color:red distance_bucket:=3m
```

Then UE5 Play. Record:

```bash
ros2 topic echo /ue/connected --once
ros2 topic echo /vla/expert_trajectory --once
ros2 topic echo /ue/kinematic_setpoint --once
ros2 topic info -v /ue/kinematic_setpoint
ros2 topic info /ue/thruster_command
```

Test FOLLOW, STOP, duplicate sequence and observation pause. Day 11A is complete
only after ROS evidence and visible UE5 pose behavior both pass.

## What is not started

Day 11B PC registry and Run-level split implementation is not started on this
branch. Do not start training: the single Day 10 pilot must still produce
`training_ready=false`.
