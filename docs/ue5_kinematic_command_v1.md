# Jetson to UE5 kinematic command contract v1

Status: code and static tests implemented; UE5 Blueprint runtime acceptance pending.

This is a simulation-only execution contract. It lets an expert trajectory
move the UE5 boat without using `DecisionOutput`, the controller, wrench,
thruster allocation or ESP32. It must not be used as evidence that the
physical controller can track the trajectory.

## Ownership and frequency

The deterministic expert publishes a 20-point trajectory in ROS `base_link`:

- shape: `[20, 2]`;
- `dt`: `0.2 s`;
- each `(x, y)` is a cumulative displacement from the latest planning origin;
- total prediction horizon: `4 s`.

UE5 must not iterate over those 20 points. The Jetson
`expert_kinematic_executor` owns scheduling:

1. retain the newest `/vla/expert_trajectory`;
2. at 5 Hz, take only waypoint 0;
3. publish it once for that `(Run_ID, Scene_Seed, Frame_Index)`;
4. wait for a newer UE5 frame and replan;
5. publish one invalid hold if the source is older than `0.5 s`.

This is receding-horizon expert execution. It avoids an open-loop four-second
playback and prevents UE5 frame rate from determining how many trajectory
points are consumed.

## Mutually exclusive outbound modes

The ObjectDeliverer bridge parameter `outbound_command_mode` is one of:

- `thruster`: subscribe only to `/ue/thruster_command` (legacy);
- `kinematic`: subscribe only to `/ue/kinematic_setpoint`;
- `disabled`: send no actuation command.

The Day 11 launch selects `kinematic`. Do not run the legacy full-system launch
at the same time. Exactly one bridge may own TCP port `8080`.

## Jetson launch

Build and source the workspace, then start Jetson before pressing UE5 Play:

```bash
cd /home/jetson/jetson_asv_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch asv_bringup day11_expert_kinematic.launch.py \
  action:=follow \
  target_attribute:=color:red \
  distance_bucket:=3m
```

The expert defaults to `max_speed_mps=1.5`. With `dt=0.2`, the first step is
at most `0.3 m`; the executor rejects a step above `max_step_m=0.35`.

## JSON sent to UE5

Each command uses the existing `__OD_END__` terminator followed by `NUL`.
Example:

```json
{
  "Command_Type": "Kinematic_Setpoint",
  "Stamp_Us": 12345678,
  "Source_Stamp_Us": 12000000,
  "Run_ID": "A1D7...",
  "Scene_Seed": 12345,
  "Source_Frame_Index": 215,
  "Sequence": 17,
  "Frame_ID": "ue_actor_local",
  "Source_Model_Version": "deterministic_follow_stop_expert_v1",
  "Step_Dt": 0.2,
  "Delta_X_Cm": 29.8,
  "Delta_Y_Cm": -3.4,
  "Hold_Position": false,
  "Valid": true,
  "Reason": "EXPERT_FIRST_POINT:deterministic_follow_stop_expert_v1"
}
```

The bridge performs the coordinate conversion:

```text
Delta_X_Cm =  100 * ROS delta_x_m
Delta_Y_Cm = -100 * ROS delta_y_m
```

Therefore UE5 receives actor-local `+X=forward`, `+Y=right` centimetres.

## Required UE5 Blueprint behavior

On a complete JSON command:

1. Ignore packets whose `Command_Type` is not `Kinematic_Setpoint`.
2. If `Valid=false` or `Hold_Position=true`, do not change pose.
3. Reject a `Sequence` that is not greater than the last applied sequence
   within the same `Run_ID`.
4. Read the current actor transform.
5. Transform local vector `(Delta_X_Cm, Delta_Y_Cm, 0)` into world space using
   the current actor rotation.
6. Set the new location to current location plus that world vector.
7. If the vector norm is non-zero, set world yaw to the direction of that
   world vector; preserve roll, pitch and Z.
8. Use teleport/kinematic semantics, clear physics linear and angular velocity,
   and do not apply left/right force in this mode.
9. Continue sending the normal observation packet with a strictly increasing
   `Frame_Index`.

Blueprint must apply a command at most once. Do not interpolate, queue, or loop
over 20 points: Jetson has already reduced the trajectory to one step.

## Runtime acceptance

For the first test, place a red target more than 3 m ahead and run the launch
above. Press Play only after Jetson prints that it is listening.

Pass conditions:

- `/ue/connected` becomes true;
- `/vla/expert_trajectory` is valid with horizon 20;
- `/ue/kinematic_setpoint` is valid and each source frame is used at most once;
- received JSON has `Delta_X_Cm > 0` and step norm at most `35 cm`;
- the boat advances once per new command, not once per UE render tick;
- STOP produces `Valid=true`, `Hold_Position=true`, zero displacement;
- pausing UE observations for more than `0.5 s` produces one invalid hold;
- no `/ue/thruster_command` subscriber exists in this launch.

Runtime proof must record the ROS topic output and UE5 behavior. Static tests
and `colcon build` alone do not complete this acceptance.
