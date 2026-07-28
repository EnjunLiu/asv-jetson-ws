# UE5 to Jetson frame contract v1

Status: Day 4 in progress.

This document records the packet that is currently emitted by the UE5
ObjectDeliverer blueprint and the conversion performed by the Jetson bridge.
It does not define a trajectory or actuator interface.

## Transport

- TCP server: Jetson port `8080`
- encoding: UTF-8 JSON
- packet terminator: `__OD_END__`
- observed rate: approximately 10 Hz
- maximum receive buffer: 32 MiB
- maximum JPEG payload: 8 MiB

## Existing required fields

The current bridge continues to require:

- `Run_ID`: non-empty identifier created once in UE5 `BeginPlay`
- `Scene_Seed`: signed integer that drives the run's UE5 random stream
- `Frame_Index`: non-negative integer incremented once per generated packet
- `Time`: UE simulation time in seconds
- `ASV_Location`: UE world position in centimetres
- `ASV_Rotation`: Roll, Pitch and Yaw in degrees
- `Surge_Velocity`: centimetres per second
- `Angular_Velocity`: radians per second
- `Target_Location`: compatibility-only target world position

`Camera_Capture` is optional. When present it is an array of JPEG bytes.

The bridge also accepts the compatibility spellings `Run_Id`, `RunId` and
`run_id`, but `Run_ID` is the canonical UE5 field name. `Scene_Seed` must stay
constant within a run. A new `Run_ID` must start at `Frame_Index=0`.
Duplicate or decreasing indices are rejected. A forward gap is accepted so the
latest observation remains usable, but the gap is reported in `detail` and the
node log.

Every published `/ue/asv_state`, `/ue/target_ground_truth`,
`/ue/camera_frame` and `/ue/entities` message carries the normalized
`run_id`, `scene_seed` and `frame_index`.

## Entities

`Entities` is an array. Each item currently has this shape:

```json
{
  "RelativePosition": {"x": 959.01, "y": 143.35, "z": -6.84},
  "RelativeVelocity": {"x": 9.82, "y": 1.90, "z": 0.0},
  "Entity_Id": "target_01",
  "Class": "boat",
  "Color": "red",
  "Is_Target": true,
  "Visible": true,
  "BBox_XYXY_Px": true
}
```

The Jetson bridge publishes `/ue/entities` as
`asv_jetson_interfaces/msg/UEEntityArray`.

Current conversion:

```text
relative_x =  0.01 * UE x
relative_y = -0.01 * UE y
relative_z =  0.01 * UE z
```

The same scale and signs are applied to relative velocity. The output frame is
`base_link`: +X forward, +Y port/left and +Z up.

`BBox_XYXY_Px` is currently a boolean rather than four pixel coordinates. It is
not required and is ignored by the bridge. A later visual-data contract may add
an optional real bounding box without changing entity validity.

## Validation

An entity array is valid only when:

- `Entities` is an array with at most 64 items;
- every item has a non-empty ID, class and color;
- target and visibility fields are booleans;
- position and velocity contain finite x, y and z values;
- IDs are unique within the frame.

Malformed input publishes an empty array with `valid=false` and an explicit
`detail`; partial entity frames are never accepted.

## Still required before Day 4 completion

- generate non-placeholder `Run_ID`, `Scene_Seed` and strictly increasing
  `Frame_Index` values in UE5 and validate them with a real run;
- freeze camera dimensions, FOV and mounting transform;
- add a checked-in synthetic packet validator;
- decide how the natural-language task enters the run.
