# FrameRecord v1

Status: Day 5 frozen.

`FrameRecord v1` is the JSON boundary between synchronized raw observations
and later feature/tensor construction. It is an offline data artifact, not a
new ROS control message.

## Scope

One record contains:

- one run/frame identity: `run_id`, `scene_seed`, `frame_index`, `stamp_us`;
- the latched natural-language task text;
- one UE ego-state observation;
- one JPEG file reference and the frozen Day 4 camera profile;
- zero to 64 validated UE entities in `base_link`;
- one named validity mask for task, ego, camera and entities.

It intentionally does not contain:

- bounding boxes;
- cached language or visual embeddings;
- a padded entity tensor, Top-K selection or tensor mask;
- expert labels, policy outputs, candidate trajectories or a world model;
- desired displacement, wrench or thruster commands.

Those belong to later days. In particular, Day 7 owns the fixed entity tensor
shape and Top-K/mask policy.

## Frames and units

- `stamp_us`: UE simulation time in integer microseconds;
- `ego.position_m`: metres in the current `ue_world` frame;
- `ego.rpy_ue_rad`: the current UE roll/pitch/yaw values converted to radians;
- entity positions and velocities: metres and metres/second in `base_link`;
- camera mount position: metres in `base_link`;
- camera mount rotation: frozen raw UE component angles in degrees.

The camera profile remains `1280x720`, FOV Angle `90` degrees, mount position
`(0.42, 0.0, 0.20)` metres and raw UE mount rotation
`(roll=0, pitch=-5, yaw=0)` degrees.

## Validity and synchronization

`modality_mask` has exactly four named booleans:

```json
{
  "task": true,
  "ego": true,
  "camera": true,
  "entities": true
}
```

Each value must equal its modality block's `valid` field. Top-level `valid`
must equal the logical AND of all four mask values. This permits an invalid
record to remain auditable without fabricating missing input.

Valid ego, camera and entity blocks must carry exactly the top-level
`stamp_us`. Task text is latched for a run, so its timestamp may be earlier but
must never be in the future.

JSON `NaN` and infinity, duplicate JSON keys, duplicate entity IDs, absolute
image paths and `..` path traversal are rejected.

## Files

- schema: `src/asv_vla/schema/frame_record_v1.schema.json`
- checked-in sample: `src/asv_vla/examples/frame_record_v1.json`
- read/write implementation: `src/asv_vla/asv_vla/frame_record.py`
- tests: `src/asv_vla/test/test_frame_record.py`

Validate the sample from the workspace:

```bash
source install/setup.bash
ros2 run asv_vla validate_frame_record \
  src/asv_vla/examples/frame_record_v1.json
```

Expected prefix: `FRAME_RECORD_VALID`.
