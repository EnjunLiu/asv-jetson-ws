# Jetson VLA interface contract

This document freezes the Day 1–3 Jetson boundary. UE5 blueprint design and
payload evolution are intentionally outside this contract.

## Control boundary

The VLA side may publish only a two-dimensional desired displacement trajectory.
It never publishes left/right thruster values.

```text
language + visual + task features
                |
                v
      trajectory policy
                |
                v
/vla/selected_trajectory  [H=20, 2], dt=0.2 s, base_link
                |
                v
      trajectory controller
                |
                v
/decision/output  desired_x, desired_y, valid
                |
                v
existing control / ESP32 boundary
```

`+X` is forward and `+Y` is port/left in the ASV body frame.
`delta_p_xy` contains interleaved `[dx0, dy0, ..., dx19, dy19]` values in
metres. A future learned policy must preserve this contract.

## Day 1 fail-closed semantics

`SelectedTrajectory.valid=true` means that the message container has the
expected shape, frame, timestamp and finite values. It does **not** authorize
actuation.

The Day 1 policy always publishes a well-formed zero trajectory with
`safe_stop=true`. The Day 1 trajectory controller maps it to:

```text
DecisionOutput.desired_x = 0
DecisionOutput.desired_y = 0
DecisionOutput.valid = false
```

The existing control chain must then keep `ControlInput`, wrench and thruster
messages at zero with `valid=false`. A valid zero displacement is not used as a
fallback because a controller could interpret it as position hold.

## Day 2 language contract

`/task/text` is event driven. The real encoder publishes
`/vla/language_embedding` with:

- `embedding_dim=256`;
- finite, L2-normalized `float32` values;
- `model_id=Qwen/Qwen3-Embedding-0.6B`;
- `cached=true` for repeated normalized text;
- `valid=false` and an all-zero vector for empty, oversized, unavailable-model
  or inference-failure cases.

The real encoder and `LanguageEncoderStub` must never run together because both
publish the same topic. Use `language_full_stack.launch.py` for the real model.

## Day 3 data boundary

Labels in `dataset/language/*.jsonl` exist only for dataset construction,
splitting and evaluation. They are not an online parser output and may not
bypass the embedding or trajectory policy.

## Day 9 expert-label boundary

The deterministic FOLLOW/STOP expert publishes `ExpertTrajectory` only on
`/vla/expert_trajectory`. This is a data-label topic, not the executable
`/vla/selected_trajectory` topic. It is never connected directly to the
trajectory controller, control manager, ESP32, or left/right thrusters.

`ExpertTrajectory` retains `run_id`, `scene_seed`, `frame_index`, and
`stamp_us`. The full identity is required because adjacent UE5 Frame Index
values can legitimately share one game-time timestamp.

The structured `action`, `target_attribute`, and `distance_bucket` inputs come
from offline dataset metadata:

- FOLLOW selectors: `color:red`, `color:blue`, `bearing:left`,
  `bearing:right`;
- FOLLOW standoff distances: `3m` or `10m`;
- STOP: `target_attribute=none`, `distance_bucket=none`.

FOLLOW predicts the selected target with constant relative velocity, places
the desired ASV waypoint at the requested line-of-sight standoff, and limits
each 0.2 s waypoint increment to the configured expert speed. STOP produces a
20-step zero-displacement label with `safe_stop=true`. Missing, invalid,
ambiguous, or non-finite target data produces the fixed zero shape with
`valid=false`; it never silently becomes a valid STOP label.

Bearing selection uses a 0.25 m lateral deadband around the body-frame
centerline. Sub-millimetre UE/float noise therefore cannot turn a centered
color target into a left/right training label.

## Day 10 supervised-data boundary

Day 10 pairs immutable Day 8 `FrameRecord` observations with compatible Day 3
instructions and recomputed Day 9 expert trajectories. The output contains
only `manifest.json` and `samples.jsonl`; source JSON and JPEG files remain in
their original episode and are referenced by relative path and SHA-256.

Each sample retains:

- `run_id / scene_seed / frame_index / stamp_us / frame_id`;
- source FrameRecord and JPEG paths plus SHA-256 hashes;
- instruction text, offline structured labels, and language-template split;
- expert version, `dt=0.2`, `horizon=20`, selected target, stop flag, and a
  finite nested `20x2` trajectory.

The builder overlays multiple language interventions on the same observation
deliberately. `language_split` therefore measures held-out wording families,
not visual or scene generalization. Future visual-generalization evaluation
must group independent UE runs by Run ID or Scene Seed.

Dataset construction never publishes ROS control topics. A changed source
file, duplicate sample identity, missing instruction, invalid target, hash
mismatch, or trajectory mismatch causes validation failure rather than a
partially trusted sample.

## Legacy path

`PredictedWorldState`, `state_predictor_node` and `decision_node` belong to the
existing formal non-VLA path. They remain buildable until a later integration
branch introduces an explicit `legacy|vla` launch mode. They are not the
deleted VLA world-model evaluation stage.
