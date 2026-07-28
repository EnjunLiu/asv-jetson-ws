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

## Legacy path

`PredictedWorldState`, `state_predictor_node` and `decision_node` belong to the
existing formal non-VLA path. They remain buildable until a later integration
branch introduces an explicit `legacy|vla` launch mode. They are not the
deleted VLA world-model evaluation stage.
