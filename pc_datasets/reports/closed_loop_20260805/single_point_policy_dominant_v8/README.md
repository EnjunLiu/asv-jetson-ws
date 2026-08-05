# Single-Point Policy Dominant Closed Loop v8

This directory contains the independent red-4m and blue-3m L7/S2/seed-230908
closed-loop evidence produced with the synthetic-geometry decision-head
checkpoint. Earlier v3-v7 reports are preserved.

## Model

- PC checkpoint: `C:\Temp\asv_vla_synthetic_qwen_l7_20260805\policy_synthetic_qwen_l7_seed23.pt`
- Jetson checkpoint: `/home/jetson/jetson_asv_ws/models/policy_synthetic_qwen_l7_seed23_20260805.pt`
- checkpoint SHA-256: `f2dc38a141a3f230b2ddf55cef26841f00812bbd350f28aa84c84f5d5d1e2483`
- Qwen embedding table SHA-256: `c144affbb0b18ab61cd135179b54e3564a91b6c0fc97c5baa965037664ed5958`
- model inputs: language embedding, structured entity geometry, previous action
- model output: one `[desired_x, desired_y]` displacement in `base_link`

The checkpoint was trained on PC CUDA Python only. Jetson was used for model
loading, ROS build, CUDA inference, and closed-loop execution; no Jetson
training was performed.

## Runtime evidence

Both scenes used real Qwen CUDA encoding, image-conditioned CUDA perception,
the new CUDA policy, and headless UE5 `SceneAuto`. UE truth appears only in the
offline world-coordinate log and plot.

| scene | UE runtime | apply count | policy-driven | backstop | fail-closed | final standoff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RED 4m | 185.00 s | 700 | 677 / 700 (96.7%) | 15 (2.1%) | 0 | 3.786 m |
| BLUE 3m | 185.01 s | 700 | 691 / 700 (98.7%) | 8 (1.1%) | 0 | 3.521 m |

The combined plot is:
`C:\Users\LIU\Desktop\track_world_single_point_policy_dominant_red4_blue3_20260805.png`.
The standoff curves are perception/runtime measurements, not smoothed or
interpolated success claims; see `combined_metrics.json` for mean and P95
error.
