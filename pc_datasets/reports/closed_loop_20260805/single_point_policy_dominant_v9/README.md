# Single-Point Policy Dominant Closed Loop v9

This directory contains independent red-4m, blue-3m, and red-3m L7/S2/seed-230908
closed-loop evidence produced with the synthetic-geometry decision-head
checkpoint. Earlier v3-v8 reports are preserved. The red-3m v1 attempt is
excluded because a stale duplicate ROS node invalidated its identity checks;
red-3m v2 is the clean rerun used here.

## Model

- PC checkpoint: `C:\Temp\asv_vla_synthetic_qwen_l7_20260805\policy_synthetic_qwen_l7_seed23.pt`
- Jetson checkpoint: `/home/jetson/jetson_asv_ws/models/policy_synthetic_qwen_l7_seed23_20260805.pt`
- checkpoint SHA-256: `f2dc38a141a3f230b2ddf55cef26841f00812bbd350f28aa84c84f5d5d1e2483`
- Qwen embedding table SHA-256: `c144affbb0b18ab61cd135179b54e3564a91b6c0fc97c5baa965037664ed5958`
- checkpoint reproduction manifest: `checkpoint_manifest.json`
- model inputs: language embedding, structured entity geometry, previous action
- model output: one `[desired_x, desired_y]` displacement in `base_link`
- checkpoint contract: `language + entity_geometry + previous_action` and validity
  masks; output is one bounded `[B, 2]` action, with no visual or ego input

The checkpoint was trained on PC CUDA Python only. Jetson was used for model
loading, ROS build, CUDA inference, and closed-loop execution; no Jetson
training was performed.

The delivered checkpoint predates the embedded metadata extension in the
training script, so `checkpoint_manifest.json` records its exact training
parameters and the validated model contract without rewriting the hashed
weight file.

## Runtime evidence

All three scenes used real Qwen CUDA encoding, image-conditioned CUDA perception,
the new CUDA policy, and headless UE5 `SceneAuto`. UE truth appears only in the
offline world-coordinate log and plot.

| scene | UE runtime | apply count | policy-driven | backstop | hold | fail-closed | final standoff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RED 4m | 185.00 s | 700 | 677 / 700 (96.7%) | 15 (2.1%) | 8 | 0 | 3.786 m |
| BLUE 3m | 185.01 s | 700 | 691 / 700 (98.7%) | 8 (1.1%) | 1 | 0 | 3.521 m |
| RED 3m | 185.01 s | 700 | 698 / 700 (99.7%) | 1 (0.1%) | 1 | 0 | 3.446 m |

The audit categories close exactly as `policy-driven + backstop + hold +
fail-closed = apply count`. Per-scene metrics are in `red4m/metrics.json`,
`blue3m/metrics.json`, and `red3m/metrics.json`; `combined_metrics.json` is
their committed combined copy. All committed log references are relative to
this report directory.

The red-3m execution-window Jetson log ends at the UE disconnect and therefore
keeps its `events=700` audit aligned with 700 UE apply records. The complete
captured Jetson log, including the later ROS shutdown tail, is preserved as
`red3m/jetson_full_shutdown.log`; its post-disconnect shutdown audit is not
counted in the closed-loop metrics.

The combined plot is:
`C:\Users\LIU\Desktop\track_world_single_point_policy_dominant_red4_blue3_red3_20260806.png`.
The standoff curves are perception/runtime measurements, not smoothed or
interpolated success claims; see `combined_metrics.json` for mean and P95
error.
