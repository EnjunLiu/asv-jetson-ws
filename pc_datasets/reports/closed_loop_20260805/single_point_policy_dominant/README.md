# Single-Point Policy-Dominant Closed Loop

This directory contains real UE5/Jetson online closed-loop evidence for RED 4m,
BLUE 3m, and RED 3m in L7/S2 with `seed=230908`. All scenes ran for 185 seconds
with the same motion parameters: wavelength 6000 cm, amplitude 200 cm, speed
60 cm/s, and delay 40 s. The RED 3m log is a fresh rerun; the earlier run with
different sine parameters is not used and no entity trajectory was copied from
another scene.

## Active Models

- policy: `models/policy_single_point.pt`
- policy SHA-256: `f2dc38a141a3f230b2ddf55cef26841f00812bbd350f28aa84c84f5d5d1e2483`
- perception: `models/perception_image_conditioned.npz`
- perception SHA-256: `a1e7451642c51b879e8b9ce1d7037567c2057d534bcb547c483716188ceb5e6e`
- language model: `models/Qwen3-Embedding-0.6B`

The RED 4m and BLUE 3m source logs retain their historical `V1` slot strings
and old checkpoint pathname. Their checkpoint SHA-256 is identical to the
canonical `policy_single_point.pt`; the raw log text is preserved rather than
rewritten. RED 3m used the canonical model path and unversioned slot directly.

The policy inputs are the online Qwen task embedding, image-derived structured
entity geometry, previous action, and validity masks. Its output is one bounded
body-frame desired displacement point. Jetson performed ROS build and CUDA
inference only; no training ran on Jetson.

## Results

| scene | samples | mean abs error | P95 abs error | final standoff | policy-driven | backstop | hold | fail-closed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RED 4m | 186 | 0.347 m | 0.847 m | 3.786 m | 677/700 (96.7%) | 15 | 8 | 0 |
| BLUE 3m | 186 | 0.568 m | 1.452 m | 3.521 m | 691/700 (98.7%) | 8 | 1 | 0 |
| RED 3m | 186 | 0.534 m | 1.317 m | 3.034 m | 713/721 (98.9%) | 7 | 1 | 0 |

`combined_metrics.json` contains the machine-readable metrics. The final 2x3
plot is `C:\Users\LIU\Desktop\track_world_single_point_policy_dominant_2x3.png`.
The top row shows each ASV trajectory together with all four entity trajectories;
the bottom row shows signed `actual - desired` standoff error.

The plot was generated with `--require-shared-entity-tracks` and a 5 cm limit.
The measured maximum cross-scene deviation was 1.016 cm across all four entity
tracks. This is a hard comparison gate, not trajectory substitution or
interpolation.
