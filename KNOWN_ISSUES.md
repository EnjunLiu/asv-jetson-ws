# Known Issues — ASV VLA UE5 Demo

Last updated: 2026-07-30

## Color grounding (Day 16)

- **Blue 10m standoff**: The model correctly follows blue at 3m (Dir=0.79)
  but the 10m holdout was only fixed after adding 4 STOP-held 10m training
  runs.  Without sufficient 10m standoff data in the training distribution the
  model defaults to "approach" behavior.
- **Response ratio**: Even when directional/assignment accuracy passes, the
  response ratio (prediction magnitude relative to target) can be low (~0.03),
  indicating the model produces conservative trajectories in novel scenes.

## UE5 automation (Day 19)

- **`-game` mode connection**: The ObjectDeliverer plugin may not connect to
  Jetson in `-game -RenderOffscreen` mode.  The same workflow succeeded during
  Day 12–15 collection.  Root cause not confirmed; suspected UE5 project state
  or network initialization timing.  PIE (Play In Editor) mode always works.
- **Workaround**: Use the UE5 editor and press Play manually after Jetson
  reports "Listening".

## Jetson memory

- **GPU memory leak**: Repeated ROS node restarts can leak CUDA allocations.
  `sudo reboot` is the only reliable cleanup.  Check `free -h` before every
  session; ensure ≥3 GB available.
- **Qwen OOM**: The full Qwen3-Embedding-0.6B model (~2 GB) cannot load
  simultaneously with MobileNet (~200 MB) and the ROS bridge on an 8 GB Orin
  Nano.  Use the language stub for live inference or pre-compute embeddings.
- **`/tmp` cleared on reboot**: Checkpoint and other temporary files are lost.
  Store persistent artifacts under `~/jetson_asv_ws/models/`.

## Training

- **expected_run_count hardcoded**: `train.py` only accepts 12, 30, or 34
  runs.  Adding new data requires updating the allowed set in
  `_build_dataset_bundle`.
- **Cross-run pairing requires train-set L3/L4 runs**: The
  `paired_trajectory_contrastive_loss` only works when L3 and L4 layouts
  appear in the training split.  Ensure counterbalanced layouts are
  represented.
- **PC training only**: Training requires an NVIDIA GPU (RTX 5060 tested).
  Jetson lacks the memory for full 30-run training.  Feature caches are built
  on Jetson and transferred to PC for training.

## Launch parameters (Day 11)

- **ROS parameter pass-through**: Some launch arguments (e.g. `action:=stop`,
  `distance_bucket:=10m`) may not correctly override node defaults.  Verify
  with `ros2 param get` at runtime.

## Validation

- **PC/Jetson feature consistency**: Cosine similarity ≥0.999 confirmed for
  20 frames.  Not bitwise-exact due to floating-point differences between
  NVIDIA PyTorch (Jetson) and desktop PyTorch (PC).
- **N1 holdout runs must stay in test split**: The `group_split_v1.json`
  auto-assignment may put new runs into train/validation.  Always verify with
  `grep "N1\|R14"` after running `make_group_splits`.

## Not implemented / out of scope

- Six-candidate trajectory evaluation
- Learned world model
- End-to-end thruster output
- Real-vessel sea trial
- Multi-vessel collision avoidance
- Open-world generalization beyond UE5 simulation
