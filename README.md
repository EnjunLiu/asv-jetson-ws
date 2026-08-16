# Jetson ROS 2 Deployment

This repository is the Jetson online runtime for the ASV hardware-in-the-loop platform: image-conditioned perception, language embedding, policy inference, safety gating, trajectory control and the UE bridge.

## Scope

- ROS 2 packages: `asv_jetson_interfaces`, `asv_ue_bridge`, `asv_vla`, `asv_bringup`.
- Policy output: two-dimensional body-frame desired displacement in meters.
- Unified control boundary: `DesiredDisplacement` on
  `/control/desired_displacement`; UE5 and future ESP32 adapters branch here.
- `/ue/entities`: collection and offline supervision validation only; never an online privileged-truth feature.
- Model binaries, datasets, caches, install/build/log directories and device credentials are intentionally absent.
- Data collection, replay, expert labeling and policy training belong in the separate PC training repository; they are intentionally absent here.

The corresponding PC-side repository is `asv-vla-training`; it owns dataset
generation, offline evaluation and training workflows. This repository only
ships the artifacts and nodes needed after a model has been prepared.

The model manifest records the selected artifact names and SHA-256 values. Copy artifacts into the controlled deployment location and verify hashes before a Jetson run. Runtime results are environment-dependent and are not inferred from this source checkout.

## Deployment layout

The standalone Jetson repository is intended to sit beside the external model directory:

```text
asv-hil-runtime/
├── models/
└── jetson/       # clone of this repository
```

From the `jetson` repository root, `vla_closed_loop.launch.py` resolves the three
artifacts from `../models` by default. Use `models_dir:=<relative-or-absolute-path>`
when the model directory is elsewhere. The required files are:

```text
models/
├── policy_single_point.pt
├── perception_image_conditioned.npz
└── Qwen3-Embedding-0.6B/
```

The repository does not contain model weights or credentials.

## Build and run

```bash
cd jetson
source /opt/ros/humble/setup.bash
colcon build --merge-install --symlink-install \
  --packages-select asv_jetson_interfaces asv_ue_bridge asv_vla asv_bringup
source install/setup.bash

ros2 launch asv_bringup vla_closed_loop.launch.py \
  models_dir:=../models \
  execution_address:=<UE5_HOST_IP> execution_port:=8081 \
  language_model_id:=Qwen/Qwen3-Embedding-0.6B \
  language_device:=cuda visual_device:=cuda policy_device:=cuda \
  language_release_after_encode:=true \
  task_text:="跟随红色目标船，保持3米距离"
```

The UE5 project must be running its TCP executor on port `8081` before the launch
can complete the software HIL loop. This launch intentionally does not start ESP32,
thruster allocation, or a physical control manager.

## Verification boundary

The imported Python contracts are covered by the host test suite. A current Jetson result requires building on the target, confirming CUDA/Torch loading, and retaining same-run launch and marker logs. Those steps are environment-dependent and are not inferred from PC tests.
