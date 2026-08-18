# ASV Jetson Runtime

Jetson-side ROS 2 runtime for the UE5 ASV simulation loop.

## Runtime nodes

Only four nodes are exposed:

| Node | Algorithm | Subscribes | Publishes |
| --- | --- | --- | --- |
| `language` | `language.py` | `/task/text` | `/vla/language_embedding` |
| `perception` | `perception.py` | `/ue/camera_frame`, `/vla/language_embedding` | `/vla/entities` |
| `decision` | `decision.py` | `/vla/entities`, `/vla/language_embedding`, `/ue/asv_state` | `/control/desired_displacement` |
| `bridge_node` | `src/bridge/src/bridge_node.cpp` | TCP JSON, `/control/desired_displacement` | `/ue/camera_frame`, `/ue/asv_state`, `/ue/entities` |

Temporal tracking, entity feature construction, policy inference and safety
checks are internal algorithms. They are not separate ROS nodes or console
entry points.

The decision output is a bounded two-dimensional body-frame displacement in
meters. Invalid, stale, mismatched or unavailable inputs fail closed to hold
position.

## Image contract

UE5 sends a standard sRGB JPEG. Exposure, tone mapping and the linear-to-sRGB
conversion are performed in the UE5 capture path. The Jetson perception path
decodes the JPEG directly and does not apply brightness, gamma or low-light
preprocessing.

## Build and run

Models are external deployment artifacts and are intentionally not committed.
The expected deployment artifacts are `policy_single_point.pt`,
`perception_image_conditioned.npz` and `Qwen3-Embedding-0.6B/`.

```bash
source /opt/ros/humble/setup.bash
colcon build --merge-install --symlink-install \
  --packages-select interfaces bridge vla bringup
source install/setup.bash

ros2 launch bringup vla_closed_loop.launch.py \
  models_dir:=../models \
  execution_address:=<UE5_HOST_IP> execution_port:=8081 \
  task_text:="跟随红色目标船，保持3米距离"
```

The launch starts `bridge_node`, `language`, `perception` and `decision`. It
does not start an ESP32 controller, thruster allocator or physical control
manager.

## Verification

Run the host tests with the package source on `PYTHONPATH`:

```bash
PYTHONPATH=src/vla python -m pytest -q src/vla/test
```

CUDA readiness and the complete UE5 closed loop must be verified on the target
Jetson/UE5 setup with same-run logs. Model identity or CUDA memory failures
remain fail-closed and are not hidden by a CPU fallback.
