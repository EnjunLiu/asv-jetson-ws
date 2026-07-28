# Jetson ASV ROS 2 workspace

ROS 2 Humble workspace for the Jetson side of a twin-thruster unmanned surface
vessel. UE5 sends simulation observations; the Jetson publishes only
two-dimensional desired displacement/trajectory commands to the existing
control boundary.

## Current paths

- `full_system.launch.py` is the existing legacy perception/prediction/control
  path.
- `smoke_full_stack.launch.py` is the Day 1 fail-closed VLA contract test.
- `language_full_stack.launch.py` replaces only the language stub with the
  frozen Day 2 embedding model.

Never run the formal and smoke launches at the same time: they share control
topics. The direct VLA policy publishes one `[20,2]` trajectory; the removed
six-candidate and learned-world-model evaluation stages are not part of the
architecture.

The frozen interfaces and fail-closed semantics are documented in
[`docs/interfaces.md`](docs/interfaces.md). The execution plan and acceptance
gates are in [`TODO.md`](TODO.md).

## Build

Run on the Jetson:

```bash
cd ~/jetson_asv_ws
source /opt/ros/humble/setup.bash
source /home/jetson/microros_ws/install/setup.bash

colcon build --symlink-install
source install/setup.bash
```

## Day 1: direct-trajectory fail-closed contract

Terminal A:

```bash
cd ~/jetson_asv_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch asv_bringup smoke_full_stack.launch.py \
  jetson_git_sha:="$(git rev-parse HEAD)"
```

Terminal B:

```bash
cd ~/jetson_asv_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run asv_vla contract_probe
```

Acceptance marker:

```text
DAY1_CONTRACT_PASS
```

This verifies a well-formed single safe-stop trajectory followed by invalid
zero `DecisionOutput`, invalid zero control/wrench and invalid zero thrusters.

## Day 2: language embedding

The model is `Qwen/Qwen3-Embedding-0.6B`, loaded locally through the
PyTorch-only `sentence-transformers` path. Do not replace the NVIDIA Jetson
PyTorch installation with a desktop wheel.

Run the lightweight unit tests first:

```bash
cd ~/jetson_asv_ws
source .venv/bin/activate
PYTHONPATH=src/asv_vla python -m pytest -q -p no:cacheprovider src/asv_vla/test
```

Run the real offline model test:

```bash
PYTHONPATH=src/asv_vla \
  python -m asv_vla.evaluate_language_similarity \
  --model-path models/Qwen3-Embedding-0.6B \
  --device cuda
```

Acceptance marker:

```text
LANGUAGE_EMBEDDING_OFFLINE_PASS
```

The command writes the ignored runtime artifact
`artifacts/language_embedding/language_similarity.csv`.

For the ROS path, start `language_full_stack.launch.py` and run
`ros2 run asv_vla language_embedding_probe`. Expected marker:

```text
LANGUAGE_EMBEDDING_PASS
```

Run the language model headlessly: close VS Code language servers, Jupyter and
other large processes before benchmarking. If Qwen still cannot load without
memory allocation failures, switch to the documented MiniLM fallback while
keeping the 256-dimensional ROS contract.

## Day 3: language intervention data

```bash
cd ~/jetson_asv_ws
source .venv/bin/activate

PYTHONPATH=src/asv_vla \
  python -m asv_vla.generate_language_interventions --check

PYTHONPATH=src/asv_vla \
  python -m asv_vla.evaluate_language_coverage
```

Acceptance marker:

```text
LANGUAGE_INTERVENTION_COVERAGE_PASS
```

Dataset labels are used only for organization, splitting and evaluation. They
are not an online task parser and cannot bypass the language embedding or VLA
policy.

## Platform

- Jetson Orin Nano 8 GB
- Ubuntu 22.04 / ROS 2 Humble
- micro-ROS agent Humble

Record the actual L4T, CUDA, TensorRT and NVIDIA PyTorch versions in every
deployment benchmark; do not infer them only from a nominal JetPack release.
