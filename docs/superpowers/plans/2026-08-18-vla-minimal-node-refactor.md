# VLA Minimal Node Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce VLA to `language`, `perception`, and `decision`, make each a thin ROS wrapper around its same-named algorithm file, rename the bridge to `bridge_node`, and remove Jetson-side image enhancement.

**Architecture:** `language` publishes `TaskEmbedding`; `perception` consumes camera plus language context, performs detection and internal temporal tracking, and publishes final `EntityArray`; `decision` consumes language plus entities, performs inference and safety checks, and publishes final `DesiredDisplacement`. The C++ bridge remains the UE5 transport boundary.

**Tech Stack:** ROS 2, `rclpy`, Python/NumPy/PyTorch, setuptools console scripts, `ament_cmake`, C++17, Unreal Engine 5 JPEG capture.

---

## Rules

- Work in `C:\Users\LIU\Documents\ChatGPT\实习面试项目整理\asv-jetson-ws`.
- UE5 changes belong to the separate repository `C:\Users\LIU\Documents\ChatGPT\实习面试项目整理\asv-unreal-simulation`; inspect and commit that repository independently.
- Preserve unrelated changes and stage only task files.
- Run ROS tests on Jetson; Windows has no `rclpy`, so Windows checks are syntax/static only.
- Do not delete an old module until its replacement exists, imports are updated, and `rg` shows no runtime consumer.
- Every migration starts with a failing contract test, then implementation, focused tests, and a commit.

## Target Structure

```text
src/vla/vla/
  language.py
  language_node.py
  perception.py
  perception_node.py
  decision.py
  decision_node.py
```

Algorithm ownership:

- `language_encoder.py` -> `language.py`.
- `image_entity_perception.py`, `visual_encoder.py`, `temporal_entity_tracker.py` -> `perception.py`.
- `vla_policy_node.py`, `policy_model.py`, `trajectory_contract.py`, `visual_standoff_guard.py`, `safety_gate.py` -> `decision.py` plus thin `decision_node.py`.
- `task_instruction_node.py` is deleted; its initial parameter is read by `language_node.py`.
- `bridge_node.cpp` remains the bridge implementation; its executable and ROS name become `bridge_node`.

## Task 1: Structure Contract

**Files:** Create `src/vla/test/test_minimal_node_structure.py`; modify runtime identity, language, perception, and policy contract tests.

- [ ] Add failing assertions for exactly these entry points: `language = vla.language_node:main`, `perception = vla.perception_node:main`, and `decision = vla.decision_node:main`.
- [ ] Assert that `task_instruction`, `temporal_entity_tracker`, `safety_gate`, `language_qwen`, `image_entity_perception`, and `vla_policy` are absent; target modules exist; support modules do not; and `/vla/perceived_entities`, `/vla/tracked_entities`, `/vla/policy_displacement` are absent from runtime files.
- [ ] Run `python -m pytest -q src/vla/test/test_minimal_node_structure.py` on Jetson and confirm failure against the old structure.
- [ ] Commit the red tests with `git add src/vla/test && git commit -m "test: define minimal VLA node contract"`.

## Task 2: Language

**Files:** Create `src/vla/vla/language.py` and `language_node.py`; update language tests, `src/vla/setup.py`, and launch; delete `language_encoder.py` and `language_qwen_node.py` after migration.

- [ ] Move `LanguageEncoderError` subclasses, `EncodingResult`, `USVLanguageEncoder`, embedding constants, `LanguageEmbeddingState`, and pure payload/validation helpers into `language.py`; it must not import `rclpy` or ROS messages.
- [ ] Create `language_node.py` containing only parameters, `/task/text` subscription, `/vla/language_embedding` publisher, timer, ROS conversion, logs, lifecycle, and `main()`.
- [ ] Preserve CUDA-only behavior, invalid embedding publication, caching, and `release_model_after_encode=true`: a released encoder accepts only the same cached task and a changed task requires restart.
- [ ] Set the setup entry point to `"language = vla.language_node:main"` and launch executable/name to `language`; add no compatibility alias.
- [ ] Update imports and run `python -m pytest -q src/vla/test/test_language_encoder.py src/vla/test/test_language_qwen_node.py`; expected: all pass and old module imports are gone.
- [ ] Commit with `git add src/vla src/bringup/launch/vla_closed_loop.launch.py && git rm src/vla/vla/language_encoder.py src/vla/vla/language_qwen_node.py && git commit -m "refactor: make language node and algorithm explicit"`.

## Task 3: Perception and Tracking

**Files:** Create `src/vla/vla/perception.py` and `perception_node.py`; update perception/tracker tests, setup, and launch; delete the old perception, visual encoder, and tracker modules after migration.

- [ ] Move image contracts, JPEG decode, task parsing, model/prediction code, `CameraProfile`, visual encoder classes/errors, and projection helpers into `perception.py`.
- [ ] Move `FrameMetadata`, `GeometryObservation`, `TrackedEntity`, `TemporalEntityTracker`, `_DropoutRecovery`, and tracker errors into `perception.py`. Preserve first-frame invalid velocity, monotonic timestamp checks, identity reset, bounded dropout prediction, and expiration without fabricated entities.
- [ ] Create `perception_node.py` with only ROS parameters, subscriptions to `/ue/camera_frame` and `/vla/language_embedding`, publisher `/vla/entities`, callbacks, logs, lifecycle, and `main()`.
- [ ] In the frame callback, decode JPEG, call pure prediction, convert predictions to observations, call tracker and dropout recovery, then publish one final `EntityArray`. Remove `/task/text` subscription and both perceived/tracked topics.
- [ ] Remove `enhance_low_light_image()` and all image preprocessing parameters; pass decoded sRGB directly to the model while retaining invalid-frame/model fail-closed handling.
- [ ] Set setup entry point to `"perception = vla.perception_node:main"` and launch executable/name to `perception`; remove the tracker launch node.
- [ ] Run `python -m pytest -q src/vla/test/test_image_entity_perception.py src/vla/test/test_image_entity_perception_node.py src/vla/test/test_temporal_entity_tracker.py`; expected: all pass and deleted-module imports are gone.
- [ ] Commit the migration after adding new files/tests and removing `image_entity_perception.py`, `image_entity_perception_node.py`, `visual_encoder.py`, and `temporal_entity_tracker.py`.

## Task 4: Decision and Safety

**Files:** Create `src/vla/vla/decision.py` and `decision_node.py`; update policy/safety/trajectory/guard/entity tests, setup, and launch; delete old policy/support/safety modules after migration.

- [ ] Move policy model classes and runner, action contracts, identity synchronization, entity feature construction, smoothing, and input validation into `decision.py`.
- [ ] Move guard functions and pure safety functions (`SafetyGateConfig`, `SafetyGateResult`, rejection codes, collision checks, rate limiting, and `evaluate_safety_gate`) into `decision.py` without changing rejection semantics.
- [ ] Create `decision_node.py` with only ROS parameters, subscriptions to `/vla/language_embedding` and `/vla/entities`, publisher `/control/desired_displacement`, timers, ROS conversion, logs, lifecycle, and `main()`.
- [ ] Remove the policy feedback subscription to `/control/desired_displacement`; store the last accepted action internally for smoothing/rate limiting. Publish an invalid zero command on invalid, stale, non-finite, over-limit, collision-risk, or E-STOP conditions.
- [ ] Replace temporary `EntityFeatures` ROS messages with a Python result dataclass/NumPy arrays inside `decision.py`.
- [ ] Set setup entry point to `"decision = vla.decision_node:main"`, launch executable/name to `decision`, remove the safety node, and remove `/vla/policy_displacement`.
- [ ] Run the existing policy, smoothing, sync, safety, trajectory, guard, and entity-feature tests after updating imports. Expected: all pass and only the direct final control topic remains.
- [ ] Commit after removing `vla_policy_node.py`, `policy_model.py`, `trajectory_contract.py`, `visual_standoff_guard.py`, and `safety_gate.py`.

## Task 5: Remove Obsolete Nodes and Message

**Files:** Delete `src/vla/vla/task_instruction_node.py` and `src/interfaces/msg/EntityFeatures.msg`; modify setup, launch, `src/interfaces/CMakeLists.txt`, interface tests, and all references.

- [ ] Remove `task_instruction` from setup and launch. `language` reads the initial `task_text` parameter; no internal task publisher remains.
- [ ] Before deleting the message, run `rg -n "EntityFeatures|entity_features|/vla/entity_features|task_instruction" src`; expected: only migration tests/configuration references remain.
- [ ] Remove the message file and CMake entry, add absence assertions, then run `colcon build --packages-select interfaces vla` and the interface/structure tests.
- [ ] Commit the deletion only after the build passes and no runtime import or generated configuration reference remains.

## Task 6: Rename the C++ Bridge

**Files:** Modify `src/bridge/CMakeLists.txt`, `src/bridge/src/bridge_node.cpp`, `src/bridge/config/ue_bridge.yaml`, launch, and bridge/runtime identity tests.

- [ ] Add a failing naming test for CMake target/install target, C++ `Node("bridge_node")`, YAML root `bridge_node:`, and launch executable/name `bridge_node`.
- [ ] Replace every CMake target occurrence of `ue_object_deliverer_bridge_node` with `bridge_node`; keep source path `src/bridge/src/bridge_node.cpp`.
- [ ] Rename the C++ ROS node, YAML root, launch executable/name, tests, docs, and scripts. Do not change TCP ports, JSON keys, coordinate signs, or `DesiredDisplacement` fields.
- [ ] Run `colcon build --packages-select bridge` and the runtime identity tests; expected: bridge builds and the old executable/name is absent.
- [ ] Commit with `git add src/bridge src/bringup/launch/vla_closed_loop.launch.py src/vla/test && git commit -m "refactor: rename UE bridge node"`.

## Task 7: UE5 JPEG Contract

**Files:** In the separate `asv-unreal-simulation` repository, modify `Source/EDGE/Private/ImageCompressionLibrary.cpp`, its public header, and any Blueprint/config asset with an explicit JPEG quality pin.

- [ ] Enter `C:\Users\LIU\Documents\ChatGPT\实习面试项目整理\asv-unreal-simulation`, verify `git status --short`, then locate controls with `rg -n "JPEG|Quality|Gamma|gamma|Brightness|brightness|Contrast|contrast|EnhanceCapture|LinearToGamma" Source Content`; check Blueprint pins before relying on C++ defaults.
- [ ] Remove gamma/brightness/contrast enhancement and `EnhanceCaptureChannel()`/`EnhanceCapturePixels()`. Keep required linear-to-sRGB conversion and set the final JPEG quality to 95.
- [ ] Keep dimensions, channel statistics, and optional JPEG dump diagnostics. Do not add Jetson compensation.
- [ ] Build the actual UE5 project, capture a JPEG, inspect the saved output and confirm no second gamma transform darkens it.
- [ ] In the UE5 repository, stage only inspected source/assets and commit with `git commit -m "fix: emit standard sRGB JPEG frames from UE5"`. Do not include UE5 files in the Jetson repository commit.

## Task 8: Full Verification and UE5 Closed Loop

- [ ] In the Jetson repository, search for old names/topics and preprocessing with `rg -n "task_instruction|temporal_entity_tracker|safety_gate|language_qwen|image_entity_perception|vla_policy|ue_object_deliverer_bridge_node|/vla/perceived_entities|/vla/tracked_entities|/vla/policy_displacement|EntityFeatures|enhance_low_light_image|image_preprocess_" src`; in the UE5 repository search the enhancement terms under `Source` and `Content`. Expected: no runtime residue.
- [ ] Run `python -m compileall -q src/vla/vla src/vla/test`, `colcon build --symlink-install --packages-select interfaces vla bridge bringup`, and `python -m pytest -q src/vla/test` on Jetson.
- [ ] Start launch and verify `ros2 node list` contains `/language`, `/perception`, `/decision`, `/bridge_node`; inspect publisher ownership for `/vla/language_embedding`, `/vla/entities`, and `/control/desired_displacement`.
- [ ] Exercise invalid language, entity, and model states; verify `decision` publishes invalid zero displacement and never emits non-finite or over-limit action.
- [ ] With real UE5 running, record same-run `run_id`, `scene_seed`, and `frame_index` across `CameraFrame -> EntityArray -> DesiredDisplacement -> bridge JSON`; verify actual UE5 motion and image brightness.
- [ ] Report static checks, offline tests, Jetson build/tests, ROS graph checks, and UE5 closed-loop evidence separately. Do not call complete without same-run UE5 evidence.
