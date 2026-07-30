"""Day 19 VLA policy inference node.

Bridges frozen encoder outputs (language, visual, entities) to the
learned trajectory policy.  Publishes to ``/vla/policy_trajectory``.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import rclpy
from rclpy.node import Node
from asv_jetson_interfaces.msg import (
    SelectedTrajectory,
    TaskEmbedding,
    TaskFeatures,
    VisualFeatures,
)
import torch

from .trajectory_contract import ACTION_DIM, DT_SEC, FRAME_ID, HORIZON

POLICY_MODEL_VERSION = "day16_cross_loader_v4_seed17"

# Maximum staleness for each modality (seconds).
STALE_SEC = 1.0


class VLAPolicyNode(Node):
    """Subscribes to encoder topics and publishes one trajectory per frame."""

    def __init__(self, checkpoint_path: str = "") -> None:
        super().__init__("vla_policy")

        self.declare_parameter("checkpoint_path", checkpoint_path)

        # Latest encoder messages.
        self._language: TaskEmbedding | None = None
        self._visual: VisualFeatures | None = None
        self._entities: TaskFeatures | None = None
        self._language_stamp = 0.0
        self._visual_stamp = 0.0
        self._entities_stamp = 0.0

        # Subscribers.
        self._lang_sub = self.create_subscription(
            TaskEmbedding, "/vla/language_embedding", self._on_language, 10
        )
        self._vis_sub = self.create_subscription(
            VisualFeatures, "/vla/visual_features", self._on_visual, 10
        )
        self._ent_sub = self.create_subscription(
            TaskFeatures, "/vla/task_features", self._on_entities, 10
        )

        # Publisher.
        self._pub = self.create_publisher(
            SelectedTrajectory, "/vla/policy_trajectory", 10
        )

        # Load model.
        ckpt_path = (
            str(self.get_parameter("checkpoint_path")
                 .get_parameter_value().string_value)
            or checkpoint_path
        )
        self._model = self._load_model(ckpt_path) if ckpt_path else None
        if self._model is not None:
            self.get_logger().info(f"VLA policy loaded from {ckpt_path}")
        else:
            self.get_logger().warn("no checkpoint — publishing safe stop only")

        self._frame_seq = 0
        self._last_inference_ms = 0.0

    def _load_model(self, path: str) -> Any:
        import sys, os
        repo = os.path.expanduser("~/jetson_asv_ws")
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from training.model import SmallTrajectoryPolicy, SmallPolicyConfig

        cfg = SmallPolicyConfig(
            entity_attention_mode="language_additive",
            language_conditioned_entity_attention=True,
        )
        model = SmallTrajectoryPolicy(cfg)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval()
        model._device = device
        return model

    def _on_language(self, msg: TaskEmbedding) -> None:
        self._language = msg
        self._language_stamp = time.monotonic()

    def _on_visual(self, msg: VisualFeatures) -> None:
        self._visual = msg
        self._visual_stamp = time.monotonic()

    def _on_entities(self, msg: TaskFeatures) -> None:
        self._entities = msg
        self._entities_stamp = time.monotonic()
        # Entities arrive last (after visual), trigger inference.
        self._maybe_infer()

    def _maybe_infer(self) -> None:
        now = time.monotonic()
        lang = self._language
        vis = self._visual
        ent = self._entities

        # Staleness check.
        if lang is None or vis is None or ent is None:
            return
        if any(
            now - t > STALE_SEC
            for t in (self._language_stamp, self._visual_stamp, self._entities_stamp)
        ):
            return

        # Build fake ego (assume stationary for now — Day 4 interface has ego).
        ego = np.array([0.0, 0.0], dtype=np.float32)
        ego_valid = True

        msg = SelectedTrajectory()
        msg.stamp_us = int(ent.stamp_us)
        msg.run_id = str(ent.run_id)
        msg.frame_id = FRAME_ID
        msg.model_version = POLICY_MODEL_VERSION
        msg.dt = DT_SEC
        msg.horizon = HORIZON

        if self._model is None:
            # No model — safe stop.
            msg.delta_p_xy = [0.0] * (HORIZON * ACTION_DIM)
            msg.safe_stop = True
            msg.valid = True
            msg.reason = "NO_MODEL_LOADED"
            self._pub.publish(msg)
            return

        # Prepare inputs.
        device = self._model._device
        with torch.no_grad():
            language = torch.from_numpy(
                np.array(lang.embedding, dtype=np.float32).copy()
            ).unsqueeze(0).to(device)

            vf = np.array(vis.feature, dtype=np.float32).copy()
            vis_dim = int(vis.feature_dim)
            tok_count = int(vis.token_count)
            global_visual = torch.from_numpy(vf[:vis_dim]).unsqueeze(0).to(device)
            entity_count = max(tok_count - 1, 0)
            ev = np.zeros((entity_count, vis_dim), dtype=np.float32)
            if entity_count > 0:
                ev_flat = vf[vis_dim:vis_dim + entity_count * vis_dim]
                ev[:min(entity_count, len(ev_flat)//vis_dim)] = ev_flat.reshape(-1, vis_dim)[:entity_count]
            entity_visual = torch.from_numpy(ev).unsqueeze(0).to(device)

            ent_feat = np.array(ent.features, dtype=np.float32).copy()
            # Zero out color truth columns (14, 15) for policy input.
            ent_feat = ent_feat.reshape(1, ent.max_entities, ent.feature_dim)
            ent_feat[:, :, 14] = 0.0
            ent_feat[:, :, 15] = 0.0
            entity_geometry = torch.from_numpy(ent_feat).to(device)

            ego_t = torch.from_numpy(ego.copy()).unsqueeze(0).to(device)

            language_valid = torch.tensor(
                [lang.valid], dtype=torch.bool, device=device
            )
            global_visual_mask = torch.tensor(
                [vis.valid], dtype=torch.bool, device=device
            )
            vis_mask = np.array(vis.mask, dtype=bool).copy()
            ev_mask = torch.from_numpy(
                vis_mask[1:1+entity_count] if len(vis_mask) > 1 else np.zeros(entity_count, dtype=bool)
            ).unsqueeze(0).to(device)
            if ev_mask.shape[1] < entity_count:
                pad = torch.zeros(1, entity_count - ev_mask.shape[1], dtype=torch.bool, device=device)
                ev_mask = torch.cat([ev_mask, pad], dim=1)
            eg_mask = torch.from_numpy(
                np.array(ent.mask, dtype=bool).copy()
            ).unsqueeze(0).to(device)
            ego_valid_t = torch.tensor(
                [ego_valid], dtype=torch.bool, device=device
            )
            policy_valid = (
                language_valid & global_visual_mask & ego_valid_t
            )

            try:
                output = self._model(
                    language=language,
                    global_visual=global_visual,
                    entity_visual=entity_visual,
                    entity_geometry=entity_geometry,
                    ego=ego_t,
                    language_valid=language_valid,
                    global_visual_mask=global_visual_mask,
                    entity_visual_mask=ev_mask,
                    entity_geometry_mask=eg_mask,
                    ego_valid=ego_valid_t,
                    policy_input_valid=policy_valid,
                )

                traj = output.trajectory.cpu().numpy()[0].flatten().tolist()
                stop_logit = float(output.stop_logit.cpu().numpy()[0, 0])
                valid = bool(output.valid_mask.cpu().numpy()[0])

                msg.delta_p_xy = [float(v) for v in traj]
                msg.safe_stop = stop_logit > 0.0
                msg.valid = valid
                msg.reason = (
                    "POLICY_STOP" if msg.safe_stop else "POLICY_INFERRED"
                )
            except Exception as exc:
                msg.delta_p_xy = [0.0] * (HORIZON * ACTION_DIM)
                msg.safe_stop = True
                msg.valid = False
                msg.reason = f"INFERENCE_ERROR:{exc}"

        self._pub.publish(msg)
        self._frame_seq += 1


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VLAPolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
