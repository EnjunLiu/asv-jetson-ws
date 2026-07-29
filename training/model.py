"""Day 14 small multimodal policy with one bounded trajectory output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SmallPolicyConfig:
    language_dim: int = 256
    visual_dim: int = 576
    entity_count: int = 16
    entity_geometry_dim: int = 16
    ego_dim: int = 2
    horizon: int = 20
    action_dim: int = 2
    language_hidden: int = 128
    visual_hidden: int = 128
    entity_geometry_hidden: int = 64
    entity_hidden: int = 192
    ego_hidden: int = 32
    fusion_hidden: int = 256
    maximum_step_m: float = 0.3
    invalid_stop_logit: float = 20.0
    maximum_trainable_parameters: int = 2_000_000

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SmallPolicyConfig":
        model = value.get("model", value)
        if not isinstance(model, Mapping):
            raise ValueError("model configuration must be a mapping")
        known = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = set(model) - known
        if unknown:
            raise ValueError(f"unknown model configuration keys: {sorted(unknown)}")
        return cls(**dict(model))

    def __post_init__(self) -> None:
        positive_ints = (
            self.language_dim,
            self.visual_dim,
            self.entity_count,
            self.entity_geometry_dim,
            self.ego_dim,
            self.horizon,
            self.action_dim,
            self.language_hidden,
            self.visual_hidden,
            self.entity_geometry_hidden,
            self.entity_hidden,
            self.ego_hidden,
            self.fusion_hidden,
            self.maximum_trainable_parameters,
        )
        if any(value <= 0 for value in positive_ints):
            raise ValueError("all dimensions and parameter limits must be positive")
        if self.maximum_step_m <= 0.0:
            raise ValueError("maximum_step_m must be positive")
        if not torch.isfinite(torch.tensor(self.invalid_stop_logit)):
            raise ValueError("invalid_stop_logit must be finite")


@dataclass(frozen=True)
class PolicyOutput:
    """Policy tensors plus an explicit fail-closed sample-validity mask."""

    trajectory: Tensor
    stop_logit: Tensor
    valid_mask: Tensor
    increments: Tensor


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
        nn.GELU(),
    )


class SmallTrajectoryPolicy(nn.Module):
    """Fuse frozen cache features into one structurally bounded trajectory.

    Language and visual tensors are detached at the model boundary. This keeps
    the Day 13 backbones frozen even if a caller accidentally supplies tensors
    connected to a live encoder graph.
    """

    def __init__(self, config: SmallPolicyConfig | None = None) -> None:
        super().__init__()
        self.config = config or SmallPolicyConfig()
        cfg = self.config

        self.language_encoder = _mlp(
            cfg.language_dim, cfg.language_hidden, cfg.language_hidden
        )
        self.global_visual_encoder = nn.Sequential(
            nn.Linear(cfg.visual_dim, cfg.visual_hidden),
            nn.GELU(),
        )
        self.entity_visual_encoder = _mlp(
            cfg.visual_dim, cfg.visual_hidden, cfg.visual_hidden
        )
        self.entity_geometry_encoder = _mlp(
            cfg.entity_geometry_dim,
            cfg.entity_geometry_hidden,
            cfg.entity_geometry_hidden,
        )
        self.entity_fusion = nn.Sequential(
            nn.Linear(
                cfg.visual_hidden + cfg.entity_geometry_hidden,
                cfg.entity_hidden,
            ),
            nn.GELU(),
        )
        self.entity_attention = nn.Linear(cfg.entity_hidden, 1)
        self.ego_encoder = _mlp(cfg.ego_dim, cfg.ego_hidden, cfg.ego_hidden)

        fusion_input_dim = (
            cfg.language_hidden
            + cfg.visual_hidden
            + cfg.entity_hidden
            + cfg.ego_hidden
        )
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, cfg.fusion_hidden),
            nn.GELU(),
            nn.LayerNorm(cfg.fusion_hidden),
            nn.Linear(cfg.fusion_hidden, cfg.fusion_hidden),
            nn.GELU(),
        )
        self.trajectory_head = nn.Linear(
            cfg.fusion_hidden, cfg.horizon * cfg.action_dim
        )
        self.stop_head = nn.Linear(cfg.fusion_hidden, 1)

        parameter_count = self.trainable_parameter_count()
        if parameter_count > cfg.maximum_trainable_parameters:
            raise ValueError(
                f"policy has {parameter_count} trainable parameters; "
                f"limit is {cfg.maximum_trainable_parameters}"
            )

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    @staticmethod
    def _expect_shape(tensor: Tensor, shape: tuple[int, ...], name: str) -> None:
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"{name} shape {tuple(tensor.shape)} does not match {shape}"
            )

    @staticmethod
    def _as_mask(
        mask: Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
        name: str,
    ) -> Tensor:
        if mask is None:
            return torch.ones(batch_size, dtype=torch.bool, device=device)
        if tuple(mask.shape) != (batch_size,):
            raise ValueError(
                f"{name} shape {tuple(mask.shape)} does not match "
                f"{(batch_size,)}"
            )
        return mask.to(device=device, dtype=torch.bool)

    @staticmethod
    def _sanitize_masked(values: Tensor, mask: Tensor, name: str) -> Tensor:
        expanded = mask
        while expanded.ndim < values.ndim:
            expanded = expanded.unsqueeze(-1)
        expanded = expanded.expand_as(values)
        active_values = values[expanded]
        if active_values.numel() and not torch.isfinite(active_values).all():
            raise ValueError(f"{name} contains NaN or Inf in an active position")
        return torch.where(expanded, values, torch.zeros_like(values))

    def forward(
        self,
        *,
        language: Tensor,
        global_visual: Tensor,
        entity_visual: Tensor,
        entity_geometry: Tensor,
        ego: Tensor,
        language_valid: Tensor | None = None,
        global_visual_mask: Tensor | None = None,
        entity_visual_mask: Tensor | None = None,
        entity_geometry_mask: Tensor | None = None,
        ego_valid: Tensor | None = None,
        policy_input_valid: Tensor | None = None,
    ) -> PolicyOutput:
        cfg = self.config
        if language.ndim != 2:
            raise ValueError("language must have shape [B, language_dim]")
        batch_size = int(language.shape[0])
        device = language.device
        dtype = language.dtype
        expected_device_dtype = (
            global_visual,
            entity_visual,
            entity_geometry,
            ego,
        )
        if any(tensor.device != device for tensor in expected_device_dtype):
            raise ValueError("all policy inputs must be on the same device")
        if any(tensor.dtype != dtype for tensor in expected_device_dtype):
            raise ValueError("all floating policy inputs must share one dtype")

        self._expect_shape(
            language, (batch_size, cfg.language_dim), "language"
        )
        self._expect_shape(
            global_visual,
            (batch_size, cfg.visual_dim),
            "global_visual",
        )
        self._expect_shape(
            entity_visual,
            (batch_size, cfg.entity_count, cfg.visual_dim),
            "entity_visual",
        )
        self._expect_shape(
            entity_geometry,
            (
                batch_size,
                cfg.entity_count,
                cfg.entity_geometry_dim,
            ),
            "entity_geometry",
        )
        self._expect_shape(ego, (batch_size, cfg.ego_dim), "ego")

        language_mask = self._as_mask(
            language_valid,
            batch_size=batch_size,
            device=device,
            name="language_valid",
        )
        global_mask = self._as_mask(
            global_visual_mask,
            batch_size=batch_size,
            device=device,
            name="global_visual_mask",
        )
        ego_mask = self._as_mask(
            ego_valid,
            batch_size=batch_size,
            device=device,
            name="ego_valid",
        )
        input_mask = self._as_mask(
            policy_input_valid,
            batch_size=batch_size,
            device=device,
            name="policy_input_valid",
        )

        if entity_visual_mask is None:
            entity_visual_mask = torch.ones(
                batch_size,
                cfg.entity_count,
                dtype=torch.bool,
                device=device,
            )
        if entity_geometry_mask is None:
            entity_geometry_mask = torch.ones(
                batch_size,
                cfg.entity_count,
                dtype=torch.bool,
                device=device,
            )
        self._expect_shape(
            entity_visual_mask,
            (batch_size, cfg.entity_count),
            "entity_visual_mask",
        )
        self._expect_shape(
            entity_geometry_mask,
            (batch_size, cfg.entity_count),
            "entity_geometry_mask",
        )
        visual_entity_mask = entity_visual_mask.to(
            device=device, dtype=torch.bool
        )
        geometry_entity_mask = entity_geometry_mask.to(
            device=device, dtype=torch.bool
        )

        valid_mask = language_mask & global_mask & ego_mask & input_mask
        geometry_entity_mask = geometry_entity_mask & valid_mask.unsqueeze(1)
        visual_entity_mask = (
            visual_entity_mask
            & geometry_entity_mask
            & valid_mask.unsqueeze(1)
        )

        language_clean = self._sanitize_masked(
            language.detach(), language_mask, "language"
        )
        global_clean = self._sanitize_masked(
            global_visual.detach(), global_mask, "global_visual"
        )
        entity_visual_clean = self._sanitize_masked(
            entity_visual.detach(), visual_entity_mask, "entity_visual"
        )
        entity_geometry_clean = self._sanitize_masked(
            entity_geometry, geometry_entity_mask, "entity_geometry"
        )
        ego_clean = self._sanitize_masked(ego, ego_mask, "ego")

        language_token = self.language_encoder(language_clean)
        global_token = self.global_visual_encoder(global_clean)
        entity_visual_token = self.entity_visual_encoder(entity_visual_clean)
        entity_geometry_token = self.entity_geometry_encoder(
            entity_geometry_clean
        )
        entity_token = self.entity_fusion(
            torch.cat((entity_visual_token, entity_geometry_token), dim=-1)
        )

        attention_score = self.entity_attention(entity_token).squeeze(-1)
        attention_weight = torch.exp(attention_score.clamp(-20.0, 20.0))
        attention_weight = attention_weight * geometry_entity_mask.to(
            dtype=dtype
        )
        attention_weight = attention_weight / attention_weight.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0e-12)
        pooled_entity = torch.sum(
            entity_token * attention_weight.unsqueeze(-1), dim=1
        )
        ego_token = self.ego_encoder(ego_clean)

        fused = self.fusion(
            torch.cat(
                (language_token, global_token, pooled_entity, ego_token),
                dim=-1,
            )
        )
        stop_logit = self.stop_head(fused)
        raw_increments = self.trajectory_head(fused).reshape(
            batch_size, cfg.horizon, cfg.action_dim
        )
        raw_norm = torch.linalg.vector_norm(
            raw_increments, dim=-1, keepdim=True
        )
        radial_scale = torch.where(
            raw_norm > 1.0e-6,
            torch.tanh(raw_norm) / raw_norm.clamp_min(1.0e-6),
            torch.ones_like(raw_norm),
        )
        movement_gate = torch.sigmoid(-stop_logit).unsqueeze(1)
        increments = (
            raw_increments
            * radial_scale
            * cfg.maximum_step_m
            * movement_gate
        )
        trajectory = torch.cumsum(increments, dim=1)

        sample_mask = valid_mask.view(batch_size, 1, 1)
        increments = torch.where(
            sample_mask, increments, torch.zeros_like(increments)
        )
        trajectory = torch.where(
            sample_mask, trajectory, torch.zeros_like(trajectory)
        )
        stop_logit = torch.where(
            valid_mask.view(batch_size, 1),
            stop_logit,
            torch.full_like(stop_logit, cfg.invalid_stop_logit),
        )
        return PolicyOutput(
            trajectory=trajectory,
            stop_logit=stop_logit,
            valid_mask=valid_mask,
            increments=increments,
        )
