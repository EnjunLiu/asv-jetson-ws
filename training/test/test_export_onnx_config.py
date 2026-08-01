from __future__ import annotations

import ast
from pathlib import Path

import pytest


EXPORT_SOURCE = Path(__file__).resolve().parents[1] / "export_onnx.py"
MODEL_V1_SOURCE = (
    Path(__file__).resolve().parents[1] / "config" / "model_small_v1.yaml"
)


def test_onnx_export_resolves_config_from_checkpoint_or_legacy_defaults() -> None:
    source = EXPORT_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "_checkpoint_model_config" in function_names
    assert "model_config" in source
    assert "dataclass_defaults_legacy" in source
    assert 'entity_attention_mode="language_additive"' not in source
    assert "strict=True" in source


def test_model_small_v1_omits_attention_overrides_and_uses_dataclass_defaults() -> None:
    source = MODEL_V1_SOURCE.read_text(encoding="utf-8")
    assert "entity_attention_mode:" not in source
    assert "language_conditioned_entity_attention:" not in source
    # SmallPolicyConfig defaults are the legacy architecture used by v1.
    assert 'entity_attention_mode: str = "legacy"' in (
        Path(__file__).resolve().parents[1] / "model.py"
    ).read_text(encoding="utf-8")
    assert "language_conditioned_entity_attention: bool = False" in (
        Path(__file__).resolve().parents[1] / "model.py"
    ).read_text(encoding="utf-8")


def test_checkpoint_config_preserves_attention_provenance() -> None:
    pytest.importorskip("torch")
    from training.export_onnx import _checkpoint_model_config

    legacy, legacy_source = _checkpoint_model_config({})
    assert legacy_source == "dataclass_defaults_legacy"
    assert legacy.entity_attention_mode == "legacy"
    assert legacy.language_conditioned_entity_attention is False

    additive, additive_source = _checkpoint_model_config(
        {
            "model_config": {
                "entity_attention_mode": "language_additive",
                "language_conditioned_entity_attention": True,
            }
        }
    )
    assert additive_source == "checkpoint:model_config"
    assert additive.entity_attention_mode == "language_additive"
    assert additive.language_conditioned_entity_attention is True
