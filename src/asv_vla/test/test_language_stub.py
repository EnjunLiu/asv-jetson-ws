"""Pure tests for language-stub embedding validation (no ROS daemon)."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np


LANGUAGE_STUB = (
    Path(__file__).resolve().parents[1] / "asv_vla" / "language_stub_node.py"
)


def _load_reader():
    tree = ast.parse(LANGUAGE_STUB.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_read_embedding"
    ]
    namespace = {"Path": Path, "np": np, "EMBEDDING_DIM": 256}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(LANGUAGE_STUB), "exec"), namespace)
    return namespace["_read_embedding"]


read_embedding = _load_reader()


def test_valid_embedding_is_loaded_without_ros(tmp_path):
    path = tmp_path / "red.npy"
    expected = np.arange(256, dtype=np.float32)
    np.save(path, expected)

    embedding, model_id = read_embedding(path)

    assert embedding == expected.tolist()
    assert model_id == "stub:file:red"


def test_missing_or_wrong_shape_embedding_is_rejected(tmp_path):
    missing = read_embedding(tmp_path / "missing.npy")
    assert missing is None

    wrong_shape = tmp_path / "wrong.npy"
    np.save(wrong_shape, np.zeros(255, dtype=np.float32))
    assert read_embedding(wrong_shape) is None
