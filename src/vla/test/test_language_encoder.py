import numpy as np
import pytest

from vla.language_encoder import (
    EmptyInstructionError,
    InstructionTooLongError,
    InvalidEmbeddingError,
    LanguageEncoderMemoryError,
    USVLanguageEncoder,
)


class FakeParameter:
    def __init__(self):
        self.requires_grad = True

    def requires_grad_(self, enabled):
        self.requires_grad = enabled
        return self


class FakeModel:
    def __init__(self, values=None, native_dim=1024):
        self.calls = 0
        self.call_sentences = []
        self.call_kwargs = []
        self.eval_called = False
        self.parameter = FakeParameter()
        self.native_dim = native_dim
        if values is None:
            values = np.arange(1, native_dim + 1, dtype=np.float32)
        self.values = np.asarray(values)

    def eval(self):
        self.eval_called = True
        return self

    def parameters(self):
        return [self.parameter]

    def get_sentence_embedding_dimension(self):
        return self.native_dim

    def encode(self, sentences, **kwargs):
        self.calls += 1
        self.call_sentences.append(tuple(sentences))
        self.call_kwargs.append(dict(kwargs))
        return np.vstack([self.values for _ in sentences])


def make_encoder(model=None, **kwargs):
    if model is None:
        model = FakeModel()
    return USVLanguageEncoder(
        "unused-in-injected-model-test",
        model=model,
        inference_context=lambda: __import__("contextlib").nullcontext(),
        **kwargs,
    )


def test_fixed_shape_dtype_norm_and_cache():
    model = FakeModel()
    encoder = make_encoder(model=model)

    first = encoder.encode_with_metadata("跟随红色目标船，保持5米")
    second = encoder.encode_with_metadata("跟随红色目标船，保持5米")

    assert first.embedding.shape == (256,)
    assert first.embedding.dtype == np.float32
    assert np.linalg.norm(first.embedding) == pytest.approx(1.0, abs=1.0e-6)
    assert first.cached is False
    assert second.cached is True
    assert np.array_equal(first.embedding, second.embedding)
    assert model.calls == 1
    assert model.call_sentences == [
        ("Instruct: Encode an instruction for a twin-thruster unmanned "
         "surface vessel performing follow or stop tasks.\n"
         "Query: 跟随红色目标船，保持5米",)
    ]
    assert model.call_kwargs[0]["batch_size"] == 1
    assert encoder.cache_entries == 1


def test_changed_instruction_reencodes_one_item_and_keeps_cache_isolated():
    model = FakeModel()
    encoder = make_encoder(model=model)

    first = encoder.encode_with_metadata("跟随红色目标船")
    changed = encoder.encode_with_metadata("立即停止")
    repeated = encoder.encode_with_metadata("跟随红色目标船")

    assert first.cached is False
    assert changed.cached is False
    assert repeated.cached is True
    assert model.calls == 2
    assert [len(sentences) for sentences in model.call_sentences] == [1, 1]
    assert [kwargs["batch_size"] for kwargs in model.call_kwargs] == [1, 1]
    assert encoder.cache_entries == 2


def test_model_is_put_in_eval_mode_and_frozen():
    model = FakeModel()
    make_encoder(model=model)

    assert model.eval_called is True
    assert model.parameter.requires_grad is False


def test_whitespace_is_normalized_for_cache_key():
    model = FakeModel()
    encoder = make_encoder(model=model)

    encoder.encode("  立即停止  ")
    result = encoder.encode_with_metadata("立即停止")

    assert result.cached is True
    assert model.calls == 1


def test_empty_instruction_is_rejected():
    encoder = make_encoder()
    with pytest.raises(EmptyInstructionError):
        encoder.encode(" \n ")


def test_overlong_instruction_is_rejected():
    encoder = make_encoder(max_chars=8)
    with pytest.raises(InstructionTooLongError):
        encoder.encode("跟随红色目标船保持五米")


def test_short_backend_vector_is_rejected():
    model = FakeModel(
        values=np.arange(1, 129, dtype=np.float32),
        native_dim=None,
    )
    encoder = make_encoder(model=model)
    with pytest.raises(InvalidEmbeddingError):
        encoder.encode("立即停止")


def test_nonfinite_backend_vector_is_rejected():
    values = np.ones(1024, dtype=np.float32)
    values[3] = np.nan
    encoder = make_encoder(model=FakeModel(values=values))
    with pytest.raises(InvalidEmbeddingError):
        encoder.encode("立即停止")


def test_zero_backend_vector_is_rejected():
    values = np.zeros(1024, dtype=np.float32)
    encoder = make_encoder(model=FakeModel(values=values))
    with pytest.raises(InvalidEmbeddingError):
        encoder.encode("立即停止")


def test_cuda_memory_failure_is_diagnostic_and_not_swallowed():
    class OutOfMemoryModel(FakeModel):
        def encode(self, sentences, **kwargs):
            raise RuntimeError("CUDA out of memory while allocating input")

    encoder = make_encoder(model=OutOfMemoryModel(), device="cuda")
    with pytest.raises(LanguageEncoderMemoryError, match="CUDA_MEMORY_ERROR"):
        encoder.encode("立即停止")
