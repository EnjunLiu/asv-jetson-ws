from types import SimpleNamespace

from asv_vla.kinematic_executor import expert_source_identity


def test_expert_identity_includes_full_source_frame_identity():
    source = SimpleNamespace(
        run_id="run-1",
        scene_seed=42,
        frame_index=7,
        stamp_us=123456,
    )

    assert expert_source_identity(source) == (
        "run-1",
        42,
        7,
        123456,
    )


def test_same_frame_index_with_new_stamp_is_a_new_action_frame():
    first = SimpleNamespace(
        run_id="run-1",
        scene_seed=42,
        frame_index=7,
        stamp_us=123456,
    )
    second = SimpleNamespace(
        run_id="run-1",
        scene_seed=42,
        frame_index=7,
        stamp_us=123457,
    )

    assert expert_source_identity(first) != expert_source_identity(second)
