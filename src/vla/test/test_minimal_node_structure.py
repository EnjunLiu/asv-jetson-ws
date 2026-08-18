from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[3]
VLA = REPOSITORY / "src/vla/vla"
SETUP = REPOSITORY / "src/vla/setup.py"
LAUNCH = REPOSITORY / "src/bringup/launch/vla_closed_loop.launch.py"
RUNTIME_FILES = [SETUP, LAUNCH, *VLA.glob("*.py")]


def test_setup_exposes_only_target_vla_nodes() -> None:
    source = SETUP.read_text(encoding="utf-8")
    expected = {
        "language": "vla.language_node:main",
        "perception": "vla.perception_node:main",
        "decision": "vla.decision_node:main",
    }
    removed = {
    }
    for name, target in expected.items():
        assert f'"{name} = {target}"' in source
    for name in removed:
        assert f'"{name} =' not in source


def test_target_modules_exist_and_support_modules_are_absent() -> None:
    for name in (
        "language",
        "language_node",
        "perception",
        "perception_node",
        "decision",
        "decision_node",
    ):
        assert (VLA / f"{name}.py").exists()
    for name in (
        "language_encoder",
        "visual_encoder",
        "policy_model",
        "trajectory_contract",
        "visual_standoff_guard",
    ):
        assert not (VLA / f"{name}.py").exists()


def test_old_topics_and_node_names_are_absent_from_runtime_sources() -> None:
    source = "\n".join(path.read_text(encoding="utf-8") for path in RUNTIME_FILES)
    for value in (
        "/vla/perceived_entities",
        "/vla/tracked_entities",
        "/vla/policy_displacement",
    ):
        assert value not in source


def test_launch_uses_minimal_node_names() -> None:
    source = LAUNCH.read_text(encoding="utf-8")
    for name in ("language", "perception", "decision", "bridge_node"):
        assert f'executable="{name}"' in source
        assert f'name="{name}"' in source
