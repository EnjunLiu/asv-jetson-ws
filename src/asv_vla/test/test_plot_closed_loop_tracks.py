"""Focused contract tests for the closed-loop track plotting tool."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pytest


REPOSITORY = Path(__file__).resolve().parents[3]
TOOL_PATH = REPOSITORY / "tools" / "plot_closed_loop_tracks.py"
REPORT = (
    REPOSITORY
    / "pc_datasets"
    / "reports"
    / "closed_loop_20260805"
    / "single_point_policy_dominant_v9"
)
ENTITY_IDS = ("target_red", "target_blue", "target_left", "target_right")


@pytest.fixture(scope="module")
def plot_tool():
    spec = importlib.util.spec_from_file_location("plot_closed_loop_tracks", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_scene_log(path: Path, offset_cm: float = 0.0) -> None:
    positions = {
        "target_red": (500.0 + offset_cm, 0.0),
        "target_blue": (200.0 + offset_cm, 100.0),
        "target_left": (350.0 + offset_cm, -200.0),
        "target_right": (350.0 + offset_cm, 200.0),
    }
    lines: list[str] = []
    for time_s in (0.0, 1.0):
        for entity, (x_cm, y_cm) in positions.items():
            lines.append(
                f"SCENE_TARGET_POS t={time_s} entity={entity} "
                f"world=X={x_cm} Y={y_cm} Z=0"
            )
        lines.append(f"SCENE_ASV_POS t={time_s} world=X={time_s * 10} Y=0 Z=0")
    path.write_text("\n".join(lines), encoding="utf-8")


def _scene(plot_tool, tmp_path: Path, name: str, offset_cm: float = 0.0):
    path = tmp_path / (name.replace(" ", "_") + ".log")
    _write_scene_log(path, offset_cm)
    return plot_tool.parse_scene(f"{name}={path}")


def test_parse_scene_retains_all_entity_tracks(plot_tool, tmp_path: Path) -> None:
    scene = _scene(plot_tool, tmp_path, "RED 4m")

    assert set(scene.all_entity_tracks) == set(ENTITY_IDS)
    assert all(len(scene.all_entity_tracks[entity]) == 2 for entity in ENTITY_IDS)
    assert scene.target == scene.all_entity_tracks["target_red"]
    assert [time_s for time_s, _ in scene.standoff_error()] == [0.0, 1.0]
    assert [error_m for _, error_m in scene.standoff_error()] == pytest.approx([1.0, 0.9])


def test_plot_scenes_uses_fixed_layout_and_signed_error(plot_tool, tmp_path: Path, monkeypatch) -> None:
    scenes = [
        _scene(plot_tool, tmp_path, "RED 4m"),
        _scene(plot_tool, tmp_path, "BLUE 3m"),
        _scene(plot_tool, tmp_path, "RED 3m"),
    ]
    original_close = plot_tool.plt.close
    monkeypatch.setattr(plot_tool.plt, "close", lambda figure: None)
    output = tmp_path / "tracks.png"
    plot_tool.plot_scenes(scenes, output)
    figure = plt.gcf()
    try:
        assert output.is_file()
        assert len(figure.axes) == 6
        top_axes, error_axes = figure.axes[:3], figure.axes[3:]
        assert [axis.get_title() for axis in top_axes] == [scene.name for scene in scenes]
        for axis in top_axes:
            assert {line.get_label() for line in axis.lines} >= {"ASV", *ENTITY_IDS}
            assert axis.get_xlim() == pytest.approx(top_axes[0].get_xlim())
            assert axis.get_ylim() == pytest.approx(top_axes[0].get_ylim())
            assert axis.get_aspect() == pytest.approx(1.0)
        for axis in error_axes:
            assert axis.get_ylim() == pytest.approx(error_axes[0].get_ylim())
        red_error = next(line for line in error_axes[0].lines if line.get_label() == "target_red error")
        assert list(red_error.get_ydata()) == pytest.approx([1.0, 0.9])
        assert any(all(value == 0 for value in line.get_ydata()) for line in error_axes[0].lines)
    finally:
        original_close(figure)


def test_shared_entity_tracks_accepts_matching_pair_and_rejects_old_red3(plot_tool) -> None:
    red4 = plot_tool.parse_scene(f"RED 4m={REPORT / 'red4m' / 'ue_TRACK-SYNTH-RED-4m-V1.log'}")
    blue3 = plot_tool.parse_scene(f"BLUE 3m={REPORT / 'blue3m' / 'ue_TRACK-SYNTH-BLUE-3m-V1.log'}")
    red3 = plot_tool.parse_scene(f"RED 3m={REPORT / 'red3m' / 'ue_TRACK-SYNTH-RED-3m-V2.log'}")

    plot_tool.require_shared_entity_tracks([red4, blue3], tolerance_cm=5.0)
    with pytest.raises(ValueError, match=r"shared entity track mismatch for target_red"):
        plot_tool.require_shared_entity_tracks([red4, blue3, red3], tolerance_cm=5.0)


def test_shared_entity_tracks_reports_missing_tracks(plot_tool, tmp_path: Path) -> None:
    scene = _scene(plot_tool, tmp_path, "RED 4m")
    scene.entity_tracks.pop("target_right")

    with pytest.raises(ValueError, match=r"RED 4m is missing required entity tracks: target_right"):
        plot_tool.require_shared_entity_tracks([scene])
