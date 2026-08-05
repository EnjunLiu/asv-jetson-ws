#!/usr/bin/env python3
"""Plot UE world-coordinate closed-loop tracks and policy audit evidence.

The parser consumes runtime evidence only. UE ``SCENE_*`` records provide the
world trajectory and Jetson ``POLICY_*`` records provide audit counts. Missing
optional target or audit fields are represented as missing values in the plot
and metrics; no trajectory or counter is inferred from another scene.

Typical use::

    python plot_closed_loop_tracks.py \
        --input-dir pc_datasets/reports/closed_loop_20260805/single_point_policy_dominant \
        --output /tmp/closed_loop_tracks.png \
        --metrics /tmp/closed_loop_tracks.json

For an explicit pair of files, the backwards-compatible form is::

    python plot_closed_loop_tracks.py \
        --scene "RED 3m=/path/ue.log=/path/jetson.log" \
        --output /tmp/red3m.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
ASV_RE = re.compile(
    rf"SCENE_ASV_POS\b.*?\bt\s*=\s*({FLOAT}).*?"
    rf"\bworld\s*=\s*X\s*=\s*({FLOAT})\s+Y\s*=\s*({FLOAT})\s+"
    rf"Z\s*=\s*({FLOAT})",
    re.IGNORECASE,
)
TARGET_RE = re.compile(
    rf"SCENE_TARGET_POS\b.*?\bt\s*=\s*({FLOAT}).*?"
    rf"\bentity\s*=\s*([A-Za-z0-9_-]+).*?"
    rf"\bworld\s*=\s*X\s*=\s*({FLOAT})\s+Y\s*=\s*({FLOAT})\s+"
    rf"Z\s*=\s*({FLOAT})",
    re.IGNORECASE,
)
KEY_VALUE_RE = re.compile(
    r"(?P<key>[A-Za-z][A-Za-z0-9_]*)=(?P<value>\"[^\"]*\"|'[^']*'|[^\s,}]+)"
)
SCENE_TOKEN_RE = re.compile(
    r"(?:track[-_ ]*)?(red|blue)[-_ ]*([34])m(?:\b|[-_])", re.IGNORECASE
)
TRACK_EXTENSIONS = {".csv", ".json", ".jsonl", ".log", ".ndjson", ".txt", ".track"}
DEFAULT_SCENE_ORDER = ("RED 4m", "BLUE 3m", "RED 3m")
SCENE_COLORS = {"red": "#b23a48", "blue": "#2878a8"}
LINE_COLORS = ("#59636e", "#b23a48", "#2878a8")


@dataclass(frozen=True)
class Point:
    time_s: float
    x_cm: float
    y_cm: float


@dataclass(frozen=True)
class AuditCounts:
    """Audit counters, with ``None`` meaning that the source did not say."""

    events: int | None
    raw_observed: int | None
    policy_driven: int | None
    backstop: int | None
    hold: int | None
    fail_closed: int | None
    policy_stop: int | None
    source: str
    complete: bool
    observed_trace_records: int = 0

    @property
    def classified_events(self) -> int | None:
        values = (self.policy_driven, self.backstop, self.hold, self.fail_closed)
        return sum(values) if all(value is not None for value in values) else None

    @property
    def policy_dominance_rate(self) -> float | None:
        total = self.classified_events
        return self.policy_driven / total if total and self.policy_driven is not None else None

    @property
    def backstop_rate(self) -> float | None:
        total = self.classified_events
        return self.backstop / total if total and self.backstop is not None else None


@dataclass
class SceneTrack:
    name: str
    target_color: str
    target_entity: str
    desired_standoff_m: float
    ue_log: Path | None
    jetson_logs: tuple[Path, ...]
    asv: list[Point]
    target: list[Point]
    audits: AuditCounts | None

    @property
    def available(self) -> bool:
        return bool(self.asv)

    def standoff(self) -> list[tuple[float, float]]:
        if not self.target:
            return []
        target_by_time = {round(point.time_s, 3): point for point in self.target}
        result: list[tuple[float, float]] = []
        for asv in self.asv:
            target = target_by_time.get(round(asv.time_s, 3))
            if target is None:
                target = min(self.target, key=lambda item: abs(item.time_s - asv.time_s))
            distance_m = math.hypot(target.x_cm - asv.x_cm, target.y_cm - asv.y_cm) / 100.0
            result.append((asv.time_s, distance_m))
        return result


@dataclass
class _TrajectoryRecords:
    asv: dict[float, Point]
    targets: dict[str, dict[float, Point]]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_lines(path: Path) -> Iterable[str]:
    return _read_text(path).splitlines()


def _finite_point(time_s: Any, x_cm: Any, y_cm: Any) -> Point | None:
    try:
        point = Point(float(time_s), float(x_cm), float(y_cm))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (point.time_s, point.x_cm, point.y_cm)):
        return None
    return point


def _parse_text_trajectory(text: str, target_entity: str) -> _TrajectoryRecords:
    asv: dict[float, Point] = {}
    targets: dict[str, dict[float, Point]] = {}
    for line in text.splitlines():
        asv_match = ASV_RE.search(line)
        if asv_match:
            point = _finite_point(asv_match.group(1), asv_match.group(2), asv_match.group(3))
            if point is not None:
                asv[point.time_s] = point
        target_match = TARGET_RE.search(line)
        if target_match:
            entity = target_match.group(2).casefold()
            point = _finite_point(target_match.group(1), target_match.group(3), target_match.group(4))
            if point is not None:
                targets.setdefault(entity, {})[point.time_s] = point
    return _TrajectoryRecords(asv, targets)


def _normal_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).casefold())


def _mapping_value(mapping: Mapping[str, Any], *names: str) -> Any:
    wanted = {_normal_key(name) for name in names}
    for key, value in mapping.items():
        if _normal_key(key) in wanted:
            return value
    return None


def _structured_point(mapping: Mapping[str, Any], target_entity: str) -> tuple[str, Point] | None:
    entity_value = _mapping_value(mapping, "entity", "entity_id", "target_id", "actor", "name", "kind", "type", "role")
    entity = str(entity_value).casefold() if entity_value is not None else ""
    is_asv = "asv" in entity or any(_normal_key(key) in {"asv", "vessel", "vehicle"} for key in mapping)
    is_target = entity.startswith("target_") or "target" in entity
    if not is_asv and not is_target:
        return None
    time_value = _mapping_value(mapping, "time_s", "time", "t", "timestamp", "stamp")
    world = _mapping_value(mapping, "world", "position", "world_position")
    world_mapping = world if isinstance(world, Mapping) else {}
    x_value = _mapping_value(mapping, "x_cm", "world_x_cm", "x_m", "world_x_m", "world_x", "x")
    y_value = _mapping_value(mapping, "y_cm", "world_y_cm", "y_m", "world_y_m", "world_y", "y")
    if x_value is None:
        x_value = _mapping_value(world_mapping, "x_cm", "x_m", "x")
    if y_value is None:
        y_value = _mapping_value(world_mapping, "y_cm", "y_m", "y")
    if time_value is None or x_value is None or y_value is None:
        return None
    x_scale = 100.0 if _mapping_value(mapping, "x_m", "world_x_m") is not None else (
        100.0 if _mapping_value(world_mapping, "x_m") is not None else 1.0
    )
    y_scale = 100.0 if _mapping_value(mapping, "y_m", "world_y_m") is not None else (
        100.0 if _mapping_value(world_mapping, "y_m") is not None else 1.0
    )
    try:
        x_cm = float(x_value) * x_scale
        y_cm = float(y_value) * y_scale
    except (TypeError, ValueError):
        return None
    point = _finite_point(time_value, x_cm, y_cm)
    if point is None:
        return None
    if is_asv:
        return "asv", point
    if entity.startswith("target_"):
        return entity, point
    return target_entity, point


def _walk_mappings(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_mappings(child)


def _parse_structured_trajectory(text: str, path: Path, target_entity: str) -> _TrajectoryRecords:
    records = _TrajectoryRecords({}, {})
    parsed: Any = None
    if path.suffix.casefold() in {".json", ".jsonl", ".ndjson"}:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = []
            for line in text.splitlines():
                try:
                    parsed.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if parsed is not None:
        for mapping in _walk_mappings(parsed):
            point = _structured_point(mapping, target_entity)
            if point is None:
                continue
            kind, value = point
            if kind == "asv":
                records.asv[value.time_s] = value
            else:
                records.targets.setdefault(kind, {})[value.time_s] = value
        return records
    if path.suffix.casefold() == ".csv":
        try:
            for row in csv.DictReader(text.splitlines()):
                point = _structured_point(row, target_entity)
                if point is None:
                    continue
                kind, value = point
                if kind == "asv":
                    records.asv[value.time_s] = value
                else:
                    records.targets.setdefault(kind, {})[value.time_s] = value
        except (csv.Error, ValueError):
            pass
    return records


def parse_trajectory_file(path: Path, target_entity: str) -> tuple[list[Point], list[Point]]:
    """Return sorted ASV and requested-target points from one evidence file."""

    text = _read_text(path)
    records = _parse_text_trajectory(text, target_entity)
    if path.suffix.casefold() in {".csv", ".json", ".jsonl", ".ndjson"}:
        structured = _parse_structured_trajectory(text, path, target_entity)
        records.asv.update(structured.asv)
        for entity, points in structured.targets.items():
            records.targets.setdefault(entity, {}).update(points)
    target = records.targets.get(target_entity.casefold(), {})
    return (
        [records.asv[key] for key in sorted(records.asv)],
        [target[key] for key in sorted(target)],
    )


def parse_ue_log(path: Path, target_color: str) -> tuple[list[Point], list[Point]]:
    """Backwards-compatible UE parser using ``target_<color>``."""

    return parse_trajectory_file(path, f"target_{target_color}")


def _key_values(line: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for match in KEY_VALUE_RE.finditer(line):
        value = match.group("value").strip("\"'")
        result[match.group("key").casefold()] = value
    return result


def _int_field(fields: Mapping[str, Any], *names: str) -> int | None:
    for name in names:
        value = fields.get(name.casefold())
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _audit_fields(fields: Mapping[str, Any]) -> dict[str, int | None]:
    return {
        "events": _int_field(fields, "events", "total_events", "event_count"),
        "raw": _int_field(fields, "raw", "raw_count", "raw_actions", "raw_observed"),
        "policy_driven": _int_field(fields, "policy_driven", "policy_driven_count"),
        "backstop": _int_field(fields, "backstop", "backstop_count"),
        "hold": _int_field(fields, "hold", "hold_count", "deadband_hold"),
        "fail_closed": _int_field(fields, "fail_closed", "fail_closed_count"),
        "policy_stop": _int_field(fields, "policy_stop", "policy_stop_count"),
    }


def _normal_guard(value: str | None) -> str | None:
    if not value:
        return None
    normalized = re.sub(r"[^a-z]", "", value.casefold())
    if normalized in {"policydriven", "notapplied", "passthrough", "policyraw", "raw"}:
        return "policy_driven" if normalized != "raw" else "raw"
    if normalized in {"backstop", "standoffbackstop"}:
        return "backstop"
    if normalized in {"hold", "deadbandhold"}:
        return "hold"
    if normalized in {"failclosed", "policystop", "policyinvalid", "invalid", "visualtargetmissing"}:
        return "fail_closed"
    return None


def _audit_from_mapping(value: Any) -> dict[str, int | None] | None:
    for mapping in _walk_mappings(value):
        fields = {str(key).casefold(): item for key, item in mapping.items()}
        counts = _audit_fields(fields)
        if any(item is not None for item in counts.values()):
            return counts
    return None


def parse_audit_log(path: Path) -> AuditCounts | None:
    """Parse cumulative ``POLICY_AUDIT`` or explicitly sampled policy markers."""

    latest: dict[str, int | None] | None = None
    trace_counts = {"policy_driven": 0, "backstop": 0, "hold": 0, "fail_closed": 0}
    trace_records = 0
    raw_records = 0
    explicit_raw: int | None = None
    saw_cumulative_audit = False
    saw_marker = False

    for line in _read_lines(path):
        fields = _key_values(line)
        has_raw = "raw_dx=" in line.casefold() or "raw_dy=" in line.casefold() or "raw_action" in line.casefold()
        if has_raw:
            raw_records += 1
        if "policy_audit" in line.casefold():
            candidate = _audit_fields(fields)
            if any(item is not None for item in candidate.values()):
                latest = candidate
                explicit_raw = candidate["raw"]
                saw_cumulative_audit = True
                saw_marker = True
            continue

        marker = None
        if "policy_trace" in line.casefold():
            marker = _normal_guard(fields.get("guard_reason") or fields.get("guard_result") or fields.get("guard"))
        elif "policy_periodic_trace" in line.casefold():
            marker = _normal_guard(fields.get("guard_reason") or fields.get("guard"))
        if marker is not None:
            trace_records += 1
            if marker in trace_counts:
                trace_counts[marker] += 1
            elif marker == "raw":
                raw_records += 1
            saw_marker = True
            candidate = _audit_fields(fields)
            if any(item is not None for item in candidate.values()):
                latest = candidate
                explicit_raw = candidate["raw"]

        if '"hold_position":true' in line.casefold() or "hold_position=true" in line.casefold():
            trace_counts["hold"] += 1
            trace_records += 1
            saw_marker = True

    if not saw_marker and path.suffix.casefold() in {".json", ".jsonl", ".ndjson"}:
        try:
            mapping_counts = _audit_from_mapping(json.loads(_read_text(path)))
        except json.JSONDecodeError:
            mapping_counts = None
        if mapping_counts is not None:
            latest = mapping_counts
            explicit_raw = mapping_counts["raw"]
            saw_cumulative_audit = True
            saw_marker = True

    if not saw_marker:
        return None
    if latest is not None:
        raw_value = explicit_raw if explicit_raw is not None else (raw_records or None)
        return AuditCounts(
            events=latest["events"],
            raw_observed=raw_value,
            policy_driven=latest["policy_driven"],
            backstop=latest["backstop"],
            hold=latest["hold"],
            fail_closed=latest["fail_closed"],
            policy_stop=latest["policy_stop"],
            source="POLICY_AUDIT" if saw_cumulative_audit else "POLICY_TRACE samples",
            complete=latest["events"] is not None,
            observed_trace_records=trace_records,
        )
    return AuditCounts(
        events=None,
        raw_observed=raw_records or None,
        policy_driven=trace_counts["policy_driven"],
        backstop=trace_counts["backstop"],
        hold=trace_counts["hold"],
        fail_closed=trace_counts["fail_closed"],
        policy_stop=None,
        source="POLICY_TRACE samples",
        complete=False,
        observed_trace_records=trace_records,
    )


def _scene_slot(value: str) -> str | None:
    match = SCENE_TOKEN_RE.search(value)
    if not match:
        return None
    return f"{match.group(1).casefold()}{match.group(2)}m"


def _scene_definition(name: str) -> tuple[str, str, float]:
    slot = _scene_slot(name)
    if slot is None:
        raise ValueError(f"scene name must contain RED/BLUE and 3m/4m: {name}")
    color = "blue" if slot.startswith("blue") else "red"
    desired = float(slot[slot.find(color) + len(color) : -1])
    return slot, color, desired


def _candidate_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(root)
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() in TRACK_EXTENSIONS
    )


def _has_audit_marker(path: Path) -> bool:
    try:
        text = _read_text(path)
    except OSError:
        return False
    return any(marker in text.casefold() for marker in ("policy_audit", "policy_trace", "policy_periodic_trace"))


def _choose_candidate(candidates: list[tuple[Path, int]], role: str) -> Path | None:
    if not candidates:
        return None

    def score(item: tuple[Path, int]) -> tuple[int, int, str]:
        path, samples = item
        name = path.name.casefold()
        role_hint = 0
        if role == "ue":
            role_hint = 100 if any(token in name for token in ("ue", "scene", "track")) else 0
        else:
            role_hint = 100 if any(token in name for token in ("jetson", "vla", "policy", "audit")) else 0
        if "err" in name:
            role_hint -= 50
        return role_hint, samples, str(path)

    return max(candidates, key=score)[0]


def _discover_scenes(input_dir: Path, names: tuple[str, ...]) -> list[SceneTrack]:
    files = _candidate_files(input_dir)
    parsed_ue: dict[Path, tuple[list[Point], list[Point]]] = {}
    ue_candidates: dict[str, list[tuple[Path, int]]] = {}
    audit_candidates: dict[str, list[tuple[Path, int]]] = {}
    for path in files:
        slot = _scene_slot(path.name)
        if slot is None:
            continue
        slot_name = slot
        _, color, _ = _scene_definition(path.name)
        target_entity = f"target_{color}"
        try:
            asv, target = parse_trajectory_file(path, target_entity)
        except OSError:
            continue
        if asv:
            parsed_ue[path] = (asv, target)
            ue_candidates.setdefault(slot_name, []).append((path, len(asv)))
        if _has_audit_marker(path) or path.suffix.casefold() in {".json", ".jsonl", ".ndjson"}:
            audit = parse_audit_log(path)
            if audit is not None:
                score = audit.events if audit.events is not None else audit.observed_trace_records
                audit_candidates.setdefault(slot_name, []).append((path, score or 0))

    scenes: list[SceneTrack] = []
    for name in names:
        slot, color, desired = _scene_definition(name)
        ue_path = _choose_candidate(ue_candidates.get(slot, []), "ue")
        asv, target = parsed_ue.get(ue_path, ([], [])) if ue_path is not None else ([], [])
        jetson_path = _choose_candidate(audit_candidates.get(slot, []), "audit")
        jetson_logs = (jetson_path,) if jetson_path is not None else ()
        audits = parse_audit_log(jetson_path) if jetson_path is not None else None
        scenes.append(
            SceneTrack(
                name=name,
                target_color=color,
                target_entity=f"target_{color}",
                desired_standoff_m=desired,
                ue_log=ue_path,
                jetson_logs=jetson_logs,
                asv=asv,
                target=target,
                audits=audits,
            )
        )
    return scenes


def parse_scene(spec: str) -> SceneTrack:
    """Parse ``NAME=UE_LOG[=JETSON_LOG]`` for explicit CLI use."""

    parts = spec.split("=", 2)
    if len(parts) not in {2, 3} or not parts[0].strip() or not parts[1].strip():
        raise ValueError("scene must use NAME=UE_LOG[=JETSON_LOG]")
    name = parts[0].strip()
    _, color, desired = _scene_definition(name)
    ue_log = Path(parts[1]).expanduser()
    if not ue_log.is_file():
        raise FileNotFoundError(ue_log)
    jetson_logs: tuple[Path, ...] = ()
    audits: AuditCounts | None = None
    if len(parts) == 3 and parts[2].strip():
        jetson_log = Path(parts[2]).expanduser()
        if not jetson_log.is_file():
            raise FileNotFoundError(jetson_log)
        jetson_logs = (jetson_log,)
        audits = parse_audit_log(jetson_log)
    target_entity = f"target_{color}"
    asv, target = parse_trajectory_file(ue_log, target_entity)
    if not asv:
        raise ValueError(f"no UE ASV trajectory records found for {name}")
    return SceneTrack(name, color, target_entity, desired, ue_log, jetson_logs, asv, target, audits)


def _fmt_count(value: int | None) -> str:
    return "n/a" if value is None else str(value)


def _scene_metrics(scene: SceneTrack) -> dict[str, object]:
    samples = scene.standoff()
    errors = [abs(distance - scene.desired_standoff_m) for _, distance in samples]
    path_length_m = sum(
        math.hypot(curr.x_cm - prev.x_cm, curr.y_cm - prev.y_cm) / 100.0
        for prev, curr in zip(scene.asv, scene.asv[1:])
    )
    metrics: dict[str, object] = {
        "scene": scene.name,
        "target_color": scene.target_color,
        "target_entity": scene.target_entity,
        "desired_standoff_m": scene.desired_standoff_m,
        "available": scene.available,
        "ue_log": str(scene.ue_log) if scene.ue_log is not None else None,
        "jetson_logs": [str(path) for path in scene.jetson_logs],
        "asv_samples": len(scene.asv),
        "target_samples": len(scene.target),
        "path_length_m": path_length_m if scene.asv else None,
        "standoff_mean_abs_error_m": sum(errors) / len(errors) if errors else None,
        "standoff_p95_abs_error_m": sorted(errors)[int(0.95 * (len(errors) - 1))] if errors else None,
        "standoff_final_m": samples[-1][1] if samples else None,
    }
    if scene.audits is None:
        metrics["audit"] = None
    else:
        audit = scene.audits
        metrics["audit"] = {
            "events": audit.events,
            "events_complete": audit.complete,
            "raw_action_observed": audit.raw_observed,
            "policy_driven": audit.policy_driven,
            "backstop": audit.backstop,
            "hold": audit.hold,
            "fail_closed": audit.fail_closed,
            "policy_stop_subset": audit.policy_stop,
            "policy_dominance_rate": audit.policy_dominance_rate,
            "backstop_rate": audit.backstop_rate,
            "source": audit.source,
            "observed_trace_records": audit.observed_trace_records,
        }
    return metrics


def _audit_text(scene: SceneTrack) -> str:
    audit = scene.audits
    if audit is None:
        return f"{scene.name}\nno policy audit"
    status = "cumulative" if audit.complete else f"observed samples ({audit.observed_trace_records})"
    policy_rate = "n/a" if audit.policy_dominance_rate is None else f"{audit.policy_dominance_rate:.0%}"
    return (
        f"{scene.name} [{status}]\n"
        f"raw obs: {_fmt_count(audit.raw_observed)}\n"
        f"policy-driven: {_fmt_count(audit.policy_driven)} ({policy_rate})\n"
        f"backstop: {_fmt_count(audit.backstop)}\n"
        f"hold: {_fmt_count(audit.hold)}\n"
        f"fail-closed: {_fmt_count(audit.fail_closed)}"
    )


def plot_scenes(scenes: list[SceneTrack], output: Path) -> None:
    fig, (track_axis, distance_axis, audit_axis) = plt.subplots(
        1,
        3,
        figsize=(17, 7.5),
        gridspec_kw={"width_ratios": (1.55, 1.0, 0.9)},
    )
    all_x: list[float] = []
    all_y: list[float] = []
    for index, scene in enumerate(scenes):
        color = LINE_COLORS[index % len(LINE_COLORS)]
        target_color = SCENE_COLORS[scene.target_color]
        if scene.asv:
            asv_x = [point.x_cm / 100.0 for point in scene.asv]
            asv_y = [point.y_cm / 100.0 for point in scene.asv]
            track_axis.plot(asv_x, asv_y, color=color, linewidth=2.0, label=f"{scene.name} ASV")
            track_axis.scatter(asv_x[0], asv_y[0], color=color, marker="o", s=28, zorder=4)
            track_axis.scatter(asv_x[-1], asv_y[-1], color=color, marker="x", s=48, zorder=4)
            all_x.extend(asv_x)
            all_y.extend(asv_y)
        if scene.target:
            target_x = [point.x_cm / 100.0 for point in scene.target]
            target_y = [point.y_cm / 100.0 for point in scene.target]
            track_axis.plot(
                target_x,
                target_y,
                color=target_color,
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
                label=f"{scene.name} target",
            )
            all_x.extend(target_x)
            all_y.extend(target_y)
        samples = scene.standoff()
        if samples:
            distance_axis.plot(
                [time for time, _ in samples],
                [distance for _, distance in samples],
                color=color,
                linewidth=1.8,
                label=scene.name,
            )
            distance_axis.axhline(
                scene.desired_standoff_m,
                color=color,
                linestyle=":",
                linewidth=1.0,
                alpha=0.7,
            )

    track_axis.set_title("UE5 world-coordinate online closed loop")
    track_axis.set_xlabel("World X (m)")
    track_axis.set_ylabel("World Y (m)")
    track_axis.grid(True, alpha=0.25)
    track_axis.set_aspect("equal", adjustable="box")
    if all_x and all_y:
        track_axis.legend(fontsize=8, loc="best")
        margin = max(1.0, 0.04 * max(max(all_x) - min(all_x), max(all_y) - min(all_y)))
        track_axis.set_xlim(min(all_x) - margin, max(all_x) + margin)
        track_axis.set_ylim(min(all_y) - margin, max(all_y) + margin)
    else:
        track_axis.text(0.5, 0.5, "No UE ASV trajectory", ha="center", va="center", transform=track_axis.transAxes)

    distance_axis.set_title("Standoff distance")
    distance_axis.set_xlabel("Runtime (s)")
    distance_axis.set_ylabel("Target distance (m)")
    distance_axis.grid(True, alpha=0.25)
    if any(scene.standoff() for scene in scenes):
        distance_axis.legend(fontsize=8, loc="best")
    else:
        distance_axis.text(0.5, 0.5, "Target trajectory unavailable", ha="center", va="center", transform=distance_axis.transAxes)

    audit_axis.axis("off")
    audit_axis.set_title("Policy audit", pad=10)
    for index, scene in enumerate(scenes):
        y = 0.86 - index * 0.30
        audit_axis.text(
            0.02,
            y,
            _audit_text(scene),
            transform=audit_axis.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            color="#30343b",
            linespacing=1.35,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f5f6f7", "edgecolor": "#d6d9dd"},
        )
    fig.text(
        0.5,
        0.015,
        "raw obs is counted only when raw_action/raw_dx/raw_dy is present; n/a means the field was absent",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#59636e",
    )
    fig.suptitle("Single-point policy dominant closed-loop tracking", fontsize=15)
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _scene_order(value: str) -> tuple[str, ...]:
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    if not names:
        raise ValueError("scene order cannot be empty")
    for name in names:
        _scene_definition(name)
    return names


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot UE world tracks and non-fabricated policy audit evidence for closed-loop scenes."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input-dir",
        type=Path,
        help="Evidence directory; recursively discovers RED/BLUE 3m/4m UE and Jetson files.",
    )
    source.add_argument(
        "--scene",
        action="append",
        metavar="NAME=UE_LOG[=JETSON_LOG]",
        help="Explicit scene input; repeat for each scene. Jetson log is optional.",
    )
    parser.add_argument(
        "--scene-order",
        default=",".join(DEFAULT_SCENE_ORDER),
        help="Comma-separated order for --input-dir (default: RED 4m,BLUE 3m,RED 3m).",
    )
    parser.add_argument("--output", type=Path, required=True, help="PNG output path.")
    parser.add_argument("--metrics", type=Path, help="Optional JSON metrics output path.")
    parser.add_argument(
        "--require-scenes",
        action="store_true",
        help="Fail if any requested scene has no UE ASV trajectory.",
    )
    args = parser.parse_args()

    names = _scene_order(args.scene_order)
    if args.input_dir is not None:
        scenes = _discover_scenes(args.input_dir.expanduser(), names)
    else:
        scenes = [parse_scene(spec) for spec in args.scene or []]
        if not scenes:
            parser.error("at least one --scene is required")
    missing = [scene.name for scene in scenes if not scene.available]
    if args.require_scenes and missing:
        raise ValueError("missing UE ASV trajectory for: " + ", ".join(missing))
    if not any(scene.available for scene in scenes):
        raise ValueError("no UE ASV trajectory records found in the requested inputs")

    plot_scenes(scenes, args.output)
    metrics = {scene.name: _scene_metrics(scene) for scene in scenes}
    if args.metrics is not None:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"TRACK_PLOT_WRITTEN {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
