from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .language_encoder import USVLanguageEncoder


INSTRUCTIONS = (
    ("red_5_01", "follow_red_5m", "跟随红色目标船，保持5米距离"),
    ("red_5_02", "follow_red_5m", "跟在红船后面，与它相距五米"),
    ("red_5_03", "follow_red_5m", "追踪红色无人船，距离保持在5米左右"),
    ("red_5_04", "follow_red_5m", "锁定红船并维持五米跟随距离"),
    ("blue_5_01", "follow_blue_5m", "跟随蓝色目标船，保持5米距离"),
    ("blue_5_02", "follow_blue_5m", "跟在蓝船后面，与它相距五米"),
    ("blue_5_03", "follow_blue_5m", "追踪蓝色无人船，距离保持在5米左右"),
    ("blue_5_04", "follow_blue_5m", "锁定蓝船并维持五米跟随距离"),
    ("red_10_01", "follow_red_10m", "跟随红色目标船，保持10米距离"),
    ("red_10_02", "follow_red_10m", "跟在红船后面，与它相距十米"),
    ("red_10_03", "follow_red_10m", "追踪红色无人船，距离保持在10米左右"),
    ("red_10_04", "follow_red_10m", "锁定红船并维持十米跟随距离"),
    ("blue_10_01", "follow_blue_10m", "跟随蓝色目标船，保持10米距离"),
    ("blue_10_02", "follow_blue_10m", "跟在蓝船后面，与它相距十米"),
    ("blue_10_03", "follow_blue_10m", "追踪蓝色无人船，距离保持在10米左右"),
    ("blue_10_04", "follow_blue_10m", "锁定蓝船并维持十米跟随距离"),
    ("stop_01", "stop", "立即停止"),
    ("stop_02", "stop", "停止跟随并保持安全停机"),
    ("stop_03", "stop", "不要继续前进，马上停船"),
    ("stop_04", "stop", "终止当前任务并安全停止"),
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate USV language embeddings."
    )
    parser.add_argument(
        "--model-path",
        default="models/Qwen3-Embedding-0.6B",
        help="Local Qwen3 embedding model directory.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/language_embedding/language_similarity.csv",
        help="CSV output path.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dim", type=int, default=256)
    parser.add_argument("--repeat-count", type=int, default=10)
    parser.add_argument("--repeat-tolerance", type=float, default=1.0e-6)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.repeat_count < 2:
        raise SystemExit("--repeat-count must be at least 2")

    encoder = USVLanguageEncoder(
        args.model_path,
        output_dim=args.output_dim,
        device=args.device,
    )

    vectors = []
    cache_hits = 0
    for _, _, text in INSTRUCTIONS:
        result = encoder.encode_with_metadata(text)
        vectors.append(result.embedding)
        cache_hits += int(result.cached)
    matrix = np.vstack(vectors)
    similarity = matrix @ matrix.T

    repeated = []
    for _ in range(args.repeat_count):
        result = encoder.encode_with_metadata(INSTRUCTIONS[0][2])
        repeated.append(result.embedding)
        cache_hits += int(result.cached)
    first = repeated[0]
    max_repeat_difference = max(
        float(np.max(np.abs(first - vector)))
        for vector in repeated[1:]
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["instruction_id", "label", "text"]
            + [item[0] for item in INSTRUCTIONS]
        )
        for index, (instruction_id, label, text) in enumerate(INSTRUCTIONS):
            writer.writerow(
                [instruction_id, label, text]
                + [f"{value:.8f}" for value in similarity[index]]
            )

    print(f"model_path={args.model_path}")
    print(f"embedding_shape={matrix.shape}")
    print(f"cache_hits={cache_hits}")
    print(f"max_repeat_difference={max_repeat_difference:.10g}")
    print(f"csv={output_path}")

    checks = {
        "shape": matrix.shape == (len(INSTRUCTIONS), args.output_dim),
        "finite": bool(np.all(np.isfinite(matrix))),
        "normalized": bool(
            np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1.0e-4)
        ),
        "repeat_deterministic":
            max_repeat_difference <= args.repeat_tolerance,
    }
    for name, passed in checks.items():
        print(f"{name}={'PASS' if passed else 'FAIL'}")
    if not all(checks.values()):
        raise SystemExit(1)
    print("LANGUAGE_EMBEDDING_OFFLINE_PASS")


if __name__ == "__main__":
    main()
