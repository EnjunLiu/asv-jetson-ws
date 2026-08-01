"""Day 20 Jetson ONNX benchmark and stress test."""

from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np


def benchmark(
    model_path: str,
    duration_sec: float = 60.0,
    warmup_iters: int = 10,
) -> dict:
    import onnxruntime as ort

    session = ort.InferenceSession(model_path)
    inputs = {
        inp.name: np.random.randn(
            *[1 if isinstance(d, str) else d for d in inp.shape]
        ).astype(np.float32)
        for inp in session.get_inputs()
    }
    # bool inputs
    for k, v in inputs.items():
        if "valid" in k or "mask" in k:
            inputs[k] = v > 0

    # Warmup.
    for _ in range(warmup_iters):
        session.run(None, inputs)

    # Benchmark.
    latencies: list[float] = []
    start = time.perf_counter()
    iterations = 0
    while time.perf_counter() - start < duration_sec:
        t0 = time.perf_counter()
        session.run(None, inputs)
        latencies.append(time.perf_counter() - t0)
        iterations += 1

    elapsed = time.perf_counter() - start
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]

    return {
        "model_path": model_path,
        "iterations": iterations,
        "duration_sec": round(elapsed, 2),
        "hz": round(iterations / elapsed, 2),
        "latency_p50_ms": round(p50 * 1000, 2),
        "latency_p95_ms": round(p95 * 1000, 2),
        "latency_p99_ms": round(p99 * 1000, 2),
        "latency_min_ms": round(min(latencies) * 1000, 2),
        "latency_max_ms": round(max(latencies) * 1000, 2),
        "passed_2hz": (iterations / elapsed) >= 2.0,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Day 20 ONNX benchmark")
    p.add_argument(
        "--model",
        type=Path,
        default=Path("models/policy_sine_near_image_color_seed42.onnx"),
    )
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    try:
        r = benchmark(str(args.model), duration_sec=args.duration)
    except Exception as e:
        print(f"ONNX_BENCHMARK_FAIL: {e}")
        return 1

    status = "PASS" if r["passed_2hz"] else "FAIL"
    print(
        f"ONNX_BENCHMARK_{status} "
        f"hz={r['hz']} p50={r['latency_p50_ms']}ms "
        f"p95={r['latency_p95_ms']}ms iterations={r['iterations']}"
    )
    print(json.dumps(r, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(r, indent=2))
    return 0 if r["passed_2hz"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
