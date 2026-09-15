# benchmark.py

from dataclasses import dataclass
from typing import Callable, Dict
import statistics
import torch


@dataclass
class BenchmarkResult:
    name: str
    median_ms: float
    mean_ms: float
    min_ms: float
    p20_ms: float
    p80_ms: float
    p90_ms: float

    def __str__(self):
        return (
            f"{self.name:<16} "
            f"median={self.median_ms:.4f} ms  "
            f"mean={self.mean_ms:.4f} ms  "
            f"min={self.min_ms:.4f} ms  "
            f"p20={self.p20_ms:.4f} ms  "
            f"p80={self.p80_ms:.4f} ms  "
            f"p90={self.p90_ms:.4f} ms"
        )


def _percentile(sorted_values, q):
    idx = int((len(sorted_values) - 1) * q)
    return sorted_values[idx]


def benchmark(
    name: str,
    fn: Callable[[], None],
    warmup: int = 50,
    repeat: int = 200,
) -> BenchmarkResult:

    # -----------------------------
    # 1. Warmup
    # -----------------------------
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()

    # -----------------------------
    # 2. CUDA Event
    # -----------------------------
    starts = [
        torch.cuda.Event(enable_timing=True)
        for _ in range(repeat)
    ]

    ends = [
        torch.cuda.Event(enable_timing=True)
        for _ in range(repeat)
    ]

    # -----------------------------
    # 3. Benchmark
    # -----------------------------
    for i in range(repeat):
        starts[i].record()

        fn()

        ends[i].record()

    torch.cuda.synchronize()

    # -----------------------------
    # 4. Collect latency
    # -----------------------------
    times = [
        starts[i].elapsed_time(ends[i])
        for i in range(repeat)
    ]

    times.sort()

    return BenchmarkResult(
        name=name,
        median_ms=statistics.median(times),
        mean_ms=statistics.mean(times),
        min_ms=times[0],
        p20_ms=_percentile(times, 0.20),
        p80_ms=_percentile(times, 0.80),
        p90_ms=_percentile(times, 0.90),
    )


def benchmark_all(
    implementations: Dict[str, Callable[[], None]],
    warmup: int = 50,
    repeat: int = 200,
):
    results = []

    for name, fn in implementations.items():

        result = benchmark(
            name=name,
            fn=fn,
            warmup=warmup,
            repeat=repeat,
        )

        results.append(result)

    # latency 从小到大排序
    results.sort(key=lambda x: x.median_ms)

    print()
    print("=" * 100)
    print("GPU Benchmark")
    print("=" * 100)

    for result in results:
        print(result)

    # 最快实现作为 baseline
    if results:
        fastest = results[0].median_ms

        print()
        print("Relative Performance")

        for result in results:
            slowdown = result.median_ms / fastest

            print(
                f"{result.name:<16} "
                f"{slowdown:.3f}x vs fastest"
            )

    return results