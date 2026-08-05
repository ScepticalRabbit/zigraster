#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter

import bench_dicuq


# Input assumptions.
SINGLE_THREAD_RUN_SECONDS = 45.0
RUNS_PER_COMBINATION = 10
PARALLEL_FRACTIONS = [0.99, 0.95, 0.90]

# Choose which experiments to include in the estimate.
EXPERIMENT_IDS = ["1", "2", "3", "4"]


def time_per_combination(total_threads: int, parallel_fraction: float) -> float:
    serial_fraction = 1.0 - parallel_fraction
    baseline = SINGLE_THREAD_RUN_SECONDS * RUNS_PER_COMBINATION
    return baseline * (serial_fraction + parallel_fraction / total_threads)


def case_builder_map() -> dict[str, callable]:
    return {
        "1": bench_dicuq.experiment_1_cases,
        "2": bench_dicuq.experiment_2_cases,
        "3": bench_dicuq.experiment_3_cases,
        "4": bench_dicuq.experiment_4_cases,
    }


def thread_histogram(cases: list[dict[str, object]]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for case in cases:
        case_name = str(case["case_name"])
        for part in case_name.split("_"):
            if part.startswith("threads-"):
                counts[int(part.split("-")[1])] += 1
                break
        else:
            raise ValueError(f"Could not parse thread count from {case_name}")
    return counts


def format_seconds(seconds: float) -> str:
    hours = seconds / 3600.0
    minutes = seconds / 60.0
    if hours >= 1.0:
        return f"{hours:.2f} h"
    return f"{minutes:.1f} min"


def main() -> int:
    builders = case_builder_map()
    selected = [(exp_id, builders[exp_id]) for exp_id in EXPERIMENT_IDS]

    print(
        "DIC UQ Runtime Estimate\n"
        f"  single-thread run time = {SINGLE_THREAD_RUN_SECONDS:.3f} s\n"
        f"  runs per combination   = {RUNS_PER_COMBINATION}\n"
        f"  parallel fractions     = {PARALLEL_FRACTIONS}\n"
    )

    total_hist: Counter[int] = Counter()

    for exp_id, builder in selected:
        cases = builder()
        hist = thread_histogram(cases)
        total_hist.update(hist)
        print(f"Experiment {exp_id}: {len(cases)} cases")
        print(f"  thread histogram: {dict(sorted(hist.items()))}")
        for pf in PARALLEL_FRACTIONS:
            total_seconds = sum(
                count * time_per_combination(threads, pf)
                for threads, count in hist.items()
            )
            print(f"  p={pf:.2f}: {total_seconds:.1f} s ({format_seconds(total_seconds)})")
        print()

    print(f"All selected experiments: {sum(total_hist.values())} cases")
    print(f"  thread histogram: {dict(sorted(total_hist.items()))}")
    for pf in PARALLEL_FRACTIONS:
        total_seconds = sum(
            count * time_per_combination(threads, pf)
            for threads, count in total_hist.items()
        )
        print(f"  p={pf:.2f}: {total_seconds:.1f} s ({format_seconds(total_seconds)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
