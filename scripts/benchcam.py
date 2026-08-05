#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import time

from bench_common import build_run_root
from bench_common import format_duration
from bench_common import run_case
from bench_common import timestamp_string
from bench_common import write_timing_csv


BENCHMARK_NAME = "benchcam"

SIMD_VARIANT_LABELS = [
    "scalar",
    "simd",
]
SUB_SAMPLE_VALUES = [1, 2]
SAVE_STRATEGIES = ["memory"]

RUN_EXPERIMENT_1 = True

# Experiment 1: default camera-path sweep.
# This runs the full benchcam case matrix once per executable/save
# combination. The compiled binaries globally turn SIMD off/on.
EXPERIMENT_1_SIMD_VARIANT_LABELS = SIMD_VARIANT_LABELS
EXPERIMENT_1_SUB_SAMPLE_VALUES = SUB_SAMPLE_VALUES
EXPERIMENT_1_SAVE_STRATEGIES = SAVE_STRATEGIES


def experiment_1_cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for simd_variant_label in EXPERIMENT_1_SIMD_VARIANT_LABELS:
        for sub_sample in EXPERIMENT_1_SUB_SAMPLE_VALUES:
            for save_strategy in EXPERIMENT_1_SAVE_STRATEGIES:
                case_name = (
                    f"{BENCHMARK_NAME}_{simd_variant_label}"
                    f"_ssaa-{sub_sample}"
                    f"_save-{save_strategy}"
                )
                cases.append(
                    {
                        "experiment": "experiment_1",
                        "case_name": case_name,
                        "executable": f"{BENCHMARK_NAME}_{simd_variant_label}",
                        "args": [
                            "--sub-sample",
                            str(sub_sample),
                            "--save-strategy",
                            save_strategy,
                        ],
                    }
                )
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=pathlib.Path, default=None)
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    timing_timestamp = timestamp_string()
    run_root = build_run_root(BENCHMARK_NAME, args.out_root)
    cases: list[dict[str, object]] = []

    if RUN_EXPERIMENT_1:
        cases.extend(experiment_1_cases())

    if not args.dry_run:
        run_root.mkdir(parents=True, exist_ok=True)

    script_start = time.perf_counter()
    timing_rows: list[dict[str, object]] = []

    if RUN_EXPERIMENT_1:
        exp_start = time.perf_counter()
        for case in cases:
            run_case(
                BENCHMARK_NAME,
                case,
                run_root,
                args.runs,
                args.dry_run,
            )
        exp_seconds = time.perf_counter() - exp_start
        print(
            f"[{BENCHMARK_NAME}] Experiment 1 summary: "
            f"{format_duration(exp_seconds)}"
        )
        timing_rows.append(
            {
                "benchmark": BENCHMARK_NAME,
                "kind": "experiment",
                "name": "experiment_1",
                "seconds": f"{exp_seconds:.6f}",
            }
        )

    total_seconds = time.perf_counter() - script_start
    print(f"[{BENCHMARK_NAME}] Total summary: {format_duration(total_seconds)}")
    timing_rows.append(
        {
            "benchmark": BENCHMARK_NAME,
            "kind": "total",
            "name": "all",
            "seconds": f"{total_seconds:.6f}",
        }
    )

    if not args.dry_run:
        timing_csv = write_timing_csv(
            BENCHMARK_NAME,
            timing_rows,
            timing_timestamp,
        )
        print(f"[{BENCHMARK_NAME}] Timing written to {timing_csv}")
        print(f"Results written under {run_root}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
