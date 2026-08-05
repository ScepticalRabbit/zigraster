#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import pathlib
import shlex
import subprocess
import time

from perf_common import command_path, repo_root

DEFAULT_OUT_ROOT = pathlib.Path("out") / "bench_stats_perf"
DEFAULT_IMAGE_OUT_DIR = pathlib.Path("out") / "bench_images_perf"
DEFAULT_RUNS = 25
DEFAULT_PIXELS_X: int | None = None
DEFAULT_PIXELS_Y: int | None = None
DEFAULT_TOTAL_THREADS = 1
DEFAULT_MAX_GEOM_WORKERS_PER_JOB = 1
DEFAULT_MAX_RASTER_WORKERS_PER_JOB = 1
DEFAULT_RENDER_GROUP_COUNT = 1
DEFAULT_FRAME_BATCH_SIZE_PER_GROUP = 1
DEFAULT_MAX_GEOM_JOBS_IN_FLIGHT_PER_GROUP = 1
DEFAULT_EXPERIMENT1 = True
DEFAULT_EXPERIMENT2 = True

EXPERIMENT1_SUB_SAMPLES = [2]
EXPERIMENT2_SUB_SAMPLES = [1, 2]

EXPERIMENT1_CASES_BASE: list[dict[str, object]] = [
    {
        "study_group": "experiment1",
        "experiment": "precision",
        "case_name": "tiltraster_precision_f64_simd_v8",
        "precision": "f64",
        "lanes": 8,
        "texture_storage": "u8",
        "shader_subset": "all",
        "mesh_subset": "all",
    },
    # {
    #     "study_group": "experiment1",
    #     "experiment": "precision",
    #     "case_name": "tiltraster_precision_f64_simd_v4",
    #     "precision": "f64",
    #     "lanes": 4,
    #     "texture_storage": "u8",
    #     "shader_subset": "all",
    #     "mesh_subset": "all",
    # },
    # {
    #     "study_group": "experiment1",
    #     "experiment": "precision",
    #     "case_name": "tiltraster_precision_f32_simd_v16",
    #     "precision": "f32",
    #     "lanes": 16,
    #     "texture_storage": "u8",
    #     "shader_subset": "all",
    #     "mesh_subset": "all",
    # },
    {
        "study_group": "experiment1",
        "experiment": "precision",
        "case_name": "tiltraster_precision_f32_simd_v8",
        "precision": "f32",
        "lanes": 8,
        "texture_storage": "u8",
        "shader_subset": "all",
        "mesh_subset": "all",
    },
    # {
    #     "study_group": "experiment1",
    #     "experiment": "texstore",
    #     "case_name": "tiltraster_texstore_u8",
    #     "precision": "f64",
    #     "lanes": 8,
    #     "texture_storage": "u8",
    #     "shader_subset": "texture",
    #     "mesh_subset": "all",
    # },
    # {
    #     "study_group": "experiment1",
    #     "experiment": "texstore",
    #     "case_name": "tiltraster_texstore_u16",
    #     "precision": "f64",
    #     "lanes": 8,
    #     "texture_storage": "u16",
    #     "shader_subset": "texture",
    #     "mesh_subset": "all",
    # },
    {
        "study_group": "experiment1",
        "experiment": "distortion",
        "case_name": "tiltraster_distortion_brown_f64_simd_v8",
        "precision": "f64",
        "lanes": 8,
        "texture_storage": "u8",
        "shader_subset": "all",
        "mesh_subset": "all",
        "distortion": "brown",
    },
    {
        "study_group": "experiment1",
        "experiment": "distortion",
        "case_name": "tiltraster_distortion_brownext_f64_simd_v8",
        "precision": "f64",
        "lanes": 8,
        "texture_storage": "u8",
        "shader_subset": "all",
        "mesh_subset": "all",
        "distortion": "brownext",
    },
]


def build_experiment1_cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for base_case in EXPERIMENT1_CASES_BASE:
        for sub_sample in EXPERIMENT1_SUB_SAMPLES:
            case = dict(base_case)
            case["sub_sample"] = sub_sample
            case["case_name"] = f"{base_case['case_name']}_ssaa{sub_sample}"
            cases.append(case)
    return cases


def build_experiment2_cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for sub_sample in EXPERIMENT2_SUB_SAMPLES:
        cases.append(
            {
                "study_group": "experiment2",
                "experiment": "llvmpipe_compare",
                "case_name": (
                    "tiltraster_llvmpipe_compare_"
                    f"tri3_f32_simd_v8_ssaa{sub_sample}"
                ),
                "precision": "f32",
                "lanes": 8,
                "texture_storage": "u8",
                "shader_subset": "all",
                "mesh_subset": "tri3_compare",
                "sub_sample": sub_sample,
            }
        )
    return cases


EXPERIMENT1_CASES = build_experiment1_cases()
EXPERIMENT2_CASES = build_experiment2_cases()


def build_run_root(out_root: pathlib.Path | None) -> pathlib.Path:
    root_dir = out_root or DEFAULT_OUT_ROOT
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return root_dir / timestamp


def default_image_out_dir() -> pathlib.Path:
    return DEFAULT_IMAGE_OUT_DIR


def default_lanes(precision: str) -> int:
    return 16 if precision == "f32" else 8


def binary_name(precision: str, lanes: int) -> str:
    if lanes == default_lanes(precision):
        return f"bench_tiltraster_{precision}_simd"
    return f"bench_tiltraster_{precision}_simd_v{lanes}"


def binary_path(precision: str, lanes: int) -> pathlib.Path:
    path = repo_root() / "bin" / binary_name(precision, lanes)
    if not path.exists():
        raise SystemExit(
            f"Missing binary {path}. Run "
            "scripts/compile_perf_all.py first.",
        )
    return path


def timestamp_string() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def write_command_file(output_dir: pathlib.Path, command: list[str]) -> None:
    command_file = output_dir / "command.txt"
    command_file.write_text(
        " ".join(shlex.quote(part) for part in command) + "\n",
    )


def write_experiment_meta(
    output_dir: pathlib.Path,
    case: dict[str, object],
    command: list[str],
) -> None:
    meta_path = output_dir / "experiment_meta.txt"
    lines = [
        f"experiment={case['experiment']}",
        f"case_name={case['case_name']}",
        f"binary={case['binary']}",
        f"precision={case['precision']}",
        "simd=on",
        f"texture_storage={case['texture_storage']}",
        f"shader_subset={case['shader_subset']}",
        f"mesh_subset={case['mesh_subset']}",
        f"lanes={case['lanes']}",
        f"sub_sample={case.get('sub_sample', 2)}",
        f"distortion={case.get('distortion', 'none')}",
        "command=" + " ".join(shlex.quote(part) for part in command),
    ]
    meta_path.write_text("\n".join(lines) + "\n")


def write_timing_csv(
    rows: list[dict[str, object]],
    timestamp: str,
) -> pathlib.Path:
    out_dir = repo_root() / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"time_bench_perf_raster_{timestamp}.csv"
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["experiment", "case_name", "seconds"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return csv_path


def run_case(
    case: dict[str, object],
    run_root: pathlib.Path,
    image_out_dir: pathlib.Path,
    runs: int,
    pixels_x: int | None,
    pixels_y: int | None,
    dry_run: bool,
) -> float:
    output_dir = run_root / str(case["case_name"])
    binary = binary_path(
        str(case["precision"]),
        int(case["lanes"]),
    )
    binary_tag = binary_name(
        str(case["precision"]),
        int(case["lanes"]),
    )
    case_with_binary = dict(case)
    case_with_binary["binary"] = binary_tag
    sub_sample = int(case.get("sub_sample", 2))
    command = [
        str(binary),
        "--out-dir",
        command_path(output_dir),
        "--image-out-dir",
        command_path(image_out_dir),
        "--total-threads",
        str(DEFAULT_TOTAL_THREADS),
        "--max-geom-workers-per-job",
        str(DEFAULT_MAX_GEOM_WORKERS_PER_JOB),
        "--max-raster-workers-per-job",
        str(DEFAULT_MAX_RASTER_WORKERS_PER_JOB),
        "--render-group-count",
        str(DEFAULT_RENDER_GROUP_COUNT),
        "--frame-batch-size-per-group",
        str(DEFAULT_FRAME_BATCH_SIZE_PER_GROUP),
        "--max-geom-jobs-in-flight-per-group",
        str(DEFAULT_MAX_GEOM_JOBS_IN_FLIGHT_PER_GROUP),
        "--runs",
        str(runs),
        "--sub-sample",
        str(sub_sample),
        "--texture-storage",
        str(case["texture_storage"]),
        "--shader-subset",
        str(case["shader_subset"]),
        "--mesh-subset",
        str(case["mesh_subset"]),
    ]
    if pixels_x is not None:
        command.extend(["--pixels-x", str(pixels_x)])
    if pixels_y is not None:
        command.extend(["--pixels-y", str(pixels_y)])
    if "distortion" in case:
        command.extend(["--distortion", str(case["distortion"])])

    print(f"[bench_perf_raster] {case['case_name']}")
    if dry_run:
        print("  dry-run:", " ".join(shlex.quote(part) for part in command))
        return 0.0

    output_dir.mkdir(parents=True, exist_ok=True)
    write_command_file(output_dir, command)
    write_experiment_meta(output_dir, case_with_binary, command)

    stdout_path = output_dir / "stdout.txt"
    stderr_path = output_dir / "stderr.txt"
    start = time.perf_counter()
    with stdout_path.open("w") as stdout_file, stderr_path.open("w") as stderr_file:
        subprocess.run(
            command,
            check=True,
            cwd=repo_root(),
            stdout=stdout_file,
            stderr=stderr_file,
        )
    return time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=pathlib.Path, default=None)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--pixels-x", type=int, default=DEFAULT_PIXELS_X)
    parser.add_argument("--pixels-y", type=int, default=DEFAULT_PIXELS_Y)
    parser.add_argument(
        "--experiment1",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_EXPERIMENT1,
    )
    parser.add_argument(
        "--experiment2",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_EXPERIMENT2,
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected_cases: list[dict[str, object]] = []
    if args.experiment1:
        selected_cases.extend(EXPERIMENT1_CASES)
    if args.experiment2:
        selected_cases.extend(EXPERIMENT2_CASES)
    if not selected_cases:
        raise SystemExit("No experiments selected.")

    run_root = build_run_root(args.out_root)
    image_out_dir = default_image_out_dir()
    timing_rows: list[dict[str, object]] = []
    timing_stamp = timestamp_string()

    if not args.dry_run:
        run_root.mkdir(parents=True, exist_ok=True)

    for case in selected_cases:
        seconds = run_case(
            case,
            run_root,
            image_out_dir,
            args.runs,
            args.pixels_x,
            args.pixels_y,
            args.dry_run,
        )
        timing_rows.append(
            {
                "experiment": case["experiment"],
                "case_name": case["case_name"],
                "seconds": f"{seconds:.6f}",
            }
        )

    if not args.dry_run:
        timing_csv = write_timing_csv(timing_rows, timing_stamp)
        print(f"Timing written to {timing_csv}")
        print(f"Results written under {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
