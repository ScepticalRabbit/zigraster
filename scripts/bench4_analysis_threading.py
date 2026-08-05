#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import pathlib
import re
import statistics
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper_bench_common import latest_run_dir_with_paths, row_float
from paper_const import (
    PLOT_LINE_AXIS_FONT_SIZE,
    PLOT_LINE_FIG_SIZE_IN,
    PLOT_LINE_LEGEND_FONT_SIZE,
    PLOT_LINE_SECONDARY_AXIS_FONT_SIZE,
    PLOT_LINE_TICK_FONT_SIZE,
    PLOT_LINE_TITLE_FONT_SIZE,
    PLOT_RESOLUTION_DPI,
    repo_root,
)


BENCH_NAME = "bench_dicuq"
EXPERIMENT_DIR = "experiment_5_offline_sweet_spot"
OUT_DIR = repo_root() / "verif"

BENCH4_AMDAHL_CSV = OUT_DIR / "bench4_amdahl.csv"

CASE_DIR_RE = re.compile(
    r"^bench_dicuq_threads-(?P<threads>\d+)"
    r"_groups-(?P<groups>\d+)"
    r"_workerspg-(?P<workerspg>\d+)"
    r"_batch-(?P<batch>\d+)"
    r"_geomjobs-(?P<geomjobs>\d+)"
    r"_geomw-(?P<geomw>\d+)"
    r"_geommode-(?P<geommode>[a-z_]+)"
    r"_rasterw-(?P<rasterw>\d+)"
    r"_render-(?P<render>[a-z_]+)"
    r"_save-(?P<save>[a-z_]+?)"
    r"(?:_overlap-(?P<overlap>[a-z]+))?$"
)

BATCH_MODE_ORDER = ["1", "W", "2W"]
GEOMJOBS_MODE_ORDER = ["1", "W"]
SAVE_MODE_ORDER = [
    "disk",
    "disk_overlap",
    "memory",
]
SAVE_MODE_LABELS = {
    "disk": "Disk",
    "disk_overlap": "Disk Overlap",
    "memory": "Memory",
}
SAVE_MODE_COLORS = {
    "disk": "tab:orange",
    "disk_overlap": "tab:green",
    "memory": "tab:blue",
}
REFERENCE_SAVE_MODE = "memory"

LEGACY_SAVE_MODE_MAP = {
    "memory_direct_write": "memory",
    "memory_per_frame_copy": "memory",
    "both_direct_write": "both",
    "both_per_frame_copy": "both",
}


@dataclass(slots=True)
class CaseStats:
    case_dir: pathlib.Path
    case_name: str
    threads: int
    groups: int
    workers_per_group: int
    batch: int
    geom_jobs: int
    geom_workers: int
    geom_mode: str
    raster_workers: int
    render_mode: str
    save_mode: str
    e2e_median_ms: float
    e2e_min_ms: float
    e2e_max_ms: float
    raster_median_ms: float
    raster_min_ms: float
    raster_max_ms: float
    e2e_throughput_median_mpx_s: float
    e2e_throughput_min_mpx_s: float
    e2e_throughput_max_mpx_s: float
    raster_throughput_median_mpx_s: float
    raster_throughput_min_mpx_s: float
    raster_throughput_max_mpx_s: float


@dataclass(slots=True)
class SeriesPoint:
    threads: int
    time_median_ms: float
    time_min_ms: float
    time_max_ms: float
    throughput_median_mpx_s: float
    throughput_min_mpx_s: float
    throughput_max_mpx_s: float


def load_csv_rows(csv_path: pathlib.Path) -> list[dict[str, str]]:
    with csv_path.open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def find_camera_all_row(csv_path: pathlib.Path) -> dict[str, str]:
    rows = load_csv_rows(csv_path)
    camera_all = [row for row in rows if row["Camera"] == "all"]
    if len(camera_all) != 1:
        raise ValueError(
            f"expected exactly one Camera=all row in {csv_path}, got {len(camera_all)}"
        )
    return camera_all[0]


def load_config_map(config_path: pathlib.Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in config_path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key] = value
    return result


def normalize_save_mode(save_mode: str) -> str:
    return LEGACY_SAVE_MODE_MAP.get(save_mode, save_mode)


def parse_case_stats(case_dir: pathlib.Path) -> CaseStats | None:
    match = CASE_DIR_RE.match(case_dir.name)
    if match is None:
        return None

    required_paths = [
        case_dir / "config.txt",
        case_dir / "bench_run0_byframe.csv",
    ]
    if not all(path.exists() for path in required_paths):
        return None

    config_map = load_config_map(case_dir / "config.txt")
    frame_rows = load_csv_rows(case_dir / "bench_run0_byframe.csv")
    pixels_x = int(config_map["pixels_x"])
    pixels_y = int(config_map["pixels_y"])
    frame_count = len(frame_rows)
    total_mpx = (pixels_x * pixels_y * frame_count) / 1.0e6

    run_paths = sorted(case_dir.glob("bench_run*_e2e.csv"))
    # Discard run0
    run_paths = [p for p in run_paths if p.name != "bench_run0_e2e.csv"]

    if not run_paths:
        # Fallback to loading precomputed summary files if run files are missing
        required_summary_paths = [
            case_dir / "bench_e2e_overruns_median.csv",
            case_dir / "bench_e2e_overruns_min.csv",
            case_dir / "bench_e2e_overruns_max.csv",
        ]
        if not all(path.exists() for path in required_summary_paths):
            return None
        median_row = find_camera_all_row(
            case_dir / "bench_e2e_overruns_median.csv"
        )
        min_row = find_camera_all_row(
            case_dir / "bench_e2e_overruns_min.csv"
        )
        max_row = find_camera_all_row(
            case_dir / "bench_e2e_overruns_max.csv"
        )
        case_name = median_row["Case"]
        e2e_median_ms = row_float(median_row, "E2E Time")
        e2e_min_ms = row_float(min_row, "E2E Time")
        e2e_max_ms = row_float(max_row, "E2E Time")
        e2e_throughput_median = row_float(
            median_row, "E2E TP"
        )
        e2e_throughput_min = row_float(
            min_row, "E2E TP"
        )
        e2e_throughput_max = row_float(
            max_row, "E2E TP"
        )
        raster_median_ms = row_float(
            median_row, "Raster Time"
        )
        raster_min_ms = row_float(min_row, "Raster Time")
        raster_max_ms = row_float(max_row, "Raster Time")
        raster_throughput_median = row_float(
            median_row, "Raster TP"
        )
        raster_throughput_min = row_float(
            min_row, "Raster TP"
        )
        raster_throughput_max = row_float(
            max_row, "Raster TP"
        )
    else:
        e2e_times = []
        raster_times = []
        e2e_tps = []
        raster_tps = []
        case_name = ""

        for run_path in run_paths:
            with run_path.open(newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    if row.get("Camera") == "all":
                        case_name = row["Case"]
                        e2e_ms = row_float(row, "E2E Time")
                        e2e_times.append(e2e_ms)
                        raster_ms = row_float(
                            row, "Raster Time"
                        )
                        raster_times.append(raster_ms)
                        e2e_tp = row_float(
                            row, "E2E TP"
                        )
                        e2e_tps.append(e2e_tp)
                        raster_tp = row_float(
                            row, "Raster TP"
                        )
                        raster_tps.append(raster_tp)
                        break

        e2e_median_ms = statistics.median(e2e_times)
        e2e_min_ms = min(e2e_times)
        e2e_max_ms = max(e2e_times)
        raster_median_ms = statistics.median(raster_times)
        raster_min_ms = min(raster_times)
        raster_max_ms = max(raster_times)
        e2e_throughput_median = statistics.median(e2e_tps)
        e2e_throughput_min = min(e2e_tps)
        e2e_throughput_max = max(e2e_tps)
        raster_throughput_median = statistics.median(raster_tps)
        raster_throughput_min = min(raster_tps)
        raster_throughput_max = max(raster_tps)

    save_mode = normalize_save_mode(match.group("save"))
    overlap = match.groupdict().get("overlap")
    if save_mode == "disk" and overlap == "true":
        save_mode = "disk_overlap"

    return CaseStats(
        case_dir=case_dir,
        case_name=case_name,
        threads=int(match.group("threads")),
        groups=int(match.group("groups")),
        workers_per_group=int(match.group("workerspg")),
        batch=int(match.group("batch")),
        geom_jobs=int(match.group("geomjobs")),
        geom_workers=int(match.group("geomw")),
        geom_mode=match.group("geommode"),
        raster_workers=int(match.group("rasterw")),
        render_mode=match.group("render"),
        save_mode=save_mode,
        e2e_median_ms=e2e_median_ms,
        e2e_min_ms=e2e_min_ms,
        e2e_max_ms=e2e_max_ms,
        raster_median_ms=raster_median_ms,
        raster_min_ms=raster_min_ms,
        raster_max_ms=raster_max_ms,
        e2e_throughput_median_mpx_s=e2e_throughput_median,
        e2e_throughput_min_mpx_s=e2e_throughput_min,
        e2e_throughput_max_mpx_s=e2e_throughput_max,
        raster_throughput_median_mpx_s=raster_throughput_median,
        raster_throughput_min_mpx_s=raster_throughput_min,
        raster_throughput_max_mpx_s=raster_throughput_max,
    )


def collect_case_stats(include_disk_overlap: bool = True) -> list[CaseStats]:
    required_paths = ["experiment_5_offline_sweet_spot"]
    if include_disk_overlap:
        required_paths.append(
            "experiment_7_offline_sweet_spot_disk_overlap"
        )

    latest_run = latest_run_dir_with_paths(
        BENCH_NAME,
        required_paths,
    )

    stats: list[CaseStats] = []

    exp5_dir = latest_run / "experiment_5_offline_sweet_spot"
    print(f"Using benchmark run: {exp5_dir}")
    for case_dir in sorted(
        path for path in exp5_dir.iterdir() if path.is_dir()
    ):
        case_stats = parse_case_stats(case_dir)
        if case_stats is not None:
            stats.append(case_stats)

    if include_disk_overlap:
        exp7_dir = latest_run / (
            "experiment_7_offline_sweet_spot_disk_overlap"
        )
        print(f"Using benchmark run: {exp7_dir}")
        for case_dir in sorted(
            path for path in exp7_dir.iterdir() if path.is_dir()
        ):
            case_stats = parse_case_stats(case_dir)
            if case_stats is not None:
                stats.append(case_stats)

    if not stats:
        raise FileNotFoundError(
            f"no case directories found in {latest_run}"
        )

    case_names = sorted({case.case_name for case in stats})
    if len(case_names) != 1:
        raise ValueError(
            f"expected one benchmark case in {latest_run}, "
            f"found {case_names}"
        )

    return stats


def save_mode_label(save_mode: str) -> str:
    return SAVE_MODE_LABELS.get(save_mode, save_mode.replace("_", " ").title())


def batch_mode(case: CaseStats) -> str:
    if case.batch == 1:
        return "1"
    if case.batch == case.workers_per_group:
        return "W"
    if case.batch == 2 * case.workers_per_group:
        return "2W"
    return str(case.batch)


def geomjobs_mode(case: CaseStats) -> str:
    if case.geom_jobs == 1:
        return "1"
    if case.geom_jobs == case.workers_per_group:
        return "W"
    return str(case.geom_jobs)


def best_case_by_threads_and_save(stats: list[CaseStats]) -> dict[tuple[int, str], CaseStats]:
    best: dict[tuple[int, str], CaseStats] = {}
    for case in stats:
        key = (case.threads, case.save_mode)
        current = best.get(key)
        if current is None or case.e2e_throughput_median_mpx_s > current.e2e_throughput_median_mpx_s:
            best[key] = case
    return best


def best_raster_case_by_threads_and_save(
    stats: list[CaseStats],
) -> dict[tuple[int, str], CaseStats]:
    best: dict[tuple[int, str], CaseStats] = {}
    for case in stats:
        key = (case.threads, case.save_mode)
        current = best.get(key)
        if (
            current is None
            or case.raster_throughput_median_mpx_s >
            current.raster_throughput_median_mpx_s
        ):
            best[key] = case
    return best


def best_case_by_partition(
    stats: list[CaseStats],
) -> dict[tuple[int, str, int, int], CaseStats]:
    best: dict[tuple[int, str, int, int], CaseStats] = {}
    for case in stats:
        key = (case.threads, case.save_mode, case.groups, case.workers_per_group)
        current = best.get(key)
        if current is None or case.e2e_throughput_median_mpx_s > current.e2e_throughput_median_mpx_s:
            best[key] = case
    return best


def _style_axes(ax: plt.Axes, x_label: str, y_label: str, title: str) -> None:
    ax.set_xlabel(x_label, fontsize=PLOT_LINE_AXIS_FONT_SIZE)
    ax.set_ylabel(y_label, fontsize=PLOT_LINE_AXIS_FONT_SIZE)
    if title:
        ax.set_title(title, fontsize=PLOT_LINE_TITLE_FONT_SIZE)
    ax.tick_params(labelsize=PLOT_LINE_TICK_FONT_SIZE)
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)


def _save_plot(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / f"{stem}.png"
    svg_path = OUT_DIR / f"{stem}.svg"
    fig.savefig(png_path, dpi=PLOT_RESOLUTION_DPI, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {svg_path}")


def fit_amdahl_serial_fraction(threads: list[int], speedups: list[float]) -> float:
    grid = np.linspace(0.0, 1.0, 200_001)
    n = np.asarray(threads, dtype=float)
    s_obs = np.asarray(speedups, dtype=float)
    preds = 1.0 / (grid[:, None] + (1.0 - grid[:, None]) / n[None, :])
    err = np.sum((preds - s_obs[None, :]) ** 2, axis=1)
    return float(grid[int(np.argmin(err))])


def point_serial_fraction(threads: int, speedup: float) -> float | None:
    if threads <= 1 or speedup <= 0.0:
        return None
    return (1.0 / speedup - 1.0 / threads) / (1.0 - 1.0 / threads)


def _add_ideal_throughput_line(
    ax: plt.Axes,
    x_values: list[int],
    baseline_throughput: float,
    color: str,
    label: str,
) -> None:
    x_sorted = sorted(set(x_values))
    if not x_sorted:
        return
    ax.plot(
        x_sorted,
        [baseline_throughput * float(x) for x in x_sorted],
        linestyle="--",
        linewidth=1.2,
        color=color,
        alpha=0.8,
        label=label,
    )


def _add_ideal_runtime_line(
    ax: plt.Axes,
    x_values: list[int],
    baseline_runtime_ms: float,
    color: str,
    label: str,
) -> None:
    x_sorted = sorted(set(x_values))
    if not x_sorted:
        return
    ax.plot(
        x_sorted,
        [baseline_runtime_ms / float(x) for x in x_sorted],
        linestyle="--",
        linewidth=1.2,
        color=color,
        alpha=0.8,
        label=label,
    )


def plot_best_e2e_throughput(stats: list[CaseStats], stem: str) -> None:
    best_map = best_case_by_threads_and_save(stats)

    fig, ax = plt.subplots(figsize=PLOT_LINE_FIG_SIZE_IN)
    right_ax = ax.twinx()
    legend_handles = []
    legend_labels = []
    all_x_vals: list[int] = []
    memory_baseline = best_map[(1, REFERENCE_SAVE_MODE)].e2e_throughput_median_mpx_s

    for save_mode in SAVE_MODE_ORDER:
        line_cases = sorted(
            [case for key, case in best_map.items() if key[1] == save_mode],
            key=lambda case: case.threads,
        )
        if not line_cases:
            continue
        x_vals = [case.threads for case in line_cases]
        y_vals = [case.e2e_throughput_median_mpx_s for case in line_cases]
        yerr_low = [
            case.e2e_throughput_median_mpx_s - case.e2e_throughput_min_mpx_s
            for case in line_cases
        ]
        yerr_high = [
            case.e2e_throughput_max_mpx_s - case.e2e_throughput_median_mpx_s
            for case in line_cases
        ]
        all_x_vals.extend(x_vals)

        err = ax.errorbar(
            x_vals,
            y_vals,
            yerr=[yerr_low, yerr_high],
            fmt="o-",
            linewidth=1.8,
            markersize=5,
            capsize=3,
            color=SAVE_MODE_COLORS[save_mode],
            label=save_mode_label(save_mode),
        )
        legend_handles.append(err.lines[0])
        legend_labels.append(save_mode_label(save_mode))

    _add_ideal_throughput_line(
        ax,
        all_x_vals,
        memory_baseline,
        "black",
        "Ideal",
    )
    legend_handles.append(ax.lines[-1])
    legend_labels.append("Ideal")
    _style_axes(
        ax,
        "Total Threads",
        "End-to-End Throughput\n[MPx/s]",
        "",
    )
    right_ax.set_ylabel(
        "Speedup vs. 1-thread",
        fontsize=PLOT_LINE_SECONDARY_AXIS_FONT_SIZE,
    )
    right_ax.tick_params(labelsize=PLOT_LINE_TICK_FONT_SIZE)
    left_min, left_max = ax.get_ylim()
    right_ax.set_ylim(left_min / memory_baseline, left_max / memory_baseline)
    ax.legend(
        legend_handles,
        legend_labels,
        fontsize=PLOT_LINE_LEGEND_FONT_SIZE,
        ncol=1,
    )
    _save_plot(fig, stem)


def plot_best_raster_throughput(stats: list[CaseStats], stem: str) -> None:
    best_map = best_raster_case_by_threads_and_save(stats)

    fig, ax = plt.subplots(figsize=PLOT_LINE_FIG_SIZE_IN)
    right_ax = ax.twinx()
    legend_handles = []
    legend_labels = []
    all_x_vals: list[int] = []
    memory_baseline = best_map[(1, REFERENCE_SAVE_MODE)].raster_throughput_median_mpx_s

    for save_mode in SAVE_MODE_ORDER:
        line_cases = sorted(
            [case for key, case in best_map.items() if key[1] == save_mode],
            key=lambda case: case.threads,
        )
        if not line_cases:
            continue
        x_vals = [case.threads for case in line_cases]
        y_vals = [case.raster_throughput_median_mpx_s for case in line_cases]
        yerr_low = [
            case.raster_throughput_median_mpx_s - case.raster_throughput_min_mpx_s
            for case in line_cases
        ]
        yerr_high = [
            case.raster_throughput_max_mpx_s - case.raster_throughput_median_mpx_s
            for case in line_cases
        ]
        all_x_vals.extend(x_vals)

        err = ax.errorbar(
            x_vals,
            y_vals,
            yerr=[yerr_low, yerr_high],
            fmt="o-",
            linewidth=1.8,
            markersize=5,
            capsize=3,
            color=SAVE_MODE_COLORS[save_mode],
            label=save_mode_label(save_mode),
        )
        legend_handles.append(err.lines[0])
        legend_labels.append(save_mode_label(save_mode))

    _add_ideal_throughput_line(
        ax,
        all_x_vals,
        memory_baseline,
        "black",
        "Ideal",
    )
    legend_handles.append(ax.lines[-1])
    legend_labels.append("Ideal")
    _style_axes(
        ax,
        "Total Threads",
        "Raster Throughput\n[MPx/s]",
        "",
    )
    right_ax.set_ylabel(
        "Speedup vs. 1-thread",
        fontsize=PLOT_LINE_SECONDARY_AXIS_FONT_SIZE,
    )
    right_ax.tick_params(labelsize=PLOT_LINE_TICK_FONT_SIZE)
    left_min, left_max = ax.get_ylim()
    right_ax.set_ylim(left_min / memory_baseline, left_max / memory_baseline)
    ax.legend(
        legend_handles,
        legend_labels,
        fontsize=PLOT_LINE_LEGEND_FONT_SIZE,
        ncol=1,
    )
    _save_plot(fig, stem)


def build_best_series(stats: list[CaseStats], save_mode: str) -> list[SeriesPoint]:
    best_map = best_case_by_threads_and_save(stats)
    line_cases = sorted(
        [case for key, case in best_map.items() if key[1] == save_mode],
        key=lambda case: case.threads,
    )
    return [
        SeriesPoint(
            threads=case.threads,
            time_median_ms=case.e2e_median_ms,
            time_min_ms=case.e2e_min_ms,
            time_max_ms=case.e2e_max_ms,
            throughput_median_mpx_s=case.e2e_throughput_median_mpx_s,
            throughput_min_mpx_s=case.e2e_throughput_min_mpx_s,
            throughput_max_mpx_s=case.e2e_throughput_max_mpx_s,
        )
        for case in line_cases
    ]


def build_partition_series(
    stats: list[CaseStats],
    save_mode: str,
) -> dict[str, list[SeriesPoint]]:
    best_partition_map = best_case_by_partition(stats)
    series: dict[str, list[SeriesPoint]] = {}
    for (_, case_save, groups, workers_per_group), case in best_partition_map.items():
        if case_save != save_mode:
            continue
        label = f"{groups}x{workers_per_group}"
        series.setdefault(label, []).append(
            SeriesPoint(
                threads=case.threads,
                time_median_ms=case.e2e_median_ms,
                time_min_ms=case.e2e_min_ms,
                time_max_ms=case.e2e_max_ms,
                throughput_median_mpx_s=case.e2e_throughput_median_mpx_s,
                throughput_min_mpx_s=case.e2e_throughput_min_mpx_s,
                throughput_max_mpx_s=case.e2e_throughput_max_mpx_s,
            )
        )
    for label in list(series):
        series[label] = sorted(series[label], key=lambda point: point.threads)
    return series


def build_raster_series(stats: list[CaseStats]) -> list[SeriesPoint]:
    grouped: dict[int, CaseStats] = {}
    for case in stats:
        if case.save_mode != REFERENCE_SAVE_MODE:
            continue
        if case.groups != 1:
            continue
        current = grouped.get(case.threads)
        if (
            current is None
            or case.raster_throughput_median_mpx_s > current.raster_throughput_median_mpx_s
        ):
            grouped[case.threads] = case
    return [
        SeriesPoint(
            threads=case.threads,
            time_median_ms=case.raster_median_ms,
            time_min_ms=case.raster_min_ms,
            time_max_ms=case.raster_max_ms,
            throughput_median_mpx_s=case.raster_throughput_median_mpx_s,
            throughput_min_mpx_s=case.raster_throughput_min_mpx_s,
            throughput_max_mpx_s=case.raster_throughput_max_mpx_s,
        )
        for _, case in sorted(grouped.items())
    ]


def plot_raster_scaling(stats: list[CaseStats], stem: str) -> None:
    series = build_raster_series(stats)
    if not series:
        raise ValueError("no raster scaling series found")

    baseline = series[0]
    fig, ax_left = plt.subplots(figsize=PLOT_LINE_FIG_SIZE_IN)
    ax_right = ax_left.twinx()

    x_vals = [point.threads for point in series]
    throughput = [point.throughput_median_mpx_s for point in series]
    throughput_err_low = [
        point.throughput_median_mpx_s - point.throughput_min_mpx_s for point in series
    ]
    throughput_err_high = [
        point.throughput_max_mpx_s - point.throughput_median_mpx_s for point in series
    ]

    ax_left.errorbar(
        x_vals,
        throughput,
        yerr=[throughput_err_low, throughput_err_high],
        fmt="o-",
        linewidth=1.8,
        markersize=5,
        capsize=3,
        label="Raster",
        color="tab:blue",
    )
    _add_ideal_throughput_line(
        ax_left,
        x_vals,
        baseline.throughput_median_mpx_s,
        "black",
        "Ideal",
    )

    _style_axes(
        ax_left,
        "Total Threads",
        "Raster Throughput\n[MPx/s]",
        "",
    )
    ax_left.set_xticks(x_vals, [str(x) for x in x_vals])
    left_min, left_max = ax_left.get_ylim()
    ax_right.set_ylabel(
        "Speedup vs. 1-thread",
        fontsize=PLOT_LINE_SECONDARY_AXIS_FONT_SIZE,
    )
    ax_right.tick_params(labelsize=PLOT_LINE_TICK_FONT_SIZE)
    ax_right.set_ylim(
        left_min / baseline.throughput_median_mpx_s,
        left_max / baseline.throughput_median_mpx_s,
    )
    ax_left.legend(fontsize=PLOT_LINE_LEGEND_FONT_SIZE, ncol=1)
    _save_plot(fig, stem)


def plot_best_runtime(stats: list[CaseStats], stem: str) -> None:
    best_map = best_case_by_threads_and_save(stats)

    fig, ax = plt.subplots(figsize=PLOT_LINE_FIG_SIZE_IN)
    legend_handles = []
    legend_labels = []

    for save_mode in SAVE_MODE_ORDER:
        line_cases = sorted(
            [case for key, case in best_map.items() if key[1] == save_mode],
            key=lambda case: case.threads,
        )
        if not line_cases:
            continue
        x_vals = [case.threads for case in line_cases]
        y_vals = [case.e2e_median_ms for case in line_cases]
        yerr_low = [case.e2e_median_ms - case.e2e_min_ms for case in line_cases]
        yerr_high = [case.e2e_max_ms - case.e2e_median_ms for case in line_cases]
        baseline = line_cases[0]

        err = ax.errorbar(
            x_vals,
            y_vals,
            yerr=[yerr_low, yerr_high],
            fmt="o-",
            linewidth=1.8,
            markersize=5,
            capsize=3,
            color=SAVE_MODE_COLORS[save_mode],
            label=save_mode_label(save_mode),
        )
        _add_ideal_runtime_line(
            ax,
            x_vals,
            baseline.e2e_median_ms,
            SAVE_MODE_COLORS[save_mode],
            f"{save_mode_label(save_mode)} ideal",
        )
        legend_handles.append(err.lines[0])
        legend_labels.append(save_mode_label(save_mode))
        legend_handles.append(ax.lines[-1])
        legend_labels.append(f"{save_mode_label(save_mode)} ideal")

    _style_axes(
        ax,
        "Total Threads",
        "End-to-End Runtime\n[ms]",
        "Best End-to-End Runtime",
    )
    ax.legend(
        legend_handles,
        legend_labels,
        fontsize=PLOT_LINE_LEGEND_FONT_SIZE,
        ncol=1,
    )
    _save_plot(fig, stem)


def plot_partition_heatmap(
    stats: list[CaseStats],
    save_mode: str,
    stem_absolute: str,
    stem_relative: str,
) -> None:
    best_partition_map = best_case_by_partition(stats)
    data: dict[tuple[str, int], CaseStats] = {}
    best_by_threads = best_case_by_threads_and_save(stats)
    partitions: set[str] = set()
    thread_values: set[int] = set()

    for (threads, case_save, groups, workers_per_group), case in best_partition_map.items():
        if case_save != save_mode:
            continue
        label = f"{groups}x{workers_per_group}"
        data[(label, threads)] = case
        partitions.add(label)
        thread_values.add(threads)

    sorted_partitions = sorted(
        partitions,
        key=lambda label: (int(label.split("x")[0]), int(label.split("x")[1])),
    )
    sorted_threads = sorted(thread_values)

    def build_matrix(relative: bool) -> np.ndarray:
        matrix = np.full((len(sorted_partitions), len(sorted_threads)), np.nan)
        for row_idx, label in enumerate(sorted_partitions):
            for col_idx, threads in enumerate(sorted_threads):
                case = data.get((label, threads))
                if case is None:
                    continue
                value = case.e2e_throughput_median_mpx_s
                if relative:
                    best_case = best_by_threads[(threads, save_mode)]
                    value /= best_case.e2e_throughput_median_mpx_s
                matrix[row_idx, col_idx] = value
        return matrix

    def render_heatmap(matrix: np.ndarray, title: str, cbar_label: str, stem: str) -> None:
        fig, ax = plt.subplots(
            figsize=(PLOT_LINE_FIG_SIZE_IN[0] * 1.55, PLOT_LINE_FIG_SIZE_IN[1] * 1.45),
            constrained_layout=True,
        )
        im = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(sorted_threads)))
        ax.set_xticklabels([str(v) for v in sorted_threads], fontsize=PLOT_LINE_TICK_FONT_SIZE)
        ax.set_yticks(range(len(sorted_partitions)))
        ax.set_yticklabels(sorted_partitions, fontsize=PLOT_LINE_TICK_FONT_SIZE)
        ax.set_xlabel("Total Threads", fontsize=PLOT_LINE_AXIS_FONT_SIZE)
        ax.set_ylabel("Partition", fontsize=PLOT_LINE_AXIS_FONT_SIZE)
        ax.set_title(title, fontsize=PLOT_LINE_TITLE_FONT_SIZE)

        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                val = matrix[row_idx, col_idx]
                if math.isnan(val):
                    continue
                text = f"{val:.2f}" if val < 100.0 else f"{val:.0f}"
                text_color = "white" if val < np.nanmean(matrix) else "black"
                ax.text(
                    col_idx,
                    row_idx,
                    text,
                    ha="center",
                    va="center",
                    fontsize=PLOT_LINE_TICK_FONT_SIZE - 1,
                    color=text_color,
                )

        cbar = fig.colorbar(im, ax=ax, pad=0.03, location="right")
        cbar.set_label(cbar_label, fontsize=PLOT_LINE_SECONDARY_AXIS_FONT_SIZE)
        cbar.ax.tick_params(labelsize=PLOT_LINE_TICK_FONT_SIZE)
        _save_plot(fig, stem)

    render_heatmap(
        build_matrix(False),
        "Partition End-to-End Throughput Heatmap "
        f"({save_mode_label(save_mode)})\n"
        "Partition = Group x Workers",
        "End-to-End Throughput [MPx/s]",
        stem_absolute,
    )
    render_heatmap(
        build_matrix(True),
        f"Partition Relative End-to-End Throughput Heatmap "
        f"({save_mode_label(save_mode)})\n"
        "Partition = Group x Workers",
        "E2E throughput / best at same thread count",
        stem_relative,
    )


def _normalized_heatmap_data(
    stats: list[CaseStats],
    save_mode: str,
) -> dict[tuple[int, str, str], list[float]]:
    best_map = best_case_by_threads_and_save(stats)
    heatmap_values: dict[tuple[int, str, str], list[float]] = {}
    for case in stats:
        if case.save_mode != save_mode:
            continue
        best_case = best_map[(case.threads, case.save_mode)]
        normalized = (
            case.e2e_throughput_median_mpx_s /
            best_case.e2e_throughput_median_mpx_s
        )
        key = (case.workers_per_group, batch_mode(case), geomjobs_mode(case))
        heatmap_values.setdefault(key, []).append(normalized)
    return heatmap_values


def plot_tuning_heatmap(
    stats: list[CaseStats],
    save_mode: str,
    stem: str,
) -> None:
    heatmap_values = _normalized_heatmap_data(stats, save_mode)
    workers_values = sorted({case.workers_per_group for case in stats if case.save_mode == save_mode})

    fig, axes = plt.subplots(
        1,
        len(GEOMJOBS_MODE_ORDER),
        figsize=(PLOT_LINE_FIG_SIZE_IN[0] * 1.8, PLOT_LINE_FIG_SIZE_IN[1] * 1.25),
        squeeze=False,
        constrained_layout=True,
    )

    vmin = 0.0
    vmax = 1.0
    im = None
    for idx, geom_mode_label in enumerate(GEOMJOBS_MODE_ORDER):
        ax = axes[0][idx]
        matrix = np.full((len(workers_values), len(BATCH_MODE_ORDER)), np.nan)
        for row_idx, workers in enumerate(workers_values):
            for col_idx, batch_label in enumerate(BATCH_MODE_ORDER):
                values = heatmap_values.get((workers, batch_label, geom_mode_label))
                if values:
                    matrix[row_idx, col_idx] = statistics.median(values)
        im = ax.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )
        ax.set_xticks(range(len(BATCH_MODE_ORDER)))
        ax.set_xticklabels(BATCH_MODE_ORDER, fontsize=PLOT_LINE_TICK_FONT_SIZE)
        ax.set_yticks(range(len(workers_values)))
        ax.set_yticklabels([str(v) for v in workers_values], fontsize=PLOT_LINE_TICK_FONT_SIZE)
        ax.set_xlabel("Batch Size Mode", fontsize=PLOT_LINE_AXIS_FONT_SIZE)
        if idx == 0:
            ax.set_ylabel("Workers per Group", fontsize=PLOT_LINE_AXIS_FONT_SIZE)
        ax.set_title(
            f"geomjobs = {geom_mode_label}",
            fontsize=PLOT_LINE_TITLE_FONT_SIZE,
        )
        for row_idx, workers in enumerate(workers_values):
            for col_idx, batch_label in enumerate(BATCH_MODE_ORDER):
                val = matrix[row_idx, col_idx]
                if math.isnan(val):
                    continue
                text_color = "white" if val < 0.6 else "black"
                ax.text(
                    col_idx,
                    row_idx,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=PLOT_LINE_TICK_FONT_SIZE - 1,
                    color=text_color,
                )

    if im is not None:
        cbar = fig.colorbar(
            im,
            ax=axes.ravel().tolist(),
            shrink=0.95,
            pad=0.03,
            location="right",
        )
        cbar.set_label(
            "Median E2E throughput / best at same thread count",
            fontsize=PLOT_LINE_SECONDARY_AXIS_FONT_SIZE,
        )
        cbar.ax.tick_params(labelsize=PLOT_LINE_TICK_FONT_SIZE)

    fig.suptitle(
        f"End-to-End Throughput Tuning Heatmap ({save_mode_label(save_mode)})",
        fontsize=PLOT_LINE_TITLE_FONT_SIZE,
    )
    _save_plot(fig, stem)


def plot_memory_disk_crossover_heatmaps(stats: list[CaseStats]) -> None:
    save_cases: dict[str, dict[tuple[int, int, int, int, int], CaseStats]] = {
        save_mode: {}
        for save_mode in SAVE_MODE_ORDER
    }
    workers_values = sorted({case.workers_per_group for case in stats})

    for case in stats:
        if case.save_mode not in SAVE_MODE_ORDER:
            continue
        key = (
            case.threads,
            case.groups,
            case.workers_per_group,
            case.batch,
            case.geom_jobs,
        )
        save_cases[case.save_mode][key] = case

    def build_value_maps(
        lhs_mode: str,
        rhs_mode: str,
    ) -> tuple[dict[tuple[int, str, str], list[float]], dict[tuple[int, str, str], list[float]]]:
        lhs_cases = save_cases[lhs_mode]
        rhs_cases = save_cases[rhs_mode]
        matched_keys = sorted(set(lhs_cases) & set(rhs_cases))
        ratio_values: dict[tuple[int, str, str], list[float]] = {}
        delta_values: dict[tuple[int, str, str], list[float]] = {}

        for key in matched_keys:
            lhs_case = lhs_cases[key]
            rhs_case = rhs_cases[key]
            bucket = (
                lhs_case.workers_per_group,
                batch_mode(lhs_case),
                geomjobs_mode(lhs_case),
            )
            ratio_values.setdefault(bucket, []).append(
                lhs_case.e2e_median_ms / rhs_case.e2e_median_ms
            )
            delta_values.setdefault(bucket, []).append(
                lhs_case.e2e_median_ms - rhs_case.e2e_median_ms
            )

        return ratio_values, delta_values

    def build_matrix(
        source: dict[tuple[int, str, str], list[float]],
    ) -> dict[str, np.ndarray]:
        matrices: dict[str, np.ndarray] = {}
        for geom_mode_label in GEOMJOBS_MODE_ORDER:
            matrix = np.full((len(workers_values), len(BATCH_MODE_ORDER)), np.nan)
            for row_idx, workers in enumerate(workers_values):
                for col_idx, batch_label in enumerate(BATCH_MODE_ORDER):
                    values = source.get((workers, batch_label, geom_mode_label))
                    if values:
                        matrix[row_idx, col_idx] = statistics.median(values)
            matrices[geom_mode_label] = matrix
        return matrices

    def render_panel_heatmap(
        matrices: dict[str, np.ndarray],
        stem: str,
        title: str,
        cbar_label: str,
        fmt_fn,
        cmap: str,
        center: float | None = None,
    ) -> None:
        fig, axes = plt.subplots(
            1,
            len(GEOMJOBS_MODE_ORDER),
            figsize=(PLOT_LINE_FIG_SIZE_IN[0] * 1.8, PLOT_LINE_FIG_SIZE_IN[1] * 1.25),
            squeeze=False,
            constrained_layout=True,
        )

        all_vals = np.concatenate(
            [
                matrix[~np.isnan(matrix)]
                for matrix in matrices.values()
                if np.any(~np.isnan(matrix))
            ]
        )
        if all_vals.size == 0:
            plt.close(fig)
            return
        vmin = float(np.min(all_vals))
        vmax = float(np.max(all_vals))
        im = None

        for idx, geom_mode_label in enumerate(GEOMJOBS_MODE_ORDER):
            ax = axes[0][idx]
            matrix = matrices[geom_mode_label]
            kwargs = {
                "origin": "lower",
                "aspect": "auto",
                "cmap": cmap,
                "vmin": vmin,
                "vmax": vmax,
            }
            if center is not None:
                max_abs = max(abs(vmin - center), abs(vmax - center))
                kwargs["vmin"] = center - max_abs
                kwargs["vmax"] = center + max_abs
            im = ax.imshow(matrix, **kwargs)
            ax.set_xticks(range(len(BATCH_MODE_ORDER)))
            ax.set_xticklabels(BATCH_MODE_ORDER, fontsize=PLOT_LINE_TICK_FONT_SIZE)
            ax.set_yticks(range(len(workers_values)))
            ax.set_yticklabels(
                [str(v) for v in workers_values],
                fontsize=PLOT_LINE_TICK_FONT_SIZE,
            )
            ax.set_xlabel("Batch Size Mode", fontsize=PLOT_LINE_AXIS_FONT_SIZE)
            if idx == 0:
                ax.set_ylabel("Workers per Group", fontsize=PLOT_LINE_AXIS_FONT_SIZE)
            ax.set_title(
                f"geomjobs = {geom_mode_label}",
                fontsize=PLOT_LINE_TITLE_FONT_SIZE,
            )
            mean_val = float(np.nanmean(matrix)) if np.any(~np.isnan(matrix)) else 0.0
            for row_idx in range(matrix.shape[0]):
                for col_idx in range(matrix.shape[1]):
                    val = matrix[row_idx, col_idx]
                    if math.isnan(val):
                        continue
                    text_color = "white" if val < mean_val else "black"
                    ax.text(
                        col_idx,
                        row_idx,
                        fmt_fn(val),
                        ha="center",
                        va="center",
                        fontsize=PLOT_LINE_TICK_FONT_SIZE - 1,
                        color=text_color,
                    )

        if im is not None:
            cbar = fig.colorbar(
                im,
                ax=axes.ravel().tolist(),
                shrink=0.95,
                pad=0.03,
                location="right",
            )
            cbar.set_label(cbar_label, fontsize=PLOT_LINE_SECONDARY_AXIS_FONT_SIZE)
            cbar.ax.tick_params(labelsize=PLOT_LINE_TICK_FONT_SIZE)

        fig.suptitle(title, fontsize=PLOT_LINE_TITLE_FONT_SIZE)
        _save_plot(fig, stem)

    comparison_pairs = [
        ("disk", "memory"),
        ("disk_overlap", "disk"),
        ("disk_overlap", "memory"),
    ]
    for lhs_mode, rhs_mode in comparison_pairs:
        if lhs_mode not in SAVE_MODE_ORDER or rhs_mode not in SAVE_MODE_ORDER:
            continue
        ratio_values, delta_values = build_value_maps(lhs_mode, rhs_mode)
        if not ratio_values:
            continue
        lhs_label = save_mode_label(lhs_mode)
        rhs_label = save_mode_label(rhs_mode)
        stem_suffix = f"{lhs_mode}_vs_{rhs_mode}"
        render_panel_heatmap(
            build_matrix(ratio_values),
            f"fig_bench4_runtime_ratio_{stem_suffix}",
            f"{lhs_label} / {rhs_label} Runtime Ratio",
            f"Median {lhs_mode} runtime / {rhs_mode} runtime",
            lambda val: f"{val:.2f}",
            "coolwarm",
            center=1.0,
        )
        render_panel_heatmap(
            build_matrix(delta_values),
            f"fig_bench4_runtime_delta_ms_{stem_suffix}",
            f"{lhs_label} - {rhs_label} Runtime Difference",
            f"Median {lhs_mode} runtime - {rhs_mode} runtime [ms]",
            lambda val: f"{val:.0f}",
            "coolwarm",
            center=0.0,
        )


def write_amdahl_csv(stats: list[CaseStats]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    series_map: dict[str, list[SeriesPoint]] = {
        "raster_single_group_memory": build_raster_series(stats),
    }
    baseline_map: dict[str, SeriesPoint] = {
        "raster_single_group_memory": build_raster_series(stats)[0],
    }
    for save_mode in SAVE_MODE_ORDER:
        best_series = build_best_series(stats, save_mode)
        series_map[f"best_{save_mode}_e2e"] = best_series
        baseline_map[f"best_{save_mode}_e2e"] = best_series[0]
        for label, series in build_partition_series(stats, save_mode).items():
            series_name = f"partition_{label}_{save_mode}"
            series_map[series_name] = series
            baseline_map[series_name] = best_series[0]

    with BENCH4_AMDAHL_CSV.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "series",
                "metric",
                "threads",
                "time_median_ms",
                "time_min_ms",
                "time_max_ms",
                "throughput_median_mpx_s",
                "throughput_min_mpx_s",
                "throughput_max_mpx_s",
                "speedup_median",
                "speedup_min",
                "speedup_max",
                "point_serial_fraction",
                "fitted_serial_fraction",
                "fitted_parallel_fraction",
                "fitted_asymptotic_speedup",
                "predicted_speedup",
                "predicted_time_ms",
                "predicted_throughput_mpx_s",
                "abs_speedup_error",
                "rel_speedup_error",
            ]
        )

        for series_name, points in sorted(series_map.items()):
            if not points:
                continue
            baseline = baseline_map[series_name]
            threads = [point.threads for point in points]
            speedups = [
                point.throughput_median_mpx_s / baseline.throughput_median_mpx_s
                for point in points
            ]
            fitted_fs = fit_amdahl_serial_fraction(threads, speedups)
            fitted_fp = 1.0 - fitted_fs
            fitted_asymptotic = math.inf if fitted_fs == 0.0 else 1.0 / fitted_fs

            for point, speedup in zip(points, speedups):
                predicted_speedup = 1.0 / (
                    fitted_fs + (1.0 - fitted_fs) / float(point.threads)
                )
                predicted_time_ms = baseline.time_median_ms / predicted_speedup
                predicted_throughput = (
                    baseline.throughput_median_mpx_s * predicted_speedup
                )
                abs_error = abs(predicted_speedup - speedup)
                rel_error = abs_error / speedup if speedup > 0.0 else 0.0
                speedup_min = (
                    point.throughput_min_mpx_s / baseline.throughput_median_mpx_s
                )
                speedup_max = (
                    point.throughput_max_mpx_s / baseline.throughput_median_mpx_s
                )
                writer.writerow(
                    [
                        series_name,
                        "throughput",
                        point.threads,
                        f"{point.time_median_ms:.6f}",
                        f"{point.time_min_ms:.6f}",
                        f"{point.time_max_ms:.6f}",
                        f"{point.throughput_median_mpx_s:.6f}",
                        f"{point.throughput_min_mpx_s:.6f}",
                        f"{point.throughput_max_mpx_s:.6f}",
                        f"{speedup:.6f}",
                        f"{speedup_min:.6f}",
                        f"{speedup_max:.6f}",
                        (
                            ""
                            if point.threads == 1
                            else f"{point_serial_fraction(point.threads, speedup):.6f}"
                        ),
                        f"{fitted_fs:.6f}",
                        f"{fitted_fp:.6f}",
                        (
                            "inf"
                            if math.isinf(fitted_asymptotic)
                            else f"{fitted_asymptotic:.6f}"
                        ),
                        f"{predicted_speedup:.6f}",
                        f"{predicted_time_ms:.6f}",
                        f"{predicted_throughput:.6f}",
                        f"{abs_error:.6f}",
                        f"{rel_error:.6f}",
                    ]
                )
    print(f"Wrote {BENCH4_AMDAHL_CSV}")


def print_best_config_summary(stats: list[CaseStats]) -> None:
    best_map = best_case_by_threads_and_save(stats)
    print("\nBest configurations by total threads:")
    for save_mode in SAVE_MODE_ORDER:
        print(f"  {save_mode_label(save_mode)}:")
        for threads in sorted({case.threads for case in stats if case.save_mode == save_mode}):
            case = best_map[(threads, save_mode)]
            print(
                "   "
                f"threads={threads:>2} "
                f"groups={case.groups:>2} "
                f"workerspg={case.workers_per_group:>2} "
                f"batch={case.batch:>3} "
                f"geomjobs={case.geom_jobs:>2} "
                f"e2e={case.e2e_throughput_median_mpx_s:>8.3f} MPx/s "
                f"raster={case.raster_throughput_median_mpx_s:>8.3f} MPx/s "
                f"runtime={case.e2e_median_ms:>9.3f} ms"
            )


def main(include_disk_overlap: bool = True) -> int:
    global SAVE_MODE_ORDER
    if not include_disk_overlap:
        SAVE_MODE_ORDER = [m for m in SAVE_MODE_ORDER if m != "disk_overlap"]

    stats = collect_case_stats(include_disk_overlap=include_disk_overlap)
    print_best_config_summary(stats)

    plot_raster_scaling(stats, "fig_bench4_raster_scaling")
    plot_best_e2e_throughput(stats, "fig_bench4_best_e2e_throughput")
    plot_best_e2e_throughput(stats, "fig_bench4_best_throughput")
    plot_best_raster_throughput(stats, "fig_bench4_best_raster_throughput")
    plot_best_runtime(stats, "fig_bench4_best_runtime")
    for save_mode in SAVE_MODE_ORDER:
        plot_partition_heatmap(
            stats,
            save_mode,
            f"fig_bench4_partition_heatmap_{save_mode}",
            f"fig_bench4_partition_heatmap_{save_mode}_relative",
        )
        plot_tuning_heatmap(
            stats,
            save_mode,
            f"fig_bench4_tuning_heatmap_{save_mode}",
        )
    plot_memory_disk_crossover_heatmaps(stats)
    write_amdahl_csv(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
