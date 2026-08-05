#!/usr/bin/env python3
"""Compare Riley performance runs against LLVMpipe benchmark outputs."""

from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

SSAA_LEVELS = (1, 2)
RILEY_ROOT = Path("out") / "bench_stats_perf"
LLVM_ROOT = Path("temp")
PERF_ROOT = Path("perf")
PNG_NAME = "Riley_vs_LLVMpipe_multiplier.png"


@dataclass(frozen=True)
class CaseStats:
    case: str
    time_ms: float
    throughput_mpx_s: float


@dataclass(frozen=True)
class StatsBundle:
    median: dict[str, CaseStats]
    min: dict[str, CaseStats]
    max: dict[str, CaseStats]
    mad: dict[str, CaseStats]


@dataclass(frozen=True)
class ComparisonRow:
    ssaa: int
    riley_case: str
    llvm_case: str
    riley_time_ms: float
    riley_mad_time_ms: float
    llvm_time_ms: float
    llvm_mad_time_ms: float
    riley_tp_mpx_s: float
    llvm_tp_mpx_s: float

    @property
    def speed_ratio(self) -> float:
        return self.llvm_time_ms / self.riley_time_ms

    @property
    def label(self) -> str:
        return self.riley_case


def find_column(headers: list[str], *keywords: str) -> str:
    for header in headers:
        for keyword in keywords:
            if keyword.lower() in header.lower():
                return header
    raise ValueError(
        f"Could not find column matching {keywords} in headers {headers}"
    )


def load_stats_csv(csv_path: Path) -> dict[str, CaseStats]:
    data: dict[str, CaseStats] = {}
    with csv_path.open(mode="r", newline="", encoding="utf-8") as file_in:
        reader = csv.reader(file_in)
        headers = next(reader)

        case_col = find_column(headers, "case")
        e2e_time_col = find_column(headers, "e2e time", "e2e_time")
        e2e_tp_col = find_column(headers, "e2e tp", "e2e_tp", "e2e throughput")

        case_idx = headers.index(case_col)
        e2e_time_idx = headers.index(e2e_time_col)
        e2e_tp_idx = headers.index(e2e_tp_col)

        for row in reader:
            if not row:
                continue
            case_name = row[case_idx]
            data[case_name] = CaseStats(
                case=case_name,
                time_ms=float(row[e2e_time_idx]),
                throughput_mpx_s=float(row[e2e_tp_idx]),
            )
    return data


def load_stats_bundle(csv_dir: Path, prefix: str = "bench_stats") -> StatsBundle:
    return StatsBundle(
        median=load_stats_csv(csv_dir / f"{prefix}_median.csv"),
        min=load_stats_csv(csv_dir / f"{prefix}_min.csv"),
        max=load_stats_csv(csv_dir / f"{prefix}_max.csv"),
        mad=load_stats_csv(csv_dir / f"{prefix}_mad.csv"),
    )


def load_llvm_stats_bundle(ssaa: int) -> StatsBundle:
    def latest_csv(kind: str) -> Path:
        patterns = (
            f"llvmpipe_stats_{kind}_ssaa{ssaa}_llvmpipe_*.csv",
            f"llvmpipe_stats_{kind}_ssaa{ssaa}_llvmpipe.csv",
        )
        matches: list[Path] = []
        for pattern in patterns:
            matches.extend(LLVM_ROOT.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"No LLVMpipe {kind} stats CSV found for SSAA {ssaa} in {LLVM_ROOT}"
            )
        return max(matches, key=lambda path: path.name)

    return StatsBundle(
        median=load_stats_csv(latest_csv("median")),
        min=load_stats_csv(latest_csv("min")),
        max=load_stats_csv(latest_csv("max")),
        mad=load_stats_csv(latest_csv("mad")),
    )


def latest_timestamp_dir(root_dir: Path) -> Path:
    dirs = [
        path
        for path in root_dir.iterdir()
        if path.is_dir() and re.fullmatch(r"\d{8}_\d{6}", path.name)
    ]
    if not dirs:
        raise FileNotFoundError(f"No timestamped run directories found in {root_dir}")
    return max(dirs, key=lambda path: path.name)


def latest_matching_subdir(parent_dir: Path, pattern: re.Pattern[str]) -> Path:
    matches = [
        path for path in parent_dir.iterdir() if path.is_dir() and pattern.fullmatch(path.name)
    ]
    if not matches:
        raise FileNotFoundError(
            f"No run directories matching {pattern.pattern} found in {parent_dir}"
        )
    return max(matches, key=lambda path: path.name)


def map_llvm_case_to_riley_case(llvm_case: str, riley_cases: set[str]) -> str | None:
    candidates = [llvm_case]

    if llvm_case.startswith("tri3_texfunc_grey_"):
        mapped = llvm_case.replace("tri3_texfunc_grey_", "tri3_func_")
        if "sinusoidal" in mapped and not mapped.endswith("_approx"):
            candidates.append(f"{mapped}_approx")
        candidates.append(mapped)

    if llvm_case.startswith("tri3_texfunc_rgb_"):
        mapped = llvm_case.replace("tri3_texfunc_rgb_", "tri3_func_rgb_")
        if "sinusoidal" in mapped and not mapped.endswith("_approx"):
            candidates.append(f"{mapped}_approx")
        candidates.append(mapped)

    if llvm_case.startswith("tri3_"):
        candidates.append(llvm_case.replace("tri3_", "tri3opt_"))

    for candidate in candidates:
        if candidate in riley_cases:
            return candidate
    return None


def collect_rows_for_ssaa(
    ssaa: int,
    riley_stats: StatsBundle,
    llvm_stats: StatsBundle,
) -> list[ComparisonRow]:
    rows: list[ComparisonRow] = []
    riley_cases = set(riley_stats.median)

    for llvm_case, llvm_case_stats in llvm_stats.median.items():
        riley_case = map_llvm_case_to_riley_case(llvm_case, riley_cases)
        if riley_case is None:
            continue

        riley_case_stats = riley_stats.median[riley_case]
        riley_mad_stats = riley_stats.mad[riley_case]
        llvm_mad_stats = llvm_stats.mad[llvm_case]
        rows.append(
            ComparisonRow(
                ssaa=ssaa,
                riley_case=riley_case,
                llvm_case=llvm_case,
                riley_time_ms=riley_case_stats.time_ms,
                riley_mad_time_ms=riley_mad_stats.time_ms,
                llvm_time_ms=llvm_case_stats.time_ms,
                llvm_mad_time_ms=llvm_mad_stats.time_ms,
                riley_tp_mpx_s=riley_case_stats.throughput_mpx_s,
                llvm_tp_mpx_s=llvm_case_stats.throughput_mpx_s,
            )
        )

    return rows


def write_comparison_csv(csv_path: Path, rows: list[ComparisonRow]) -> None:
    with csv_path.open(mode="w", newline="", encoding="utf-8") as file_out:
        writer = csv.writer(file_out)
        writer.writerow(
            [
                "SSAA",
                "Riley Case",
                "LLVMpipe Case",
                "Riley E2E [ms]",
                "Riley MAD E2E [ms]",
                "LLVMpipe E2E [ms]",
                "LLVMpipe MAD E2E [ms]",
                "Riley E2E TP [MPx/s]",
                "LLVMpipe E2E TP [MPx/s]",
                "LLVMpipe / Riley Time Ratio",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.ssaa,
                    row.riley_case,
                    row.llvm_case,
                    f"{row.riley_time_ms:.6f}",
                    f"{row.riley_mad_time_ms:.6f}",
                    f"{row.llvm_time_ms:.6f}",
                    f"{row.llvm_mad_time_ms:.6f}",
                    f"{row.riley_tp_mpx_s:.6f}",
                    f"{row.llvm_tp_mpx_s:.6f}",
                    f"{row.speed_ratio:.6f}",
                ]
            )


def plot_comparison_png(png_path: Path, rows: list[ComparisonRow]) -> None:
    if not rows:
        raise ValueError("No comparison rows available for plotting.")

    plot_rows = sorted(rows, key=lambda row: (row.ssaa, row.speed_ratio))
    labels = [row.label for row in plot_rows]
    ratios = [row.speed_ratio for row in plot_rows]
    colors = ["#1f77b4" if row.ssaa == 1 else "#ff7f0e" for row in plot_rows]

    fig_height = max(6.0, 0.33 * len(plot_rows))
    fig, axis = plt.subplots(figsize=(16, fig_height), constrained_layout=True)

    y_positions = list(range(len(plot_rows)))
    bars = axis.barh(y_positions, ratios, color=colors)

    axis.set_yticks(y_positions)
    axis.set_yticklabels(labels, fontsize=9)
    axis.set_xlabel(
        "LLVMpipe E2E time / Riley E2E time (> 1.0 means Riley is faster)"
    )
    axis.set_title(
        "Riley vs LLVMpipe multiplier (> 1.0 means Riley is faster)"
    )
    axis.axvline(1.0, color="black", linewidth=1.0, linestyle="--")
    axis.grid(axis="x", linestyle=":", alpha=0.5)

    max_ratio = max(ratios)
    axis.set_xlim(0.0, max(1.25, max_ratio * 1.12))

    for bar, ratio in zip(bars, ratios):
        axis.text(
            bar.get_width() + max_ratio * 0.01,
            bar.get_y() + bar.get_height() * 0.5,
            f"{ratio:.2f}x",
            va="center",
            ha="left",
            fontsize=8,
        )

    legend_handles = [
        plt.Rectangle((0.0, 0.0), 1.0, 1.0, color="#1f77b4", label="SSAA 1"),
        plt.Rectangle((0.0, 0.0), 1.0, 1.0, color="#ff7f0e", label="SSAA 2"),
    ]
    axis.legend(handles=legend_handles, loc="lower right")

    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def plot_absolute_time_png(
    png_path: Path,
    rows: list[ComparisonRow],
    ssaa: int,
) -> None:
    plot_rows = [row for row in rows if row.ssaa == ssaa]
    if not plot_rows:
        raise ValueError(f"No comparison rows available for SSAA {ssaa}.")

    plot_rows.sort(key=lambda row: row.riley_time_ms)

    labels = [row.riley_case for row in plot_rows]
    x_positions = list(range(len(plot_rows)))
    width = 0.38

    riley_times = [row.riley_time_ms for row in plot_rows]
    llvm_times = [row.llvm_time_ms for row in plot_rows]

    riley_err = [3.0 * row.riley_mad_time_ms for row in plot_rows]
    llvm_err = [3.0 * row.llvm_mad_time_ms for row in plot_rows]

    fig_width = max(16.0, 0.42 * len(plot_rows))
    fig, axis = plt.subplots(figsize=(fig_width, 7.5), constrained_layout=True)

    riley_x = [xx - width * 0.5 for xx in x_positions]
    llvm_x = [xx + width * 0.5 for xx in x_positions]

    axis.bar(
        riley_x,
        riley_times,
        width=width,
        color="#1f77b4",
        label="Riley",
    )
    axis.bar(
        llvm_x,
        llvm_times,
        width=width,
        color="#ff7f0e",
        label="LLVMpipe",
    )

    axis.errorbar(
        riley_x,
        riley_times,
        yerr=riley_err,
        fmt="none",
        ecolor="#0d3d66",
        elinewidth=1.2,
        capsize=4,
        capthick=1.2,
        zorder=5,
    )
    axis.errorbar(
        llvm_x,
        llvm_times,
        yerr=llvm_err,
        fmt="none",
        ecolor="#9c4d08",
        elinewidth=1.2,
        capsize=4,
        capthick=1.2,
        zorder=5,
    )

    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)
    axis.set_ylabel("Median E2E time [ms]")
    axis.set_title(
        f"Riley vs LLVMpipe absolute timings, SSAA {ssaa} (error bars: ±3×MAD)"
    )
    axis.grid(axis="y", linestyle=":", alpha=0.5)
    axis.legend(loc="upper left")

    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def main() -> int:
    if not RILEY_ROOT.exists():
        print(f"Error: Riley perf root does not exist: {RILEY_ROOT}", file=sys.stderr)
        return 1

    latest_riley_dir = latest_timestamp_dir(RILEY_ROOT)
    print(f"Latest Riley run directory: {latest_riley_dir}")

    all_rows: list[ComparisonRow] = []

    for ssaa in SSAA_LEVELS:
        riley_pattern = re.compile(
            rf"tiltraster_llvmpipe_compare_tri3_f32_simd_v8_ssaa{ssaa}"
        )
        riley_run_dir = latest_matching_subdir(latest_riley_dir, riley_pattern)
        print(f"SSAA {ssaa} Riley run dir: {riley_run_dir}")

        riley_stats = load_stats_bundle(riley_run_dir)
        llvm_stats = load_llvm_stats_bundle(ssaa)

        ssaa_rows = collect_rows_for_ssaa(ssaa, riley_stats, llvm_stats)
        if not ssaa_rows:
            print(
                f"Warning: no overlapping Riley/LLVMpipe cases found for SSAA {ssaa}",
                file=sys.stderr,
            )
        else:
            out_csv_path = LLVM_ROOT / f"llvmpipe_comparison_results_ssaa{ssaa}.csv"
            write_comparison_csv(out_csv_path, ssaa_rows)
            print(f"Wrote SSAA {ssaa} comparison CSV: {out_csv_path}")

        all_rows.extend(ssaa_rows)

    if not all_rows:
        print("Error: no comparison rows were generated.", file=sys.stderr)
        return 1

    all_rows.sort(key=lambda row: (row.ssaa, row.riley_case))

    PERF_ROOT.mkdir(parents=True, exist_ok=True)
    unified_csv_temp = LLVM_ROOT / "llvmpipe_comparison_results.csv"
    unified_csv_perf = PERF_ROOT / "llvmpipe_comparison_results.csv"
    png_path = PERF_ROOT / PNG_NAME
    abs_png_ssaa1 = PERF_ROOT / "Riley_vs_LLVMpipe_absolute_ssaa1.png"
    abs_png_ssaa2 = PERF_ROOT / "Riley_vs_LLVMpipe_absolute_ssaa2.png"

    write_comparison_csv(unified_csv_temp, all_rows)
    write_comparison_csv(unified_csv_perf, all_rows)
    plot_comparison_png(png_path, all_rows)
    plot_absolute_time_png(abs_png_ssaa1, all_rows, 1)
    plot_absolute_time_png(abs_png_ssaa2, all_rows, 2)

    print(f"Wrote unified CSV: {unified_csv_temp}")
    print(f"Wrote unified CSV: {unified_csv_perf}")
    print(f"Wrote comparison PNG: {png_path}")
    print(f"Wrote absolute timing PNG: {abs_png_ssaa1}")
    print(f"Wrote absolute timing PNG: {abs_png_ssaa2}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
