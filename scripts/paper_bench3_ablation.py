#!/usr/bin/env python3
from __future__ import annotations

import pathlib

from paper_bench_common import combined_case_dir_name
from paper_bench_common import fmt_triplet_any
from paper_bench_common import load_case_map_from_dir
from paper_bench_common import latest_stats_dir_with_candidates
from paper_bench_common import row_float
from paper_bench_common import write_tabs_tex

TABLE_RAW_CAPTION = (
    "Optimisation ablation timings for adaptive hulls and SIMD execution in "
    "benchmark case 3. Timings and throughputs are reported as "
    "median $\\pm$ median absolute deviation (MAD)." 
)

TABLE_SPEEDUP_CAPTION = (
    "Optimisation speedup matrix for adaptive hulls and SIMD execution in "
    "benchmark case 3. Speedup multipliers are computed relative to the "
    "padded-BB scalar baseline for each element type. Values greater than one "
    "indicate faster execution."
)

BENCH_NAME = "bench_sphere2000"
EXPERIMENT_DIR = "experiment_1"
SAVE_STRATEGY = "memory"
REPRESENTATIVE_SHADER_LABEL = "Catmull--Rom LUT-lerp"
REPRESENTATIVE_CASE_SUFFIX = "tex8_grey_cubic_catmull_rom_lut_lerp"

ELEMENT_CASES = [
    ("quad4", "quad4"),
    ("tri6", "tri6"),
    ("quad8", "quad8"),
    ("quad9", "quad9"),
]

CONFIG_ROWS = [
    ("padded BB", "scalar", "scalar", "off"),
    ("padded BB", "SIMD", "simd", "off"),
    ("adaptive hull", "scalar", "scalar", "on_no_fallback"),
    ("adaptive hull", "SIMD", "simd", "on_no_fallback"),
]

SPEEDUP_ROWS = [
    ("SIMD only", "simd", "off"),
    ("adaptive hull only", "scalar", "on_no_fallback"),
    ("adaptive hull + SIMD", "simd", "on_no_fallback"),
]

OUT_TABS_NAME = "tabs_bench3.tex"




def stats_path(simd_label: str, hull_mode: str) -> pathlib.Path:
    return latest_stats_dir_with_candidates(
        BENCH_NAME,
        [
            (
                EXPERIMENT_DIR,
                combined_case_dir_name(
                    BENCH_NAME,
                    simd_label,
                    hull_mode,
                    SAVE_STRATEGY,
                ),
            ),
        ],
        ["bench_stats_median.csv", "bench_stats_mad.csv"],
    )


def fmt_speedup(value: float) -> str:
    return f"${value:.2f}$"


def build_raw_table_tex() -> str:
    stats_maps: dict[tuple[str, str], tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]] = {}

    for _, _, simd_label, hull_mode in CONFIG_ROWS:
        key = (simd_label, hull_mode)
        if key in stats_maps:
            continue
        bench_stats_dir = stats_path(simd_label, hull_mode)
        median_map = load_case_map_from_dir(
            bench_stats_dir,
            "bench_stats_median.csv",
        )
        mad_map = load_case_map_from_dir(
            bench_stats_dir,
            "bench_stats_mad.csv",
        )
        stats_maps[key] = (median_map, mad_map)

    body_rows: list[str] = []
    for element_label, case_prefix in ELEMENT_CASES:
        case_name = f"{case_prefix}_{REPRESENTATIVE_CASE_SUFFIX}"
        for bounds_label, kernel_label, simd_label, hull_mode in CONFIG_ROWS:
            median_map, mad_map = stats_maps[(simd_label, hull_mode)]
            median_row = median_map[case_name]
            mad_row = mad_map[case_name]
            body_rows.append(
                "\\texttt{" + element_label + "} & " +
                bounds_label + " & " +
                kernel_label + " & " +
                fmt_triplet_any(
                    median_row,
                    mad_row,
                    "E2E Time",
                ) + " & " +
                fmt_triplet_any(
                    median_row,
                    mad_row,
                    "Geom Time",
                ) + " & " +
                fmt_triplet_any(
                    median_row,
                    mad_row,
                    "Raster Time",
                ) + " & " +
                fmt_triplet_any(
                    median_row,
                    mad_row,
                    "Raster TP",
                ) + " \\\\"
            )

    body = "\n".join(body_rows)
    return (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{TABLE_RAW_CAPTION}}}\n"
        "\\label{tab:performance_ablation_raw}\n"
        "\\begin{tabular}{lllrrrr}\n"
        "\\hline\n"
        "\\makecell{Element} &\n"
        "\\makecell{Bounds} &\n"
        "\\makecell{Kernel} &\n"
        "\\makecell{End-to-end\\\\{}[$10^{-3}$ s]} &\n"
        "\\makecell{Geometry\\\\{}[$10^{-3}$ s]} &\n"
        "\\makecell{Raster\\\\{}[$10^{-3}$ s]} &\n"
        "\\makecell{Raster\\\\{}[MPx/s]} \\\\\n"
        "\\hline\n"
        f"{body}\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def build_speedup_table_tex() -> str:
    baseline_dir = stats_path("scalar", "off")
    baseline_median_map = load_case_map_from_dir(
        baseline_dir,
        "bench_stats_median.csv",
    )

    variant_maps: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for _, simd_label, hull_mode in SPEEDUP_ROWS:
        key = (simd_label, hull_mode)
        if key in variant_maps:
            continue
        variant_dir = stats_path(simd_label, hull_mode)
        variant_maps[key] = load_case_map_from_dir(
            variant_dir,
            "bench_stats_median.csv",
        )

    body_rows: list[str] = []
    for element_label, case_prefix in ELEMENT_CASES:
        case_name = f"{case_prefix}_{REPRESENTATIVE_CASE_SUFFIX}"
        baseline_row = baseline_median_map[case_name]
        baseline_e2e = row_float(
            baseline_row,
            "E2E Time",
        )
        baseline_raster = row_float(
            baseline_row,
            "Raster Time",
        )

        for config_label, simd_label, hull_mode in SPEEDUP_ROWS:
            variant_row = variant_maps[(simd_label, hull_mode)][case_name]
            variant_e2e = row_float(
                variant_row,
                "E2E Time",
            )
            variant_raster = row_float(
                variant_row,
                "Raster Time",
            )
            e2e_speedup = 0.0 if variant_e2e == 0.0 else baseline_e2e / variant_e2e
            raster_speedup = (
                0.0 if variant_raster == 0.0 else baseline_raster / variant_raster
            )
            body_rows.append(
                "\\texttt{" + element_label + "} & " +
                config_label + " & " +
                fmt_speedup(e2e_speedup) + " & " +
                fmt_speedup(raster_speedup) + " \\\\"
            )

    body = "\n".join(body_rows)
    return (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{TABLE_SPEEDUP_CAPTION}}}\n"
        "\\label{tab:performance_ablation_speedup}\n"
        "\\begin{tabular}{llrr}\n"
        "\\hline\n"
        "\\makecell{Element} &\n"
        "\\makecell{Configuration} &\n"
        "\\makecell{End-to-end\\\\speedup} &\n"
        "\\makecell{Raster\\\\speedup} \\\\\n"
        "\\hline\n"
        f"{body}\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def main() -> int:
    baseline_stats_dir = stats_path("scalar", "off")
    print(f"Using ablation benchmark stats from {baseline_stats_dir.parent}...")
    print(f"Using representative shader: {REPRESENTATIVE_SHADER_LABEL}")
    tabs_tex = build_raw_table_tex() + "\n" + build_speedup_table_tex()
    write_tabs_tex(OUT_TABS_NAME, tabs_tex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
