#!/usr/bin/env python3
from __future__ import annotations

import csv
import pathlib
from paper_bench_common import combined_case_dir_name
from paper_bench_common import fmt_triplet
from paper_bench_common import fmt_triplet_any
from paper_bench_common import calc_median_mad
from paper_bench_common import legacy_hull_case_dir_name
from paper_bench_common import legacy_simd_case_dir_name
from paper_bench_common import load_case_map_from_dir
from paper_bench_common import load_run_case_rows_from_dir
from paper_bench_common import latest_stats_dir_with_candidates
from paper_bench_common import write_tabs_tex
from paper_bench_common import row_float
from paper_const import repo_root
from paper_const import TABLE_MAD_DECIMAL_PLACES
from paper_const import TABLE_MEDIAN_DECIMAL_PLACES


BENCH_NAME = "bench_geom"
SIMD_LABEL = "simd"
HULL_MODE = "on_no_fallback"
SAVE_STRATEGY = "memory"
GEOM_SHADER_LABEL = "nodal interp."
GEOM_SHADER_CASE_SUFFIX = "nodal_grey"
GEOM_CASES = [
    ("tri3", "tri3", "tri3"),
    ("tri6", "tri6", "tri6"),
    ("quad4", "quad4", "quad4"),
    ("quad8", "quad8", "quad8"),
    ("quad9", "quad9", "quad9"),
]
OUT_TABS_NAME = "tabs_bench2.tex"

TABLE_CAPTION = (
    "Geometry preprocessing performance results for all element types in "
    "benchmark 2. Timings and throughputs are reported as median $\\pm$ median "
    "absolute deviation (MAD)."
)


def stats_path() -> pathlib.Path:
    candidates: list[tuple[str, str]] = [
        (
            "experiment_1",
            combined_case_dir_name(
                BENCH_NAME,
                SIMD_LABEL,
                HULL_MODE,
                SAVE_STRATEGY,
            ),
        ),
    ]
    if HULL_MODE == "off":
        candidates.append(
            (
                "experiment_1",
                legacy_simd_case_dir_name(
                    BENCH_NAME,
                    SIMD_LABEL,
                    SAVE_STRATEGY,
                ),
            ),
        )
    else:
        candidates.append(
            (
                "experiment_2",
                legacy_hull_case_dir_name(
                    BENCH_NAME,
                    SIMD_LABEL,
                    HULL_MODE,
                    SAVE_STRATEGY,
                ),
            ),
        )
    return latest_stats_dir_with_candidates(
        BENCH_NAME,
        candidates,
    )


def count_elements(mesh_dir_name: str) -> int:
    connect_path = (
        repo_root() /
        "data" /
        "bench" /
        f"{mesh_dir_name}_geom" /
        "connect.csv"
    )
    with connect_path.open(newline="") as csv_file:
        rows = [
            row for row in csv.reader(csv_file)
            if any(cell.strip() for cell in row)
        ]
    return len(rows)


def count_nodes(mesh_dir_name: str) -> int:
    mesh_dir = (
        repo_root() /
        "data" /
        "bench" /
        f"{mesh_dir_name}_geom"
    )
    coords_path = mesh_dir / "coords.csv"
    connect_path = mesh_dir / "connect.csv"
    with coords_path.open(newline="") as csv_file:
        coords_rows = [
            row for row in csv.reader(csv_file)
            if any(cell.strip() for cell in row)
        ]
    coords_count = len(coords_rows)
    used_nodes: set[int] = set()
    with connect_path.open(newline="") as csv_file:
        for row in csv.reader(csv_file):
            if not any(cell.strip() for cell in row):
                continue
            for cell in row:
                cell = cell.strip()
                if cell:
                    used_nodes.add(int(cell))
    if used_nodes and max(used_nodes) >= coords_count:
        raise ValueError(
            f"Connectivity in {mesh_dir_name}_geom references node index "
            f"{max(used_nodes)} but coords.csv only has {coords_count} rows."
        )
    return len(used_nodes)


def fmt_float_pair(
    median_val: float,
    mad_val: float,
) -> str:
    return (
        f"${median_val:.{TABLE_MEDIAN_DECIMAL_PLACES}f} "
        f"\\pm {mad_val:.{TABLE_MAD_DECIMAL_PLACES}f}$"
    )


def build_table_tex() -> str:
    bench_stats_dir = stats_path()
    median_map = load_case_map_from_dir(
        bench_stats_dir,
        "bench_stats_median.csv",
    )
    mad_map = load_case_map_from_dir(
        bench_stats_dir,
        "bench_stats_mad.csv",
    )
    run_case_maps = load_run_case_rows_from_dir(
        bench_stats_dir,
    )

    body_rows: list[str] = []
    for element_label, mesh_dir_name, case_prefix in GEOM_CASES:
        case_name = f"{case_prefix}_{GEOM_SHADER_CASE_SUFFIX}"
        median_row = median_map[case_name]
        mad_row = mad_map[case_name]
        elems_num = count_elements(mesh_dir_name)
        nodes_num = count_nodes(mesh_dir_name)
        e2e_text = fmt_triplet_any(
            median_row,
            mad_row,
            "E2E Time",
        )
        geom_text = fmt_triplet_any(
            median_row,
            mad_row,
            "Geom Time",
        )
        throughput_vals: list[float] = []
        for case_map in run_case_maps:
            run_row = case_map[case_name]
            geom_ms = row_float(run_row, "Geom Time")
            if geom_ms > 0.0:
                throughput_vals.append(nodes_num / geom_ms / 1.0e3)
        throughput_median, throughput_mad = calc_median_mad(
            throughput_vals,
        )
        elem_throughput_vals: list[float] = []
        for case_map in run_case_maps:
            run_row = case_map[case_name]
            geom_ms = row_float(run_row, "Geom Time")
            if geom_ms > 0.0:
                elem_throughput_vals.append(elems_num / geom_ms / 1.0e3)
        elem_throughput_median, elem_throughput_mad = calc_median_mad(
            elem_throughput_vals,
        )
        elem_throughput_text = fmt_float_pair(
            elem_throughput_median,
            elem_throughput_mad,
        )
        node_throughput_text = fmt_float_pair(
            throughput_median,
            throughput_mad,
        )
        body_rows.append(
            "\\texttt{" + element_label + "} & " +
            str(nodes_num) + " & " +
            str(elems_num) + " & " +
            e2e_text + " & " +
            geom_text + " & " +
            node_throughput_text + " & " +
            elem_throughput_text + " \\\\"
        )

    body = "\n".join(body_rows)
    return (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Geometry-preprocessing benchmark results for "
        "representative element types. The image size is fixed at "
        "$1600 \\times 1000$ pixels, SSAA is fixed at $1 \\times 1$, and "
        "all runs are single threaded. Timings and throughputs are reported "
        "as median $\\pm$ MAD over 10 runs. Wall-clock timings are reported "
        "in $10^{-3}$ seconds. The geometry throughput is reported in both "
        "MNodes/s and MElem/s, computed from the exact input node and element "
        "counts and the geometry preprocessing time.}\n"
        "\\label{tab:performance_geometry_benchmark}\n"
        "\\begin{tabular}{lrrllll}\n"
        "\\hline\n"
        "\\makecell[c]{Element\\\\ Type} & "
        "\\makecell[c]{Node\\\\ Count} & "
        "\\makecell[c]{Element\\\\ Count} & "
        "\\makecell[c]{End-to-end\\\\ {[$10^{-3}$ s]}} & "
        "\\makecell[c]{Geometry\\\\ {[$10^{-3}$ s]}} & "
        "\\makecell[c]{Throughput\\\\ {[MNodes/s]}} & "
        "\\makecell[c]{Throughput\\\\ {[MElem/s]}} \\\\\n"
        "\\hline\n"
        f"{body}\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
def main() -> int:
    bench_stats_dir = stats_path()
    print(f"Using benchmark stats from {bench_stats_dir}...")
    print(f"Using shader selection: {GEOM_SHADER_LABEL}")
    tabs_tex = build_table_tex()
    write_tabs_tex(OUT_TABS_NAME, tabs_tex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
