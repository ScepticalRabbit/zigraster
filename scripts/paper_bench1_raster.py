#!/usr/bin/env python3
from __future__ import annotations

import pathlib

from paper_bench_common import combined_case_dir_name
from paper_bench_common import fmt_triplet_any
from paper_bench_common import legacy_hull_case_dir_name
from paper_bench_common import legacy_simd_case_dir_name
from paper_bench_common import load_case_map_from_dir
from paper_bench_common import latest_stats_dir_with_candidates
from paper_bench_common import write_tabs_tex


BENCH_NAME = "bench_fullraster"
SIMD_LABEL = "simd"
HULL_MODE = "on_no_fallback"
SAVE_STRATEGY = "memory"

TABLE_CAPTION = (
    "Raster stage performance results for the full sensor rendering cases in "
    "benchmark 1. Timings and throughputs are reported as median $\\pm$ median "
    "absolute deviation (MAD)."
)

ELEMENT_CASES = [
    (
        "tri3",
        2,
        [
            ("nodal interp.", "tri3_nodal_grey"),
            (
                "\\makecell[c]{Catmull--Rom\\\\ direct}",
                "tri3_tex8_grey_cubic_catmull_rom_direct",
            ),
            (
                "\\makecell[c]{Catmull--Rom\\\\ LUT-lerp}",
                "tri3_tex8_grey_cubic_catmull_rom_lut_lerp",
            ),
        ],
    ),
    (
        "tri6",
        2,
        [
            ("nodal interp.", "tri6_nodal_grey"),
            (
                "\\makecell[c]{Catmull--Rom\\\\ direct}",
                "tri6_tex8_grey_cubic_catmull_rom_direct",
            ),
            (
                "\\makecell[c]{Catmull--Rom\\\\ LUT-lerp}",
                "tri6_tex8_grey_cubic_catmull_rom_lut_lerp",
            ),
        ],
    ),
    (
        "quad4",
        1,
        [
            ("nodal interp.", "quad4_nodal_grey"),
            (
                "\\makecell[c]{Catmull--Rom\\\\ direct}",
                "quad4_tex8_grey_cubic_catmull_rom_direct",
            ),
            (
                "\\makecell[c]{Catmull--Rom\\\\ LUT-lerp}",
                "quad4_tex8_grey_cubic_catmull_rom_lut_lerp",
            ),
        ],
    ),
    (
        "quad8",
        1,
        [
            ("nodal interp.", "quad8_nodal_grey"),
            (
                "\\makecell[c]{Catmull--Rom\\\\ direct}",
                "quad8_tex8_grey_cubic_catmull_rom_direct",
            ),
            (
                "\\makecell[c]{Catmull--Rom\\\\ LUT-lerp}",
                "quad8_tex8_grey_cubic_catmull_rom_lut_lerp",
            ),
        ],
    ),
    (
        "quad9",
        1,
        [
            ("nodal interp.", "quad9_nodal_grey"),
            (
                "\\makecell[c]{Catmull--Rom\\\\ direct}",
                "quad9_tex8_grey_cubic_catmull_rom_direct",
            ),
            (
                "\\makecell[c]{Catmull--Rom\\\\ LUT-lerp}",
                "quad9_tex8_grey_cubic_catmull_rom_lut_lerp",
            ),
        ],
    ),
]

OUT_TABS_NAME = "tabs_bench1.tex"




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

    body_rows: list[str] = []
    for element_name, elems_num, shader_cases in ELEMENT_CASES:
        for shader_label, case_name in shader_cases:
            median_row = median_map[case_name]
            mad_row = mad_map[case_name]
            e2e_text = fmt_triplet_any(
                median_row,
                mad_row,
                "E2E Time",
            )
            raster_text = fmt_triplet_any(
                median_row,
                mad_row,
                "Raster Time",
            )
            throughput_text = fmt_triplet_any(
                median_row,
                mad_row,
                "Raster TP",
            )
            body_rows.append(
                "\\texttt{" + element_name + "} & " +
                shader_label + " & " +
                str(elems_num) + " & " +
                e2e_text + " & " +
                raster_text + " & " +
                throughput_text + " \\\\"
            )

    body = "\n".join(body_rows)
    return (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{TABLE_CAPTION}}}\n"
        "\\label{tab:performance_raster_benchmark}\n"
        "\\begin{tabular}{llrlll}\n"
        "\\hline\n"
        "\\makecell[c]{Element\\\\ Type} & Shader & "
        "\\makecell[c]{Element\\\\ Count} & "
        "\\makecell[c]{End-to-end\\\\ {[$10^{-3}$ s]}} & "
        "\\makecell[c]{Raster\\\\ {[$10^{-3}$ s]}} & "
        "\\makecell[c]{Throughput\\\\ {[MPx/s]}} \\\\\n"
        "\\hline\n"
        f"{body}\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def main() -> int:
    bench_stats_dir = stats_path()
    print(f"Using benchmark stats from {bench_stats_dir}...")
    tabs_tex = build_table_tex()
    write_tabs_tex(OUT_TABS_NAME, tabs_tex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
