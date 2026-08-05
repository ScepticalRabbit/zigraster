#!/usr/bin/env python3
from __future__ import annotations

import csv
import pathlib
import statistics

from paper_bench_common import combined_case_dir_name
from paper_bench_common import load_stable_row_map
from paper_bench_common import latest_run_dir_with_paths
from paper_const import repo_root


SAVE_STRATEGY = "memory"
EXPERIMENT_DIR = "experiment_1"

ANALYSE_RASTER = True
ANALYSE_GEOM = True
ANALYSE_SPHERE2000 = True
ANALYSE_SPHERE2000ZOOM = True

BASE_SIMD_LABEL = "scalar"
BASE_HULL_MODE = "off"

ABLATION_VARIANTS = {
    "simdon": ("simd", "off"),
    "hullson": ("scalar", "on_no_fallback"),
    "simdon_hullson": ("simd", "on_no_fallback"),
}

BENCH_SPECS = {
    "bench_fullraster": {
        "prefix": "raster",
        "metrics": [
            "E2E TP [MPx/s]",
            "E2E Time [ms]",
            "Geom Time [ms]",
            "Raster Time [ms]",
            "Raster TP [MPx/s]",
        ],
        "speed_metrics": ["E2E TP [MPx/s]", "Raster TP [MPx/s]"],
    },
    "bench_geom": {
        "prefix": "geom",
        "metrics": [
            "E2E TP [MPx/s]",
            "E2E Time [ms]",
            "Geom Time [ms]",
            "Geom TP [MElem/s]",
        ],
        "speed_metrics": ["Geom TP [MElem/s]"],
    },
    "bench_sphere2000": {
        "prefix": "sphere2000",
        "metrics": [
            "E2E TP [MPx/s]",
            "E2E Time [ms]",
            "Geom Time [ms]",
            "Raster Time [ms]",
            "Geom TP [MElem/s]",
            "Raster TP [MPx/s]",
        ],
        "speed_metrics": ["Geom TP [MElem/s]", "Raster TP [MPx/s]"],
    },
    "bench_sphere2000zoom": {
        "prefix": "sphere2000zoom",
        "metrics": [
            "E2E TP [MPx/s]",
            "E2E Time [ms]",
            "Geom Time [ms]",
            "Raster Time [ms]",
            "Geom TP [MElem/s]",
            "Raster TP [MPx/s]",
        ],
        "speed_metrics": ["Geom TP [MElem/s]", "Raster TP [MPx/s]"],
    },
}

BENCH_ENABLED = {
    "bench_fullraster": ANALYSE_RASTER,
    "bench_geom": ANALYSE_GEOM,
    "bench_sphere2000": ANALYSE_SPHERE2000,
    "bench_sphere2000zoom": ANALYSE_SPHERE2000ZOOM,
}


def case_dir_name(
    bench_name: str,
    simd_label: str,
    hull_mode: str,
) -> str:
    return combined_case_dir_name(
        bench_name,
        simd_label,
        hull_mode,
        SAVE_STRATEGY,
    )


def required_paths_for_variant(
    bench_name: str,
    variant_name: str,
) -> list[str]:
    base_case_dir = case_dir_name(
        bench_name,
        BASE_SIMD_LABEL,
        BASE_HULL_MODE,
    )
    variant_simd_label, variant_hull_mode = ABLATION_VARIANTS[variant_name]
    variant_case_dir = case_dir_name(
        bench_name,
        variant_simd_label,
        variant_hull_mode,
    )

    required_rel_paths: list[str] = []
    for case_dir in (base_case_dir, variant_case_dir):
        for stat_name in ("median", "mad", "min", "max"):
            required_rel_paths.append(
                f"{EXPERIMENT_DIR}/{case_dir}/bench_stats_{stat_name}.csv"
            )
    return required_rel_paths


def resolve_variant_run_dir(
    bench_name: str,
    variant_name: str,
) -> tuple[pathlib.Path, str, str]:
    run_dir = latest_run_dir_with_paths(
        bench_name,
        required_paths_for_variant(bench_name, variant_name),
    )
    base_case_dir = case_dir_name(
        bench_name,
        BASE_SIMD_LABEL,
        BASE_HULL_MODE,
    )
    variant_simd_label, variant_hull_mode = ABLATION_VARIANTS[variant_name]
    variant_case_dir = case_dir_name(
        bench_name,
        variant_simd_label,
        variant_hull_mode,
    )
    return run_dir, base_case_dir, variant_case_dir


def load_stats_bundle(
    run_dir: pathlib.Path,
    case_dir: str,
) -> dict[str, dict[str, dict[str, str]]]:
    stats_dir = run_dir / EXPERIMENT_DIR / case_dir
    return {
        stat_name: load_stable_row_map(
            stats_dir / f"bench_stats_{stat_name}.csv"
        )
        for stat_name in ("median", "mad", "min", "max")
    }


def fmt_float(value: float) -> str:
    return f"{value:.6f}"


def calc_speedup(
    base_row: dict[str, str],
    variant_row: dict[str, str],
    metric_name: str,
) -> float:
    base_val = float(base_row[metric_name])
    variant_val = float(variant_row[metric_name])
    if base_val == 0.0:
        return 0.0
    return variant_val / base_val


def build_ablation_rows(
    run_dir: pathlib.Path,
    base_case_dir: str,
    variant_case_dir: str,
    metrics: list[str],
    speed_metrics: list[str],
) -> list[dict[str, str]]:
    base_stats = load_stats_bundle(run_dir, base_case_dir)
    variant_stats = load_stats_bundle(run_dir, variant_case_dir)

    stable_keys = [
        key
        for key in base_stats["median"].keys()
        if key in variant_stats["median"]
    ]

    rows: list[dict[str, str]] = []
    for stable_key in stable_keys:
        base_meta = base_stats["median"][stable_key]
        out_row = {
            "CaseStableKey": stable_key,
            "Case": base_meta["Case"],
            "CaseOccurrence": base_meta["CaseOccurrence"],
            "Element": base_meta["Element"],
            "Shader": base_meta["Shader"],
            "Interpolator": base_meta["Interpolator"],
            "BaseConfig": base_case_dir,
            "VariantConfig": variant_case_dir,
        }

        for metric_name in metrics:
            for stat_name in ("median", "mad", "min", "max"):
                out_row[f"{metric_name}_{stat_name}_base"] = (
                    base_stats[stat_name][stable_key][metric_name]
                )
                out_row[f"{metric_name}_{stat_name}_variant"] = (
                    variant_stats[stat_name][stable_key][metric_name]
                )

        for metric_name in speed_metrics:
            speedup = calc_speedup(
                base_stats["median"][stable_key],
                variant_stats["median"][stable_key],
                metric_name,
            )
            out_row[f"{metric_name}_speedup_x"] = fmt_float(speedup)

        out_row["CaseStableKeyTail"] = stable_key
        rows.append(out_row)

    return rows


def write_csv(
    file_name: str,
    rows: list[dict[str, str]],
) -> pathlib.Path:
    out_path = repo_root() / "verif" / file_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows to write for {file_name}")
    field_names = list(rows[0].keys())
    with out_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path}")
    return out_path


def summarize_speedups(
    rows: list[dict[str, str]],
    metric_name: str,
) -> tuple[float, float, float]:
    speedups = [float(row[f"{metric_name}_speedup_x"]) for row in rows]
    return min(speedups), statistics.median(speedups), max(speedups)


def analyse_bench_variant(
    bench_name: str,
    prefix: str,
    variant_name: str,
    metrics: list[str],
    speed_metrics: list[str],
) -> tuple[pathlib.Path, list[dict[str, str]]]:
    run_dir, base_case_dir, variant_case_dir = resolve_variant_run_dir(
        bench_name,
        variant_name,
    )
    print(
        f"{bench_name}: {variant_name} run dir = {run_dir} "
        f"({base_case_dir} -> {variant_case_dir})"
    )
    rows = build_ablation_rows(
        run_dir,
        base_case_dir,
        variant_case_dir,
        metrics,
        speed_metrics,
    )
    csv_path = write_csv(
        f"{prefix}_{variant_name}_ablation.csv",
        rows,
    )
    return csv_path, rows


def print_raster_overview(
    variant_rows: dict[str, list[dict[str, str]]],
) -> None:
    print("\nRaster overview")
    for variant_name in ("simdon", "hullson", "simdon_hullson"):
        rows = variant_rows[variant_name]
        spd_min, spd_med, spd_max = summarize_speedups(
            rows,
            "Raster TP [MPx/s]",
        )
        print(
            f"{variant_name} Raster TP [MPx/s] speedup min/median/max = "
            f"{spd_min:.3f} / {spd_med:.3f} / {spd_max:.3f}"
        )


def main() -> int:
    raster_rows_by_variant: dict[str, list[dict[str, str]]] = {}

    for bench_name, spec in BENCH_SPECS.items():
        if not BENCH_ENABLED[bench_name]:
            print(f"{bench_name}: skipped by top-level selection constant")
            continue

        for variant_name in ("simdon", "hullson", "simdon_hullson"):
            _, rows = analyse_bench_variant(
                bench_name,
                spec["prefix"],
                variant_name,
                spec["metrics"],
                spec["speed_metrics"],
            )
            if bench_name == "bench_fullraster":
                raster_rows_by_variant[variant_name] = rows

    if raster_rows_by_variant:
        print_raster_overview(raster_rows_by_variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
