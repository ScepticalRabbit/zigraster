#!/usr/bin/env python3
"""Generate compact analytic oracle data for the Zig verification suite."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from types import ModuleType


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_helper(repo_root: Path) -> ModuleType:
    helper_path = repo_root / "scripts" / "paper_verif_2_helper.py"
    spec = importlib.util.spec_from_file_location("verif_silhouette_helper", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_case_specs() -> list[tuple[str, str, int]]:
    return [
        ("tri3", "shear", 0),
        ("tri6", "bulge", 5),
        ("quad4", "shear", 0),
        ("quad8", "bulge", 5),
        ("quad9", "bulge", 5),
    ]


def generate_oracles() -> Path:
    repo_root = get_repo_root()
    input_root = repo_root / "out" / "verif_oracle_inputs"
    gold_root = repo_root / "gold" / "verif"
    gold_root.mkdir(parents=True, exist_ok=True)
    helper = load_helper(repo_root)

    rows: list[list[float]] = []
    for mesh_name, distort_name, frame_idx in get_case_specs():
        stats_path = (
            input_root
            / f"b_{mesh_name}_{distort_name}"
            / "cam0_frame0_field0_stats.csv"
        )
        result = helper.analyse_stats_file(stats_path, None, 1)
        rows.append(
            [
                float(result["centroid_ref_x_px"]),
                float(result["centroid_ref_y_px"]),
                float(result["area_ref_px2"]),
            ]
        )
        if frame_idx < 0:
            raise ValueError("frame index must be non-negative")

    out_path = gold_root / "silhouette_metrics.csv"
    with out_path.open("w", newline="") as out_file:
        writer = csv.writer(out_file, lineterminator="\n")
        writer.writerows(rows)
    return out_path


def main() -> int:
    out_path = generate_oracles()
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
