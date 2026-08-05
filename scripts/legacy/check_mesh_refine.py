#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FRAMES_NUM = 63
OUT_DIR = pathlib.Path("verif")
FIG_DPI = 300.0


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent


def fe_root() -> pathlib.Path:
    return repo_root() / "data" / "FE"


def out_root() -> pathlib.Path:
    return repo_root() / OUT_DIR


def parse_geo_vars(geo_path: pathlib.Path) -> dict[str, float]:
    vars_map: dict[str, float] = {}
    vars_map["Pi"] = math.pi

    assign_re = re.compile(
        r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+);"
    )

    with geo_path.open() as geo_file:
        for raw_line in geo_file:
            line = raw_line.split("//", 1)[0].strip()
            if not line:
                continue
            match = assign_re.match(line)
            if match is None:
                continue
            var_name = match.group(1)
            expr = match.group(2).strip()
            try:
                value = eval(
                    expr,
                    {"__builtins__": {}},
                    vars_map,
                )
            except Exception:
                continue
            vars_map[var_name] = float(value)

    return vars_map


def load_post_csv(csv_path: pathlib.Path) -> list[dict[str, float]]:
    with csv_path.open(newline="") as csv_file:
        rows_raw = list(csv.DictReader(csv_file))

    rows: list[dict[str, float]] = []
    for row_raw in rows_raw:
        row_num: dict[str, float] = {}
        for key, value in row_raw.items():
            if value is None or value == "":
                continue
            row_num[key] = float(value)
        rows.append(row_num)
    return rows


def refinement_csv_paths(frames_num: int) -> list[tuple[int, pathlib.Path]]:
    paths: list[tuple[int, pathlib.Path]] = []
    pattern = re.compile(
        rf"platehole3d_(\d+)mr_{frames_num}f\.csv$"
    )
    for csv_path in sorted(fe_root().glob(f"platehole3d_*mr_{frames_num}f.csv")):
        match = pattern.match(csv_path.name)
        if match is None:
            continue
        mesh_ref = int(match.group(1))
        paths.append((mesh_ref, csv_path))
    return paths


def calc_truth_scf(plate_width: float, hole_rad: float) -> float:
    hole_diam = 2.0 * hole_rad
    hole_ratio = hole_diam / plate_width
    return (
        3.00
        - 3.13 * hole_ratio
        + 3.7 * hole_ratio * hole_ratio
    )


def fmt_mesh_ref(mesh_ref: int) -> str:
    return str(mesh_ref)


def write_results_csv(
    out_path: pathlib.Path,
    rows: list[dict[str, float]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    field_names = [
        "mesh_refinement",
        "final_time",
        "reaction_force_abs_N",
        "net_nominal_stress_Pa",
        "max_vm_stress_Pa",
        "max_yy_stress_Pa",
        "vm_scf",
        "yy_scf",
        "scf_truth",
        "vm_scf_error_pct",
        "yy_scf_error_pct",
    ]
    with out_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_stress_plot(
    out_path: pathlib.Path,
    mesh_refs: list[int],
    stress_vals: list[float],
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=FIG_DPI)
    ax.plot(mesh_refs, stress_vals, marker="o", linewidth=1.6)
    ax.set_xlabel("Mesh refinement number, $M$")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)


def save_scf_plot(
    out_path: pathlib.Path,
    mesh_refs: list[int],
    scf_vals: list[float],
    scf_truth: float,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=FIG_DPI)
    ax.plot(mesh_refs, scf_vals, marker="o", linewidth=1.6, label="FE")
    ax.axhline(
        scf_truth,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="Analytical",
    )
    ax.set_xlabel("Mesh refinement number, $M$")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)


def main() -> int:
    geo_vars = parse_geo_vars(fe_root() / "platehole3d.geo")
    plate_width = geo_vars["plate_width"]
    plate_thick = geo_vars["plate_thick"]
    hole_rad = geo_vars["hole_rad"]
    scf_truth = calc_truth_scf(plate_width, hole_rad)
    hole_diam = 2.0 * hole_rad

    csv_paths = refinement_csv_paths(FRAMES_NUM)
    if not csv_paths:
        raise FileNotFoundError(
            f"no platehole3d_*mr_{FRAMES_NUM}f.csv files found in {fe_root()}"
        )

    results: list[dict[str, float]] = []
    mesh_refs: list[int] = []
    max_vm_vals: list[float] = []
    max_yy_vals: list[float] = []
    vm_scf_vals: list[float] = []
    yy_scf_vals: list[float] = []

    for mesh_ref, csv_path in csv_paths:
        rows = load_post_csv(csv_path)
        if not rows:
            raise ValueError(f"no rows found in {csv_path}")
        final_row = rows[-1]
        reaction_force_abs = abs(final_row["react_y_top"])
        max_vm_stress = final_row["stress_vm_max"]
        max_yy_stress = final_row["stress_yy_max"]
        net_nominal_stress = reaction_force_abs / (
            (plate_width - hole_diam) * plate_thick
        )
        vm_scf = (
            max_vm_stress / net_nominal_stress
            if net_nominal_stress != 0.0
            else 0.0
        )
        yy_scf = (
            max_yy_stress / net_nominal_stress
            if net_nominal_stress != 0.0
            else 0.0
        )
        vm_scf_error_pct = (
            100.0 * (vm_scf - scf_truth) / scf_truth
            if scf_truth != 0.0
            else 0.0
        )
        yy_scf_error_pct = (
            100.0 * (yy_scf - scf_truth) / scf_truth
            if scf_truth != 0.0
            else 0.0
        )

        results.append(
            {
                "mesh_refinement": mesh_ref,
                "final_time": final_row["time"],
                "reaction_force_abs_N": reaction_force_abs,
                "net_nominal_stress_Pa": net_nominal_stress,
                "max_vm_stress_Pa": max_vm_stress,
                "max_yy_stress_Pa": max_yy_stress,
                "vm_scf": vm_scf,
                "yy_scf": yy_scf,
                "scf_truth": scf_truth,
                "vm_scf_error_pct": vm_scf_error_pct,
                "yy_scf_error_pct": yy_scf_error_pct,
            }
        )
        mesh_refs.append(mesh_ref)
        max_vm_vals.append(max_vm_stress)
        max_yy_vals.append(max_yy_stress)
        vm_scf_vals.append(vm_scf)
        yy_scf_vals.append(yy_scf)

    out_dir = out_root()
    out_dir.mkdir(parents=True, exist_ok=True)

    write_results_csv(out_dir / "meshref.csv", results)
    save_stress_plot(
        out_dir / "meshref_vmstress.png",
        mesh_refs,
        max_vm_vals,
        "Max. von Mises stress [Pa]",
    )
    save_scf_plot(
        out_dir / "meshref_vmscf.png",
        mesh_refs,
        vm_scf_vals,
        scf_truth,
        "Von Mises SCF, $K$",
    )
    save_stress_plot(
        out_dir / "meshref_yystress.png",
        mesh_refs,
        max_yy_vals,
        "Max. $\\sigma_{yy}$ [Pa]",
    )
    save_scf_plot(
        out_dir / "meshref_yyscf.png",
        mesh_refs,
        yy_scf_vals,
        scf_truth,
        "$\\sigma_{yy}$ SCF, $K$",
    )

    print(f"Frames selected: {FRAMES_NUM}")
    print(f"Plate width [m]: {plate_width:.6e}")
    print(f"Plate thickness [m]: {plate_thick:.6e}")
    print(f"Hole radius [m]: {hole_rad:.6e}")
    print(f"Hole diameter [m]: {hole_diam:.6e}")
    print(f"Analytical SCF: {scf_truth:.6f}")
    print(f"Wrote {out_dir / 'meshref.csv'}")
    print(f"Wrote {out_dir / 'meshref_vmstress.png'}")
    print(f"Wrote {out_dir / 'meshref_vmscf.png'}")
    print(f"Wrote {out_dir / 'meshref_yystress.png'}")
    print(f"Wrote {out_dir / 'meshref_yyscf.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
