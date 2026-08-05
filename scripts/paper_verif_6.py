#!/usr/bin/env python3
from __future__ import annotations

import csv
import pathlib
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from paper_const import SHEAR_SHEAR, bulge_in_frame, repo_root


VERIF_DIR = pathlib.Path("verif/verif_6")
SUMMARY_PATH = VERIF_DIR / "verif_6_summary.csv"
OUT_TABS_TEX_PATH = VERIF_DIR / "tabs_verif_6.tex"
OUT_FIGS_TEX_PATH = VERIF_DIR / "figs_verif_6.tex"
OUT_DIR = repo_root() / VERIF_DIR
CAMERA_CASE_ORDER = [
    "none",
    "mild_barrel",
    "mild_pincushion",
    "strong_barrel",
    "mixed_asymmetric",
]
CAMERA_CASE_LABELS = {
    "none": "none",
    "mild_barrel": "mild barrel",
    "mild_pincushion": "mild pincushion",
    "strong_barrel": "strong barrel",
    "mixed_asymmetric": "mixed asymmetric",
}


def python_path() -> pathlib.Path:
    venv_python = repo_root() / ".venv" / "bin" / "python"
    if venv_python.exists():
        return venv_python
    return pathlib.Path(sys.executable)


def load_summary(summary_path: pathlib.Path) -> list[dict[str, str]]:
    with summary_path.open(newline="") as summary_file:
        return list(csv.DictReader(summary_file))


def ensure_summary_rows() -> list[dict[str, str]]:
    summary_path = repo_root() / SUMMARY_PATH
    if not summary_path.exists():
        print(
            "Missing verif_6 summary, generating it with "
            "scripts/paper_verif_6_helper.py..."
        )
        subprocess.run(
            [
                str(python_path()),
                "scripts/paper_verif_6_helper.py",
            ],
            cwd=repo_root(),
            check=True,
        )
    return load_summary(summary_path)


def find_row(
    rows: list[dict[str, str]],
    mesh_name: str,
    geom_name: str,
    camera_case: str,
    frame_idx: int,
) -> dict[str, str]:
    for row in rows:
        if (
            row["mesh_name"] == mesh_name and
            row["geom_name"] == geom_name and
            row["camera_case"] == camera_case and
            int(row["frame_idx"]) == frame_idx
        ):
            return row
    raise KeyError(
        f"missing row for {mesh_name} {geom_name} {camera_case} "
        f"frame {frame_idx}",
    )


def save_png_from_bmp(src_path: pathlib.Path, dst_name: str) -> pathlib.Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dst_path = OUT_DIR / dst_name
    with Image.open(src_path) as image:
        image.save(
            dst_path,
            format="PNG",
            optimize=True,
            dpi=(300, 300),
        )
        fig, ax = plt.subplots(figsize=(3.1, 3.1), constrained_layout=False)
        ax.imshow(image, cmap="gray")
        ax.set_axis_off()
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        fig.savefig(
            dst_path.with_suffix(".svg"),
            bbox_inches="tight",
            pad_inches=0.0,
        )
        plt.close(fig)
    return dst_path


def verif_case_dir(
    mesh_name: str,
    geom_name: str,
    camera_case: str,
) -> pathlib.Path:
    return OUT_DIR / f"verif_6_{mesh_name}_{geom_name}_{camera_case}"


def make_table_row(row: dict[str, str]) -> str:
    area_err_pct = float(row["area_err_pct"])
    area_field = "--" if area_err_pct != area_err_pct else f"{area_err_pct:.3f}"
    return (
        f"\\texttt{{{row['mesh_name']}}} & {row['geom_name']} & "
        f"{CAMERA_CASE_LABELS[row['camera_case']]} & "
        f"{float(row['centroid_dist_px']):.3f} & "
        f"{area_field} & "
        f"{float(row['mask_diff_pct']):.3f} \\\\"
    )


def build_table_tex(rows: list[dict[str, str]]) -> str:
    body_rows: list[str] = []
    for mesh_name in ["tri3", "quad4", "tri6", "quad8", "quad9"]:
        body_rows.append(
            make_table_row(
                find_row(rows, mesh_name, "shear", "none", SHEAR_SHEAR),
            )
        )
        for camera_case in CAMERA_CASE_ORDER[1:]:
            body_rows.append(
                make_table_row(
                    find_row(
                        rows,
                        mesh_name,
                        "shear",
                        camera_case,
                        SHEAR_SHEAR,
                    )
                )
            )
        if mesh_name in {"tri6", "quad8", "quad9"}:
            for camera_case in CAMERA_CASE_ORDER:
                body_rows.append(
                    make_table_row(
                        find_row(
                            rows,
                            mesh_name,
                            "bulge",
                            camera_case,
                            bulge_in_frame(mesh_name),
                        )
                    )
                )
    body = "\n".join(body_rows)
    return (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Selected distorted silhouette verification statistics "
        "for verification case 6.}\n"
        "\\label{tab:verification_case_6_silhouette}\n"
        "\\begin{tabular}{lllrrr}\n"
        "\\hline\n"
        "Element & Geometry & Distortion & Centroid [px] & "
        "Area err. [\\%] & Mask diff. [\\%] \\\\\n"
        "\\hline\n"
        f"{body}\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def export_selected_figures(rows: list[dict[str, str]]) -> list[str]:
    figure_names: list[str] = []
    selected_specs = [
        ("tri6", "shear", SHEAR_SHEAR),
        ("tri6", "bulge", bulge_in_frame("tri6")),
    ]
    for mesh_name, geom_name, frame_idx in selected_specs:
        for camera_case in CAMERA_CASE_ORDER:
            row = find_row(rows, mesh_name, geom_name, camera_case, frame_idx)
            case_dir = verif_case_dir(mesh_name, geom_name, camera_case)
            bmp_name = f"cam0_frame{int(row['frame_idx'])}_field0.bmp"
            out_name = (
                f"fig_verif_6_{mesh_name}_{geom_name}_{camera_case}.png"
            )
            save_png_from_bmp(case_dir / bmp_name, out_name)
            figure_names.append(out_name)
    return figure_names


def subfigure_block(file_name: str) -> str:
    return (
        "\\begin{subfigure}[c]{0.19\\textwidth}\n"
        "\\centering\n"
        f"\\includegraphics[width=\\linewidth]{{{file_name}}}\n"
        "\\caption{}\n"
        "\\end{subfigure}"
    )


def build_figs_tex(figure_names: list[str]) -> str:
    return (
        "\\begin{figure}[htbp]\n"
        "\\centering\n"
        + "\n".join(subfigure_block(file_name) for file_name in figure_names)
        + "\n\\caption{Selected distorted silhouette renders for "
        "verification case 6.}\n"
        "\\label{fig:verification_case_6_silhouette}\n"
        "\\end{figure}\n"
    )


def write_outputs(tabs_tex: str, figs_tex: str) -> None:
    out_tabs_path = repo_root() / OUT_TABS_TEX_PATH
    out_figs_path = repo_root() / OUT_FIGS_TEX_PATH
    out_tabs_path.parent.mkdir(parents=True, exist_ok=True)
    out_tabs_path.write_text(tabs_tex)
    out_figs_path.write_text(figs_tex)
    print(f"Wrote {out_tabs_path}")
    print(f"Wrote {out_figs_path}")


def main() -> int:
    print("Loading verif_6 summary...")
    rows = ensure_summary_rows()
    print(f"Loaded {len(rows)} summary rows.")
    print("Building verif_6 LaTeX table...")
    tabs_tex = build_table_tex(rows)
    print("Exporting verif_6 figures...")
    figure_names = export_selected_figures(rows)
    print("Building verif_6 figure TeX...")
    figs_tex = build_figs_tex(figure_names)
    write_outputs(tabs_tex, figs_tex)
    print(f"Saved figure assets to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
