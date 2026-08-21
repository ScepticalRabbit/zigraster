#!/usr/bin/env python3
"""Report whether every CSV mesh below ``data`` follows Riley's convention."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from riley.python import meshconv


DATA_DIR = Path(__file__).resolve().parent
CONNECT_FILENAMES = ("connect.csv", "connectivity.csv")


def main() -> int:
    results = [audit_mesh(mesh_dir) for mesh_dir in find_mesh_dirs()]
    width = max(len(mesh_name) for status, mesh_name, _ in results)

    print("Riley mesh convention audit")
    print()
    for status, mesh_name, detail in results:
        print(f"{status:<11} {mesh_name:<{width}}  {detail}")

    conforms = sum(status == "CONFORMS" for status, _, _ in results)
    errors = len(results) - conforms
    print()
    print(f"Summary: {conforms} conform, {errors} do not ({len(results)} meshes total).")
    return 1 if errors else 0


def find_mesh_dirs() -> list[Path]:
    return sorted(
        {
            path.parent
            for filename in CONNECT_FILENAMES
            for path in DATA_DIR.rglob(filename)
            if path.parent.joinpath("coords.csv").is_file()
        },
    )


def audit_mesh(mesh_dir: Path) -> tuple[str, str, str]:
    mesh_name = mesh_dir.relative_to(DATA_DIR).as_posix()
    try:
        coords = np.loadtxt(mesh_dir / "coords.csv", delimiter=",", dtype=np.float64)
        connect_paths = [mesh_dir / name for name in CONNECT_FILENAMES]
        connect_paths = [path for path in connect_paths if path.is_file()]
        connect_tables = [np.loadtxt(path, delimiter=",", dtype=np.int64) for path in connect_paths]
        if len(connect_tables) == 2 and not np.array_equal(*connect_tables):
            raise ValueError("connect.csv and connectivity.csv differ")

        mesh = meshconv.MeshData(
            coords=np.atleast_2d(coords),
            connect={"connect1": np.atleast_2d(connect_tables[0])},
            mesh_type=_mesh_type_hint(mesh_dir),
        )
        report = meshconv.check_mesh_convention(mesh)
    except (OSError, ValueError, NotImplementedError) as err:
        return "DOES NOT", mesh_name, str(err)

    if report.is_valid:
        return "CONFORMS", mesh_name, ""
    return "DOES NOT", mesh_name, ", ".join(report.failed_checks)


def _mesh_type_hint(mesh_dir: Path) -> str | None:
    if mesh_dir.parts[-2] == "FE":
        return "volume"
    if mesh_dir.name.startswith(("tri", "quad")):
        return "surface"
    return None


if __name__ == "__main__":
    sys.exit(main())
