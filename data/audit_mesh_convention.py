#!/usr/bin/env python3
"""Audit CSV meshes below ``data`` against Riley's mesh convention."""


import importlib
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType

import numpy as np


DATA_DIR = Path(__file__).resolve().parent
SRC_DIR = DATA_DIR.parent / "src"
CONNECT_FILENAMES = ("connect.csv", "connectivity.csv")

# Bypass riley/__init__.py so the audit needs no compiled Cython extension.
for package_name in ("riley", "riley.python"):
    package = ModuleType(package_name)
    package.__path__ = [str(SRC_DIR.joinpath(*package_name.split(".")))]
    sys.modules[package_name] = package
meshconv = importlib.import_module("riley.python.meshconv")

ElementType = meshconv.EElementType
_ELEMENT_TYPES = {
    "FE/platehole3d_2mr_63f": ElementType.HEX8,
    "cubes/tet4": ElementType.TET4,
    "cubes/tet10": ElementType.TET10,
    "cubes/hex8": ElementType.HEX8,
    "cubes/hex20": ElementType.HEX20,
    "cubes/hex27": ElementType.HEX27,
    "tri3": ElementType.TRI3,
    "tri3opt": ElementType.TRI3,
    "tri6": ElementType.TRI6,
    "quad4": ElementType.QUAD4,
    "quad4ibi": ElementType.QUAD4,
    "quad4newton": ElementType.QUAD4,
    "quad8": ElementType.QUAD8,
    "quad9": ElementType.QUAD9,
}
_SURFACE_ROOTS = {"bench", "calplate", "edge", "min", "simple", "small", "tilt"}
_RABBIT_SOURCES = {"feebs", "rabbit", "riley"}


def audit_mesh(mesh_dir: Path) -> tuple[str, str, str]:
    mesh_name = mesh_dir.relative_to(DATA_DIR).as_posix()
    if mesh_name == "cubes/tet14":
        return (
            "UNSUPPORTED",
            mesh_name,
            "TET14 is unsupported; its fixture remains one-based and node-major.",
        )

    root_name = mesh_name.partition("/")[0]
    mapping_name = mesh_name
    if root_name in _SURFACE_ROOTS:
        mapping_name = mesh_dir.name.split("_", maxsplit=1)[0]
    elif root_name == "rabbits":
        source_name, separator, mapping_name = mesh_dir.name.partition("_")
        if not separator or source_name not in _RABBIT_SOURCES:
            mapping_name = ""

    element_type = _ELEMENT_TYPES.get(mapping_name)
    if element_type is None:
        return (
            "UNKNOWN",
            mesh_name,
            f"No explicit element-type mapping for '{mesh_name}'.",
        )

    try:
        coords = np.loadtxt(
            mesh_dir / "coords.csv",
            delimiter=",",
            dtype=np.float64,
            ndmin=2,
        )
        # Float loading admits integral scientific notation; conversion validates it.
        connect_tables = [
            np.loadtxt(
                mesh_dir / filename,
                delimiter=",",
                dtype=np.float64,
                ndmin=2,
            )
            for filename in CONNECT_FILENAMES
            if (mesh_dir / filename).is_file()
        ]
        if len(connect_tables) == 2 and not np.array_equal(*connect_tables):
            raise ValueError("connect.csv and connectivity.csv differ")

        report = meshconv.check_mesh_convention(
            meshconv.SimData(
                coords=coords,
                blocks={
                    "connect1": meshconv.convert_source_block(
                        connect_tables[0],
                        node_count=coords.shape[0],
                        spec=meshconv.SourceBlockSpec(
                            element_type=element_type,
                            indexing=meshconv.EConnectIndexing.ZERO_BASED,
                            layout=meshconv.EConnectLayout.ROW_MAJOR,
                        ),
                    ),
                },
            ),
        )
    except (OSError, TypeError, ValueError, NotImplementedError) as err:
        return "DOES NOT", mesh_name, str(err)

    if report:
        detail = "; ".join(
            f"{block_name}: {', '.join(check.value for check in failed_checks)}"
            for block_name, failed_checks in sorted(report.items())
        )
        return "DOES NOT", mesh_name, detail
    return "CONFORMS", mesh_name, ""


def main() -> int:
    mesh_dirs = sorted(
        {
            path.parent
            for filename in CONNECT_FILENAMES
            for path in DATA_DIR.rglob(filename)
            if path.parent.joinpath("coords.csv").is_file()
        },
    )
    results = [audit_mesh(mesh_dir) for mesh_dir in mesh_dirs]
    width = max((len(mesh_name) for _, mesh_name, _ in results), default=0)

    print("Riley mesh convention audit\n")
    for status, mesh_name, detail in results:
        print(f"{status:<11} {mesh_name:<{width}}  {detail}")

    counts = Counter(status for status, _, _ in results)
    print(
        f"\nSummary: {counts['CONFORMS']} conform, {counts['DOES NOT']} do not, "
        f"{counts['UNSUPPORTED']} unsupported, {counts['UNKNOWN']} unknown "
        f"({len(results)} meshes total)."
    )
    return int(counts["CONFORMS"] != len(results))


if __name__ == "__main__":
    sys.exit(main())
