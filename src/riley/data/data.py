# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
from __future__ import annotations

from importlib.resources import files
from pathlib import Path


_SHAPE_NAMES = ("cube", "cylinder", "platewithhole", "platehole")
_SHAPE_ELEM_TYPES = ("hex8", "hex20", "hex27", "tet4", "tet10")
_SPHERE200_CASE_NAMES = (
    "tri3_sphere200",
    "tri6_sphere200",
    "quad4newton_sphere200",
    "quad8_sphere200",
    "quad9_sphere200",
)


def _package_data_root_path() -> Path:
    return Path(str(files("riley.data")))


def _repo_root_path() -> Path:
    return Path(__file__).resolve().parents[3]


def _fallback_repo_data_path(rel_path: str) -> Path:
    repo_root = _repo_root_path()
    candidate = repo_root / rel_path
    return candidate


def _resolve_data_path(
    package_rel_path: str,
    fallback_repo_rel_path: str,
) -> Path:
    package_path = _package_data_root_path() / package_rel_path
    if package_path.exists():
        return package_path

    fallback_path = _fallback_repo_data_path(fallback_repo_rel_path)
    if fallback_path.exists():
        return fallback_path

    raise FileNotFoundError(
        "Riley packaged data could not be found at either "
        f"{package_path} or {fallback_path}.",
    )


def speckle_texture_path() -> Path:
    return _resolve_data_path("textures/speckle.bmp", "texture/speckle.bmp")


def cal_target_texture_path() -> Path:
    return _resolve_data_path(
        "textures/cal_target-simple.tiff",
        "texture/cal_target-simple.tiff",
    )


def _normalize_shape_and_prefix(
    shape: str,
    elem_type: str,
    pure: bool = False,
) -> tuple[str, str]:
    shape_norm = shape.lower()
    if shape_norm == "platehole":
        shape_norm = "platewithhole"

    if shape_norm not in ("cube", "cylinder", "platewithhole"):
        raise ValueError(
            f"Unsupported shape: {shape!r}. Expected one of "
            "('cube', 'cylinder', 'platewithhole')."
        )

    elem_norm = elem_type.lower()
    if elem_norm.startswith("pure_"):
        pure = True
        elem_norm = elem_norm[5:]

    if elem_norm not in _SHAPE_ELEM_TYPES:
        raise ValueError(
            f"Unsupported element type: {elem_type!r}. "
            f"Expected one of {_SHAPE_ELEM_TYPES}."
        )

    if pure and shape_norm != "cube":
        raise ValueError("Pure MOOSE mesh is only supported for 'cube'.")

    prefix = (
        f"{shape_norm}_pure_{elem_norm}"
        if pure
        else f"{shape_norm}_{elem_norm}"
    )
    return shape_norm, prefix


def shape_dir_path(shape: str) -> Path:
    shape_norm = shape.lower()
    if shape_norm == "platehole":
        shape_norm = "platewithhole"

    if shape_norm not in ("cube", "cylinder", "platewithhole"):
        raise ValueError(
            f"Unsupported shape: {shape!r}. Expected one of "
            "('cube', 'cylinder', 'platewithhole')."
        )

    rel_path = f"shapes/{shape_norm}"
    return _resolve_data_path(rel_path, f"data/{rel_path}")


def shape_case_path(
    shape: str,
    elem_type: str | None = None,
    pure: bool = False,
) -> Path:
    if elem_type is not None:
        _normalize_shape_and_prefix(shape, elem_type, pure=pure)
    return shape_dir_path(shape)


def shape_exodus_path(
    shape: str,
    elem_type: str,
    pure: bool = False,
) -> Path:
    shape_norm, prefix = _normalize_shape_and_prefix(
        shape, elem_type, pure=pure
    )
    dir_path = shape_dir_path(shape_norm)
    candidate = dir_path / f"{prefix}_out.e"
    if candidate.is_file():
        return candidate

    alt_cand = dir_path / f"{prefix}.e"
    if alt_cand.is_file():
        return alt_cand

    raise FileNotFoundError(f"Exodus file not found: {candidate}")


def shape_coords_path(
    shape: str,
    elem_type: str,
    pure: bool = False,
) -> Path:
    shape_norm, prefix = _normalize_shape_and_prefix(
        shape, elem_type, pure=pure
    )
    dir_path = shape_dir_path(shape_norm)
    candidate = dir_path / f"{prefix}_coords.csv"
    if candidate.is_file():
        return candidate

    raise FileNotFoundError(f"Coords file not found: {candidate}")


def shape_connectivity_path(
    shape: str,
    elem_type: str,
    pure: bool = False,
) -> Path:
    shape_norm, prefix = _normalize_shape_and_prefix(
        shape, elem_type, pure=pure
    )
    dir_path = shape_dir_path(shape_norm)
    candidate = dir_path / f"{prefix}_connectivity.csv"
    if candidate.is_file():
        return candidate

    raise FileNotFoundError(f"Connectivity file not found: {candidate}")


def shape_field_path(
    shape: str,
    elem_type: str,
    field_name: str,
    pure: bool = False,
) -> Path:
    shape_norm, prefix = _normalize_shape_and_prefix(
        shape, elem_type, pure=pure
    )
    dir_path = shape_dir_path(shape_norm)
    candidate = dir_path / f"{prefix}_{field_name}.csv"
    if candidate.is_file():
        return candidate

    raise FileNotFoundError(f"Field file not found: {candidate}")


def shape_disp_path(
    shape: str,
    elem_type: str,
    component: str = "x",
    pure: bool = False,
) -> Path:
    comp_clean = component.lower().removeprefix("disp_")
    if comp_clean not in ("x", "y", "z"):
        raise ValueError(
            f"Invalid displacement component: {component!r}. "
            "Expected 'x', 'y', 'z', 'disp_x', 'disp_y', or 'disp_z'."
        )
    return shape_field_path(
        shape, elem_type, f"disp_{comp_clean}", pure=pure
    )


def shape_temperature_path(
    shape: str,
    elem_type: str,
    pure: bool = False,
) -> Path:
    return shape_field_path(shape, elem_type, "temperature", pure=pure)


def shape_msh_path(
    shape: str,
    elem_type: str,
    pure: bool = False,
) -> Path:
    if pure:
        raise ValueError("Pure MOOSE cases do not use a Gmsh .msh file.")
    shape_norm, prefix = _normalize_shape_and_prefix(
        shape, elem_type, pure=False
    )
    dir_path = shape_dir_path(shape_norm)
    candidate = dir_path / f"{prefix}.msh"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Mesh file not found: {candidate}")


def shape_geo_path(
    shape: str,
    elem_type: str,
    pure: bool = False,
) -> Path:
    if pure:
        raise ValueError("Pure MOOSE cases do not use a Gmsh .geo file.")
    shape_norm, prefix = _normalize_shape_and_prefix(
        shape, elem_type, pure=False
    )
    dir_path = shape_dir_path(shape_norm)
    candidate = dir_path / f"{prefix}.geo"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Geo file not found: {candidate}")


def shape_moose_input_path(
    shape: str,
    elem_type: str,
    pure: bool = False,
) -> Path:
    shape_norm, prefix = _normalize_shape_and_prefix(
        shape, elem_type, pure=pure
    )
    dir_path = shape_dir_path(shape_norm)
    candidate = dir_path / f"{prefix}.i"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"MOOSE input file not found: {candidate}")


def cube_case_path(
    case_name: str | None = None,
    pure: bool = False,
) -> Path:
    if case_name is not None:
        _normalize_shape_and_prefix("cube", case_name, pure=pure)
    return shape_dir_path("cube")


def cube_exodus_path(case_name: str, pure: bool = False) -> Path:
    return shape_exodus_path("cube", case_name, pure=pure)


def cube_coords_path(case_name: str, pure: bool = False) -> Path:
    return shape_coords_path("cube", case_name, pure=pure)


def cube_connectivity_path(case_name: str, pure: bool = False) -> Path:
    return shape_connectivity_path("cube", case_name, pure=pure)


def cube_disp_path(
    case_name: str,
    component: str = "x",
    pure: bool = False,
) -> Path:
    return shape_disp_path("cube", case_name, component, pure=pure)


def cube_temperature_path(case_name: str, pure: bool = False) -> Path:
    return shape_temperature_path("cube", case_name, pure=pure)


def cube_msh_path(case_name: str) -> Path:
    return shape_msh_path("cube", case_name, pure=False)


def cube_geo_path(case_name: str) -> Path:
    return shape_geo_path("cube", case_name, pure=False)


def cube_moose_input_path(case_name: str, pure: bool = False) -> Path:
    return shape_moose_input_path("cube", case_name, pure=pure)


def cylinder_case_path() -> Path:
    return shape_dir_path("cylinder")


def cylinder_exodus_path(elem_type: str) -> Path:
    return shape_exodus_path("cylinder", elem_type)


def cylinder_coords_path(elem_type: str) -> Path:
    return shape_coords_path("cylinder", elem_type)


def cylinder_connectivity_path(elem_type: str) -> Path:
    return shape_connectivity_path("cylinder", elem_type)


def cylinder_disp_path(elem_type: str, component: str = "x") -> Path:
    return shape_disp_path("cylinder", elem_type, component)


def cylinder_temperature_path(elem_type: str) -> Path:
    return shape_temperature_path("cylinder", elem_type)


def cylinder_msh_path(elem_type: str) -> Path:
    return shape_msh_path("cylinder", elem_type, pure=False)


def cylinder_geo_path(elem_type: str) -> Path:
    return shape_geo_path("cylinder", elem_type, pure=False)


def cylinder_moose_input_path(elem_type: str) -> Path:
    return shape_moose_input_path("cylinder", elem_type, pure=False)


def platewithhole_case_path() -> Path:
    return shape_dir_path("platewithhole")


def platewithhole_exodus_path(elem_type: str) -> Path:
    return shape_exodus_path("platewithhole", elem_type)


def platewithhole_coords_path(elem_type: str) -> Path:
    return shape_coords_path("platewithhole", elem_type)


def platewithhole_connectivity_path(elem_type: str) -> Path:
    return shape_connectivity_path("platewithhole", elem_type)


def platewithhole_disp_path(elem_type: str, component: str = "x") -> Path:
    return shape_disp_path("platewithhole", elem_type, component)


def platewithhole_temperature_path(elem_type: str) -> Path:
    return shape_temperature_path("platewithhole", elem_type)


def platewithhole_msh_path(elem_type: str) -> Path:
    return shape_msh_path("platewithhole", elem_type, pure=False)


def platewithhole_geo_path(elem_type: str) -> Path:
    return shape_geo_path("platewithhole", elem_type, pure=False)


def platewithhole_moose_input_path(elem_type: str) -> Path:
    return shape_moose_input_path("platewithhole", elem_type, pure=False)


def sphere200_case_path(case_name: str = "tri6_sphere200") -> Path:
    if case_name not in _SPHERE200_CASE_NAMES:
        raise ValueError(
            f"Unsupported sphere200 data case: {case_name!r}. "
            f"Expected one of {_SPHERE200_CASE_NAMES}.",
        )
    return _resolve_data_path(
        f"min/{case_name}",
        f"data/min/{case_name}",
    )


def platehole_csv_case_path() -> Path:
    return _resolve_data_path(
        "fe/platehole3d_2mr_63f",
        "data/FE/platehole3d_2mr_63f",
    )


def platehole_exodus_path() -> Path:
    return _resolve_data_path(
        "fe/platehole3d_2mr_63f.e",
        "data/FE/platehole3d_2mr_63f.e",
    )


def stereocal_case_path() -> Path:
    return _resolve_data_path(
        "calplate/tri3_calplate3d",
        "data/calplate/tri3_calplate3d",
    )


def rabbits_root_path() -> Path:
    return _resolve_data_path("rabbits", "data/rabbits")


def rabbit_case_path(
    rabbit_name: str,
    mesh_name: str,
) -> Path:
    case_dir = rabbits_root_path() / f"{rabbit_name}_{mesh_name}"
    if not case_dir.is_dir():
        raise FileNotFoundError(
            f"Packaged rabbit case does not exist: {case_dir}",
        )
    return case_dir


__all__ = [
    "cal_target_texture_path",
    "cube_case_path",
    "cube_connectivity_path",
    "cube_coords_path",
    "cube_disp_path",
    "cube_exodus_path",
    "cube_geo_path",
    "cube_moose_input_path",
    "cube_msh_path",
    "cube_temperature_path",
    "cylinder_case_path",
    "cylinder_connectivity_path",
    "cylinder_coords_path",
    "cylinder_disp_path",
    "cylinder_exodus_path",
    "cylinder_geo_path",
    "cylinder_moose_input_path",
    "cylinder_msh_path",
    "cylinder_temperature_path",
    "platehole_csv_case_path",
    "platehole_exodus_path",
    "platewithhole_case_path",
    "platewithhole_connectivity_path",
    "platewithhole_coords_path",
    "platewithhole_disp_path",
    "platewithhole_exodus_path",
    "platewithhole_geo_path",
    "platewithhole_moose_input_path",
    "platewithhole_msh_path",
    "platewithhole_temperature_path",
    "rabbit_case_path",
    "rabbits_root_path",
    "shape_case_path",
    "shape_connectivity_path",
    "shape_coords_path",
    "shape_dir_path",
    "shape_disp_path",
    "shape_exodus_path",
    "shape_field_path",
    "shape_geo_path",
    "shape_moose_input_path",
    "shape_msh_path",
    "shape_temperature_path",
    "speckle_texture_path",
    "sphere200_case_path",
    "stereocal_case_path",
]
