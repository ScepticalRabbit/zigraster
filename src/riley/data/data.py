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

_SHAPE_NAMES = (
    "cube",
    "cylinder",
    "sphere",
    "platewithhole",
    "platehole",
    "platewithhole2d",
    "platehole2d",
    "platewithhole_2d",
)
_SHAPE_ELEM_TYPES = ("hex8", "hex20", "hex27", "tet4", "tet10")
_2D_SHAPE_ELEM_TYPES = ("quad4", "quad8", "quad9", "tri3", "tri6")
_SURF_ELEM_TYPES = (
    "tri3",
    "tri6",
    "quad4",
    "quad8",
    "quad9",
)
_SPHERE200_CASE_NAMES = (
    "tri3_sphere200",
    "tri6_sphere200",
    "quad4_sphere200",
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
    return _resolve_data_path(
        "textures/speckle_mono.bmp", "texture/speckle_mono.bmp"
    )


def cal_target_texture_path() -> Path:
    return _resolve_data_path(
        "textures/cal_target.tiff",
        "texture/cal_target.tiff",
    )


def texture_dir_path() -> Path:
    return _resolve_data_path("textures", "texture")


def _normalize_shape_name(shape: str) -> str:
    shape_norm = shape.lower()
    if shape_norm in ("platehole", "platewithhole"):
        return "platewithhole"
    if shape_norm in ("platehole2d", "platewithhole2d", "platewithhole_2d"):
        return "platewithhole_2d"
    if shape_norm in ("cube", "cylinder", "sphere"):
        return shape_norm
    raise ValueError(
        f"Unsupported shape: {shape!r}. Expected one of "
        "('cube', 'cylinder', 'sphere', 'platewithhole', "
        "'platewithhole_2d')."
    )


def _normalize_shape_and_prefix(
    shape: str,
    elem_type: str,
    pure: bool = False,
) -> tuple[str, str]:
    shape_norm = _normalize_shape_name(shape)

    if shape_norm == "platewithhole_2d":
        elem_norm = (
            elem_type.lower()
            .removeprefix("platewithhole_2d_")
            .removeprefix("platewithhole2d_")
        )
        if elem_norm not in _2D_SHAPE_ELEM_TYPES:
            raise ValueError(
                f"Unsupported 2D element type: {elem_type!r}. "
                f"Expected one of {_2D_SHAPE_ELEM_TYPES}."
            )
        prefix = f"platewithhole2d_{elem_norm}"
        return shape_norm, prefix

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


def shape_dir_path(shape: str, kind: str = "vol") -> Path:
    shape_norm = _normalize_shape_name(shape)
    if shape_norm == "platewithhole_2d":
        candidates = ("shapes/platewithhole_2d", "shapes/platewithhole2d")
    elif kind == "surf":
        candidates = (f"shapes/{shape_norm}_surf",)
    else:
        candidates = (
            f"shapes/{shape_norm}_vol",
            f"shapes/{shape_norm}",
        )

    for rel_path in candidates:
        try:
            return _resolve_data_path(rel_path, f"data/{rel_path}")
        except FileNotFoundError:
            continue

    raise FileNotFoundError(
        f"Shape directory for {shape_norm} ({kind}) could not be found."
    )


def shape_vol_dir_path(shape: str) -> Path:
    return shape_dir_path(shape, kind="vol")


def shape_surf_dir_path(shape: str) -> Path:
    return shape_dir_path(shape, kind="surf")


def shape_case_path(
    shape: str,
    elem_type: str | None = None,
    pure: bool = False,
) -> Path:
    if elem_type is not None:
        _normalize_shape_and_prefix(shape, elem_type, pure=pure)
    return shape_vol_dir_path(shape)


def _find_shape_file(
    shape_norm: str,
    filename: str,
    kind: str = "vol",
) -> Path:
    if shape_norm == "platewithhole_2d":
        candidates = (
            "shapes/platewithhole_2d",
            "shapes/platewithhole2d",
        )
    elif kind == "surf":
        candidates = (f"shapes/{shape_norm}_surf",)
    else:
        candidates = (
            f"shapes/{shape_norm}_vol",
            f"shapes/{shape_norm}",
        )
    for rel_path in candidates:
        pkg_file = _package_data_root_path() / rel_path / filename
        if pkg_file.is_file():
            return pkg_file
        repo_file = _fallback_repo_data_path(f"data/{rel_path}/{filename}")
        if repo_file.is_file():
            return repo_file
    raise FileNotFoundError(
        f"Shape file '{filename}' for shape '{shape_norm}' ({kind}) not found."
    )


def shape_exodus_path(
    shape: str,
    elem_type: str,
    pure: bool = False,
) -> Path:
    shape_norm, prefix = _normalize_shape_and_prefix(
        shape, elem_type, pure=pure
    )
    try:
        return _find_shape_file(shape_norm, f"{prefix}_out.e")
    except FileNotFoundError:
        return _find_shape_file(shape_norm, f"{prefix}.e")


def shape_coords_path(
    shape: str,
    elem_type: str,
    pure: bool = False,
) -> Path:
    shape_norm, prefix = _normalize_shape_and_prefix(
        shape, elem_type, pure=pure
    )
    return _find_shape_file(shape_norm, f"{prefix}_coords.csv")


def shape_connectivity_path(
    shape: str,
    elem_type: str,
    pure: bool = False,
    block: str | None = None,
) -> Path:
    shape_norm, prefix = _normalize_shape_and_prefix(
        shape, elem_type, pure=pure
    )
    if block is not None:
        try:
            return _find_shape_file(
                shape_norm, f"{prefix}_{block}_connectivity.csv"
            )
        except FileNotFoundError:
            pass
    return _find_shape_file(shape_norm, f"{prefix}_connectivity.csv")


def shape_field_path(
    shape: str,
    elem_type: str,
    field_name: str,
    pure: bool = False,
) -> Path:
    shape_norm, prefix = _normalize_shape_and_prefix(
        shape, elem_type, pure=pure
    )
    return _find_shape_file(shape_norm, f"{prefix}_{field_name}.csv")


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
    return _find_shape_file(shape_norm, f"{prefix}.msh")


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
    return _find_shape_file(shape_norm, f"{prefix}.geo")


def shape_moose_input_path(
    shape: str,
    elem_type: str,
    pure: bool = False,
) -> Path:
    shape_norm, prefix = _normalize_shape_and_prefix(
        shape, elem_type, pure=pure
    )
    return _find_shape_file(shape_norm, f"{prefix}.i")


def shape_surface_dataset_path(shape: str, surf_type: str) -> Path:
    surf_dir = shape_surf_dir_path(shape)
    type_norm = surf_type.lower()
    sub_dir = surf_dir / type_norm
    if sub_dir.is_dir():
        return sub_dir
    raise FileNotFoundError(
        f"Surface dataset directory not found for {shape}/{surf_type} "
        f"at {sub_dir}"
    )


# --- Cube helpers ---
def cube_case_path(
    case_name: str | None = None,
    pure: bool = False,
) -> Path:
    if case_name is not None:
        _normalize_shape_and_prefix("cube", case_name, pure=pure)
    return shape_vol_dir_path("cube")


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


# --- Cylinder helpers ---
def cylinder_case_path() -> Path:
    return shape_vol_dir_path("cylinder")


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


# --- Sphere helpers ---
def sphere_case_path() -> Path:
    return shape_vol_dir_path("sphere")


def sphere_exodus_path(elem_type: str) -> Path:
    return shape_exodus_path("sphere", elem_type)


def sphere_coords_path(elem_type: str) -> Path:
    return shape_coords_path("sphere", elem_type)


def sphere_connectivity_path(elem_type: str) -> Path:
    return shape_connectivity_path("sphere", elem_type)


def sphere_disp_path(elem_type: str, component: str = "x") -> Path:
    return shape_disp_path("sphere", elem_type, component)


def sphere_temperature_path(elem_type: str) -> Path:
    return shape_temperature_path("sphere", elem_type)


def sphere_msh_path(elem_type: str) -> Path:
    return shape_msh_path("sphere", elem_type, pure=False)


def sphere_geo_path(elem_type: str) -> Path:
    return shape_geo_path("sphere", elem_type, pure=False)


def sphere_moose_input_path(elem_type: str) -> Path:
    return shape_moose_input_path("sphere", elem_type, pure=False)


# --- Plate with hole helpers ---
def platewithhole_case_path() -> Path:
    return shape_vol_dir_path("platewithhole")


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


# --- 2D Plate with hole helpers ---
def platewithhole2d_case_path() -> Path:
    return shape_vol_dir_path("platewithhole_2d")


def platewithhole2d_exodus_path(elem_type: str) -> Path:
    return shape_exodus_path("platewithhole_2d", elem_type)


def platewithhole2d_coords_path(elem_type: str) -> Path:
    return shape_coords_path("platewithhole_2d", elem_type)


def platewithhole2d_connectivity_path(elem_type: str) -> Path:
    return shape_connectivity_path("platewithhole_2d", elem_type)


def platewithhole2d_disp_path(
    elem_type: str, component: str = "x"
) -> Path:
    return shape_disp_path("platewithhole_2d", elem_type, component)


def platewithhole2d_temperature_path(elem_type: str) -> Path:
    return shape_temperature_path("platewithhole_2d", elem_type)


def platewithhole2d_msh_path(elem_type: str) -> Path:
    return shape_msh_path("platewithhole_2d", elem_type, pure=False)


def platewithhole2d_geo_path(elem_type: str) -> Path:
    return shape_geo_path("platewithhole_2d", elem_type, pure=False)


def platewithhole2d_moose_input_path(elem_type: str) -> Path:
    return shape_moose_input_path("platewithhole_2d", elem_type, pure=False)


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
    "platewithhole2d_case_path",
    "platewithhole2d_connectivity_path",
    "platewithhole2d_coords_path",
    "platewithhole2d_disp_path",
    "platewithhole2d_exodus_path",
    "platewithhole2d_geo_path",
    "platewithhole2d_moose_input_path",
    "platewithhole2d_msh_path",
    "platewithhole2d_temperature_path",
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
    "shape_surf_dir_path",
    "shape_surface_dataset_path",
    "shape_temperature_path",
    "shape_vol_dir_path",
    "speckle_texture_path",
    "sphere200_case_path",
    "sphere_case_path",
    "sphere_connectivity_path",
    "sphere_coords_path",
    "sphere_disp_path",
    "sphere_exodus_path",
    "sphere_geo_path",
    "sphere_moose_input_path",
    "sphere_msh_path",
    "sphere_temperature_path",
    "stereocal_case_path",
    "texture_dir_path",
]
