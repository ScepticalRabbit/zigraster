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


def sphere200_case_path() -> Path:
    return _resolve_data_path(
        "min/tri6_sphere200",
        "data/min/tri6_sphere200",
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
    "platehole_csv_case_path",
    "platehole_exodus_path",
    "rabbit_case_path",
    "rabbits_root_path",
    "speckle_texture_path",
    "sphere200_case_path",
    "stereocal_case_path",
]
