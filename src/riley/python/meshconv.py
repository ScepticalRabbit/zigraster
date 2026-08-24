# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
"""Riley's public mesh-convention interface.

:func:`check_mesh_convention` reports the failed convention checks for a
mesh, :func:`enforce_mesh_convention` normalises a mesh to Riley's
convention, and :func:`extract_surf_mesh`/:func:`extract_surf_between`
extract surface meshes. All implementation details live in the private
:mod:`riley.python._meshconv` module.
"""

from __future__ import annotations

import numpy as np

from riley.python import _meshconv
from riley.python._meshconv import (
    MeshCheckCode,
    EElementType,
    EMeshType,
    MeshConvention,
    MeshConvCheck,
    SimData,
)


def check_mesh_convention(
    mesh_in: SimData,
    source_convention: MeshConvention | None = None,
) -> MeshConvCheck:
    """Return failed checks for each non-conforming connectivity table.

    An empty dictionary means the mesh conforms.  This mapping is the only
    convention-check result: it preserves every failure without duplicating
    aggregate flags or summaries.
    """
    return _meshconv.check_mesh_convention(mesh_in, source_convention)


def enforce_mesh_convention(
    mesh_in: SimData,
    source_convention: MeshConvention | None = None,
) -> SimData:
    """Normalises a mesh to the mesh convention:
    - 0-based indexing
    - CCW node ordering when viewed from the outward/visible side
    - Right-handed geometry conventions
    - Check that all indices in the connectivity table map to a row in coords
    """
    return _meshconv.enforce_mesh_convention(mesh_in, source_convention)


def extract_surf_mesh(
    mesh_in: SimData,
    enforce_convention: bool = True,
) -> SimData:
    """Extracts the external surface mesh from supported 3D volume elements."""
    return _meshconv.extract_surf_mesh(mesh_in, enforce_convention)


def extract_surf_between(
    mesh_in: SimData,
    point: np.ndarray | list[float] | tuple[float, ...],
    normal: np.ndarray | list[float] | tuple[float, ...],
    distance: float | None = None,
    tolerance: float = 1.0e-6,
    enforce_convention: bool = True,
) -> SimData:
    """Extracts a surface mesh between two planes defined by point, normal,
    and distance.

    Parameters
    ----------
    mesh_in : SimData
        The input simulation data/mesh.
    point : np.ndarray | list[float] | tuple[float, ...]
        A point on the first plane.
    normal : np.ndarray | list[float] | tuple[float, ...]
        The normal vector of the planes.
    distance : float | None, optional
        The distance along the normal to the second plane. If None, the
        surface is extracted at +/- tolerance about the first plane.
    tolerance : float, optional
        Numerical tolerance for checking if nodes lie between the planes.
        Defaults to 1.0e-6.
    enforce_convention : bool, optional
        If True, normalizes the output mesh to the Riley convention.
        Defaults to True.

    Returns
    -------
    SimData
        The extracted surface mesh.
    """
    return _meshconv.extract_surf_between(
        mesh_in,
        point,
        normal,
        distance=distance,
        tolerance=tolerance,
        enforce_convention=enforce_convention,
    )


__all__ = [
    "MeshCheckCode",
    "MeshConvCheck",
    "EMeshType",
    "EElementType",
    "MeshConvention",
    "SimData",
    "check_mesh_convention",
    "enforce_mesh_convention",
    "extract_surf_mesh",
    "extract_surf_between",
]
