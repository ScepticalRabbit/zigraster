# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
"""Public canonical mesh-block and mesh-convention API.

``SimData`` contains explicitly typed ``ElementBlock`` objects. Connectivity in
those blocks is always row-major, zero-based, contiguous ``int64``. Use
:func:`convert_source_block` at source-adapter boundaries when input layout,
indexing, or local-node order differs.
"""

from __future__ import annotations

import numpy as np

from riley.python import _meshconv
from riley.python._meshconv import (
    EConnectLayout,
    EElementType,
    ElementBlock,
    MeshCheckCode,
    MeshConvention,
    MeshConvCheck,
    MeshConvErr,
    SimData,
    SourceBlockSpec,
)
from riley.python.meshio import EConnectIndexing


def convert_source_block(
    connect: np.ndarray,
    node_count: int,
    spec: SourceBlockSpec,
) -> ElementBlock:
    """Convert explicitly described source connectivity to a canonical block.

    No layout, indexing, or topology is inferred. ``target_to_source_perm`` in
    ``spec`` maps each Riley target slot to its source slot.

    Examples
    --------
    >>> source = np.array([[1], [2], [3], [4]], dtype=np.int32)
    >>> spec = SourceBlockSpec(
    ...     element_type=EElementType.QUAD4,
    ...     indexing=EConnectIndexing.ONE_BASED,
    ...     layout=EConnectLayout.NODE_MAJOR,
    ... )
    >>> block = convert_source_block(source, node_count=4, spec=spec)
    >>> block.connect
    array([[0, 1, 2, 3]])
    >>> coords = np.array(
    ...     [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
    ...     dtype=np.float64,
    ... )
    >>> mesh = SimData(coords=coords, blocks={"surface": block})
    >>> mesh.blocks["surface"].element_type is EElementType.QUAD4
    True
    """
    return _meshconv.convert_source_block(connect, node_count, spec)


def check_mesh_convention(
    mesh_in: SimData,
    src_convention: MeshConvention | None = None,
) -> MeshConvCheck:
    """Report geometric convention failures by block.

    Canonical representation errors are rejected by ``ElementBlock``,
    ``SimData``, or :func:`convert_source_block`; this function reports only
    ``NODE_ORDER``, orientation, and surface-topology conditions.
    """
    return _meshconv.check_mesh_convention(mesh_in, src_convention)


def enforce_connectivity(
    coords: np.ndarray,
    connect: np.ndarray,
    element_type: EElementType,
    src_convention: MeshConvention | None = None,
) -> np.ndarray:
    """Return one canonical connectivity table using Riley conventions."""
    mesh = SimData(
        coords=coords,
        blocks={"mesh": ElementBlock(element_type, connect)},
    )
    return _meshconv.enforce_mesh_convention(
        mesh,
        src_convention,
    ).blocks["mesh"].connect


def enforce_mesh_convention(
    mesh_in: SimData,
    src_convention: MeshConvention | None = None,
) -> SimData:
    """Return a mesh using Riley local-node and orientation conventions.

    The original object is returned when it already conforms. A supplied
    ``MeshConvention`` remains keyed by ``EElementType``.
    """
    return _meshconv.enforce_mesh_convention(mesh_in, src_convention)


def infer_mesh_convention(mesh_in: SimData) -> MeshConvention:
    """Infer optional geometric local-slot mappings for canonical typed blocks.

    Layout, index base, and element topology are never inferred.
    """
    return _meshconv.infer_mesh_convention(mesh_in)


def extract_surf_mesh(mesh_in: SimData) -> SimData:
    """Extract typed external faces from a canonical volume mesh.

    Surface input is rejected. Block-local element variables are mapped from
    parent elements, node variables are subset, and side sets are dropped.
    """
    return _meshconv.extract_surf_mesh(mesh_in)


def extract_surf_between(
    mesh_in: SimData,
    point: np.ndarray | list[float] | tuple[float, ...],
    normal: np.ndarray | list[float] | tuple[float, ...],
    distance: float | None = None,
    tolerance: float = 1.0e-6,
) -> SimData:
    """Extract typed surface faces between two parallel planes.

    Surface block types are preserved; volume blocks emit their explicit face
    types. Associated block/node fields are subset and side sets are dropped.
    """
    return _meshconv.extract_surf_between(
        mesh_in,
        point,
        normal,
        distance=distance,
        tolerance=tolerance,
    )


__all__ = [
    "EConnectIndexing",
    "EConnectLayout",
    "EElementType",
    "ElementBlock",
    "MeshCheckCode",
    "MeshConvention",
    "MeshConvCheck",
    "MeshConvErr",
    "SimData",
    "SourceBlockSpec",
    "check_mesh_convention",
    "convert_source_block",
    "enforce_connectivity",
    "enforce_mesh_convention",
    "extract_surf_between",
    "extract_surf_mesh",
    "infer_mesh_convention",
]
