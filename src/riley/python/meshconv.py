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
convention, :func:`infer_mesh_convention` diagnoses source ordering, and
:func:`extract_surf_mesh`/:func:`extract_surf_between` extract surface
meshes. All implementation details live in the private
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
    MeshConventionInferenceError,
    MeshConvCheck,
    SimData,
)


def check_mesh_convention(
    mesh_in: SimData,
    src_convention: MeshConvention | None = None,
) -> MeshConvCheck:
    """Check a mesh for conformance to Riley's mesh convention.

    This function inspects all connectivity tables in the mesh and reports
    which convention checks fail. An empty dictionary means the mesh fully
    conforms to Riley's convention.

    Parameters
    ----------
    mesh_in : SimData
        The mesh to check. Must have ``coords`` (N x 3 array) and ``connect``
        (dict of connectivity tables). Each connectivity table must be
        2D with one element per row and one local node per column.
    src_convention : MeshConvention | None, optional
        Source ordering for specific element types. Each permutation satisfies
        ``riley_row[target_slot] = source_row[permutation[target_slot]]``.
        Omitted element types are assumed to use Riley ordering. If None, all
        connectivity is assumed to use Riley ordering.

    Returns
    -------
    MeshConvCheck
        Dictionary mapping connectivity table names to lists of failed
        ``MeshCheckCode`` conditions. Empty dict means the mesh conforms.
        Possible failure codes:

        - ``ROW_MAJOR_CONNECTIVITY``: table is column-major, needs transpose
        - ``ZERO_BASED_INDEXING``: indices are 1-based, need 0-based shift
        - ``CONNECTIVITY_INDICES``: indices outside valid range [0, N-1]
        - ``CCW_WINDING``: surface elements not wound counter-clockwise
        - ``RIGHT_HANDED_GEOMETRY``: volume elements not right-handed
        - ``SURFACE_TOPOLOGY``: non-manifold or unorientable surface
        - ``NODE_ORDER``: higher-order nodes in wrong slots for element type

    Examples
    --------
    >>> from riley.python import meshconv
    >>> import numpy as np
    >>> coords = np.array(
    ...     [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64
    ... )
    >>> mesh = meshconv.SimData(
    ...     coords=coords, connect={"connect1": np.array([[0, 1, 2, 3]])}
    ... )
    >>> report = meshconv.check_mesh_convention(mesh)
    >>> report == {}
    True
    """
    return _meshconv.check_mesh_convention(mesh_in, src_convention)


def enforce_mesh_convention(
    mesh_in: SimData,
    src_convention: MeshConvention | None = None,
) -> SimData:
    """Normalise a mesh to Riley's mesh convention.

    Applies all necessary corrections to bring a mesh into conformance with
    Riley's convention. Only fixes the conditions reported by
    :func:`check_mesh_convention`, and applies them in a fixed order
    so the result is deterministic.

    Parameters
    ----------
    mesh_in : SimData
        The mesh to normalise. Must have ``coords`` and ``connect`` set.
    src_convention : MeshConvention | None, optional
        Source ordering for specific element types. Each permutation satisfies
        ``riley_row[target_slot] = source_row[permutation[target_slot]]``.
        Omitted element types are assumed to use Riley ordering. If None, all
        connectivity is assumed to use Riley ordering.

    Returns
    -------
    SimData
        A new ``SimData`` instance with normalised connectivity tables, or the
        original mesh instance if it already conformed (idempotent).

    Notes
    -----
    The normalisation enforces:

    - **Row-major connectivity**: each element is one row,
      local nodes are columns
    - **Zero-based indexing**: all indices in range [0, N-1] where N = num nodes
    - **Node order**: higher-order nodes (mid-edge, mid-face, centre) placed in
      Riley's standard slots per element type
    - **Surface winding**: CCW when viewed from material-facing side (outward
      for closed shells, inward for cavity boundaries)
    - **Volume handedness**: right-handed coordinate system for TET/HEX elements
    - **Index validity**: all indices reference existing coordinate rows

    If ``CONNECTIVITY_INDICES`` would be violated (index out of bounds after
    normalization), a ``ValueError`` is raised instead of silently corrupting
    the mesh.

    Examples
    --------
    >>> from riley.python import meshconv
    >>> import numpy as np
    >>> coords = np.array(
    ...     [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64
    ... )
    >>> # 1-based indexing, needs conversion
    >>> mesh = meshconv.SimData(
    ...     coords=coords, connect={"connect1": np.array([[1, 2, 3, 4]])}
    ... )
    >>> mesh_out = meshconv.enforce_mesh_convention(mesh)
    >>> mesh_out.connect["connect1"]
    array([[0, 1, 2, 3]])
    >>> meshconv.check_mesh_convention(mesh_out) == {}
    True
    """
    return _meshconv.enforce_mesh_convention(mesh_in, src_convention)


def infer_mesh_convention(mesh_in: SimData) -> MeshConvention:
    """Infer a source convention from affine element geometry.

    Inference is intentionally conservative and rejects ambiguous or mixed
    source layouts. Prefer an explicitly declared :class:`MeshConvention`
    whenever the source format is known.

    Parameters
    ----------
    mesh_in : SimData
        Mesh whose source connectivity order should be inferred.

    Returns
    -------
    MeshConvention
        Inferred source-slot mapping for each element family in the mesh.

    Raises
    ------
    MeshConventionInferenceError
        If the source node roles cannot be inferred unambiguously.
    """
    return _meshconv.infer_mesh_convention(mesh_in)


def extract_surf_mesh(
    mesh_in: SimData,
    enforce_convention: bool = True,
) -> SimData:
    """Extract the external surface mesh from a 3D volume mesh.

    For volume element types (TET4, TET10, HEX8, HEX20, HEX27), this function
    identifies boundary faces (those belonging to only one element) and returns
    them as a surface mesh with proper outward-facing normals.

    Parameters
    ----------
    mesh_in : SimData
        The input 3D volume mesh. Must have ``coords`` and ``connect`` with
        volume element types.
    enforce_convention : bool, optional
        If True (default), the output surface mesh is normalised to Riley's
        convention (0-based, CCW winding, proper node order). If False, the
        output retains the input mesh's indexing style (1-based vs 0-based,
        row-major vs column-major).

    Returns
    -------
    SimData
        A surface mesh containing only the boundary faces. The returned mesh
        has:

        - ``coords``: subset of input coordinates used by boundary faces
        - ``connect``: boundary face connectivity (QUAD4/8/9 or TRI3/6/7)
        - ``mesh_type``: ``EMeshType.SURF``
        - ``node_vars``: interpolated nodal variables for surface nodes
        - ``elem_vars``: element variables for boundary faces

    Raises
    ------
    ValueError
        If the input mesh is 2D (surface elements only) or has no connectivity.

    Examples
    --------
    >>> from riley.python import meshconv
    >>> import numpy as np
    >>> coords = np.array((
    ...     (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
    ...     (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    ...     (0.0, 0.0, 1.0), (1.0, 0.0, 1.0),
    ...     (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
    ... ))
    >>> connect = np.arange(8, dtype=np.int64).reshape(1, 8)
    >>> mesh = meshconv.SimData(coords=coords, connect={"connect1": connect})
    >>> surf = meshconv.extract_surf_mesh(mesh)
    >>> surf.mesh_type is meshconv.EMeshType.SURF
    True
    >>> list(surf.connect.values())[0].shape  # (faces, nodes per face)
    (6, 4)
    """
    return _meshconv.extract_surf_mesh(mesh_in, enforce_convention)


def extract_surf_between(
    mesh_in: SimData,
    point: np.ndarray | list[float] | tuple[float, ...],
    normal: np.ndarray | list[float] | tuple[float, ...],
    distance: float | None = None,
    tolerance: float = 1.0e-6,
    enforce_convention: bool = True,
) -> SimData:
    """Extract a surface mesh from a slice between two parallel planes.

    For volume meshes, extracts internal faces between elements where nodes
    lie between the two planes. For surface meshes, extracts faces whose
    nodes lie between the planes. The output is a surface mesh with proper
    outward-facing normals.

    Parameters
    ----------
    mesh_in : SimData
        The input simulation data/mesh. Can be volume or surface elements.
    point : np.ndarray | list[float] | tuple[float, ...]
        A point on the first plane. Only the first 3 components are used.
    normal : np.ndarray | list[float] | tuple[float, ...]
        The normal vector defining the plane orientation. Must be non-zero.
        Only the first 3 components are used.
    distance : float | None, optional
        Distance along the normal to the second plane. If None (default),
        extracts a thin slice at the first plane with ``+/- tolerance``
        thickness. If provided, extracts the slab between the two planes.
    tolerance : float, optional
        Numerical tolerance for checking if nodes lie between the planes.
        Defaults to 1.0e-6. Nodes with projected distance in
        ``[min_bound - tol, max_bound + tol]`` are included.
    enforce_convention : bool, optional
        If True (default), normalises the output mesh to Riley's convention.
        If False, retains the input mesh's indexing style.

    Returns
    -------
    SimData
        The extracted surface mesh containing faces/elements between the
        planes. The mesh has ``mesh_type = EMeshType.SURF`` and conforming
        connectivity (if ``enforce_convention=True``).

    Raises
    ------
    ValueError
        If no elements/faces are found between the planes, if normal is zero,
        or if mesh lacks required coordinates/connectivity.

    Examples
    --------
    >>> from riley.python import meshconv
    >>> import numpy as np
    >>> coords = np.array((
    ...     (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
    ...     (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    ...     (0.0, 0.0, 1.0), (1.0, 0.0, 1.0),
    ...     (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
    ... ))
    >>> connect = np.arange(8, dtype=np.int64).reshape(1, 8)
    >>> mesh = meshconv.SimData(coords=coords, connect={"connect1": connect})
    >>> slab = meshconv.extract_surf_between(
    ...     mesh, point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)
    ... )
    >>> slab.connect["connect1"].shape
    (1, 4)
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
    "MeshConventionInferenceError",
    "SimData",
    "check_mesh_convention",
    "enforce_mesh_convention",
    "extract_surf_mesh",
    "infer_mesh_convention",
    "extract_surf_between",
]
