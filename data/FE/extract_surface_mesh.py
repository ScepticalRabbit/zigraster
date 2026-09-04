"""Bridge MooseHerder simulation data to Riley's mesh tools."""

from __future__ import annotations

import warnings

import numpy as np
import pyvale.mooseherder as mh

from riley.python import meshconv


def extract_surf_mesh(
    sim_data: mh.SimData,
    convention: meshconv.ConnectConvention,
) -> mh.SimData:
    """Convert one volume connectivity table and extract its Riley surface.

    Parameters
    ----------
    sim_data
        MooseHerder data containing exactly one volume connectivity table.
    convention
        Complete source convention for that volume table.

    Returns
    -------
    mh.SimData
        Surface data with column-major, one-based connectivity as expected by
        MooseHerder. Its connectivity is already in Riley local-node order.
    """
    if sim_data.coords is None:
        raise ValueError("Simulation data does not contain coordinates.")
    if sim_data.connect is None or len(sim_data.connect) != 1:
        raise ValueError(
            "Surface extraction requires exactly one connectivity table."
        )

    source_connect = np.asarray(next(iter(sim_data.connect.values())))
    volume = meshconv.convert_mesh(
        sim_data.coords,
        source_connect,
        convention,
    )
    meshconv.verify_mesh(volume)
    surface = meshconv.extract_surface(volume)
    meshconv.verify_mesh(surface)
    source_node_idxs = _find_source_node_idxs(volume.coords, surface.coords)

    face_data = mh.SimData(
        coords=surface.coords,
        connect={"connect1": surface.connect.T + 1},
        time=sim_data.time,
        num_spat_dims=sim_data.num_spat_dims,
        glob_vars=sim_data.glob_vars,
    )
    if sim_data.node_vars is not None:
        face_data.node_vars = {
            name: np.asarray(values)[source_node_idxs]
            for name, values in sim_data.node_vars.items()
        }
    if sim_data.elem_vars:
        warnings.warn(
            "Element variables are not supported by Riley surface meshes "
            "and were removed.",
            UserWarning,
            stacklevel=2,
        )
    return face_data


def _find_source_node_idxs(
    source_coords: np.ndarray,
    surface_coords: np.ndarray,
) -> np.ndarray:
    """Find source indices for the unchanged coordinates on a surface."""
    coord_to_idx: dict[tuple[float, ...], int] = {}
    for node_idx, coord in enumerate(source_coords):
        key = tuple(float(value) for value in coord)
        if key in coord_to_idx:
            raise ValueError(
                "Source coordinates contain duplicate node locations; field "
                "data cannot be mapped unambiguously."
            )
        coord_to_idx[key] = node_idx

    source_node_idxs = []
    for coord in surface_coords:
        key = tuple(float(value) for value in coord)
        if key not in coord_to_idx:
            raise ValueError("A surface node was not found in the source mesh.")
        source_node_idxs.append(coord_to_idx[key])
    return np.asarray(source_node_idxs, dtype=np.int64)
