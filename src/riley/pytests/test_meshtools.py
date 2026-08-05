# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
from __future__ import annotations

import numpy as np

import riley


def test_enforce_mesh_convention_flips_clockwise_tri3() -> None:
    coords = np.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    connect = np.array(((0, 2, 1),), dtype=np.uintp)

    _, connect_fixed = riley.enforce_mesh_convention(coords, connect)

    np.testing.assert_array_equal(
        connect_fixed,
        np.array(((0, 1, 2),), dtype=np.uintp),
    )


def test_extract_surface_mesh_hex8_cube() -> None:
    coords = np.array(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
        ),
        dtype=np.float64,
    )
    connect = np.array(((0, 1, 2, 3, 4, 5, 6, 7),), dtype=np.uintp)

    surf_coords, surf_connect = riley.extract_surface_mesh(coords, connect)

    assert surf_coords.shape == (8, 3)
    assert surf_connect.shape == (6, 4)
    assert np.unique(surf_connect).shape[0] == 8


def test_project_uvs_planar_centered_xy() -> None:
    coords = np.array(
        (
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        dtype=np.float64,
    )

    uvs = riley.project_uvs_planar_centered(
        coords,
        (200, 100),
        uv_span_max=0.8,
        projection_plane=riley.ProjectionPlane.xy,
    )

    assert uvs.shape == (4, 2)
    assert np.all(uvs >= 0.0)
    assert np.all(uvs <= 1.0)
