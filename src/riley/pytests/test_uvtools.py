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
import pytest

import riley


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
        projection_plane=riley.EProjectionPlane.XY,
    )

    assert uvs.shape == (4, 2)
    assert np.all(uvs >= 0.0)
    assert np.all(uvs <= 1.0)


@pytest.mark.parametrize(
    "plane",
    [
        riley.EProjectionPlane.XY,
        riley.EProjectionPlane.YZ,
        riley.EProjectionPlane.XZ,
    ],
)
def test_project_uvs_planar_centered_axis_planes(
    plane: riley.EProjectionPlane,
) -> None:
    coords = np.array(
        ((0.0, 0.0, 0.0), (2.0, 3.0, 4.0), (1.0, 1.0, 1.0)),
    )

    uvs = riley.project_uvs_planar_centered(coords, (100, 100), 0.8, plane)

    assert uvs.shape == (3, 2)
    assert np.all((uvs >= 0.0) & (uvs <= 1.0))


def test_project_uvs_planar_bbox_best_fits_inside_bbox() -> None:
    coords = np.array(
        ((0.0, 0.0, 0.0), (4.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
    )

    uvs = riley.project_uvs_planar_bbox(
        coords,
        (101, 101),
        (20.0, 20.0, 80.0, 80.0),
        riley.EProjectionPlane.XY,
    )

    pixels_x = uvs[:, 0] * 100.0
    pixels_y = (1.0 - uvs[:, 1]) * 100.0
    assert np.all((pixels_x >= 20.0) & (pixels_x <= 80.0))
    assert np.all((pixels_y >= 20.0) & (pixels_y <= 80.0))


def test_project_uvs_planar_custom_plane() -> None:
    coords = np.array(
        ((0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)),
    )
    plane = riley.ProjectionPlane(
        normal=np.array((0.0, 0.0, 1.0)),
        origin=np.array((0.0, 0.0, 1.0)),
    )

    uvs = riley.project_uvs_planar_centered(coords, (100, 100), 1.0, plane)

    assert np.all(np.isfinite(uvs))


@pytest.mark.parametrize("texture_size", [(1, 10), (10, 1), (0, 10)])
def test_project_uvs_rejects_invalid_texture_size(
    texture_size: tuple[int, int],
) -> None:
    coords = np.array(
        ((0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
    )

    with pytest.raises(ValueError, match="at least 2"):
        riley.project_uvs_planar_centered(coords, texture_size)


def test_project_uvs_rejects_zero_normal() -> None:
    coords = np.array(
        ((0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
    )

    with pytest.raises(ValueError, match="nonzero"):
        riley.project_uvs_planar_centered(
            coords,
            (100, 100),
            projection_plane=(np.zeros(3), np.zeros(3)),
        )


def test_project_uvs_rejects_degenerate_projection() -> None:
    coords = np.array(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
    )

    with pytest.raises(ValueError, match="zero area"):
        riley.project_uvs_planar_bbox(
            coords,
            (100, 100),
            (0.0, 0.0, 99.0, 99.0),
            riley.EProjectionPlane.XY,
        )
