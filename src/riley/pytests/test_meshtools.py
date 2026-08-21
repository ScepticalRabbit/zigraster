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
