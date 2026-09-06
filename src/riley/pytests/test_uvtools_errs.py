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
            proj_plane=(np.zeros(3), np.zeros(3)),
        )


def test_project_uvs_rejects_degenerate_proj() -> None:
    coords = np.array(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
    )

    with pytest.raises(ValueError, match="zero area"):
        riley.project_uvs_planar_bbox(
            coords,
            (100, 100),
            (0.0, 0.0, 99.0, 99.0),
            riley.EProjPlane.XY,
        )
