"""Tests for Python scene positioning operations."""

import numpy as np
import pytest

from riley.python import sceneops


def _coords(offset: tuple[float, float, float]) -> np.ndarray:
    coords = np.array(((0.0, 0.0, 0.0), (2.0, 2.0, 2.0)), dtype=np.float64)
    return coords + np.asarray(offset, dtype=np.float64)


@pytest.mark.parametrize(
    ("start", "length"),
    [(-1, 1), (0, 0), (0, -1)],
)
def test_create_mesh_group_span_rejects_invalid_range(
    start: int,
    length: int,
) -> None:
    with pytest.raises(ValueError):
        sceneops.create_mesh_group_span(start, length)


def test_calc_bounds_for_mesh_group_rejects_out_of_range() -> None:
    with pytest.raises(IndexError):
        sceneops.calc_bounds_for_mesh_group(
            [_coords((0.0, 0.0, 0.0))], sceneops.MeshGroup(1, 1),
        )


def test_calc_bounds_for_coords_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="not be empty"):
        sceneops.calc_bounds_for_coords(np.empty((0, 3)))


def test_overlap_mesh_group_bounds_rejects_invalid_fraction() -> None:
    coords_list = [
        _coords((0.0, 0.0, 0.0)),
        _coords((2.0, 0.0, 0.0)),
    ]

    with pytest.raises(ValueError, match="overlap_frac"):
        sceneops.overlap_mesh_group_bounds(
            coords_list,
            sceneops.create_mesh_group_single(0),
            sceneops.create_mesh_group_single(1),
            sceneops.BoundsOverlapSpec((1.1, 0.0, 0.0)),
        )


@pytest.mark.parametrize("divisions", [(0, 1, 1), (1, 1, 1)])
def test_arrange_mesh_groups_grid_rejects_invalid_capacity(
    divisions: tuple[int, int, int],
) -> None:
    coords_list = [_coords((0.0, 0.0, 0.0)), _coords((2.0, 0.0, 0.0))]
    groups = [
        sceneops.create_mesh_group_single(0),
        sceneops.create_mesh_group_single(1),
    ]

    with pytest.raises(ValueError):
        sceneops.arrange_mesh_groups_grid(
            coords_list,
            groups,
            sceneops.GridSpec((0.0, 0.0, 0.0), divisions),
        )
