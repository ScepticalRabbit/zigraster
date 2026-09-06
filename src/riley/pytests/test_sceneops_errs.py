"""Tests for Python scene positioning operations."""

from dataclasses import dataclass

import numpy as np
import pytest

from riley.python import sceneops


@dataclass(slots=True)
class _Mesh:
    coords: np.ndarray


def _mesh(offset: tuple[float, float, float]) -> _Mesh:
    coords = np.array(((0.0, 0.0, 0.0), (2.0, 2.0, 2.0)))
    return _Mesh(coords + np.asarray(offset))


@pytest.mark.parametrize(
    ("start", "length"),
    [(-1, 1), (0, 0), (0, -1)],
)
def test_mesh_group_span_rejects_invalid_range(start: int, length: int) -> None:
    with pytest.raises(ValueError):
        sceneops.mesh_group_span(start, length)


def test_bounds_for_mesh_group_rejects_out_of_range() -> None:
    with pytest.raises(IndexError):
        sceneops.bounds_for_mesh_group(
            [_mesh((0.0, 0.0, 0.0))], sceneops.MeshGroup(1, 1),
        )


def test_bounds_for_coords_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="not be empty"):
        sceneops.bounds_for_coords(np.empty((0, 3)))


def test_overlap_mesh_group_bounds_rejects_invalid_fraction() -> None:
    meshes = [
        _mesh((0.0, 0.0, 0.0)),
        _mesh((2.0, 0.0, 0.0)),
    ]

    with pytest.raises(ValueError, match="overlap_frac"):
        sceneops.overlap_mesh_group_bounds(
            meshes,
            sceneops.mesh_group_single(0),
            sceneops.mesh_group_single(1),
            sceneops.BoundsOverlapSpec((1.1, 0.0, 0.0)),
        )


@pytest.mark.parametrize("divisions", [(0, 1, 1), (1, 1, 1)])
def test_arrange_mesh_groups_grid_rejects_invalid_capacity(
    divisions: tuple[int, int, int],
) -> None:
    meshes = [_mesh((0.0, 0.0, 0.0)), _mesh((2.0, 0.0, 0.0))]
    groups = [sceneops.mesh_group_single(0), sceneops.mesh_group_single(1)]

    with pytest.raises(ValueError):
        sceneops.arrange_mesh_groups_grid(
            meshes, groups, sceneops.GridSpec((0.0, 0.0, 0.0), divisions),
        )
