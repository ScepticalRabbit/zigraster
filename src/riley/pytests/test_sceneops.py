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


def test_bounds_for_meshes_reduces_mesh_bounds() -> None:
    bounds = sceneops.bounds_for_meshes(
        [_mesh((0.0, 0.0, 0.0)), _mesh((4.0, -2.0, 1.0))],
    )

    np.testing.assert_allclose(bounds.minimum, (0.0, -2.0, 0.0))
    np.testing.assert_allclose(bounds.maximum, (6.0, 2.0, 3.0))


def test_center_mesh_group_at_translates_in_place() -> None:
    meshes = [_mesh((2.0, 4.0, 6.0))]

    sceneops.center_mesh_group_at(
        meshes, sceneops.mesh_group_single(0), (0.0, 0.0, 0.0),
    )

    np.testing.assert_allclose(
        sceneops.bounds_for_meshes(meshes).center, (0.0, 0.0, 0.0),
    )


def test_overlap_mesh_group_bounds_respects_direct_and_offset() -> None:
    meshes = [_mesh((0.0, 0.0, 0.0)), _mesh((10.0, 0.0, 0.0))]
    spec = sceneops.BoundsOverlapSpec(
        overlap_frac=(0.5, 0.0, 0.0),
        enabled_axes=(True, False, False),
        direct=(
            sceneops.EOverlapDirect.NEGATIVE,
            sceneops.EOverlapDirect.CURRENT,
            sceneops.EOverlapDirect.CURRENT,
        ),
        extra_offset=(0.25, 1.0, 0.0),
    )

    sceneops.overlap_mesh_group_bounds(
        meshes, sceneops.mesh_group_single(0),
        sceneops.mesh_group_single(1), spec,
    )

    fixed = sceneops.bounds_for_coords(meshes[0].coords)
    moving = sceneops.bounds_for_coords(meshes[1].coords)
    assert moving.center[0] == pytest.approx(fixed.center[0] - 0.75)
    assert moving.center[1] == pytest.approx(2.0)


def test_arrange_mesh_groups_grid_places_groups() -> None:
    meshes = [_mesh((float(index), 0.0, 0.0)) for index in range(4)]
    groups = [sceneops.mesh_group_single(index) for index in range(4)]

    sceneops.arrange_mesh_groups_grid(
        meshes,
        groups,
        sceneops.GridSpec(gap=(1.0, 1.0, 1.0), max_divs=(2, 1, 2)),
    )

    centers = [
        sceneops.bounds_for_coords(mesh.coords).center for mesh in meshes
    ]
    np.testing.assert_allclose(
        centers,
        ((0.0, 0.0, 0.0), (3.0, 0.0, 0.0),
         (0.0, 0.0, 3.0), (3.0, 0.0, 3.0)),
    )
