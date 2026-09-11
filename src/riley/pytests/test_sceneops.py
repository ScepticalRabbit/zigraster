"""Tests for Python scene positioning operations."""

import numpy as np
import pytest

from riley.python import sceneops


def _coords(offset: tuple[float, float, float]) -> np.ndarray:
    coords = np.array(((0.0, 0.0, 0.0), (2.0, 2.0, 2.0)), dtype=np.float64)
    return coords + np.asarray(offset, dtype=np.float64)


def test_calc_bounds_for_coords_list_reduces_bounds() -> None:
    bounds = sceneops.scene_calc_bounds_for_coords_list(
        [_coords((0.0, 0.0, 0.0)), _coords((4.0, -2.0, 1.0))],
    )

    np.testing.assert_allclose(bounds.minimum, (0.0, -2.0, 0.0))
    np.testing.assert_allclose(bounds.maximum, (6.0, 2.0, 3.0))


def test_center_mesh_group_at_translates_in_place() -> None:
    coords_list = [_coords((2.0, 4.0, 6.0))]

    sceneops.scene_center_mesh_group_at(
        coords_list,
        sceneops.scene_create_mesh_group_single(0),
        (0.0, 0.0, 0.0),
    )

    np.testing.assert_allclose(
        sceneops.scene_calc_bounds_for_coords_list(coords_list).center,
        (0.0, 0.0, 0.0),
    )


def test_overlap_mesh_group_bounds_respects_direct_and_offset() -> None:
    coords_list = [_coords((0.0, 0.0, 0.0)), _coords((10.0, 0.0, 0.0))]
    spec = sceneops.SceneBoundsOverlapSpec(
        overlap_frac=(0.5, 0.0, 0.0),
        enabled_axes=(True, False, False),
        direct=(
            sceneops.ESceneOverlapDirect.NEGATIVE,
            sceneops.ESceneOverlapDirect.CURRENT,
            sceneops.ESceneOverlapDirect.CURRENT,
        ),
        extra_offset=(0.25, 1.0, 0.0),
    )

    sceneops.scene_overlap_mesh_group_bounds(
        coords_list,
        sceneops.scene_create_mesh_group_single(0),
        sceneops.scene_create_mesh_group_single(1),
        spec,
    )

    fixed = sceneops.scene_calc_bounds_for_coords(coords_list[0])
    moving = sceneops.scene_calc_bounds_for_coords(coords_list[1])
    assert moving.center[0] == pytest.approx(fixed.center[0] - 0.75)
    assert moving.center[1] == pytest.approx(2.0)


def test_arrange_mesh_groups_grid_places_groups() -> None:
    coords_list = []
    groups = []
    for index in range(4):
        coords_list.append(_coords((float(index), 0.0, 0.0)))
        groups.append(sceneops.scene_create_mesh_group_single(index))

    sceneops.scene_arrange_mesh_groups_grid(
        coords_list,
        groups,
        sceneops.SceneGridSpec(gap=(1.0, 1.0, 1.0), max_divs=(2, 1, 2)),
    )

    centers = []
    for coords in coords_list:
        centers.append(
            sceneops.scene_calc_bounds_for_coords(coords).center
        )

    np.testing.assert_allclose(
        centers,
        ((0.0, 0.0, 0.0), (3.0, 0.0, 0.0),
         (0.0, 0.0, 3.0), (3.0, 0.0, 3.0)),
    )
