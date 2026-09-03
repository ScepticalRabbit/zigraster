# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
# --------------------------------------------------------------------------
"""Hard-coded cross-convention cases for every supported element topology."""

from __future__ import annotations

import numpy as np
import pytest

from riley.python import _meshconv, meshconv


def _tri_coords() -> np.ndarray:
    return np.array(((0., 0., 0.), (2., 0., 0.), (0., 1., 0.),
                     (1., 0., 0.), (1., .5, 0.), (0., .5, 0.),
                     (2. / 3., 1. / 3., 0.)), dtype=np.float64)


def _quad_coords() -> np.ndarray:
    return np.array(((0., 0., 0.), (2., 0., 0.), (2., 1., 0.), (0., 1., 0.),
                     (1., 0., 0.), (2., .5, 0.), (1., 1., 0.), (0., .5, 0.),
                     (1., .5, 0.)), dtype=np.float64)


def _tet_coords() -> np.ndarray:
    return np.array(((0., 0., 0.), (2., 0., 0.), (0., 1., 0.), (0., 0., 1.),
                     (1., 0., 0.), (1., .5, 0.), (0., .5, 0.), (.0, .5, .5),
                     (1., 0., .5), (0., .5, .5)), dtype=np.float64)


def _hex_coords() -> np.ndarray:
    points = [
        (x, y, z)
        for z in (0., 1.)
        for y in (0., 1.)
        for x in (0., 2.)
    ]
    corners = (points[0], points[1], points[3], points[2],
               points[4], points[5], points[7], points[6])
    edges = ((1., 0., 0.), (2., .5, 0.), (1., 1., 0.), (0., .5, 0.),
             (1., 0., 1.), (2., .5, 1.), (1., 1., 1.), (0., .5, 1.),
             (0., 0., .5), (2., 0., .5), (2., 1., .5), (0., 1., .5))
    faces = ((1., .5, 0.), (0., .5, .5), (1., .5, 1.),
             (2., .5, .5), (1., 0., .5), (1., 1., .5), (1., .5, .5))
    return np.array(corners + edges + faces, dtype=np.float64)


_PERMUTATIONS = {
    meshconv.EElementType.TRI3: (1, 2, 0),
    meshconv.EElementType.TRI6: (2, 0, 1, 5, 3, 4),
    meshconv.EElementType.TRI7: (2, 0, 1, 5, 3, 4, 6),
    meshconv.EElementType.QUAD4: (2, 3, 0, 1),
    meshconv.EElementType.QUAD8: (2, 3, 0, 1, 6, 7, 4, 5),
    meshconv.EElementType.QUAD9: (2, 3, 0, 1, 6, 7, 4, 5, 8),
    meshconv.EElementType.TET4: (1, 2, 0, 3),
    meshconv.EElementType.TET10: (1, 2, 0, 3, 5, 6, 4, 8, 9, 7),
    meshconv.EElementType.HEX8: (1, 2, 3, 0, 5, 6, 7, 4),
    meshconv.EElementType.HEX20: (1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8,
                                  13, 14, 15, 12, 17, 18, 19, 16),
    meshconv.EElementType.HEX27: (1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8,
                                  13, 14, 15, 12, 17, 18, 19, 16, 21, 22,
                                  23, 20, 25, 26, 24),
}
_ELEMENT_COORDS = {
    meshconv.EElementType.TRI3: _tri_coords()[:3],
    meshconv.EElementType.TRI6: _tri_coords()[:6],
    meshconv.EElementType.TRI7: _tri_coords(),
    meshconv.EElementType.QUAD4: _quad_coords()[:4],
    meshconv.EElementType.QUAD8: _quad_coords()[:8],
    meshconv.EElementType.QUAD9: _quad_coords(),
    meshconv.EElementType.TET4: _tet_coords()[:4],
    meshconv.EElementType.TET10: _tet_coords(),
    meshconv.EElementType.HEX8: _hex_coords()[:8],
    meshconv.EElementType.HEX20: _hex_coords()[:20],
    meshconv.EElementType.HEX27: _hex_coords(),
}
_HIGH_ORDER_TYPES = (
    meshconv.EElementType.TRI6, meshconv.EElementType.QUAD8,
    meshconv.EElementType.TET10, meshconv.EElementType.HEX20,
)


def _mesh(
    element_type: meshconv.EElementType,
    coords: np.ndarray,
    connect: np.ndarray,
) -> meshconv.SimData:
    return meshconv.SimData(
        coords=coords,
        blocks={"connect1": meshconv.ElementBlock(element_type, connect)},
    )


@pytest.mark.parametrize("element_type", _HIGH_ORDER_TYPES)
def test_source_conversion_then_enforcement_repairs_geometry(
    element_type: meshconv.EElementType,
) -> None:
    coords = _ELEMENT_COORDS[element_type]
    expected = np.arange(coords.shape[0], dtype=np.int64)
    elem_spec = _meshconv.ELEMENT_SPECS[element_type]
    if elem_spec.is_surf:
        changed = _meshconv._reverse_surf_row(expected, elem_spec)
    else:
        changed = _meshconv._reverse_handedness_row(expected, elem_spec)
    permutation = _PERMUTATIONS[element_type]
    source = (changed[np.argsort(permutation)] + 1)[:, None]
    source_spec = meshconv.SourceBlockSpec(
        element_type=element_type,
        indexing=meshconv.EConnectIndexing.ONE_BASED,
        layout=meshconv.EConnectLayout.NODE_MAJOR,
        target_to_source_perm=permutation,
    )
    block = meshconv.convert_source_block(source, coords.shape[0], source_spec)
    mesh = meshconv.SimData(coords=coords, blocks={"connect1": block})

    report = meshconv.check_mesh_convention(mesh)
    mesh_out = meshconv.enforce_mesh_convention(mesh)

    expected_failures = {meshconv.MeshCheckCode.RIGHT_HANDED_GEOMETRY}
    if elem_spec.is_surf:
        expected_failures.add(meshconv.MeshCheckCode.CCW_WINDING)
    assert set(report["connect1"]) == expected_failures
    assert np.array_equal(mesh_out.blocks["connect1"].connect, expected[None, :])
    assert meshconv.enforce_mesh_convention(mesh_out) is mesh_out


def test_infer_mesh_convention_recovers_a_single_affine_source_layout() -> None:
    element_type = meshconv.EElementType.TRI6
    permutation = _PERMUTATIONS[element_type]
    expected = np.arange(6, dtype=np.int64)[None, :]
    mesh = _mesh(
        element_type,
        _tri_coords()[:6],
        expected[:, np.argsort(permutation)],
    )

    inferred = meshconv.infer_mesh_convention(mesh)
    mesh_out = meshconv.enforce_mesh_convention(mesh, inferred)

    assert inferred.get_target_to_source_perm(element_type) is not None
    assert np.array_equal(mesh_out.blocks["connect1"].connect, expected)


def test_inference_accepts_rows_that_differ_only_by_a_valid_rotation() -> None:
    coords = np.vstack((_tri_coords()[:6], _tri_coords()[:6] + (3., 0., 0.)))
    first = np.arange(6, dtype=np.int64)
    rotation = _meshconv._get_elem_symmetries(
        meshconv.EElementType.TRI6
    )[1]
    second = np.arange(6, 12, dtype=np.int64)[np.argsort(rotation)]
    mesh = _mesh(
        meshconv.EElementType.TRI6,
        coords,
        np.vstack((first, second)),
    )

    inferred = meshconv.infer_mesh_convention(mesh)

    assert inferred.standardise_equiv_orients
    assert inferred.get_target_to_source_perm(
        meshconv.EElementType.TRI6,
    ) is not None


@pytest.mark.parametrize("split_blocks", (False, True))
def test_inference_rejects_conflicting_source_layouts(
    split_blocks: bool,
) -> None:
    coords = np.vstack((
        _tri_coords()[:6],
        _tri_coords()[:6] + (3., 0., 0.),
    ))
    first = np.arange(6, dtype=np.int64)
    second = np.arange(6, 12, dtype=np.int64)
    second[[4, 5]] = second[[5, 4]]
    if split_blocks:
        blocks = {
            "connect1": meshconv.ElementBlock(
                meshconv.EElementType.TRI6, first[None, :]
            ),
            "connect2": meshconv.ElementBlock(
                meshconv.EElementType.TRI6, second[None, :]
            ),
        }
    else:
        blocks = {
            "connect1": meshconv.ElementBlock(
                meshconv.EElementType.TRI6,
                np.vstack((first, second)),
            ),
        }
    mesh = meshconv.SimData(coords=coords, blocks=blocks)

    with pytest.raises(
        meshconv.MeshConvErr,
        match="disagree|multiple source layouts",
    ):
        meshconv.infer_mesh_convention(mesh)


def test_hex27_registry_uses_vtk_face_and_volume_centre_slots() -> None:
    spec = _meshconv.ELEMENT_SPECS[meshconv.EElementType.HEX27]

    assert spec.cell_centre_idx == 26
    assert spec.face_centre_idxs == (24, 23, 25, 21, 20, 22)


def test_hex27_inference_recovers_swapped_face_centre_slots() -> None:
    expected = np.arange(27, dtype=np.int64)[None, :]
    source = expected.copy()
    source[:, [20, 21]] = source[:, [21, 20]]
    mesh = _mesh(
        meshconv.EElementType.HEX27,
        meshconv.EElementType.HEX27.calc_ref_coords(),
        source,
    )

    inferred = meshconv.infer_mesh_convention(mesh)
    mesh_out = meshconv.enforce_mesh_convention(mesh, inferred)

    assert np.array_equal(mesh_out.blocks["connect1"].connect, expected)


@pytest.mark.parametrize(
    ("element_type", "expected_count"),
    (
        (meshconv.EElementType.TRI3, 3),
        (meshconv.EElementType.TRI6, 3),
        (meshconv.EElementType.TRI7, 3),
        (meshconv.EElementType.QUAD4, 4),
        (meshconv.EElementType.QUAD8, 4),
        (meshconv.EElementType.QUAD9, 4),
        (meshconv.EElementType.TET4, 12),
        (meshconv.EElementType.TET10, 12),
        (meshconv.EElementType.HEX8, 24),
        (meshconv.EElementType.HEX20, 24),
        (meshconv.EElementType.HEX27, 24),
    ),
)
def test_element_symmetry_registry_has_every_proper_orient(
    element_type: meshconv.EElementType,
    expected_count: int,
) -> None:
    permutations = _meshconv._get_elem_symmetries(element_type)

    assert len(permutations) == expected_count
    expected_slots = set(range(
        _meshconv.ELEMENT_SPECS[element_type].nodes_per_elem
    ))
    for permutation in permutations:
        assert set(permutation) == expected_slots


@pytest.mark.parametrize("element_type", _ELEMENT_COORDS)
def test_explicit_and_proper_source_orients_convert_to_riley_slots(
    element_type: meshconv.EElementType,
) -> None:
    coords = _ELEMENT_COORDS[element_type]
    expected = np.arange(coords.shape[0], dtype=np.int64)[None, :]
    permutations = dict.fromkeys((
        _PERMUTATIONS[element_type],
        *_meshconv._get_elem_symmetries(element_type),
    ))
    for permutation in permutations:
        source = expected[:, np.argsort(permutation)]
        block = meshconv.convert_source_block(
            source,
            node_count=coords.shape[0],
            spec=meshconv.SourceBlockSpec(
                element_type=element_type,
                indexing=meshconv.EConnectIndexing.ZERO_BASED,
                layout=meshconv.EConnectLayout.ROW_MAJOR,
                target_to_source_perm=permutation,
            ),
        )
        convention = meshconv.MeshConvention({element_type: permutation})
        mesh = _mesh(element_type, coords, source)
        report = meshconv.check_mesh_convention(mesh, convention)
        mesh_out = meshconv.enforce_mesh_convention(mesh, convention)

        assert block.element_type is element_type
        assert np.array_equal(block.connect, expected)
        assert bool(report) == (permutation != tuple(range(expected.shape[1])))
        assert np.array_equal(mesh_out.blocks["connect1"].connect, expected)
