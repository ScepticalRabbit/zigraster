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
    points: list[tuple[float, float, float]] = []
    for z in (0., 1.):
        for y in (0., 1.):
            for x in (0., 2.):
                points.append((x, y, z))
    # Riley corner sequence: 0,1,2,3 bottom then 4,5,6,7 top.
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


@pytest.mark.parametrize(
    ("element_type", "coords", "mesh_type"),
    (
        (
            meshconv.EElementType.TRI3,
            _tri_coords()[:3],
            meshconv.EMeshType.SURF,
        ),
        (
            meshconv.EElementType.TRI6,
            _tri_coords()[:6],
            meshconv.EMeshType.SURF,
        ),
        (meshconv.EElementType.TRI7, _tri_coords(), meshconv.EMeshType.SURF),
        (
            meshconv.EElementType.QUAD4,
            _quad_coords()[:4],
            meshconv.EMeshType.SURF,
        ),
        (
            meshconv.EElementType.QUAD8,
            _quad_coords()[:8],
            meshconv.EMeshType.SURF,
        ),
        (meshconv.EElementType.QUAD9, _quad_coords(), meshconv.EMeshType.SURF),
        (meshconv.EElementType.TET4, _tet_coords()[:4], meshconv.EMeshType.VOL),
        (meshconv.EElementType.TET10, _tet_coords(), meshconv.EMeshType.VOL),
        (meshconv.EElementType.HEX8, _hex_coords()[:8], meshconv.EMeshType.VOL),
        (
            meshconv.EElementType.HEX20,
            _hex_coords()[:20],
            meshconv.EMeshType.VOL,
        ),
        (meshconv.EElementType.HEX27, _hex_coords(), meshconv.EMeshType.VOL),
    ),
)
def test_explicit_conventions_normalise_every_supported_element(
    element_type: meshconv.EElementType,
    coords: np.ndarray,
    mesh_type: meshconv.EMeshType,
) -> None:
    std = np.arange(coords.shape[0], dtype=np.int64)[None, :]
    permutation = _PERMUTATIONS[element_type]
    source = std[:, np.argsort(permutation)]
    convention = meshconv.MeshConvention({element_type: permutation})
    mesh = meshconv.SimData(
        coords=coords,
        connect={"connect1": source},
        mesh_type=mesh_type,
    )

    assert meshconv.MeshCheckCode.NODE_ORDER in meshconv.check_mesh_convention(
        mesh, convention,
    )["connect1"]
    mesh_out = meshconv.enforce_mesh_convention(mesh, convention)

    assert mesh_out.connect is not None
    assert np.array_equal(mesh_out.connect["connect1"], std)
    assert not meshconv.check_mesh_convention(mesh_out)


@pytest.mark.parametrize(
    ("elem_type", "coords", "mesh_type"),
    (
        (
            meshconv.EElementType.TRI6,
            _tri_coords()[:6],
            meshconv.EMeshType.SURF,
        ),
        (
            meshconv.EElementType.QUAD8,
            _quad_coords()[:8],
            meshconv.EMeshType.SURF,
        ),
        (meshconv.EElementType.TET10, _tet_coords(), meshconv.EMeshType.VOL),
        (
            meshconv.EElementType.HEX20,
            _hex_coords()[:20],
            meshconv.EMeshType.VOL,
        ),
    ),
)
def test_enforce_repairs_combined_convention_changes(
    elem_type: meshconv.EElementType,
    coords: np.ndarray,
    mesh_type: meshconv.EMeshType,
) -> None:
    std = np.arange(coords.shape[0], dtype=np.int64)
    spec = _meshconv.ELEMENT_SPECS[elem_type]
    if spec.is_surf:
        changed_std = _meshconv._reverse_surf_row(std)
    else:
        changed_std = _meshconv._reverse_handedness_row(std)
    src_perm = _PERMUTATIONS[elem_type]
    src_slots = np.argsort(src_perm)
    connect = (changed_std[src_slots] + 1)[:, None]
    src_convention = meshconv.MeshConvention({elem_type: src_perm})
    mesh = meshconv.SimData(
        coords=coords,
        connect={"connect1": connect},
        mesh_type=mesh_type,
    )

    report = meshconv.check_mesh_convention(mesh, src_convention)
    mesh_out = meshconv.enforce_mesh_convention(mesh, src_convention)

    expected_failures = {
        meshconv.MeshCheckCode.ROW_MAJOR_CONNECTIVITY,
        meshconv.MeshCheckCode.ZERO_BASED_INDEXING,
        meshconv.MeshCheckCode.NODE_ORDER,
        meshconv.MeshCheckCode.RIGHT_HANDED_GEOMETRY,
    }
    if spec.is_surf:
        expected_failures.add(meshconv.MeshCheckCode.CCW_WINDING)
    assert set(report["connect1"]) == expected_failures
    assert mesh_out.connect is not None
    expected = std[None, :]
    assert np.array_equal(mesh_out.connect["connect1"], expected)
    assert meshconv.enforce_mesh_convention(mesh_out) is mesh_out


def test_infer_mesh_convention_recovers_a_single_affine_source_layout() -> None:
    element_type = meshconv.EElementType.TRI6
    permutation = _PERMUTATIONS[element_type]
    std = np.arange(6, dtype=np.int64)[None, :]
    mesh = meshconv.SimData(
        coords=_tri_coords()[:6],
        connect={"connect1": std[:, np.argsort(permutation)]},
        mesh_type=meshconv.EMeshType.SURF,
    )

    inferred = meshconv.infer_mesh_convention(mesh)

    assert inferred.get_src_perm(element_type) is not None


def test_public_inference_reports_incomplete_mesh() -> None:
    with pytest.raises(
        meshconv.MeshConvErr,
        match="requires coordinates and connectivity",
    ):
        meshconv.infer_mesh_convention(meshconv.SimData())


def test_inference_accepts_rows_that_differ_only_by_a_valid_rotation() -> None:
    coords = np.vstack((_tri_coords()[:6], _tri_coords()[:6] + (3., 0., 0.)))
    first = np.arange(6, dtype=np.int64)
    rotation = _meshconv._get_elem_symmetries(
        meshconv.EElementType.TRI6
    )[1]
    second = np.arange(6, 12, dtype=np.int64)[np.argsort(rotation)]
    mesh = meshconv.SimData(
        coords=coords,
        connect={"connect1": np.vstack((first, second))},
        mesh_type=meshconv.EMeshType.SURF,
    )

    inferred = meshconv.infer_mesh_convention(mesh)

    assert inferred.standardise_equiv_orients
    assert inferred.get_src_perm(meshconv.EElementType.TRI6) is not None


@pytest.mark.parametrize("split_tables", (False, True))
def test_inference_rejects_conflicting_src_layouts(
    split_tables: bool,
) -> None:
    coords = np.vstack((
        _tri_coords()[:6],
        _tri_coords()[:6] + (3., 0., 0.),
    ))
    first = np.arange(6, dtype=np.int64)
    second = np.arange(6, 12, dtype=np.int64)
    second[[4, 5]] = second[[5, 4]]
    if split_tables:
        connect = {
            "connect1": first[None, :],
            "connect2": second[None, :],
        }
    else:
        connect = {"connect1": np.vstack((first, second))}
    mesh = meshconv.SimData(
        coords=coords,
        connect=connect,
        mesh_type=meshconv.EMeshType.SURF,
    )

    expected = "disagree|multiple source layouts"
    with pytest.raises(meshconv.MeshConvErr, match=expected):
        meshconv.infer_mesh_convention(mesh)


def test_hex27_registry_uses_vtk_face_and_volume_centre_slots() -> None:
    spec = _meshconv.ELEMENT_SPECS[meshconv.EElementType.HEX27]

    assert spec.cell_centre_idx == 26
    assert spec.face_centre_idxs == (24, 23, 25, 21, 20, 22)


def test_hex27_inference_recovers_swapped_face_centre_slots() -> None:
    std = np.arange(27, dtype=np.int64)[None, :]
    source = std.copy()
    source[:, [20, 21]] = source[:, [21, 20]]
    mesh = meshconv.SimData(
        coords=meshconv.EElementType.HEX27.calc_ref_coords(),
        connect={"connect1": source},
        mesh_type=meshconv.EMeshType.VOL,
    )

    inferred = meshconv.infer_mesh_convention(mesh)
    mesh_out = meshconv.enforce_mesh_convention(mesh, inferred)

    assert mesh_out.connect is not None
    assert np.array_equal(mesh_out.connect["connect1"], std)


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
    for permutation in permutations:
        perm_slots = set(permutation)
        expected_slots = set(range(len(permutation)))
        assert perm_slots == expected_slots


@pytest.mark.parametrize(
    ("element_type", "coords", "mesh_type"),
    (
        (
            meshconv.EElementType.TRI3,
            _tri_coords()[:3],
            meshconv.EMeshType.SURF,
        ),
        (
            meshconv.EElementType.TRI6,
            _tri_coords()[:6],
            meshconv.EMeshType.SURF,
        ),
        (meshconv.EElementType.TRI7, _tri_coords(), meshconv.EMeshType.SURF),
        (
            meshconv.EElementType.QUAD4,
            _quad_coords()[:4],
            meshconv.EMeshType.SURF,
        ),
        (
            meshconv.EElementType.QUAD8,
            _quad_coords()[:8],
            meshconv.EMeshType.SURF,
        ),
        (meshconv.EElementType.QUAD9, _quad_coords(), meshconv.EMeshType.SURF),
        (meshconv.EElementType.TET4, _tet_coords()[:4], meshconv.EMeshType.VOL),
        (meshconv.EElementType.TET10, _tet_coords(), meshconv.EMeshType.VOL),
        (meshconv.EElementType.HEX8, _hex_coords()[:8], meshconv.EMeshType.VOL),
        (
            meshconv.EElementType.HEX20,
            _hex_coords()[:20],
            meshconv.EMeshType.VOL,
        ),
        (meshconv.EElementType.HEX27, _hex_coords(), meshconv.EMeshType.VOL),
    ),
)
def test_every_proper_src_orient_converts_to_riley_slots(
    element_type: meshconv.EElementType,
    coords: np.ndarray,
    mesh_type: meshconv.EMeshType,
) -> None:
    std = np.arange(coords.shape[0], dtype=np.int64)[None, :]
    for permutation in _meshconv._get_elem_symmetries(element_type):
        source = std[:, np.argsort(permutation)]
        mesh = meshconv.SimData(
            coords=coords,
            connect={"connect1": source},
            mesh_type=mesh_type,
        )
        src_convention = meshconv.MeshConvention({element_type: permutation})

        mesh_out = meshconv.enforce_mesh_convention(mesh, src_convention)

        assert mesh_out.connect is not None
        assert np.array_equal(mesh_out.connect["connect1"], std)
