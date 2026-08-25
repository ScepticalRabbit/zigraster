# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from riley.python import _meshconv, meshconv


DATA_DIR = Path(__file__).resolve().parents[3] / "data"
SUPPORTED_CUBES = ("tet4", "tet10", "hex8", "hex20", "hex27")
SPHERE_MESHES = (
    "tri3_sphere200",
    "tri6_sphere200",
    "quad4newton_sphere200",
    "quad8_sphere200",
    "quad9_sphere200",
)
SURF_ELEM_TYPES = (
    meshconv.EElementType.TRI3,
    meshconv.EElementType.TRI6,
    meshconv.EElementType.TRI7,
    meshconv.EElementType.QUAD4,
    meshconv.EElementType.QUAD8,
    meshconv.EElementType.QUAD9,
)
_HEX_EDGE_CORNER_IDXS = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
_HEX_FACE_CORNER_IDXS = (
    (0, 1, 2, 3), (0, 3, 7, 4), (4, 5, 6, 7),
    (1, 2, 6, 5), (0, 1, 5, 4), (3, 2, 6, 7),
)
_HEX_TO_TET_CORNER_IDXS = (
    (0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
    (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6),
)
_TET_EDGE_CORNER_IDXS = (
    (0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3),
)
_HIGH_ORDER_HEX_TYPES = {
    meshconv.EElementType.HEX20,
    meshconv.EElementType.HEX27,
}


def test_element_specs_are_complete_and_mapping_is_read_only() -> None:
    node_counts: set[int] = set()
    for spec in _meshconv.ELEMENT_SPECS.values():
        node_counts.add(spec.nodes_per_elem)
    assert node_counts == {3, 4, 6, 7, 8, 9, 10, 20, 27}
    assert (
        _meshconv.ELEMENT_SPECS[meshconv.EElementType.HEX27].centre_idx
        is None
    )

    with pytest.raises(TypeError):
        _meshconv.ELEMENT_SPECS[meshconv.EElementType.TRI3] = (
            _meshconv.ELEMENT_SPECS[meshconv.EElementType.TRI3]
        )


def test_check_mesh_convention_passes_for_std_quad() -> None:
    mesh = meshconv.SimData(
        coords=_quad_coords(),
        connect={"connect1": np.array(((0, 1, 2, 3),), dtype=np.int64)},
    )
    mesh.update_mesh_type()

    report = meshconv.check_mesh_convention(mesh)

    assert mesh.mesh_type is meshconv.EMeshType.SURF
    assert report == {}


def test_enforce_mesh_convention_corrects_legacy_connectivity() -> None:
    mesh = meshconv.SimData(
        coords=_quad_coords(),
        connect={"connect1": np.array(((1,), (2,), (3,), (4,)))},
    )

    mesh_out = meshconv.enforce_mesh_convention(mesh)

    assert mesh_out.connect is not None
    assert np.array_equal(
        mesh_out.connect["connect1"],
        np.array(((0, 1, 2, 3),), dtype=np.int64),
    )
    assert not meshconv.check_mesh_convention(mesh_out)


@pytest.mark.parametrize(
    "operation",
    (meshconv.check_mesh_convention, meshconv.enforce_mesh_convention),
)
def test_mesh_convention_rejects_mixed_indexing_between_tables(
    operation: Callable[[meshconv.SimData], object],
) -> None:
    coords = np.vstack((_quad_coords(), _quad_coords() + (2.0, 0.0, 0.0)))
    mesh = meshconv.SimData(
        coords=coords,
        connect={
            "connect_zero_based": np.array(((0, 1, 2, 3),)),
            "connect_one_based": np.array(((5, 6, 7, 8),)),
        },
        mesh_type=meshconv.EMeshType.SURF,
    )

    with pytest.raises(ValueError, match="Mixed zero-based and one-based"):
        operation(mesh)


def test_check_mesh_convention_reports_failed_checks() -> None:
    mesh = meshconv.SimData(
        coords=_quad_coords(),
        connect={"connect1": np.array(((1,), (4,), (3,), (2,)))},
    )

    report = meshconv.check_mesh_convention(mesh)

    assert report["connect1"] == [
        meshconv.MeshCheckCode.ROW_MAJOR_CONNECTIVITY,
        meshconv.MeshCheckCode.ZERO_BASED_INDEXING,
        meshconv.MeshCheckCode.CCW_WINDING,
        meshconv.MeshCheckCode.RIGHT_HANDED_GEOMETRY,
    ]


def test_enforce_mesh_convention_raises_for_invalid_indices() -> None:
    mesh = meshconv.SimData(
        coords=_quad_coords(),
        connect={"connect1": np.array(((0, 1, 2, 10),), dtype=np.int64)},
    )

    with pytest.raises(ValueError, match="invalid|outside"):
        meshconv.enforce_mesh_convention(mesh)


def test_enforce_mesh_convention_fixes_tet_handedness() -> None:
    mesh = meshconv.SimData(
        coords=np.array(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
             (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        ),
        connect={"connect1": np.array(((0, 2, 1, 3),), dtype=np.int64)},
    )

    mesh_out = meshconv.enforce_mesh_convention(mesh)

    assert not meshconv.check_mesh_convention(mesh_out)
    assert np.array_equal(
        mesh_out.connect["connect1"],
        np.array(((0, 1, 2, 3),), dtype=np.int64),
    )


def test_enforce_returns_same_object_when_mesh_conforms() -> None:
    mesh = meshconv.SimData(
        coords=_quad_coords(),
        connect={"connect1": np.array(((0, 1, 2, 3),), dtype=np.int64)},
    )
    mesh.update_mesh_type()

    assert meshconv.enforce_mesh_convention(mesh) is mesh


def test_enforce_emits_conforming_sibling_tables_untouched() -> None:
    coords = np.array(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
         (2.0, 0.0, 0.0), (3.0, 0.0, 0.0), (3.0, 1.0, 0.0), (2.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    good_connect = np.array(((0, 1, 2, 3),), dtype=np.int64)
    bad_connect = np.array(((4, 7, 6, 5),), dtype=np.int64)
    fixed_connect = np.array(((4, 5, 6, 7),), dtype=np.int64)
    mesh = meshconv.SimData(
        coords=coords,
        connect={"connect_good": good_connect, "connect_bad": bad_connect},
    )

    mesh_out = meshconv.enforce_mesh_convention(mesh)

    assert mesh_out is not mesh
    assert mesh_out.connect is not None
    assert np.array_equal(mesh_out.connect["connect_good"], good_connect)
    assert np.array_equal(mesh_out.connect["connect_bad"], fixed_connect)
    assert mesh.connect is not None
    assert np.array_equal(mesh.connect["connect_bad"], bad_connect)


def test_enforce_reports_indices_outside_coordinate_array() -> None:
    mesh = meshconv.SimData(
        coords=_quad_coords(),
        connect={"connect1": np.array(((0, 1, 2, 10),), dtype=np.int64)},
    )

    with pytest.raises(
        ValueError,
        match="contains indices outside the coordinate array",
    ):
        meshconv.enforce_mesh_convention(mesh)


def test_enforce_propagates_zero_volume_topology_errors() -> None:
    coords = np.array(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (2.0, 0.5, 0.0)),
        dtype=np.float64,
    )
    mesh = meshconv.SimData(
        coords=coords,
        connect={
            "connect1": np.array(
                ((0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)), dtype=np.int64,
            ),
        },
        mesh_type=meshconv.EMeshType.SURF,
    )

    with pytest.raises(ValueError, match="zero signed volume"):
        meshconv.enforce_mesh_convention(mesh)


def test_enforce_tolerates_nonmanifold_surface_slices() -> None:
    """Non-manifold slices have no orientable shell; consistently wound input
    falls back to the per-face behaviour and passes untouched."""

    coords = np.array(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
         (0.0, -1.0, 0.5), (0.0, 0.0, -1.0)),
        dtype=np.float64,
    )
    mesh = meshconv.SimData(
        coords=coords,
        connect={
            "connect1": np.array(
                ((0, 1, 2), (0, 3, 1), (0, 1, 4)),
                dtype=np.int64,
            ),
        },
        mesh_type=meshconv.EMeshType.SURF,
    )

    assert not meshconv.check_mesh_convention(mesh)
    assert meshconv.enforce_mesh_convention(mesh) is mesh


@pytest.mark.parametrize(
    ("elem_type", "opposite_orient"),
    (
        (meshconv.EElementType.TRI6, False),
        (meshconv.EElementType.TRI6, True),
        (meshconv.EElementType.QUAD8, False),
        (meshconv.EElementType.QUAD8, True),
    ),
)
def test_duplicate_surf_faces_finish_with_matching_orient(
    elem_type: meshconv.EElementType,
    opposite_orient: bool,
) -> None:
    coords, connect = _build_cube_ring_surf(elem_type)
    first = connect[0]
    duplicate = first.copy()
    if opposite_orient:
        duplicate = _meshconv._reverse_surf_row(duplicate)
    mesh = meshconv.SimData(
        coords=coords,
        connect={"connect1": np.vstack((first, duplicate))},
        mesh_type=meshconv.EMeshType.SURF,
    )

    mesh_out = meshconv.enforce_mesh_convention(mesh)

    assert mesh_out.connect is not None
    connect_out = mesh_out.connect["connect1"]
    assert np.array_equal(connect_out[0], connect_out[1])
    assert not meshconv.check_mesh_convention(mesh_out)


def test_enforce_fixes_mirrored_hex_handedness_and_is_idempotent() -> None:
    coords = np.array(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
         (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)),
        dtype=np.float64,
    )
    mirrored_row = np.array((0, 3, 2, 1, 4, 7, 6, 5), dtype=np.int64)[None, :]
    report = meshconv.check_mesh_convention(
        meshconv.SimData(coords=coords, connect={"connect1": mirrored_row}),
    )
    expected = {meshconv.MeshCheckCode.RIGHT_HANDED_GEOMETRY}
    assert set(report["connect1"]) == expected

    def hex_volume(row: np.ndarray) -> float:
        points = coords[row[0, :]]
        return float(np.linalg.det(np.column_stack((
            points[1] - points[0], points[3] - points[0], points[4] - points[0],
        ))))

    assert hex_volume(mirrored_row) < 0.0

    mesh = meshconv.SimData(coords=coords, connect={"connect1": mirrored_row})
    mesh_out = meshconv.enforce_mesh_convention(mesh)

    assert mesh_out.connect is not None
    assert not meshconv.check_mesh_convention(mesh_out)
    assert hex_volume(mesh_out.connect["connect1"]) > 0.0
    assert np.array_equal(
        meshconv.enforce_mesh_convention(mesh_out).connect["connect1"],
        mesh_out.connect["connect1"],
    )


def test_explicit_mesh_convention_reorders_source_slots() -> None:
    mesh = _load_cube("hex20")
    assert mesh.connect is not None
    std = mesh.connect["connect1"]
    source_to_riley = (
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        16, 17, 18, 19, 12, 13, 14, 15,
    )
    mesh.connect["connect1"] = std[:, np.argsort(source_to_riley)]
    convention = meshconv.MeshConvention({
        meshconv.EElementType.HEX20: source_to_riley,
    })

    assert meshconv.MeshCheckCode.NODE_ORDER in meshconv.check_mesh_convention(
        mesh,
        convention,
    )["connect1"]
    mesh_out = meshconv.enforce_mesh_convention(mesh, convention)

    assert mesh_out.connect is not None
    assert np.array_equal(mesh_out.connect["connect1"], std)
    assert not meshconv.check_mesh_convention(mesh_out)


@pytest.mark.parametrize(
    ("permutation", "error", "message"),
    (
        ((0, 1, 2), ValueError, "requires a 4-slot"),
        ((0, 1, 1, 3), ValueError, "exactly once"),
        ((0, 1, 2, 3.0), TypeError, "integers"),
    ),
)
def test_mesh_convention_rejects_invalid_permutations(
    permutation: tuple[object, ...],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        meshconv.MeshConvention({
            meshconv.EElementType.QUAD4: permutation,
        })


def test_mesh_convention_defensively_copies_its_mapping() -> None:
    permutations = {
        meshconv.EElementType.QUAD4: (0, 1, 2, 3),
    }
    convention = meshconv.MeshConvention(permutations)

    permutations[meshconv.EElementType.QUAD4] = (0, 3, 2, 1)

    assert convention.get_src_perm(meshconv.EElementType.QUAD4) == (
        0,
        1,
        2,
        3,
    )
    with pytest.raises(TypeError):
        convention.src_to_riley_perms[
            meshconv.EElementType.QUAD4
        ] = (0, 3, 2, 1)


def test_coincident_nodes_do_not_bypass_node_role_validation() -> None:
    coords = np.array(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.5, 0.5, 0.0),
            (0.5, 0.0, 0.0),
            (0.5, 0.0, 0.0),
        ),
        dtype=np.float64,
    )
    spec = _meshconv.ELEMENT_SPECS[meshconv.EElementType.TRI6]

    assert not _meshconv._check_std_node_roles(coords, spec)


@pytest.mark.parametrize("cube_name", SUPPORTED_CUBES)
def test_std_cube_meshes_pass_and_enforcement_is_idempotent(
    cube_name: str,
) -> None:
    mesh = _load_cube(cube_name)

    assert not meshconv.check_mesh_convention(mesh)
    enforced_once = meshconv.enforce_mesh_convention(mesh)
    enforced_twice = meshconv.enforce_mesh_convention(enforced_once)

    assert not meshconv.check_mesh_convention(enforced_once)
    assert enforced_once.connect is not None
    assert enforced_twice.connect is not None
    for name, connect in enforced_once.connect.items():
        assert np.array_equal(connect, enforced_twice.connect[name])


def test_tet14_cube_is_explicitly_unsupported() -> None:
    with pytest.raises(
        NotImplementedError,
        match="supported nodes-per-element",
    ):
        meshconv.check_mesh_convention(
            meshconv.SimData(
                coords=np.zeros((14, 3), dtype=np.float64),
                connect={
                    "connect1": np.arange(14, dtype=np.int64).reshape(1, 14),
                },
            )
        )


@pytest.mark.parametrize("cube_name", SUPPORTED_CUBES)
def test_extracted_cube_surface_passes_convention_check(cube_name: str) -> None:
    surface = meshconv.extract_surf_mesh(
        meshconv.enforce_mesh_convention(_load_cube(cube_name)),
    )

    assert not meshconv.check_mesh_convention(surface)


def test_surface_extraction_clears_volume_side_sets() -> None:
    mesh = _load_cube("hex8")
    mesh.side_sets = {("surface", "connect1"): np.array((0,), dtype=np.int64)}

    surface = meshconv.extract_surf_mesh(mesh)

    assert surface.side_sets is None


def test_surface_slice_sets_surface_mesh_type() -> None:
    mesh = _load_cube("hex8")
    mesh.mesh_type = meshconv.EMeshType.VOL

    surface = meshconv.extract_surf_between(
        mesh,
        point=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
    )

    assert surface.mesh_type is meshconv.EMeshType.SURF


def test_surface_slice_uses_first_three_vector_components() -> None:
    mesh = _load_cube("hex8")

    surface = meshconv.extract_surf_between(
        mesh,
        point=(0.0, 0.0, 0.0, 10.0),
        normal=(0.0, 0.0, 1.0, 10.0),
    )

    assert surface.connect is not None


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    (
        ("point", (0.0, 0.0), "at least three"),
        ("normal", (0.0, np.inf, 1.0), "finite"),
        ("tolerance", -1.0, "non-negative"),
        ("distance", np.nan, "finite"),
    ),
)
def test_surface_slice_rejects_invalid_arguments(
    argument: str,
    value: object,
    message: str,
) -> None:
    mesh = _load_cube("hex8")
    arguments: dict[str, object] = {
        "point": (0.0, 0.0, 0.0),
        "normal": (0.0, 0.0, 1.0),
    }
    arguments[argument] = value

    with pytest.raises(ValueError, match=message):
        meshconv.extract_surf_between(mesh, **arguments)


@pytest.mark.parametrize("mesh_name", SPHERE_MESHES)
def test_native_sphere_meshes_normalize_to_an_idempotent_convention(
    mesh_name: str,
) -> None:
    mesh = _load_native_mesh(
        DATA_DIR / "min" / mesh_name,
        mesh_type=meshconv.EMeshType.SURF,
    )

    mesh_out = meshconv.enforce_mesh_convention(mesh)
    mesh_twice = meshconv.enforce_mesh_convention(mesh_out)

    assert not meshconv.check_mesh_convention(mesh_out)
    assert np.array_equal(mesh.coords, mesh_out.coords)
    assert mesh_out.connect is not None
    assert mesh_twice.connect is not None
    for name, connect in mesh_out.connect.items():
        assert np.array_equal(connect, mesh_twice.connect[name])


def test_plate_with_hole_keeps_inward_bore_normals() -> None:
    """A closed plate surface must retain its material-facing bore wall."""

    mesh = _load_native_mesh(
        DATA_DIR / "FE" / "platehole3d_2mr_63f",
        mesh_type=meshconv.EMeshType.SURF,
    )

    assert not meshconv.check_mesh_convention(mesh)
    mesh_out = meshconv.enforce_mesh_convention(mesh)
    assert mesh_out.connect is not None
    assert mesh.connect is not None
    assert np.array_equal(
        mesh_out.connect["connect1"],
        mesh.connect["connect1"],
    )

    connect = mesh_out.connect["connect1"]
    assert mesh_out.coords is not None
    corners = mesh_out.coords[connect[:, :4]]
    normals = np.cross(
        corners[:, 1] - corners[:, 0],
        corners[:, 3] - corners[:, 0],
    )
    radial = np.mean(corners, axis=1)[:, :2] - np.array((0.0125, 0.0175))
    radial_norm = np.linalg.norm(radial, axis=1)
    wall_rows = np.abs(normals[:, 2]) < 1.0e-12
    bore_radius = radial_norm[wall_rows].min()
    bore_rows = wall_rows & np.isclose(radial_norm, bore_radius)
    outer_rows = wall_rows & ~bore_rows

    assert np.count_nonzero(bore_rows) == 64
    bore_alignment = np.sum(
        normals[bore_rows, :2] * radial[bore_rows],
        axis=1,
    )
    outer_alignment = np.sum(
        normals[outer_rows, :2] * radial[outer_rows],
        axis=1,
    )
    assert np.all(bore_alignment < 0.0)
    assert np.all(outer_alignment > 0.0)


@pytest.mark.parametrize("elem_type", SURF_ELEM_TYPES)
def test_cube_ring_orients_bore_into_void(
    elem_type: meshconv.EElementType,
) -> None:
    coords, connect = _build_cube_ring_surf(elem_type)
    reversed_rows: list[np.ndarray] = []
    for row in connect:
        reversed_rows.append(_meshconv._reverse_surf_row(row))
    mesh = meshconv.SimData(
        coords=coords,
        connect={"connect1": np.asarray(reversed_rows)},
        mesh_type=meshconv.EMeshType.SURF,
    )

    mesh_out = meshconv.enforce_mesh_convention(mesh)

    assert mesh_out.connect is not None
    assert not meshconv.check_mesh_convention(mesh_out)
    corners = _get_surf_corner_coords(mesh_out, elem_type)
    normals = np.cross(
        corners[:, 1] - corners[:, 0],
        corners[:, 2] - corners[:, 1],
    )
    centers = np.mean(corners, axis=1)
    vertical = np.isclose(normals[:, 2], 0.0)
    radial = centers[:, :2] - np.array((1.5, 1.5))
    radial_max = np.max(np.abs(radial), axis=1)
    bore = vertical & np.isclose(radial_max, 0.5)
    outside = vertical & np.isclose(radial_max, 1.5)
    bore_dot = np.sum(normals[bore, :2] * radial[bore], axis=1)
    outside_dot = np.sum(normals[outside, :2] * radial[outside], axis=1)

    expected_bore_faces = 8 if elem_type.name.startswith("TRI") else 4
    assert np.count_nonzero(bore) == expected_bore_faces
    assert np.all(bore_dot < 0.0)
    assert np.all(outside_dot > 0.0)


@pytest.mark.parametrize(
    "elem_type",
    (
        meshconv.EElementType.TET4,
        meshconv.EElementType.TET10,
        meshconv.EElementType.HEX20,
        meshconv.EElementType.HEX27,
    ),
)
def test_cube_ring_vol_extracts_complete_oriented_surf(
    elem_type: meshconv.EElementType,
) -> None:
    mesh = _build_cube_ring_vol(elem_type)

    mesh_std = meshconv.enforce_mesh_convention(mesh)
    surf = meshconv.extract_surf_mesh(mesh_std)

    assert surf.coords is not None
    assert surf.connect is not None
    assert not meshconv.check_mesh_convention(surf)
    connect = surf.connect["connect1"]
    from_tets = elem_type in (
        meshconv.EElementType.TET4,
        meshconv.EElementType.TET10,
    )
    expected_faces = 64 if from_tets else 32
    expected_nodes = 6 if elem_type is meshconv.EElementType.TET10 else 3
    if not from_tets:
        expected_nodes = 9 if elem_type is meshconv.EElementType.HEX27 else 8
    assert connect.shape == (expected_faces, expected_nodes)

    corner_count = 3 if from_tets else 4
    corners = surf.coords[connect[:, :corner_count]]
    normals = np.cross(
        corners[:, 1] - corners[:, 0],
        corners[:, 2] - corners[:, 1],
    )
    centers = np.mean(corners, axis=1)
    vertical = np.isclose(normals[:, 2], 0.0)
    radial = centers[:, :2] - np.array((1.5, 1.5))
    radial_max = np.max(np.abs(radial), axis=1)
    bore = vertical & np.isclose(radial_max, 0.5)
    outside = vertical & np.isclose(radial_max, 1.5)
    bore_dot = np.sum(normals[bore, :2] * radial[bore], axis=1)
    outside_dot = np.sum(normals[outside, :2] * radial[outside], axis=1)
    assert np.all(bore_dot < 0.0)
    assert np.all(outside_dot > 0.0)
    assert meshconv.enforce_mesh_convention(surf) is surf


@pytest.mark.parametrize("elem_type", SURF_ELEM_TYPES)
def test_nested_closed_surface_orients_cavity_into_void(
    elem_type: meshconv.EElementType,
) -> None:
    outer_coords, outer_connect = _cube_surface(2.0, 0)
    inner_coords, inner_connect = _cube_surface(1.0, 8)
    coords, connect = _upgrade_surf_elem(
        np.vstack((outer_coords, inner_coords)),
        np.vstack((outer_connect, inner_connect)),
        elem_type,
    )
    outer_rows = 12 if elem_type.name.startswith("TRI") else 6
    mesh = meshconv.SimData(
        coords=coords,
        connect={"connect1": connect},
        mesh_type=meshconv.EMeshType.SURF,
    )

    mesh_out = meshconv.enforce_mesh_convention(mesh)

    assert mesh_out.connect is not None
    assert not meshconv.check_mesh_convention(mesh_out)
    connect = mesh_out.connect["connect1"]
    corner_idxs = _meshconv.ELEMENT_SPECS[elem_type].corner_idxs
    corners = connect[:, corner_idxs]
    assert _surface_volume(mesh_out.coords, corners[:outer_rows]) > 0.0
    assert _surface_volume(mesh_out.coords, corners[outer_rows:]) < 0.0


@pytest.mark.parametrize(
    "elem_type",
    (meshconv.EElementType.TRI3, meshconv.EElementType.QUAD4),
)
def test_three_nested_surfs_alternate_material_orient(
    elem_type: meshconv.EElementType,
) -> None:
    outer_coords, outer_connect = _cube_surface(3.0, 0)
    cavity_coords, cavity_connect = _cube_surface(2.0, 8)
    island_coords, island_connect = _cube_surface(1.0, 16)
    coords, connect = _upgrade_surf_elem(
        np.vstack((outer_coords, cavity_coords, island_coords)),
        np.vstack((outer_connect, cavity_connect, island_connect)),
        elem_type,
    )
    mesh = meshconv.SimData(
        coords=coords,
        connect={"connect1": connect},
        mesh_type=meshconv.EMeshType.SURF,
    )

    mesh_out = meshconv.enforce_mesh_convention(mesh)

    assert mesh_out.coords is not None
    assert mesh_out.connect is not None
    corner_idxs = _meshconv.ELEMENT_SPECS[elem_type].corner_idxs
    corners = mesh_out.connect["connect1"][:, corner_idxs]
    rows_per_shell = 12 if elem_type is meshconv.EElementType.TRI3 else 6
    outer_end = rows_per_shell
    cavity_end = rows_per_shell * 2
    assert _surface_volume(mesh_out.coords, corners[:outer_end]) > 0.0
    assert _surface_volume(
        mesh_out.coords,
        corners[outer_end:cavity_end],
    ) < 0.0
    assert _surface_volume(mesh_out.coords, corners[cavity_end:]) > 0.0


def test_disconnected_closed_surfs_each_orient_outward() -> None:
    first_coords, first_connect = _cube_surface(1.0, 0)
    second_coords, second_connect = _cube_surface(1.0, 8)
    second_coords = second_coords + (4.0, 0.0, 0.0)
    reversed_rows: list[np.ndarray] = []
    for row in second_connect:
        reversed_rows.append(_meshconv._reverse_surf_row(row))
    second_connect = np.asarray(reversed_rows)
    mesh = meshconv.SimData(
        coords=np.vstack((first_coords, second_coords)),
        connect={"connect1": np.vstack((first_connect, second_connect))},
        mesh_type=meshconv.EMeshType.SURF,
    )

    mesh_out = meshconv.enforce_mesh_convention(mesh)

    assert mesh_out.coords is not None
    assert mesh_out.connect is not None
    connect = mesh_out.connect["connect1"]
    faces_per_cube = 6
    assert _surface_volume(mesh_out.coords, connect[:faces_per_cube]) > 0.0
    assert _surface_volume(mesh_out.coords, connect[faces_per_cube:]) > 0.0
    assert not meshconv.check_mesh_convention(mesh_out)


@pytest.mark.parametrize("nonplanar", (False, True))
def test_open_surf_component_has_stable_orient(nonplanar: bool) -> None:
    if nonplanar:
        coords = np.array(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
             (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
             (1.0, 0.0, 1.0), (1.0, 1.0, 1.0)),
        )
        connect = np.array(((0, 1, 2, 3), (1, 4, 5, 2)))
    else:
        coords = np.array(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
             (2.0, 0.0, 0.0), (0.0, 1.0, 0.0),
             (1.0, 1.0, 0.0), (2.0, 1.0, 0.0)),
        )
        connect_std = np.array(((0, 1, 4, 3), (1, 2, 5, 4)))
        reversed_rows: list[np.ndarray] = []
        for row in connect_std:
            reversed_rows.append(_meshconv._reverse_surf_row(row))
        connect = np.asarray(reversed_rows)
    mesh = meshconv.SimData(
        coords=coords,
        connect={"connect1": connect},
        mesh_type=meshconv.EMeshType.SURF,
    )

    mesh_out = meshconv.enforce_mesh_convention(mesh)

    assert mesh_out.connect is not None
    if nonplanar:
        assert np.array_equal(mesh_out.connect["connect1"], connect)
    else:
        assert np.array_equal(mesh_out.connect["connect1"], connect_std)
    assert meshconv.enforce_mesh_convention(mesh_out) is mesh_out


def _quad_coords() -> np.ndarray:
    return np.array(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )


def _cube_surface(
    scale: float,
    node_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    coords = scale * np.array(
        (
            (-1.0, -1.0, -1.0), (1.0, -1.0, -1.0),
            (1.0, 1.0, -1.0), (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0), (1.0, -1.0, 1.0),
            (1.0, 1.0, 1.0), (-1.0, 1.0, 1.0),
        ),
        dtype=np.float64,
    )
    connect = np.array(
        ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
         (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)),
        dtype=np.int64,
    )
    return coords, connect + node_offset


def _build_cube_ring_surf(
    elem_type: meshconv.EElementType,
) -> tuple[np.ndarray, np.ndarray]:
    coords, connect = _build_hex8_ring()
    volume_mesh = meshconv.SimData(
        coords=coords,
        connect={"connect1": connect},
        mesh_type=meshconv.EMeshType.VOL,
    )
    volume_mesh = meshconv.enforce_mesh_convention(volume_mesh)
    surf_mesh = meshconv.extract_surf_mesh(volume_mesh)
    assert surf_mesh.coords is not None
    assert surf_mesh.connect is not None
    return _upgrade_surf_elem(
        surf_mesh.coords,
        surf_mesh.connect["connect1"],
        elem_type,
    )


def _build_hex8_ring() -> tuple[np.ndarray, np.ndarray]:
    nodes_per_grid_row = 4
    nodes_per_grid_layer = 16
    coords_list: list[tuple[float, float, float]] = []
    for zz in range(2):
        for yy in range(4):
            for xx in range(4):
                coords_list.append((float(xx), float(yy), float(zz)))
    coords = np.asarray(coords_list, dtype=np.float64)

    def get_node_idx(xx: int, yy: int, zz: int) -> int:
        return (
            zz * nodes_per_grid_layer
            + yy * nodes_per_grid_row
            + xx
        )

    connect_rows: list[tuple[int, ...]] = []
    for yy in range(3):
        for xx in range(3):
            if xx == 1 and yy == 1:
                continue
            connect_rows.append((
                get_node_idx(xx, yy, 0),
                get_node_idx(xx + 1, yy, 0),
                get_node_idx(xx + 1, yy + 1, 0),
                get_node_idx(xx, yy + 1, 0),
                get_node_idx(xx, yy, 1),
                get_node_idx(xx + 1, yy, 1),
                get_node_idx(xx + 1, yy + 1, 1),
                get_node_idx(xx, yy + 1, 1),
            ))
    return (
        coords,
        np.asarray(connect_rows, dtype=np.int64),
    )


def _build_cube_ring_vol(
    elem_type: meshconv.EElementType,
) -> meshconv.SimData:
    coords, hex8_connect = _build_hex8_ring()
    if elem_type in _HIGH_ORDER_HEX_TYPES:
        coords, connect = _upgrade_hex_vol(coords, hex8_connect, elem_type)
    else:
        tet4_rows: list[list[int]] = []
        for hex_row in hex8_connect:
            for corner_idxs in _HEX_TO_TET_CORNER_IDXS:
                tet_row: list[int] = []
                for corner_idx in corner_idxs:
                    tet_row.append(int(hex_row[corner_idx]))
                tet4_rows.append(tet_row)
        connect = np.asarray(tet4_rows, dtype=np.int64)
        if elem_type is meshconv.EElementType.TET10:
            coords, connect = _upgrade_tet_vol(coords, connect)
    return meshconv.SimData(
        coords=coords,
        connect={"connect1": connect},
        mesh_type=meshconv.EMeshType.VOL,
    )


def _get_or_add_edge_node(
    node_a: int,
    node_b: int,
    coords: np.ndarray,
    coords_out: list[list[float]],
    edge_nodes: dict[tuple[int, int], int],
) -> int:
    edge = (min(node_a, node_b), max(node_a, node_b))
    edge_node = edge_nodes.get(edge)
    if edge_node is None:
        edge_node = len(coords_out)
        edge_nodes[edge] = edge_node
        midpoint = 0.5 * (coords[node_a] + coords[node_b])
        coords_out.append(midpoint.tolist())
    return edge_node


def _upgrade_tet_vol(
    coords: np.ndarray,
    tet4_connect: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    coords_out = coords.tolist()
    edge_nodes: dict[tuple[int, int], int] = {}
    connect_out: list[list[int]] = []
    for corners in tet4_connect:
        row_out = corners.tolist()
        for node_a_idx, node_b_idx in _TET_EDGE_CORNER_IDXS:
            edge_node = _get_or_add_edge_node(
                int(corners[node_a_idx]),
                int(corners[node_b_idx]),
                coords,
                coords_out,
                edge_nodes,
            )
            row_out.append(edge_node)
        connect_out.append(row_out)
    return (
        np.asarray(coords_out, dtype=np.float64),
        np.asarray(connect_out, dtype=np.int64),
    )


def _upgrade_hex_vol(
    coords: np.ndarray,
    hex8_connect: np.ndarray,
    elem_type: meshconv.EElementType,
) -> tuple[np.ndarray, np.ndarray]:
    coords_out = coords.tolist()
    edge_nodes: dict[tuple[int, int], int] = {}
    face_nodes: dict[tuple[int, ...], int] = {}
    connect_out: list[list[int]] = []
    for corners in hex8_connect:
        row_out = corners.tolist()
        for node_a_idx, node_b_idx in _HEX_EDGE_CORNER_IDXS:
            edge_node = _get_or_add_edge_node(
                int(corners[node_a_idx]),
                int(corners[node_b_idx]),
                coords,
                coords_out,
                edge_nodes,
            )
            row_out.append(edge_node)
        if elem_type is meshconv.EElementType.HEX27:
            spec = _meshconv.ELEMENT_SPECS[elem_type]
            unassigned_node = -1
            while len(row_out) < spec.nodes_per_elem:
                row_out.append(unassigned_node)
            for face_idx, corner_idxs in enumerate(_HEX_FACE_CORNER_IDXS):
                face_corner_nodes: list[int] = []
                for corner_idx in corner_idxs:
                    face_corner_nodes.append(int(corners[corner_idx]))
                face_key = tuple(sorted(face_corner_nodes))
                face_node = face_nodes.get(face_key)
                if face_node is None:
                    face_node = len(coords_out)
                    face_nodes[face_key] = face_node
                    face_coords = coords[np.asarray(face_corner_nodes)]
                    coords_out.append(np.mean(face_coords, axis=0).tolist())
                centre_slot = spec.face_centre_idxs[face_idx]
                row_out[centre_slot] = face_node
            if spec.cell_centre_idx is None:
                raise ValueError("HEX27 requires a cell-centre slot.")
            row_out[spec.cell_centre_idx] = len(coords_out)
            coords_out.append(np.mean(coords[corners], axis=0).tolist())
        connect_out.append(row_out)
    return (
        np.asarray(coords_out, dtype=np.float64),
        np.asarray(connect_out, dtype=np.int64),
    )


def _upgrade_surf_elem(
    coords: np.ndarray,
    quad_connect: np.ndarray,
    elem_type: meshconv.EElementType,
) -> tuple[np.ndarray, np.ndarray]:
    linear_rows: list[tuple[int, ...]] = []
    use_tris = elem_type.name.startswith("TRI")
    for row in quad_connect:
        corner_nodes: list[int] = []
        for node in row[:4]:
            corner_nodes.append(int(node))
        corners = tuple(corner_nodes)
        if use_tris:
            linear_rows.append((corners[0], corners[1], corners[2]))
            linear_rows.append((corners[0], corners[2], corners[3]))
        else:
            linear_rows.append(corners)

    nodes_per_elem = _meshconv.ELEMENT_SPECS[elem_type].nodes_per_elem
    corner_count = 3 if use_tris else 4
    if nodes_per_elem == corner_count:
        return coords.copy(), np.asarray(linear_rows, dtype=np.int64)

    coords_out = coords.tolist()
    edge_nodes: dict[tuple[int, int], int] = {}
    connect_out: list[list[int]] = []
    for corners in linear_rows:
        row_out = list(corners)
        for corner_idx, node_a in enumerate(corners):
            node_b = corners[(corner_idx + 1) % corner_count]
            edge = (min(node_a, node_b), max(node_a, node_b))
            edge_node = edge_nodes.get(edge)
            if edge_node is None:
                edge_node = len(coords_out)
                edge_nodes[edge] = edge_node
                midpoint = 0.5 * (coords[node_a] + coords[node_b])
                coords_out.append(midpoint.tolist())
            row_out.append(edge_node)
        if nodes_per_elem == corner_count * 2 + 1:
            row_out.append(len(coords_out))
            centre = np.mean(coords[np.asarray(corners)], axis=0)
            coords_out.append(centre.tolist())
        connect_out.append(row_out)
    return (
        np.asarray(coords_out, dtype=np.float64),
        np.asarray(connect_out, dtype=np.int64),
    )


def _get_surf_corner_coords(
    mesh: meshconv.SimData,
    elem_type: meshconv.EElementType,
) -> np.ndarray:
    assert mesh.coords is not None
    assert mesh.connect is not None
    corner_idxs = _meshconv.ELEMENT_SPECS[elem_type].corner_idxs
    return mesh.coords[mesh.connect["connect1"][:, corner_idxs]]


def _surface_volume(coords: np.ndarray, connect: np.ndarray) -> float:
    volume = 0.0
    for row in connect:
        points = coords[row]
        for point_ind in range(1, points.shape[0] - 1):
            volume += np.dot(
                points[0],
                np.cross(points[point_ind], points[point_ind + 1]),
            ) / 6.0
    return float(volume)


def _load_cube(name: str) -> meshconv.SimData:
    return _load_native_mesh(DATA_DIR / "cubes" / name)


def _load_native_mesh(
    mesh_dir: Path,
    *,
    mesh_type: meshconv.EMeshType | None = None,
) -> meshconv.SimData:
    coords = np.loadtxt(
        mesh_dir / "coords.csv",
        delimiter=",",
        dtype=np.float64,
    )
    connect_path = mesh_dir / "connectivity.csv"
    if not connect_path.is_file():
        connect_path = mesh_dir / "connect.csv"
    connect_raw = np.loadtxt(
        connect_path,
        delimiter=",",
        dtype=np.float64,
    )
    connect = connect_raw.astype(np.int64)
    return meshconv.SimData(
        coords=coords,
        connect={"connect1": connect},
        mesh_type=mesh_type,
    )
