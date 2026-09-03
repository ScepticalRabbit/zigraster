# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from riley import data
from riley.python import _meshconv, meshconv


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


def _mesh(
    coords: np.ndarray,
    element_type: meshconv.EElementType,
    connect: np.ndarray,
) -> meshconv.SimData:
    return meshconv.SimData(
        coords=coords,
        blocks={"connect1": meshconv.ElementBlock(element_type, connect)},
    )


def test_element_specs_are_complete_and_mapping_is_read_only() -> None:
    assert {
        spec.nodes_per_elem for spec in _meshconv.ELEMENT_SPECS.values()
    } == {3, 4, 6, 7, 8, 9, 10, 20, 27}
    assert (
        _meshconv.ELEMENT_SPECS[meshconv.EElementType.HEX27].centre_idx
        is None
    )

    with pytest.raises(TypeError):
        _meshconv.ELEMENT_SPECS[meshconv.EElementType.TRI3] = (
            _meshconv.ELEMENT_SPECS[meshconv.EElementType.TRI3]
        )


def test_check_mesh_convention_passes_for_std_quad() -> None:
    assert meshconv.check_mesh_convention(_mesh(
        _quad_coords(),
        meshconv.EElementType.QUAD4,
        np.array(((0, 1, 2, 3),), dtype=np.int64),
    )) == {}


@pytest.mark.parametrize(
    ("layout", "indexing", "offset", "dtype"),
    (
        (meshconv.EConnectLayout.ROW_MAJOR,
         meshconv.EConnectIndexing.ZERO_BASED, 0, np.int32),
        (meshconv.EConnectLayout.ROW_MAJOR,
         meshconv.EConnectIndexing.ONE_BASED, 1, np.int32),
        (meshconv.EConnectLayout.NODE_MAJOR,
         meshconv.EConnectIndexing.ZERO_BASED, 0, np.int32),
        (meshconv.EConnectLayout.NODE_MAJOR,
         meshconv.EConnectIndexing.ONE_BASED, 1, np.int32),
        (meshconv.EConnectLayout.ROW_MAJOR,
         meshconv.EConnectIndexing.ONE_BASED, 1, np.float64),
    ),
)
def test_convert_source_block_normalises_declared_source(
    layout: meshconv.EConnectLayout,
    indexing: meshconv.EConnectIndexing,
    offset: int,
    dtype: type[np.generic],
) -> None:
    source = (np.arange(4, dtype=dtype) + offset)[None, :]
    if layout is meshconv.EConnectLayout.NODE_MAJOR:
        source = source.T
    spec = meshconv.SourceBlockSpec(
        element_type=meshconv.EElementType.QUAD4,
        indexing=indexing,
        layout=layout,
    )

    block = meshconv.convert_source_block(source, node_count=4, spec=spec)

    assert block.connect.dtype == np.int64
    assert block.connect.flags.c_contiguous
    assert np.array_equal(block.connect, np.array(((0, 1, 2, 3),)))



def test_convert_source_block_rejects_float_outside_int64() -> None:
    source = np.array(((0.0, 1.0, 2.0, float(1 << 63)),))
    spec = meshconv.SourceBlockSpec(
        element_type=meshconv.EElementType.QUAD4,
        indexing=meshconv.EConnectIndexing.ZERO_BASED,
        layout=meshconv.EConnectLayout.ROW_MAJOR,
    )

    with pytest.raises(ValueError, match="outside the int64 range"):
        meshconv.convert_source_block(source, node_count=1 << 63, spec=spec)


def test_source_blocks_can_declare_different_index_bases() -> None:
    coords = np.vstack((_quad_coords(), _quad_coords() + (2.0, 0.0, 0.0)))
    blocks = {}
    for name, source, indexing in (
        ("zero_based", np.array(((0, 1, 2, 3),)),
         meshconv.EConnectIndexing.ZERO_BASED),
        ("one_based", np.array(((5, 6, 7, 8),)),
         meshconv.EConnectIndexing.ONE_BASED),
    ):
        source_spec = meshconv.SourceBlockSpec(
            element_type=meshconv.EElementType.QUAD4,
            indexing=indexing,
            layout=meshconv.EConnectLayout.ROW_MAJOR,
        )
        blocks[name] = meshconv.convert_source_block(source, 8, source_spec)

    mesh = meshconv.SimData(coords=coords, blocks=blocks)

    assert np.array_equal(
        mesh.blocks["one_based"].connect,
        np.array(((4, 5, 6, 7),)),
    )
    assert meshconv.check_mesh_convention(mesh) == {}


def test_source_block_spec_rejects_automatic_indexing() -> None:
    with pytest.raises(ValueError, match="AUTO is not explicit"):
        meshconv.SourceBlockSpec(
            element_type=meshconv.EElementType.QUAD4,
            indexing=meshconv.EConnectIndexing.AUTO,
            layout=meshconv.EConnectLayout.ROW_MAJOR,
        )


@pytest.mark.parametrize(
    ("connect", "error", "message"),
    (
        (np.arange(4, dtype=np.int64), ValueError, "2D array"),
        (np.arange(4, dtype=np.float64)[None, :], TypeError, "integer dtype"),
        (np.array(((0, 1, 2.5, 3),)), ValueError, "fractional"),
        (np.array(((0, 1, 2, -1),)), ValueError, "negative"),
        (np.array(((0, 1, 2),)), ValueError, "exactly 4 columns"),
        (np.empty((0, 4), dtype=np.int64), ValueError, "at least one"),
    ),
)
def test_element_block_strictly_validates_canonical_connectivity(
    connect: np.ndarray,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        meshconv.ElementBlock(meshconv.EElementType.QUAD4, connect)


def test_sim_data_rejects_out_of_bounds_block_connectivity() -> None:
    block = meshconv.ElementBlock(
        meshconv.EElementType.QUAD4,
        np.array(((0, 1, 2, 4),)),
    )

    with pytest.raises(ValueError, match="outside \\[0, 3\\]"):
        meshconv.SimData(coords=_quad_coords(), blocks={"surface": block})


def test_sim_data_rejects_mixed_surface_and_volume_blocks() -> None:
    blocks = {
        "surface": meshconv.ElementBlock(
            meshconv.EElementType.QUAD4,
            np.array(((0, 1, 2, 3),)),
        ),
        "volume": meshconv.ElementBlock(
            meshconv.EElementType.TET4,
            np.array(((0, 1, 2, 3),)),
        ),
    }

    with pytest.raises(ValueError, match="cannot mix surface and volume"):
        meshconv.SimData(coords=_quad_coords(), blocks=blocks)


def test_canonical_mesh_owns_validated_arrays() -> None:
    connect = np.array(((0, 1, 2, 3),), dtype=np.int64)
    damage = np.array((0.5,))
    block = meshconv.ElementBlock(
        meshconv.EElementType.QUAD4,
        connect,
        elem_vars={"damage": damage},
    )
    mesh = meshconv.SimData(
        coords=_quad_coords(),
        blocks={"surface": block},
    )

    connect[0, 0] = 3
    damage[0] = 2.0

    assert mesh.blocks["surface"] is block
    assert np.array_equal(
        mesh.blocks["surface"].connect,
        np.array(((0, 1, 2, 3),)),
    )
    assert block.elem_vars is not None
    assert np.array_equal(block.elem_vars["damage"], np.array((0.5,)))


def test_sim_data_rejects_complex_coordinates() -> None:
    coords = _quad_coords().astype(np.complex128)
    coords[0, 0] = 1.0j
    with pytest.raises(TypeError, match="real numeric"):
        _mesh(coords, meshconv.EElementType.QUAD4, np.array(((0, 1, 2, 3),)))


def test_sim_data_validates_block_local_element_variable_rows() -> None:
    block = meshconv.ElementBlock(
        meshconv.EElementType.QUAD4,
        np.array(((0, 1, 2, 3), (0, 1, 2, 3))),
        elem_vars={"damage": np.array((0.5,))},
    )

    with pytest.raises(ValueError, match="has 1 rows; expected 2"):
        meshconv.SimData(coords=_quad_coords(), blocks={"surface": block})


def test_check_mesh_convention_reports_geometric_failures_only() -> None:
    assert meshconv.check_mesh_convention(_mesh(
        _quad_coords(),
        meshconv.EElementType.QUAD4,
        np.array(((0, 3, 2, 1),)),
    ))["connect1"] == [
        meshconv.MeshCheckCode.CCW_WINDING,
        meshconv.MeshCheckCode.RIGHT_HANDED_GEOMETRY,
    ]


def test_enforce_connectivity_returns_a_canonical_table() -> None:
    connect = np.array(((0, 3, 2, 1),), dtype=np.int64)

    connect_out = meshconv.enforce_connectivity(
        _quad_coords(),
        connect,
        meshconv.EElementType.QUAD4,
    )

    assert np.array_equal(connect_out, np.array(((0, 1, 2, 3),)))
    assert connect_out.dtype == np.int64
    assert connect_out.flags.c_contiguous
    assert np.array_equal(connect, np.array(((0, 3, 2, 1),)))


def test_enforce_mesh_convention_fixes_tet_handedness() -> None:
    mesh = _mesh(
        np.array(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
             (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        ),
        meshconv.EElementType.TET4,
        np.array(((0, 2, 1, 3),), dtype=np.int64),
    )

    mesh_out = meshconv.enforce_mesh_convention(mesh)

    assert not meshconv.check_mesh_convention(mesh_out)
    assert np.array_equal(
        mesh_out.blocks["connect1"].connect,
        np.array(((0, 1, 2, 3),), dtype=np.int64),
    )


def test_enforce_returns_same_object_when_mesh_conforms() -> None:
    mesh = _mesh(
        _quad_coords(),
        meshconv.EElementType.QUAD4,
        np.array(((0, 1, 2, 3),), dtype=np.int64),
    )

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
        blocks={
            "connect_good": meshconv.ElementBlock(
                meshconv.EElementType.QUAD4, good_connect
            ),
            "connect_bad": meshconv.ElementBlock(
                meshconv.EElementType.QUAD4, bad_connect
            ),
        },
    )

    mesh_out = meshconv.enforce_mesh_convention(mesh)

    assert mesh_out is not mesh
    assert np.array_equal(
        mesh_out.blocks["connect_good"].connect,
        good_connect,
    )
    assert np.array_equal(
        mesh_out.blocks["connect_bad"].connect,
        fixed_connect,
    )
    assert np.array_equal(mesh.blocks["connect_bad"].connect, bad_connect)


def test_enforce_propagates_zero_volume_topology_errors() -> None:
    coords = np.array(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (2.0, 0.5, 0.0)),
        dtype=np.float64,
    )
    mesh = _mesh(
        coords,
        meshconv.EElementType.TRI3,
        np.array(
            ((0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)),
            dtype=np.int64,
        ),
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
    mesh = _mesh(
        coords,
        meshconv.EElementType.TRI3,
        np.array(
            ((0, 1, 2), (0, 3, 1), (0, 1, 4)),
            dtype=np.int64,
        ),
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
        spec = _meshconv.ELEMENT_SPECS[elem_type]
        duplicate = _meshconv._reverse_surf_row(duplicate, spec)
    mesh = _mesh(
        coords,
        elem_type,
        np.vstack((first, duplicate)),
    )

    mesh_out = meshconv.enforce_mesh_convention(mesh)

    connect_out = mesh_out.blocks["connect1"].connect
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
        _mesh(coords, meshconv.EElementType.HEX8, mirrored_row),
    )
    expected = {meshconv.MeshCheckCode.RIGHT_HANDED_GEOMETRY}
    assert set(report["connect1"]) == expected

    def hex_volume(row: np.ndarray) -> float:
        points = coords[row[0, :]]
        return float(np.linalg.det(np.column_stack((
            points[1] - points[0], points[3] - points[0], points[4] - points[0],
        ))))

    assert hex_volume(mirrored_row) < 0.0

    mesh = _mesh(coords, meshconv.EElementType.HEX8, mirrored_row)
    mesh_out = meshconv.enforce_mesh_convention(mesh)

    assert not meshconv.check_mesh_convention(mesh_out)
    connect_out = mesh_out.blocks["connect1"].connect
    assert hex_volume(connect_out) > 0.0
    assert np.array_equal(
        meshconv.enforce_mesh_convention(mesh_out).blocks["connect1"].connect,
        connect_out,
    )


def test_source_block_spec_reorders_source_slots() -> None:
    mesh = _load_cube("hex20")
    expected = mesh.blocks["connect1"].connect
    target_to_source = (
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        16, 17, 18, 19, 12, 13, 14, 15,
    )
    source = expected[:, np.argsort(target_to_source)]
    block = meshconv.convert_source_block(
        source,
        node_count=mesh.coords.shape[0],
        spec=meshconv.SourceBlockSpec(
            element_type=meshconv.EElementType.HEX20,
            indexing=meshconv.EConnectIndexing.ZERO_BASED,
            layout=meshconv.EConnectLayout.ROW_MAJOR,
            target_to_source_perm=target_to_source,
        ),
    )

    assert np.array_equal(block.connect, expected)


@pytest.mark.parametrize(
    ("permutation", "error", "message"),
    (
        ((0, 1, 2), ValueError, "requires 4 slots"),
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

    assert convention.get_target_to_source_perm(
        meshconv.EElementType.QUAD4,
    ) == (
        0,
        1,
        2,
        3,
    )
    with pytest.raises(TypeError):
        convention.target_to_source_perms[
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
    assert not _meshconv._check_std_node_roles(
        coords,
        _meshconv.ELEMENT_SPECS[meshconv.EElementType.TRI6],
    )


@pytest.mark.parametrize("cube_name", SUPPORTED_CUBES)
def test_std_cube_meshes_pass_and_enforcement_is_idempotent(
    cube_name: str,
) -> None:
    mesh = _load_cube(cube_name)

    assert not meshconv.check_mesh_convention(mesh)
    enforced_once = meshconv.enforce_mesh_convention(mesh)
    enforced_twice = meshconv.enforce_mesh_convention(enforced_once)

    assert not meshconv.check_mesh_convention(enforced_once)
    for name, block in enforced_once.blocks.items():
        assert np.array_equal(
            block.connect,
            enforced_twice.blocks[name].connect,
        )


def test_enforce_preserves_block_local_and_mesh_metadata() -> None:
    elem_vars = {"damage": np.array(((0.25, 0.5),))}
    mesh = meshconv.SimData(
        coords=_quad_coords(),
        blocks={
            "surface": meshconv.ElementBlock(
                meshconv.EElementType.QUAD4,
                np.array(((0, 3, 2, 1),)),
                elem_vars=elem_vars,
            ),
        },
        time=np.array((0.0, 1.0)),
        side_sets={("surface", "edge"): np.array((0,))},
        node_vars={"disp": np.arange(12).reshape(4, 3)},
        glob_vars={"load": np.array((1.0, 2.0))},
    )

    mesh_out = meshconv.enforce_mesh_convention(mesh)

    assert mesh_out is not mesh
    elem_vars_out = mesh_out.blocks["surface"].elem_vars
    assert elem_vars_out is not None
    assert np.array_equal(elem_vars_out["damage"], elem_vars["damage"])
    assert mesh_out.time is mesh.time
    assert mesh_out.side_sets is mesh.side_sets
    assert mesh_out.node_vars is mesh.node_vars
    assert mesh_out.glob_vars is mesh.glob_vars


@pytest.mark.parametrize("cube_name", SUPPORTED_CUBES)
def test_extracted_cube_surface_passes_convention_check(cube_name: str) -> None:
    assert not meshconv.check_mesh_convention(meshconv.extract_surf_mesh(
        meshconv.enforce_mesh_convention(_load_cube(cube_name)),
    ))


def test_one_hex_surface_stays_six_quad4_rows_through_check_enforce() -> None:
    coords, _ = _cube_surface(1.0, 0)
    elem_vars = {"material": np.array(((7, 8),))}
    mesh = meshconv.SimData(
        coords=coords,
        blocks={
            "hex": meshconv.ElementBlock(
                meshconv.EElementType.HEX8,
                np.arange(8, dtype=np.int64)[None, :],
                elem_vars=elem_vars,
            ),
        },
        time=np.array((0.0,)),
        node_vars={"temperature": np.arange(8, dtype=np.float64)},
        glob_vars={"load": np.array((3.0,))},
    )

    surface = meshconv.extract_surf_mesh(mesh)
    block = surface.blocks["hex"]
    connect_before = block.connect.copy()

    assert block.element_type is meshconv.EElementType.QUAD4
    assert block.connect.shape == (6, 4)
    assert meshconv.check_mesh_convention(surface) == {}
    enforced = meshconv.enforce_mesh_convention(surface)
    assert enforced is surface
    assert np.array_equal(enforced.blocks["hex"].connect, connect_before)
    assert block.elem_vars is not None
    assert np.array_equal(
        block.elem_vars["material"],
        np.repeat(elem_vars["material"], 6, axis=0),
    )
    assert surface.node_vars is not None
    assert np.array_equal(surface.node_vars["temperature"], np.arange(8))
    assert surface.time is mesh.time
    assert surface.glob_vars is mesh.glob_vars


def test_surface_extraction_clears_volume_side_sets() -> None:
    mesh = _load_cube("hex8")
    mesh.side_sets = {("surface", "connect1"): np.array((0,), dtype=np.int64)}

    assert meshconv.extract_surf_mesh(mesh).side_sets is None


def test_surface_slice_emits_explicit_surface_block_type() -> None:
    surface = meshconv.extract_surf_between(
        _load_cube("hex8"),
        point=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
    )

    assert surface.blocks["connect1"].element_type is meshconv.EElementType.QUAD4


def test_surface_slice_uses_first_three_vector_components() -> None:
    surface = meshconv.extract_surf_between(
        _load_cube("hex8"),
        point=(0.0, 0.0, 0.0, 10.0),
        normal=(0.0, 0.0, 1.0, 10.0),
    )

    assert surface.blocks["connect1"].connect.size


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
    arguments: dict[str, object] = {
        "point": (0.0, 0.0, 0.0),
        "normal": (0.0, 0.0, 1.0),
    }
    arguments[argument] = value

    with pytest.raises(ValueError, match=message):
        meshconv.extract_surf_between(_load_cube("hex8"), **arguments)


@pytest.mark.parametrize("mesh_name", SPHERE_MESHES)
def test_native_sphere_meshes_normalize_to_an_idempotent_convention(
    mesh_name: str,
) -> None:
    elem_name = mesh_name.split("_", maxsplit=1)[0].replace("newton", "")
    mesh = _load_native_mesh(
        data.sphere200_case_path(mesh_name),
        meshconv.EElementType(elem_name),
    )

    mesh_out = meshconv.enforce_mesh_convention(mesh)
    mesh_twice = meshconv.enforce_mesh_convention(mesh_out)

    assert not meshconv.check_mesh_convention(mesh_out)
    assert np.array_equal(mesh.coords, mesh_out.coords)
    for name, block in mesh_out.blocks.items():
        assert np.array_equal(block.connect, mesh_twice.blocks[name].connect)


def test_plate_with_hole_keeps_inward_bore_normals() -> None:
    """A closed plate surface must retain its material-facing bore wall."""

    mesh = _load_native_mesh(
        data.platehole_csv_case_path(),
        meshconv.EElementType.QUAD8,
    )

    assert not meshconv.check_mesh_convention(mesh)
    mesh_out = meshconv.enforce_mesh_convention(mesh)
    assert np.array_equal(
        mesh_out.blocks["connect1"].connect,
        mesh.blocks["connect1"].connect,
    )

    connect = mesh_out.blocks["connect1"].connect
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
    spec = _meshconv.ELEMENT_SPECS[elem_type]
    reversed_rows = [
        _meshconv._reverse_surf_row(row, spec) for row in connect
    ]
    mesh = _mesh(coords, elem_type, np.asarray(reversed_rows))

    mesh_out = meshconv.enforce_mesh_convention(mesh)

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

    surf = meshconv.extract_surf_mesh(meshconv.enforce_mesh_convention(mesh))

    assert not meshconv.check_mesh_convention(surf)
    connect = surf.blocks["connect1"].connect
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
    mesh = _mesh(coords, elem_type, connect)

    mesh_out = meshconv.enforce_mesh_convention(mesh)

    assert not meshconv.check_mesh_convention(mesh_out)
    connect = mesh_out.blocks["connect1"].connect
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
    mesh = _mesh(coords, elem_type, connect)

    mesh_out = meshconv.enforce_mesh_convention(mesh)

    corner_idxs = _meshconv.ELEMENT_SPECS[elem_type].corner_idxs
    corners = mesh_out.blocks["connect1"].connect[:, corner_idxs]
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
    spec = _meshconv.ELEMENT_SPECS[meshconv.EElementType.QUAD4]
    second_connect = np.asarray([
        _meshconv._reverse_surf_row(row, spec) for row in second_connect
    ])
    mesh = _mesh(
        np.vstack((first_coords, second_coords)),
        meshconv.EElementType.QUAD4,
        np.vstack((first_connect, second_connect)),
    )

    mesh_out = meshconv.enforce_mesh_convention(mesh)

    connect = mesh_out.blocks["connect1"].connect
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
        spec = _meshconv.ELEMENT_SPECS[meshconv.EElementType.QUAD4]
        connect = np.asarray([
            _meshconv._reverse_surf_row(row, spec) for row in connect_std
        ])
    mesh = _mesh(coords, meshconv.EElementType.QUAD4, connect)

    mesh_out = meshconv.enforce_mesh_convention(mesh)
    connect_out = mesh_out.blocks["connect1"].connect

    if nonplanar:
        assert np.array_equal(connect_out, connect)
    else:
        assert np.array_equal(connect_out, connect_std)
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
    surf_mesh = meshconv.extract_surf_mesh(meshconv.enforce_mesh_convention(
        _mesh(coords, meshconv.EElementType.HEX8, connect),
    ))
    return _upgrade_surf_elem(
        surf_mesh.coords,
        surf_mesh.blocks["connect1"].connect,
        elem_type,
    )


def _build_hex8_ring() -> tuple[np.ndarray, np.ndarray]:
    def get_node_idx(xx: int, yy: int, zz: int) -> int:
        return zz * 16 + yy * 4 + xx

    coords = np.asarray([
        (float(xx), float(yy), float(zz))
        for zz in range(2)
        for yy in range(4)
        for xx in range(4)
    ], dtype=np.float64)
    connect = np.asarray([
        (
            get_node_idx(xx, yy, 0),
            get_node_idx(xx + 1, yy, 0),
            get_node_idx(xx + 1, yy + 1, 0),
            get_node_idx(xx, yy + 1, 0),
            get_node_idx(xx, yy, 1),
            get_node_idx(xx + 1, yy, 1),
            get_node_idx(xx + 1, yy + 1, 1),
            get_node_idx(xx, yy + 1, 1),
        )
        for yy in range(3)
        for xx in range(3)
        if (xx, yy) != (1, 1)
    ], dtype=np.int64)
    return coords, connect


def _build_cube_ring_vol(
    elem_type: meshconv.EElementType,
) -> meshconv.SimData:
    coords, hex8_connect = _build_hex8_ring()
    if elem_type in _HIGH_ORDER_HEX_TYPES:
        coords, connect = _upgrade_hex_vol(coords, hex8_connect, elem_type)
    else:
        connect = np.asarray([
            [int(hex_row[idx]) for idx in corner_idxs]
            for hex_row in hex8_connect
            for corner_idxs in _HEX_TO_TET_CORNER_IDXS
        ], dtype=np.int64)
        if elem_type is meshconv.EElementType.TET10:
            coords, connect = _upgrade_tet_vol(coords, connect)
    return _mesh(coords, elem_type, connect)


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
        coords_out.append((0.5 * (coords[node_a] + coords[node_b])).tolist())
    return edge_node


def _upgrade_tet_vol(
    coords: np.ndarray,
    tet4_connect: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    coords_out = coords.tolist()
    edge_nodes: dict[tuple[int, int], int] = {}
    connect_out: list[list[int]] = []
    for corners in tet4_connect:
        connect_out.append(corners.tolist() + [
            _get_or_add_edge_node(
                int(corners[node_a_idx]),
                int(corners[node_b_idx]),
                coords,
                coords_out,
                edge_nodes,
            )
            for node_a_idx, node_b_idx in _TET_EDGE_CORNER_IDXS
        ])
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
        row_out = corners.tolist() + [
            _get_or_add_edge_node(
                int(corners[node_a_idx]),
                int(corners[node_b_idx]),
                coords,
                coords_out,
                edge_nodes,
            )
            for node_a_idx, node_b_idx in _HEX_EDGE_CORNER_IDXS
        ]
        if elem_type is meshconv.EElementType.HEX27:
            spec = _meshconv.ELEMENT_SPECS[elem_type]
            row_out.extend([-1] * (spec.nodes_per_elem - len(row_out)))
            for face_idx, corner_idxs in enumerate(_HEX_FACE_CORNER_IDXS):
                face_corner_nodes = [
                    int(corners[corner_idx]) for corner_idx in corner_idxs
                ]
                face_key = tuple(sorted(face_corner_nodes))
                face_node = face_nodes.get(face_key)
                if face_node is None:
                    face_node = len(coords_out)
                    face_nodes[face_key] = face_node
                    coords_out.append(np.mean(
                        coords[np.asarray(face_corner_nodes)],
                        axis=0,
                    ).tolist())
                row_out[spec.face_centre_idxs[face_idx]] = face_node
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
        corners = tuple(int(node) for node in row[:4])
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
            row_out.append(_get_or_add_edge_node(
                node_a,
                corners[(corner_idx + 1) % corner_count],
                coords,
                coords_out,
                edge_nodes,
            ))
        if nodes_per_elem == corner_count * 2 + 1:
            row_out.append(len(coords_out))
            coords_out.append(
                np.mean(coords[np.asarray(corners)], axis=0).tolist()
            )
        connect_out.append(row_out)
    return (
        np.asarray(coords_out, dtype=np.float64),
        np.asarray(connect_out, dtype=np.int64),
    )


def _get_surf_corner_coords(
    mesh: meshconv.SimData,
    elem_type: meshconv.EElementType,
) -> np.ndarray:
    corner_idxs = _meshconv.ELEMENT_SPECS[elem_type].corner_idxs
    return mesh.coords[mesh.blocks["connect1"].connect[:, corner_idxs]]


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
    return _load_native_mesh(data.cube_case_path(name), meshconv.EElementType(name))


def _load_native_mesh(
    mesh_dir: Path,
    element_type: meshconv.EElementType,
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
    connect = np.atleast_2d(connect_raw).astype(np.int64)
    return _mesh(coords, element_type, connect)
