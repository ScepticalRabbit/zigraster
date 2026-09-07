
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from riley.cython.riley import MeshType

# --------------------------------------------------------------------------
# Sources for Elem Topology
# --------------------------------------------------------------------------
# **VTK**:
# https://examples.vtk.org/site/VTKBook/05Chapter5/
# https://vtk.org/doc/nightly/release/9.7/html/classvtkCell3D.html
# https://vtk.org/doc/nightly/html/classvtkUnstructuredGridBase.html
# vtkTetra.cxx, vtkQuadraticTetra.cxx, vtkHexahedron.cxx
# vtkQuadraticHexahedron.cxx, vtkTriQuadraticHexahedron.cxx
# vtkWedge.cxx, vtkQuadraticWedge.cxx, vtkPyramid.cxx
# vtkQuadraticPyramid.cxx, vtkTriangle.cxx, vtkQuadraticTriangle.cxx
# vtkQuad.cxx, vtkQuadraticQuad.cxx, vtkBiQuadraticQuad.cxx
#
# **Exodus**:
# https://sandialabs.github.io/seacas-docs/html/elem_types.html
# https://github.com/sandialabs/seacas/blob/master/packages/seacas/
# libraries/exodus/include/exodus-elem-types.md
#
# **Gmsh**:
# https://gmsh.info/doc/texinfo/gmsh.html#Node-ordering
# https://gmsh.info/doc/texinfo/gmsh.html#x7
# --------------------------------------------------------------------------

class EElemType(Enum):

    TRI3 = "tri3"
    TRI6 = "tri6"
    TRI7 = "tri7"
    QUAD4 = "quad4"
    QUAD8 = "quad8"
    QUAD9 = "quad9"
    TET4 = "tet4"
    TET10 = "tet10"
    HEX8 = "hex8"
    HEX20 = "hex20"
    HEX27 = "hex27"

    def get_para_coords(self) -> np.ndarray:
        return np.asarray(RILEY_PARA_COORD_MAP[self], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class RileyElemTopology:

    node_count: int
    is_surf: bool
    corner_slots: tuple[int, ...]
    reverse_slots: tuple[int, ...]
    edge_corners: tuple[tuple[int, int], ...] = ()
    surf_faces: tuple[tuple[int, ...], ...] = ()
    face_corners: tuple[tuple[int, ...], ...] = ()
    face_slots: tuple[int, ...] = ()
    centre_slot: int | None = None


# Parametric coordinates in Riley node winding order
# Shape = ((N0_xi, N0_eta), ...)
RILEY_TRI_PARA_COORD_TABLE = (
    (0.0, 0.0), (1.0, 0.0), (0.0, 1.0),
    (0.5, 0.0), (0.5, 0.5), (0.0, 0.5),
)
RILEY_QUAD_PARA_COORD_TABLE = (
    (-1.0, -1.0), (1.0, -1.0),
    (1.0, 1.0), (-1.0, 1.0),
    (0.0, -1.0), (1.0, 0.0),
    (0.0, 1.0), (-1.0, 0.0),
)
RILEY_TET_PARA_COORD_TABLE = (
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
    (0.5, 0.0, 0.0), (0.5, 0.5, 0.0),
    (0.0, 0.5, 0.0), (0.0, 0.0, 0.5),
    (0.5, 0.0, 0.5), (0.0, 0.5, 0.5),
)
RILEY_HEX_PARA_COORD_TABLE = (
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0),
    (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
    (0.5, 0.0, 0.0), (1.0, 0.5, 0.0),
    (0.5, 1.0, 0.0), (0.0, 0.5, 0.0),
    (0.5, 0.0, 1.0), (1.0, 0.5, 1.0),
    (0.5, 1.0, 1.0), (0.0, 0.5, 1.0),
    (0.0, 0.0, 0.5), (1.0, 0.0, 0.5),
    (1.0, 1.0, 0.5), (0.0, 1.0, 0.5),
    (0.0, 0.5, 0.5), (1.0, 0.5, 0.5),
    (0.5, 0.0, 0.5), (0.5, 1.0, 0.5),
    (0.5, 0.5, 0.0), (0.5, 0.5, 1.0),
    (0.5, 0.5, 0.5),
)
RILEY_PARA_COORD_MAP = {
    EElemType.TRI3: RILEY_TRI_PARA_COORD_TABLE[:3],
    EElemType.TRI6: RILEY_TRI_PARA_COORD_TABLE,
    EElemType.TRI7: RILEY_TRI_PARA_COORD_TABLE
    + ((1.0 / 3.0, 1.0 / 3.0),),
    EElemType.QUAD4: RILEY_QUAD_PARA_COORD_TABLE[:4],
    EElemType.QUAD8: RILEY_QUAD_PARA_COORD_TABLE,
    EElemType.QUAD9: (
        RILEY_QUAD_PARA_COORD_TABLE + ((0.0, 0.0),)
    ),
    EElemType.TET4: RILEY_TET_PARA_COORD_TABLE[:4],
    EElemType.TET10: RILEY_TET_PARA_COORD_TABLE,
    EElemType.HEX8: RILEY_HEX_PARA_COORD_TABLE[:8],
    EElemType.HEX20: RILEY_HEX_PARA_COORD_TABLE[:20],
    EElemType.HEX27: RILEY_HEX_PARA_COORD_TABLE,
}

# Edge node index pairs in local Riley corner order
# Shape = ((node0, node1), ...)
RILEY_TRI_EDGE_TABLE = ((0, 1), (1, 2), (2, 0))
RILEY_QUAD_EDGE_TABLE = ((0, 1), (1, 2), (2, 3), (3, 0))
RILEY_TET_EDGE_TABLE = (
    (0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3),
)
RILEY_HEX_EDGE_TABLE = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)

# Boundary face node index tuples in local Riley elem order
# Shape = ((node0, node1, ...), ...)
RILEY_TET4_FACE_TABLE = (
    (0, 1, 3), (1, 2, 3), (2, 0, 3), (0, 2, 1),
)
RILEY_TET10_FACE_TABLE = (
    (0, 1, 3, 4, 8, 7), (1, 2, 3, 5, 9, 8),
    (2, 0, 3, 6, 7, 9), (0, 2, 1, 6, 5, 4),
)
RILEY_HEX8_FACE_TABLE = (
    (0, 4, 7, 3), (1, 2, 6, 5), (0, 1, 5, 4),
    (3, 7, 6, 2), (0, 3, 2, 1), (4, 5, 6, 7),
)
RILEY_HEX20_FACE_TABLE = (
    (0, 4, 7, 3, 16, 15, 19, 11),
    (1, 2, 6, 5, 9, 18, 13, 17),
    (0, 1, 5, 4, 8, 17, 12, 16),
    (3, 7, 6, 2, 19, 14, 18, 10),
    (0, 3, 2, 1, 11, 10, 9, 8),
    (4, 5, 6, 7, 12, 13, 14, 15),
)
RILEY_HEX27_FACE_TABLE = (
    (0, 4, 7, 3, 16, 15, 19, 11, 20),
    (1, 2, 6, 5, 9, 18, 13, 17, 21),
    (0, 1, 5, 4, 8, 17, 12, 16, 22),
    (3, 7, 6, 2, 19, 14, 18, 10, 23),
    (0, 3, 2, 1, 11, 10, 9, 8, 24),
    (4, 5, 6, 7, 12, 13, 14, 15, 25),
)
RILEY_HEX_FACE_CORNER_TABLE = (
    (0, 4, 7, 3), (1, 2, 6, 5), (0, 1, 5, 4),
    (3, 7, 6, 2), (0, 3, 2, 1), (4, 5, 6, 7),
)

# Elem topology specifications
RILEY_ELEM_TOP_MAP = {
    EElemType.TRI3: RileyElemTopology(
        3, True, (0, 1, 2), (0, 2, 1),
    ),
    EElemType.TRI6: RileyElemTopology(
        6, True, (0, 1, 2), (0, 2, 1, 5, 4, 3), RILEY_TRI_EDGE_TABLE,
    ),
    EElemType.TRI7: RileyElemTopology(
        7, True, (0, 1, 2), (0, 2, 1, 5, 4, 3, 6),
        RILEY_TRI_EDGE_TABLE, centre_slot=6,
    ),
    EElemType.QUAD4: RileyElemTopology(
        4, True, (0, 1, 2, 3), (0, 3, 2, 1),
    ),
    EElemType.QUAD8: RileyElemTopology(
        8, True, (0, 1, 2, 3),
        (0, 3, 2, 1, 7, 6, 5, 4), RILEY_QUAD_EDGE_TABLE,
    ),
    EElemType.QUAD9: RileyElemTopology(
        9, True, (0, 1, 2, 3),
        (0, 3, 2, 1, 7, 6, 5, 4, 8),
        RILEY_QUAD_EDGE_TABLE, centre_slot=8,
    ),
    EElemType.TET4: RileyElemTopology(
        4, False, (0, 1, 2, 3), (0, 2, 1, 3),
        RILEY_TET_EDGE_TABLE, RILEY_TET4_FACE_TABLE,
    ),
    EElemType.TET10: RileyElemTopology(
        10, False, (0, 1, 2, 3),
        (0, 2, 1, 3, 6, 5, 4, 7, 9, 8),
        RILEY_TET_EDGE_TABLE, RILEY_TET10_FACE_TABLE,
    ),
    EElemType.HEX8: RileyElemTopology(
        8, False, tuple(range(8)),
        (0, 3, 2, 1, 4, 7, 6, 5), RILEY_HEX_EDGE_TABLE,
        RILEY_HEX8_FACE_TABLE,
    ),
    EElemType.HEX20: RileyElemTopology(
        20, False, tuple(range(8)),
        (0, 3, 2, 1, 4, 7, 6, 5, 11, 10, 9, 8,
         15, 14, 13, 12, 16, 19, 18, 17),
        RILEY_HEX_EDGE_TABLE, RILEY_HEX20_FACE_TABLE,
    ),
    EElemType.HEX27: RileyElemTopology(
        27, False, tuple(range(8)),
        (0, 3, 2, 1, 4, 7, 6, 5, 11, 10, 9, 8,
         15, 14, 13, 12, 16, 19, 18, 17,
         22, 23, 20, 21, 24, 25, 26),
        RILEY_HEX_EDGE_TABLE, RILEY_HEX27_FACE_TABLE,
        RILEY_HEX_FACE_CORNER_TABLE, (20, 21, 22, 23, 24, 25), 26,
    ),
}

# 2D surface elem lookup by node count
RILEY_SURF_TYPE_BY_NODE_COUNT_MAP = {
    3: EElemType.TRI3,
    4: EElemType.QUAD4,
    6: EElemType.TRI6,
    7: EElemType.TRI7,
    8: EElemType.QUAD8,
    9: EElemType.QUAD9,
}

# Permutation maps from external formats to Riley standard ordering
VTK_TO_RILEY_MAP: dict[EElemType, tuple[int, ...]] = {}
for elem_type, spec in RILEY_ELEM_TOP_MAP.items():
    VTK_TO_RILEY_MAP[elem_type] = tuple(range(spec.node_count))

EXODUS_TO_RILEY_MAP = {
    **VTK_TO_RILEY_MAP,
    EElemType.HEX20: (
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 16, 17, 18, 19, 12, 13, 14, 15,
    ),
    EElemType.HEX27: (
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 16, 17, 18, 19, 12, 13, 14, 15,
        23, 24, 25, 26, 21, 22, 20,
    ),
}

# Mapping of Exodus element type name strings to Riley EElemType
EXODUS_ELEM_TYPE_STR_MAP = {
    "TRI": EElemType.TRI3,
    "TRI3": EElemType.TRI3,
    "TRIANGLE": EElemType.TRI3,
    "TRI6": EElemType.TRI6,
    "TRI7": EElemType.TRI7,
    "QUAD": EElemType.QUAD4,
    "QUAD4": EElemType.QUAD4,
    "QUAD8": EElemType.QUAD8,
    "QUAD9": EElemType.QUAD9,
    "TET": EElemType.TET4,
    "TET4": EElemType.TET4,
    "TETRA": EElemType.TET4,
    "TETRA4": EElemType.TET4,
    "TET10": EElemType.TET10,
    "TETRA10": EElemType.TET10,
    "HEX": EElemType.HEX8,
    "HEX8": EElemType.HEX8,
    "HEX20": EElemType.HEX20,
    "HEX27": EElemType.HEX27,
}

EXODUS_AMBIGUOUS_TYPE_MAP = {
    ("HEX", 8): EElemType.HEX8,
    ("HEX", 20): EElemType.HEX20,
    ("HEX", 27): EElemType.HEX27,
    ("TETRA", 4): EElemType.TET4,
    ("TETRA", 10): EElemType.TET10,
    ("TET", 4): EElemType.TET4,
    ("TET", 10): EElemType.TET10,
    ("QUAD", 4): EElemType.QUAD4,
    ("QUAD", 8): EElemType.QUAD8,
    ("QUAD", 9): EElemType.QUAD9,
    ("TRI", 3): EElemType.TRI3,
    ("TRI", 6): EElemType.TRI6,
    ("TRI", 7): EElemType.TRI7,
    ("TRIANGLE", 3): EElemType.TRI3,
    ("TRIANGLE", 6): EElemType.TRI6,
    ("TRIANGLE", 7): EElemType.TRI7,
}

# Elem classification and conversion lookups
RILEY_VOL_SURF_TYPE_MAP = {
    EElemType.TET4: EElemType.TRI3,
    EElemType.TET10: EElemType.TRI6,
    EElemType.HEX8: EElemType.QUAD4,
    EElemType.HEX20: EElemType.QUAD8,
    EElemType.HEX27: EElemType.QUAD9,
}
RILEY_MESH_ELEM_TYPE_MAP = {
    MeshType.tri3: EElemType.TRI3,
    MeshType.tri3opt: EElemType.TRI3,
    MeshType.tri6: EElemType.TRI6,
    MeshType.quad4ibi: EElemType.QUAD4,
    MeshType.quad4newton: EElemType.QUAD4,
    MeshType.quad8: EElemType.QUAD8,
    MeshType.quad9: EElemType.QUAD9,
}
ELEM_FAMILY_MAP = {
    EElemType.TRI3: "tri",
    EElemType.TRI6: "tri",
    EElemType.TRI7: "tri",
    EElemType.QUAD4: "quad",
    EElemType.QUAD8: "quad",
    EElemType.QUAD9: "quad",
    EElemType.TET4: "tet",
    EElemType.TET10: "tet",
    EElemType.HEX8: "hex",
    EElemType.HEX20: "hex",
    EElemType.HEX27: "hex",
}
ELEM_ORDER_MAP = {
    EElemType.TRI3: 1,
    EElemType.TRI6: 2,
    EElemType.TRI7: 2,
    EElemType.QUAD4: 1,
    EElemType.QUAD8: 2,
    EElemType.QUAD9: 2,
    EElemType.TET4: 1,
    EElemType.TET10: 2,
    EElemType.HEX8: 1,
    EElemType.HEX20: 2,
    EElemType.HEX27: 2,
}
ELEM_NODE_COUNT_MAP = {
    EElemType.TRI3: 3,
    EElemType.TRI6: 6,
    EElemType.TRI7: 7,
    EElemType.QUAD4: 4,
    EElemType.QUAD8: 8,
    EElemType.QUAD9: 9,
    EElemType.TET4: 4,
    EElemType.TET10: 10,
    EElemType.HEX8: 8,
    EElemType.HEX20: 20,
    EElemType.HEX27: 27,
}

# Triangulation stencils to subdivide 2D elems into TRI3
# Shape = ((node0, node1, node2), ...)
RILEY_TRI_STENCIL_MAP: dict[
    EElemType, tuple[tuple[int, int, int], ...]
] = {
    EElemType.TRI3: (
        (0, 1, 2),
    ),
    EElemType.TRI6: (
        (0, 3, 5),
        (3, 1, 4),
        (5, 4, 2),
        (3, 4, 5),
    ),
    EElemType.TRI7: (
        (0, 3, 6),
        (3, 1, 6),
        (1, 4, 6),
        (4, 2, 6),
        (2, 5, 6),
        (5, 0, 6),
    ),
    EElemType.QUAD4: (
        (0, 1, 2),
        (0, 2, 3),
    ),
    EElemType.QUAD8: (
        (0, 4, 7),
        (4, 1, 5),
        (5, 2, 6),
        (6, 3, 7),
        (4, 5, 7),
        (5, 6, 7),
    ),
    EElemType.QUAD9: (
        (0, 4, 8),
        (4, 1, 8),
        (1, 5, 8),
        (5, 2, 8),
        (2, 6, 8),
        (6, 3, 8),
        (3, 7, 8),
        (7, 0, 8),
    ),
}
