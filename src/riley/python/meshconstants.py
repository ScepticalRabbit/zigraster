"""Element topology and ordering constants used by Riley mesh tools."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from riley.cython.riley import MeshType


class EElementType(Enum):
    """Finite-element topologies supported by Riley's mesh tools."""

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

    def calc_ref_coords(self) -> np.ndarray:
        """Return reference-node coordinates in Riley slot order."""
        return np.asarray(RILEY_REF_COORDS[self], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class RileyElementSpec:
    """Store Riley ordering and topology for one element type."""

    node_count: int
    is_surf: bool
    corner_slots: tuple[int, ...]
    reverse_slots: tuple[int, ...]
    edge_corners: tuple[tuple[int, int], ...] = ()
    surface_faces: tuple[tuple[int, ...], ...] = ()
    face_corners: tuple[tuple[int, ...], ...] = ()
    face_slots: tuple[int, ...] = ()
    centre_slot: int | None = None


RILEY_TRI_REF_COORDS = (
    (0.0, 0.0), (1.0, 0.0), (0.0, 1.0),
    (0.5, 0.0), (0.5, 0.5), (0.0, 0.5),
)
RILEY_QUAD_REF_COORDS = (
    (-1.0, -1.0), (1.0, -1.0),
    (1.0, 1.0), (-1.0, 1.0),
    (0.0, -1.0), (1.0, 0.0),
    (0.0, 1.0), (-1.0, 0.0),
)
RILEY_TET_REF_COORDS = (
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
    (0.5, 0.0, 0.0), (0.5, 0.5, 0.0),
    (0.0, 0.5, 0.0), (0.0, 0.0, 0.5),
    (0.5, 0.0, 0.5), (0.0, 0.5, 0.5),
)
RILEY_HEX_REF_COORDS = (
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
RILEY_REF_COORDS = {
    EElementType.TRI3: RILEY_TRI_REF_COORDS[:3],
    EElementType.TRI6: RILEY_TRI_REF_COORDS,
    EElementType.TRI7: RILEY_TRI_REF_COORDS
    + ((1.0 / 3.0, 1.0 / 3.0),),
    EElementType.QUAD4: RILEY_QUAD_REF_COORDS[:4],
    EElementType.QUAD8: RILEY_QUAD_REF_COORDS,
    EElementType.QUAD9: RILEY_QUAD_REF_COORDS + ((0.0, 0.0),),
    EElementType.TET4: RILEY_TET_REF_COORDS[:4],
    EElementType.TET10: RILEY_TET_REF_COORDS,
    EElementType.HEX8: RILEY_HEX_REF_COORDS[:8],
    EElementType.HEX20: RILEY_HEX_REF_COORDS[:20],
    EElementType.HEX27: RILEY_HEX_REF_COORDS,
}

RILEY_TRI_EDGES = ((0, 1), (1, 2), (2, 0))
RILEY_QUAD_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0))
RILEY_TET_EDGES = (
    (0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3),
)
RILEY_HEX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
RILEY_TET4_FACES = (
    (0, 1, 3), (1, 2, 3), (2, 0, 3), (0, 2, 1),
)
RILEY_TET10_FACES = (
    (0, 1, 3, 4, 8, 7), (1, 2, 3, 5, 9, 8),
    (2, 0, 3, 6, 7, 9), (0, 2, 1, 6, 5, 4),
)
RILEY_HEX8_FACES = (
    (0, 4, 7, 3), (1, 2, 6, 5), (0, 1, 5, 4),
    (3, 7, 6, 2), (0, 3, 2, 1), (4, 5, 6, 7),
)
RILEY_HEX20_FACES = (
    (0, 4, 7, 3, 16, 15, 19, 11),
    (1, 2, 6, 5, 9, 18, 13, 17),
    (0, 1, 5, 4, 8, 17, 12, 16),
    (3, 7, 6, 2, 19, 14, 18, 10),
    (0, 3, 2, 1, 11, 10, 9, 8),
    (4, 5, 6, 7, 12, 13, 14, 15),
)
RILEY_HEX27_FACES = tuple(
    face + (20 + idx,) for idx, face in enumerate(RILEY_HEX20_FACES)
)
RILEY_HEX_FACE_CORNERS = (
    (0, 4, 7, 3), (1, 2, 6, 5), (0, 1, 5, 4),
    (3, 7, 6, 2), (0, 3, 2, 1), (4, 5, 6, 7),
)

RILEY_ELEMENT_SPECS = {
    EElementType.TRI3: RileyElementSpec(
        3, True, (0, 1, 2), (0, 2, 1),
    ),
    EElementType.TRI6: RileyElementSpec(
        6, True, (0, 1, 2), (0, 2, 1, 5, 4, 3), RILEY_TRI_EDGES,
    ),
    EElementType.TRI7: RileyElementSpec(
        7, True, (0, 1, 2), (0, 2, 1, 5, 4, 3, 6),
        RILEY_TRI_EDGES, centre_slot=6,
    ),
    EElementType.QUAD4: RileyElementSpec(
        4, True, (0, 1, 2, 3), (0, 3, 2, 1),
    ),
    EElementType.QUAD8: RileyElementSpec(
        8, True, (0, 1, 2, 3),
        (0, 3, 2, 1, 7, 6, 5, 4), RILEY_QUAD_EDGES,
    ),
    EElementType.QUAD9: RileyElementSpec(
        9, True, (0, 1, 2, 3),
        (0, 3, 2, 1, 7, 6, 5, 4, 8),
        RILEY_QUAD_EDGES, centre_slot=8,
    ),
    EElementType.TET4: RileyElementSpec(
        4, False, (0, 1, 2, 3), (0, 2, 1, 3),
        RILEY_TET_EDGES, RILEY_TET4_FACES,
    ),
    EElementType.TET10: RileyElementSpec(
        10, False, (0, 1, 2, 3),
        (0, 2, 1, 3, 6, 5, 4, 7, 9, 8),
        RILEY_TET_EDGES, RILEY_TET10_FACES,
    ),
    EElementType.HEX8: RileyElementSpec(
        8, False, tuple(range(8)),
        (0, 3, 2, 1, 4, 7, 6, 5), RILEY_HEX_EDGES,
        RILEY_HEX8_FACES,
    ),
    EElementType.HEX20: RileyElementSpec(
        20, False, tuple(range(8)),
        (0, 3, 2, 1, 4, 7, 6, 5, 11, 10, 9, 8,
         15, 14, 13, 12, 16, 19, 18, 17),
        RILEY_HEX_EDGES, RILEY_HEX20_FACES,
    ),
    EElementType.HEX27: RileyElementSpec(
        27, False, tuple(range(8)),
        (0, 3, 2, 1, 4, 7, 6, 5, 11, 10, 9, 8,
         15, 14, 13, 12, 16, 19, 18, 17,
         22, 23, 20, 21, 24, 25, 26),
        RILEY_HEX_EDGES, RILEY_HEX27_FACES,
        RILEY_HEX_FACE_CORNERS, (20, 21, 22, 23, 24, 25), 26,
    ),
}

RILEY_SURF_TYPES_BY_NODE_COUNT = {
    3: EElementType.TRI3,
    4: EElementType.QUAD4,
    6: EElementType.TRI6,
    7: EElementType.TRI7,
    8: EElementType.QUAD8,
    9: EElementType.QUAD9,
}
VTK_TO_RILEY = {
    elem_type: tuple(range(spec.node_count))
    for elem_type, spec in RILEY_ELEMENT_SPECS.items()
}
EXODUS_TO_RILEY = {
    **VTK_TO_RILEY,
    EElementType.HEX20: (
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 16, 17, 18, 19, 12, 13, 14, 15,
    ),
    EElementType.HEX27: (
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 16, 17, 18, 19, 12, 13, 14, 15,
        23, 24, 25, 26, 21, 22, 20,
    ),
}
RILEY_VOLUME_SURFACE_TYPES = {
    EElementType.TET4: EElementType.TRI3,
    EElementType.TET10: EElementType.TRI6,
    EElementType.HEX8: EElementType.QUAD4,
    EElementType.HEX20: EElementType.QUAD8,
    EElementType.HEX27: EElementType.QUAD9,
}
RILEY_MESH_ELEMENT_TYPES = {
    MeshType.tri3: EElementType.TRI3,
    MeshType.tri3opt: EElementType.TRI3,
    MeshType.tri6: EElementType.TRI6,
    MeshType.quad4ibi: EElementType.QUAD4,
    MeshType.quad4newton: EElementType.QUAD4,
    MeshType.quad8: EElementType.QUAD8,
    MeshType.quad9: EElementType.QUAD9,
}
ELEMENT_FAMILIES = {
    EElementType.TRI3: "tri",
    EElementType.TRI6: "tri",
    EElementType.TRI7: "tri",
    EElementType.QUAD4: "quad",
    EElementType.QUAD8: "quad",
    EElementType.QUAD9: "quad",
    EElementType.TET4: "tet",
    EElementType.TET10: "tet",
    EElementType.HEX8: "hex",
    EElementType.HEX20: "hex",
    EElementType.HEX27: "hex",
}
ELEMENT_ORDERS = {
    EElementType.TRI3: 1,
    EElementType.TRI6: 2,
    EElementType.TRI7: 2,
    EElementType.QUAD4: 1,
    EElementType.QUAD8: 2,
    EElementType.QUAD9: 2,
}
ELEMENT_NODE_COUNTS = {
    EElementType.TRI3: 3,
    EElementType.TRI6: 6,
    EElementType.TRI7: 7,
    EElementType.QUAD4: 4,
    EElementType.QUAD8: 8,
    EElementType.QUAD9: 9,
}
