# Riley standard mesh convention

This document defines the mesh convention used by Riley's Python mesh tools
and Zig renderer. `riley.python.meshconv` is the authoritative Python
implementation. The local surface-node roles below agree with
`src/riley/zig/shapefun.zig`.

Riley distinguishes between:

- renderable surface elements: TRI3, TRI6, QUAD4, QUAD8 and QUAD9;
- Python mesh-processing elements: all renderable types plus TRI7, TET4,
  TET10, HEX8, HEX20 and HEX27.

TRI7 and the volume types are supported for mesh conversion, verification and
surface extraction. They are not accepted directly by the Zig renderer.

## Mesh representation

A Riley standard mesh contains one element type and one connectivity table.
Users with different element types provide separate meshes.

- `coords` is a finite, C-contiguous `float64` array with shape `(N, 3)`.
- `connect` is a C-contiguous `int64` array with shape `(E, P)`.
- Each connectivity row is one element; each column is one local-node slot.
- Connectivity is zero-based and every index satisfies `0 <= index < N`.
- An element row must not contain duplicate global node IDs.
- Conversion never renumbers global nodes or reorders coordinate rows.
- Nodal UVs, displacements and fields remain indexed by global node ID.

Unused coordinate rows are permitted. Surface extraction creates a new mesh,
so an extracted surface contains only referenced coordinates and has compact,
zero-based connectivity.

## Reference coordinates and surface orientation

TRI reference coordinates use `(xi, eta)`:

| Slot | TRI3/TRI6 role | Reference coordinate |
| --- | --- | --- |
| 0 | corner | `(0, 0)` |
| 1 | corner | `(1, 0)` |
| 2 | corner | `(0, 1)` |
| 3 | edge 0-1 | `(1/2, 0)` |
| 4 | edge 1-2 | `(1/2, 1/2)` |
| 5 | edge 2-0 | `(0, 1/2)` |
| 6 | TRI7 centre | `(1/3, 1/3)` |

QUAD reference coordinates use `(xi, eta)`:

| Slot | QUAD4/8/9 role | Reference coordinate |
| --- | --- | --- |
| 0 | corner | `(-1, -1)` |
| 1 | corner | `(1, -1)` |
| 2 | corner | `(1, 1)` |
| 3 | corner | `(-1, 1)` |
| 4 | edge 0-1 | `(0, -1)` |
| 5 | edge 1-2 | `(1, 0)` |
| 6 | edge 2-3 | `(0, 1)` |
| 7 | edge 3-0 | `(-1, 0)` |
| 8 | QUAD9 centre | `(0, 0)` |

Slots `0..2` or `0..3` traverse a standard surface counter-clockwise when
viewed from its material-facing side.

Surface connectivity must be consistently material-facing:

- adjacent faces traverse a shared edge in opposite directions;
- a closed exterior shell has outward-pointing normals;
- a cavity boundary has normals pointing into the void;
- an open surface preserves the material side declared by source winding;
- a caller may explicitly reverse an open surface or provide a material-normal
  hint when its source convention does not determine the desired side.

Riley tracks topology by global node identity, not coordinate equality.
Coincident-coordinate nodes at intentional seams and poles may therefore be
distinct nodes. Non-manifold face sets do not have a unique shell orientation
and are rejected by strict verification.

## Tetrahedron local roles

TET corners occupy slots `0..3`. A standard TET has positive signed volume.

TET10 edge nodes are:

| Slot | Edge |
| --- | --- |
| 4 | 0-1 |
| 5 | 1-2 |
| 6 | 2-0 |
| 7 | 0-3 |
| 8 | 1-3 |
| 9 | 2-3 |

The outward surface faces extracted from a standard TET are:

```text
(0, 1, 2)
(0, 3, 1)
(0, 2, 3)
(1, 3, 2)
```

For TET10 the corresponding edge nodes follow the three face corners.

## Hexahedron local roles

HEX corners occupy slots `0..7`. Slots `0..3` form the bottom face loop,
slots `4..7` form the corresponding top face loop, and edges `0-4`, `1-5`,
`2-6` and `3-7` connect the loops. A standard HEX has positive handedness.

HEX20 and HEX27 edge nodes are:

| Slot | Edge |
| --- | --- |
| 8 | 0-1 |
| 9 | 1-2 |
| 10 | 2-3 |
| 11 | 3-0 |
| 12 | 4-5 |
| 13 | 5-6 |
| 14 | 6-7 |
| 15 | 7-4 |
| 16 | 0-4 |
| 17 | 1-5 |
| 18 | 2-6 |
| 19 | 3-7 |

HEX27 face and cell nodes are:

| Slot | Role |
| --- | --- |
| 20 | face `(0, 1, 5, 4)` |
| 21 | face `(1, 2, 6, 5)` |
| 22 | face `(2, 3, 7, 6)` |
| 23 | face `(3, 0, 4, 7)` |
| 24 | face `(0, 1, 2, 3)` |
| 25 | face `(4, 5, 6, 7)` |
| 26 | cell centre |

The outward surface faces extracted from a standard HEX are:

```text
(0, 1, 2, 3)  bottom
(0, 3, 7, 4)  left
(4, 7, 6, 5)  top
(1, 5, 6, 2)  right
(0, 4, 5, 1)  front
(2, 6, 7, 3)  back
```

Higher-order face connectivity appends the edge nodes in boundary order and,
for HEX27, the appropriate face-centre node.

## Source conversion

CSV helpers only parse files into NumPy arrays. Mesh operations happen later.
The caller supplies a `ConnectConvention` containing:

- the element type;
- whether elements occupy rows or columns;
- whether the NumPy connectivity is zero-based or one-based;
- a built-in node-order adapter or an explicit `UserTopology`;
- optional open-surface orientation instructions.

Example using a built-in adapter:

```python
from riley.python import meshconv

convention = meshconv.ConnectConvention(
    elem_type=meshconv.EElementType.HEX20,
    elem_axis=meshconv.EConnectAxis.ROW,
    index_base=1,
    node_order=meshconv.ENodeOrder.EXODUS,
)
mesh = meshconv.convert_mesh(coords, connect, convention)
meshconv.verify_mesh(mesh)
surface = meshconv.extract_surface(mesh)
```

An unknown source convention uses `UserTopology`. It describes source slots
and source corner relationships; it does not require Riley slot numbers.

Conversion performs only declared transformations. Riley never infers the
element type, element axis, index base or source package.

## Verification

`verify_mesh(mesh)` checks an already-standard mesh. It collects independent
structural, index, topology, orientation and geometry failures and
raises one `MeshVerifyErr` containing the complete report. Checks that would
be unsafe after a structural failure are skipped.

Degenerate elements are input errors. Verification never repairs a mesh.

## Surface extraction

Surface extraction accepts one verified standard volume mesh and produces:

```text
TET4  -> TRI3
TET10 -> TRI6
HEX8  -> QUAD4
HEX20 -> QUAD8
HEX27 -> QUAD9
```

Internal shared faces are removed. The output surface is material-facing,
compact and zero-based. The HEX27 cell-centre node is not part of the surface.

## Differences from VTK

VTK's cell definitions are documented in its
[data-model cell API](https://vtk.org/doc/nightly/html/classvtkCell.html); the
HEX27 details are in
[vtkTriQuadraticHexahedron](https://vtk.org/doc/nightly/html/classvtkTriQuadraticHexahedron.html).

Riley's supported local Lagrange node roles follow VTK ordering for TRI3,
TRI6, TRI7, QUAD4, QUAD8, QUAD9, TET4, TET10, HEX8, HEX20 and HEX27. This
includes HEX27 face centres `20..25`: front, right, back, left, bottom and
top. The VTK adapter remains explicit so the source convention is recorded
and tested rather than assumed.

The mesh representations and acceptance rules differ:

- VTK unstructured grids may mix cell types in flattened connectivity with
  cell offsets and per-cell type IDs. A Riley mesh contains one element type
  in one rectangular `(elements, nodes-per-element)` table.
- VTK point IDs are zero-based in memory. Riley accepts declared zero- or
  one-based NumPy input and always produces zero-based standard connectivity.
- VTK supports many cell families and arbitrary-order cells that Riley does
  not support.
- Riley requires surface winding to represent the material-facing side and
  applies explicit exterior/cavity rules to closed shells.
- Riley performs stricter topology and geometry verification for its supported
  raster and surface-extraction paths.
- Riley's Zig renderer directly accepts only TRI3, TRI6, QUAD4, QUAD8 and
  QUAD9. Riley volume types are converted and verified in Python before their
  surfaces are extracted.

Exodus must not be treated as VTK ordering merely because an Exodus reader can
produce a VTK grid. In particular, common Exodus HEX20 connectivity groups
vertical and top edge nodes differently. The Exodus adapter performs that
explicit conversion.
