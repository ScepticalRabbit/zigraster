# Riley mesh convention

Riley's rasteriser and shape functions consume the canonical mesh described
here. `riley.python.meshconv` is the authoritative implementation.

## Canonical representation

A `SimData` mesh has:

- finite, contiguous `float64` `coords` with shape `N x 3`;
- named `ElementBlock` objects, each with an explicit `EElementType`;
- contiguous `int64` connectivity with shape `E x P`, one element per row; and
- zero-based indices satisfying `0 <= index < N`.

Topology is never inferred from connectivity width or geometry, and there is no
`EMeshType`. Blocks may mix element types within one topology class, but a mesh
must not mix surface and volume blocks. Local node reordering never renumbers
global nodes or their associated fields.

## Supported element types

| Family | Types | Corner slots | Higher-order slots |
| --- | --- | --- | --- |
| Triangle | TRI3, TRI6, TRI7 | `0..2` | edges `3..5`; TRI7 centre `6` |
| Quadrilateral | QUAD4, QUAD8, QUAD9 | `0..3` | edges `4..7`; QUAD9 centre `8` |
| Tetrahedron | TET4, TET10 | `0..3` | TET10 edges `4..9` |
| Hexahedron | HEX8, HEX20, HEX27 | `0..7` | edges `8..19`; HEX27 faces `20..25`, centre `26` |

Other topologies are rejected. Degenerate elements are input errors, not cases
to repair by reordering.

## Ordering and orientation

Higher-order slots have fixed edge, face, and centre roles. Exchanging those
slots changes interpolation even when the same global node IDs are present.

Surface connectivity is consistently material-facing:

- exterior shell normals point outward, while cavity normals point into the
  void;
- adjacent faces traverse a shared edge in opposite directions;
- open planar surfaces are counter-clockwise from the material-facing side; and
- the source defines the facing side of an open non-planar surface.

Topology uses global node identity, not coordinate equality, so distinct nodes
may share coordinates at seams or poles. Non-manifold face sets have no unique
shell orientation.

TET and HEX elements require a positive right-handed signed metric. Their local
ordering defines shape-function coordinates, nodal interpolation, and extracted
surface faces.

### HEX27

Riley uses VTK HEX27 local roles:

| Slots | Role |
| --- | --- |
| `0..7` | corners |
| `8..19` | edge nodes |
| `20` | front face `(0, 1, 5, 4)` |
| `21` | right face `(1, 2, 6, 5)` |
| `22` | back face `(2, 3, 7, 6)` |
| `23` | left face `(3, 0, 4, 7)` |
| `24` | bottom face `(0, 1, 2, 3)` |
| `25` | top face `(4, 5, 6, 7)` |
| `26` | cell centre |

Exporters with another HEX27 order need an explicit permutation.

## Source boundaries and validation

At import boundaries, `SourceBlockSpec` explicitly declares element type, index
base, row- or node-major layout, and any target-to-source local-slot
permutation. `convert_source_block` produces a canonical `ElementBlock`; it also
accepts integral floating-point values from CSV text such as scientific
notation. Do not infer any of these properties from the data.

`check_mesh_convention` returns failures by block name. Use
`enforce_mesh_convention` only after constructing canonical, explicitly typed
blocks. `MeshConvention` describes, and `infer_mesh_convention` diagnoses,
geometric slot mappings for already typed blocks; neither infers topology,
index base, or table layout.

Surface extraction returns canonical, explicitly typed blocks and has no mode
to preserve source representation. Convert source blocks before extraction.

## Data generation

Emit or enforce canonical connectivity before generating UVs, displacements,
or other nodal data. A local connectivity reorder leaves global node fields in
place, but fields tied to local reference coordinates should be regenerated or
visually checked. Review render parity before updating gold images.
