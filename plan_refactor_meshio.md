# Mesh I/O and Python API Refactor Plan

## Scope

Replace the current generic Python mesh-loading machinery with a small, explicit
API that accepts NumPy arrays, converts a documented source convention into
Riley's standard mesh convention, and constructs a renderer-ready `Mesh`.

This refactor is Python-only. Verification for this phase is the Riley Python
pytest suite; the Zig suites are a separate final check performed by the user.

## Public API

### CSV loading

Retain one general-purpose function:

```python
load_csv(path, *, dtype=np.float64) -> np.ndarray
```

Remove the redundant `load_coords_csv`, `load_connect_csv`, and
`load_field_csv` functions. Integer loading must reject fractional values and
overflow rather than silently truncating them.

CSV loading only parses arrays. It does not infer, verify, or alter mesh
topology.

### Texture loading

Move texture functions from `helpers.py` into `textureio.py` and expose four
strict loaders:

```python
load_texture_mono_u8(...)
load_texture_rgb_u8(...)
load_texture_mono_u16(...)
load_texture_rgb_u16(...)
```

Their outputs are C-contiguous, channel-first arrays:

- mono: `(1, height, width)`
- RGB: `(3, height, width)`
- dtype exactly matches the function name

The default behaviour is strict. Incorrect channel counts, alpha/palette
images, or incorrect bit depths raise clear exceptions. Any conversion is
explicitly requested through a combinable `TextureCoercion` enum supporting:

- RGB to mono
- mono to RGB
- unsigned 8-bit to unsigned 16-bit
- unsigned 16-bit to unsigned 8-bit

The docstrings must document the luminance rule and integer scaling used by
each coercion. Strict RGB u16 support must preserve source precision; supported
formats will be audited and either decoded losslessly or rejected explicitly.

### Raster configuration

Move `create_raster_config` from `helpers.py` into `rileyconfig.py`.

- Move imports to module scope.
- Use `SaveStrategy.both` as the default instead of the integer `2`.
- Accept enum values, not undocumented raw integers.
- Add a complete NumPy-style docstring.

Delete `helpers.py` after all imports have migrated.

### Shader inputs

Represent every supported shader explicitly in Python:

- `TextureShader`
- `NodalShader`, supporting scalar and RGB nodal fields
- `FunctionShader`, supporting scalar and RGB functions

`Mesh` becomes a small renderer-facing value containing only:

```python
Mesh(mesh_type, coords, connect, disp, shader)
```

The Cython lowering layer derives the C shader tag and storage representation
from the shader object. Shader-specific fields do not remain flattened into
`Mesh`.

### Mesh construction

Add `create_mesh` to `meshio.py`. Its public shape is:

```python
create_mesh(
    convention,
    mesh_type,
    coords,
    connect,
    disp=None,
    shader=None,
) -> Mesh
```

`convention` describes the supplied connectivity, including its source element
topology, indexing base, axes, and ordering. `mesh_type` is the desired Riley
renderer type; this distinction is necessary because Riley has renderer
variants such as `tri3opt`, `quad4ibi`, and `quad4newton` that share topology.

`create_mesh` owns the unpleasant boundary work:

1. Check all arrays for dimensionality, shape, finite values where required,
   integer safety, dtype, and contiguity.
2. Verify that the connectivity node axis agrees with the source element type.
3. Convert the connectivity table into Riley's standard convention.
4. Decide whether the requested topology transition is supported.
5. Extract a surface from a volume or reduce polynomial order when requested.
6. Preserve source node indices and use them to remap coordinates,
   displacement, UVs, and nodal shader fields together.
7. Verify shader-specific array shapes and formats.
8. Return arrays in the exact layouts and dtypes expected by the Cython ABI.

The implementation must never remap attributes by coordinate equality because
coincident nodes are legitimate at seams.

Supported reductions are:

- `tri7` to `tri6` or `tri3`
- `tri6` to `tri3`
- `quad9` to `quad8` or `quad4`
- `quad8` to `quad4`
- `tet4` to surface `tri3`
- `tet10` to surface `tri6` or `tri3`
- `hex8` to surface `quad4`
- `hex20` to surface `quad8` or `quad4`
- `hex27` to surface `quad9`, `quad8`, or `quad4`

Reject elevation, triangle/quad family changes, surface-to-volume conversion,
tetrahedron-to-quad conversion, hexahedron-to-triangle conversion, and all
other unsupported transitions with specific errors.

Displacement is supplied as `(disp_x, disp_y, disp_z)`. The default field
layout is node-major `(nodes, time)` and is lowered to Riley's
`(time, nodes, 3)` representation. All three components are required. Any
alternate layout must be described explicitly rather than guessed.

## Demo migration

Rewrite every Python demo to show the complete public workflow explicitly:

1. Load each array with `load_csv` and an explicit dtype.
2. Declare the source connectivity convention.
3. Construct the appropriate shader object.
4. Call `create_mesh`.
5. Create its output directory locally and explicitly.

Migrate:

- `demo_sphere200`
- `demo_psf`
- `demo_rabbits`
- `demo_dicuq`
- `demo_stereocal`
- `demo_dic_from_exodus`

Remove `riley.pydemos.common`. Move only genuinely reusable frame-generation
logic to a narrowly named `demoframes.py`; inline trivial output-directory
creation.

## Tests

Exercise the public `create_mesh` route rather than testing only lower-level
helpers.

### Connectivity and topology

- Identity conversion for every Riley surface renderer type.
- Row-major and column-major connectivity inputs.
- Zero-based and one-based inputs.
- Every built-in source adapter and explicit user topology.
- Every supported order reduction and volume-to-surface transition.
- The complete prohibited-conversion matrix.
- Empty connectivity, unreferenced meshes, malformed shapes, invalid indices,
  and incompatible renderer/source topology.

### Existing cube fixtures

Reuse the committed cube families for `hex8`, `hex20`, `hex27`, `tet4`, and
`tet10`. Audit their actual indexing and table orientation before declaring the
source convention; existing descriptive text may not agree with the files.
`tet14` remains unsupported.

For volume extraction, assert:

- the expected exterior face count;
- removal of internal faces;
- outward face winding;
- correct higher-order face nodes;
- exact source-node remapping.

Encode source node IDs into displacement, UV, and nodal-field fixtures so any
attribute/connectivity misalignment is immediately visible. Include coincident
seam nodes to prove that remapping uses node identity rather than coordinates.

### Shader coverage

Cover every shader family accepted by Riley:

- mono and RGB texture shaders at u8 and u16;
- scalar and RGB nodal shaders;
- scalar and RGB function shaders;
- invalid channel counts, dtypes, layouts, UVs, and field sizes.

### Integration and demos

- Pass each constructed `Mesh` through the Python raster/ROI/position entry
  points far enough to verify Cython lowering.
- Ensure every demo constructs its mesh through `create_mesh`.
- Test strict texture loading and every opt-in coercion independently and in
  supported combinations.

Run:

```bash
.venv/bin/python -m pytest --pyargs riley.pytests -s
```

Do not regenerate datasets or run the Zig/gold suites as part of this Python
refactor. The user will perform those suites as the final compatibility check.

## Implementation order

1. Finalise the shader dataclasses and Cython lowering contract.
2. Implement strict texture loading and raster configuration modules.
3. Implement `create_mesh` and source-node-preserving surface extraction.
4. Remove obsolete CSV and helper APIs.
5. Rewrite all demos.
6. Add unit, matrix, fixture, and integration tests.
7. Run the complete Python pytest suite and resolve all failures.

## Implementation state

The refactor is implemented. The Python-only mesh, shader, texture,
configuration, Cython-boundary, cube-fixture, and demo tests pass. Two cached
Zig/Python image comparisons remain stale because the sphere mesh inputs were
regenerated after the cached Zig sphere and PSF renders. Those Zig renders are
intentionally not regenerated during this Python-only phase.
