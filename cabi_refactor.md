# C ABI and Cython API Refactor Plan

## Compatibility policy

Riley does not yet have external users. Breaking changes to the C ABI, Cython
declarations, and Python API are explicitly allowed for this refactor.

Do not retain deprecated fields, compatibility aliases, parallel legacy
representations, or conversion shims. The goal is one clean API whose structure
matches Riley's native Zig data model as closely as is practical in C.

The public function entry points can retain their current roles, but every
caller must be rebuilt against the new header because the input structure
layouts will change.

## Audit

The following public structures are currently appropriately focused and should
remain independent:

- `CVec2U32`, `CVec2F64`, and `CVec3F64`
- typed two- and three-dimensional array views
- `CDims5Usize`
- `CImageBuffF64`

Four structures are excessively flattened:

- `CCameraInput` contains camera geometry, every distortion representation,
  and every point-spread-function representation.
- `CMeshInput` contains fields for every shader and texture storage variant.
- `CFuncShaderParams` contains the parameters for every built-in function at
  once.
- `CRasterConfig` contains scheduling, raster-buffer, solver, saving, scaling,
  and diagnostic-output settings in one structure.

The camera audit also identified two representational bugs:

- Zig permits the forward and inverse polynomial distortion maps to have
  independent orders, while the current C ABI has one shared order.
- Zig's pixel-box PSF has a support radius, while the current C conversion
  discards it and reconstructs the default.

## Camera model

Introduce focused distortion payloads corresponding to the Zig types:

- `CBrownConrady`
- `CBrownConradyExt`
- `CPolynomialMap`
- `CBidirectionalPolynomial`
- `CBrownConradyPolynomial`
- `CBrownConradyExtPolynomial`

`CPolynomialMap` owns its order and coefficient arrays. A bidirectional
polynomial contains separate optional forward and inverse maps, so their orders
are represented independently.

Represent Zig's `DistortionModel` union as a C tagged union:

```c
typedef union c_distortion_payload {
    CBrownConrady brown_conrady;
    CBrownConradyExt brown_conrady_ext;
    CBidirectionalPolynomial polynomial;
    CBrownConradyPolynomial brown_conrady_polynomial;
    CBrownConradyExtPolynomial brown_conrady_ext_polynomial;
} CDistortionPayload;

typedef struct c_distortion {
    uint32_t tag;
    CDistortionPayload payload;
} CDistortion;
```

Likewise introduce:

- `CPixelBoxPSF`
- `CGaussianPSF`
- `CAnisotropicGaussianPSF`
- `CPSFPayload`
- `CPSF`

The pixel-box payload must preserve its support radius.

Reduce `CCameraInput` to camera geometry and configuration:

```c
typedef struct c_camera_input {
    CVec2U32 pixels_num;
    CVec2F64 pixels_size;
    CVec3F64 pos_world;
    CVec3F64 rot_world;
    CVec3F64 roi_cent_world;
    double focal_length;
    uint32_t sub_sample;
    CDistortion distortion;
    CPSF psf;
    uint32_t coord_sys;
    uint32_t subpixel_center_map;
} CCameraInput;
```

Retain the `Input` suffix because this type corresponds to Zig's
`CameraInput`.

## Shader and mesh model

Introduce reusable `CScaling` and focused shader payloads:

- `CTextureShaderInput`
- `CNodalShaderInput`
- `CFunctionShaderInput`

Texture data uses an explicit storage tag and a union of the supported typed
array descriptors. Channel count is explicit and must be either one or three.

Represent Zig's `ShaderInput` as:

```c
typedef union c_shader_payload {
    CTextureShaderInput texture;
    CNodalShaderInput nodal;
    CFunctionShaderInput function;
} CShaderPayload;

typedef struct c_shader_input {
    uint32_t tag;
    CShaderPayload payload;
} CShaderInput;
```

Reduce the mesh input to:

```c
typedef struct c_mesh_input {
    uint32_t mesh_type;
    CArray2DF64 coords;
    CArray2DUsize connect;
    CArray3DF64 disp;
    CShaderInput shader;
} CMeshInput;
```

## Function shaders

Replace the universal `CFuncShaderParams` field bag with the same common-plus-
variant structure used by Zig.

Introduce:

- `CConstantParams`
- `CLinearParams`
- `CQuadraticParams`
- `CSinusoidalParams`
- `CCheckerParams`
- `CCheckerSmoothParams`
- `CLambertianParams`
- `CEggboxParams`

Place these in a tagged `CFunction` union. Keep coordinate scale, coordinate
offset, output scale, and output offset as the common portion of
`CFuncShaderParams`. Only the active built-in payload is populated.

The Python API must similarly use per-function parameter dataclasses rather
than exposing parameters for every function on one object. The function kind
and its parameter type must not be capable of disagreeing.

## Raster configuration

Split `CRasterConfig` into coherent groups:

- `CExecutionConfig` for render mode, threads, job limits, and scheduling.
- `CRasterBufferConfig` for buffer mode, tile bounds, global-subpixel tiles,
  and stripe bounds.
- `CSolverConfig` for hull and Newton seed policies.
- `CImageSaveOpts` for format, bit depth, and scaling.
- `CSaveConfig` for save strategy, image mode, overlap, frame-buffer count, and
  image-save options.
- `CFullStatsOpts` for diagnostic output selections.

`CRasterConfig` then contains these groups plus report mode and background
value. Refactor native Zig `RasterConfig` into the same logical groups so the C
ABI and Zig API do not present competing organizations.

## Cython and Python API

Mirror every C struct and union exactly in `riley.pxd`.

Replace the flat Python `Camera` parameters with explicit objects:

```python
camera = riley.Camera(
    pixels_num=...,
    pixels_size=...,
    pos_world=...,
    rot_world=...,
    roi_cent_world=...,
    focal_length=...,
    sub_sample=...,
    distortion=riley.BrownConrady(...),
    psf=riley.GaussianPSF(...),
)
```

Lower `TextureShader`, `NodalShader`, and `FunctionShader` independently into
`CShaderInput`. Restructure the Python raster configuration to follow the same
groups as C and Zig.

Remove all old flat Python fields and aliases. There are no users requiring a
deprecation period.

## ABI safety

- Use fixed-width integer fields for tags and flags.
- Zero-initialize every tagged union before filling its active payload.
- Reject unknown tags before reading a payload.
- Add compile-time Zig checks for the size, alignment, and important offsets of
  all public C structures and unions.
- Add a small C compilation test using `riley.h` and `_Static_assert` where
  appropriate.
- Export a C ABI version constant or function and increment it for future
  incompatible changes.
- Treat `riley.h` as the public contract; keep the Zig extern declarations and
  Cython `.pxd` declarations mechanically identical to it.

## Verification

Add round-trip coverage for:

- every distortion variant, including different forward/inverse polynomial
  orders;
- every PSF variant, including pixel-box support radius;
- mono and RGB textures in every supported storage type;
- scalar and RGB nodal shaders;
- every scalar and RGB function-shader built-in;
- every raster configuration group and optional/zero sentinel;
- camera and stereo camera save/load behaviour;
- invalid tags, invalid channel counts, absent required payloads, and invalid
  enum values.

Rebuild the extension and run the complete Python pytest suite. The Zig test
suites remain the user's final compatibility check unless separately requested.

## Implementation order

1. Define the new C structures and tagged unions in `riley.h`.
2. Mirror them in `c-riley.zig` and add layout assertions.
3. Rewrite distortion and PSF conversion in both directions.
4. Replace shader conversion with `CShaderInput` and reduce `CMeshInput`.
5. Replace `CFuncShaderParams` with variant payloads.
6. Split native and C raster configuration together.
7. Update `riley.pxd` and the Python/Cython lowering functions.
8. Update the public Python camera, shader, and configuration dataclasses.
9. Remove every legacy field and compatibility path.
10. Add ABI, round-trip, invalid-input, and integration tests.
11. Rebuild the Cython extension and run all Python tests.

