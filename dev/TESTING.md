# Riley Testing Architecture & Test Suites

---

## 1. Core Repository Packaged Test Suites & Execution

Riley provides three core repository packaged test suites that are intended to be run during development, CI, and release verification:

| Suite Name | Execution Command | Source / Test Driver | Primary Role | Reference Data |
| :--- | :--- | :--- | :--- | :--- |
| **Verification Suite** | `zig build test-verif` | [`src/test_verif.zig`](file:///home/lloydf/riley-raster/src/test_verif.zig) | Exact mathematical validation against analytical Python oracles | `gold/verif/` |
| **Basic Test Suite** | `zig build test-basic` | [`src/test_basic.zig`](file:///home/lloydf/riley-raster/src/test_basic.zig) | Fast, comprehensive coverage of all core rasteriser features and threading | `gold/basic/` |
| **Python Pytest Suite** | `.venv/bin/pytest src/riley/pytests/` | [`src/riley/pytests/`](file:///home/lloydf/riley-raster/src/riley/pytests) | Python/Cython API, mesh pipeline, Exodus conversion, demo parity | Integrated / `gold/` |

### Running the Core Test Suites

```bash
# 1. Run the analytic verification oracle suite
zig build test-verif

# 2. Run the primary Basic test suite (oneelem, twoshapes, featurezoo)
zig build test-basic

# 3. Run the full Python / Cython test suite
.venv/bin/pytest src/riley/pytests/ -v
```
The Zig tests suites can also be run in `ReleaseSafe` with the compiler argument `-Doptimize=ReleaseSafe`.

### Regenerating Reference Gold Data for the Core Suites

```bash
# Regenerate reference data for the Basic test suite
zig build gen-gold-basic

# Regenerate reference data for the Analytic Verification suite
zig build gen-gold-verif-zig
python -m gengold.gengold_verif
```

---

## 2. Verification Oracle Suite (`test_verif`)

- **Driver**: [`src/test_verif.zig`](file:///home/lloydf/riley-raster/src/test_verif.zig)
- **Generator**: [`src/gengold/gengold_verif.py`](file:///home/lloydf/riley-raster/src/gengold/gengold_verif.py) & [`src/gen_gold_verif.zig`](file:///home/lloydf/riley-raster/src/gen_gold_verif.zig)
- **Data Location**: `gold/verif/`

### What It Covers
The **Verification Suite** rigorously verifies the core mathematical foundations of the Riley rasterisation engine against closed-form analytical equations and high-precision Python oracle solvers:

1. **Newton-Raphson Shape Function Inversion**:
   - Validates parametric coordinate $(u, v)$ convergence across linear and quadratic elements (`tri3`, `tri6`, `quad4`, `quad8`, `quad9`).
   - Evaluates analytic Jacobian computations against exact analytical derivatives.
2. **Ray-Intersection & Sub-Pixel Sampling**:
   - Tests ray-plane and ray-quadric intersection precision across arbitrary 3D orientations.
   - Evaluates sub-pixel sample box filtering, edge silhouette coverage, and conservative bounding hulls.
3. **Geometric & Depth Boundaries**:
   - Validates depth sorting ($Z$-buffering) and multi-mesh tie-breaking under exact floating-point tolerances without empirical heuristics.

---

## 3. Basic Test Suite (`test_basic`)

- **Driver**: [`src/test_basic.zig`](file:///home/lloydf/riley-raster/src/test_basic.zig)
- **Generator**: [`src/gen_gold_basic.zig`](file:///home/lloydf/riley-raster/src/gen_gold_basic.zig)
- **Data Location**: `gold/basic/`
- **Reference Formats**: Raw double-precision [`.fimg`](file:///home/lloydf/riley-raster/src/riley/zig/imageio.zig) for bit-accurate numerical testing and 24-bit RGB / 8-bit mono `.bmp` visualisations for inspection.

The **Basic Suite** is the primary end-to-end regression suite packaged with the repository. It is structured into three specialized sub-suites:

### A. Single Element Deformation & Edge Curvature (`test_oneelem`)
- **Driver**: [`src/tests/test_gold_oneelem.zig`](file:///home/lloydf/riley-raster/src/tests/test_gold_oneelem.zig)
- **Data Source**: [`data/edge/`](file:///home/lloydf/riley-raster/data/edge) (1-element meshes, $128 \times 128$)
- **Coverage**:
  - **10-Frame Continuous Deformation**: Rigid rotation (`distort_rot`), pure shear (`distort_shear`), and uniaxial stretch (`distort_stretch`) across all 5 element types (`tri3`, `tri6`, `quad4`, `quad8`, `quad9`).
  - **10-Frame Midside Curvature**: Nonlinear midside node bulging (`distort_bulge`) and tangent displacements (`distort_tan`) across quadratic elements (`tri6`, `quad8`, `quad9`).
  - **Extreme Inverted Curvature**: Extreme vertical bulging (`vertbulge`) and concave/convex rotation edge cases (`bulgein_rot`, `bulgeout_rot`) testing Newton solver robustness under severely distorted Jacobians.
- **Total Cases**: 228 images ($\approx 29.9\text{ MB}$ raw `.fimg`).

### B. Multi-Mesh Occlusion & Shader Matrix (`test_twoshapes`)
- **Driver**: [`src/tests/test_gold_twoshapes.zig`](file:///home/lloydf/riley-raster/src/tests/test_gold_twoshapes.zig)
- **Data Source**: [`data/shapes/cube_surf/`](file:///home/lloydf/riley-raster/data/shapes/cube_surf) and [`data/shapes/sphere_surf/`](file:///home/lloydf/riley-raster/data/shapes/sphere_surf) ($160 \times 100$, SSAA = 2)
- **Scene**: Multi-body occlusion between a $12\text{mm}$ diameter sphere on the left in front ($Z=0$) and a $10\text{mm}$ cube on the right behind ($Z=-10\text{mm}$), with a 20% horizontal overlap in $X$.
- **Camera**: Oblique $+20^\circ$ yaw / $-20^\circ$ pitch camera angle probing 3D cube edges and sphere silhouettes.
- **Shader Matrix**: 5 element types $\times$ 13 curated shaders:
  - **Mono Shaders (10)**:
    1. `nodal_grey`: Nodal scalar temperature field with auto-scaling across frames.
    2. `tex_u8_mono_catmull_direct`: 8-bit mono texture with direct Catmull-Rom cubic filtering.
    3. `tex_u8_mono_catmull_lutlerp`: 8-bit mono texture with LUT-interpolated Catmull-Rom cubic filtering.
    4. `tex_u8_mono_linear_direct`: 8-bit mono texture with direct bilinear filtering.
    5. `tex_u16_mono_catmull_direct`: 16-bit mono texture with direct Catmull-Rom cubic filtering.
    6. `tex_f64_mono_catmull_direct`: 64-bit float mono texture with direct Catmull-Rom cubic filtering.
    7. `func_checker_uvs`: Procedural checkerboard pattern probing `.coord_mode = .uv`.
    8. `func_eggbox_worldc`: Procedural eggbox pattern probing `.coord_mode = .world_reference`.
    9. `func_lambertian_normavg`: Lambertian lighting shader probing `.normal_type = .avg` (interpolated nodal normals).
    10. `func_lambertian_normexact`: Lambertian lighting shader probing `.normal_type = .exact` (analytical Jacobian normals).
  - **RGB Shaders (3)**:
    11. `nodal_rgb`: 3-channel vector field (temperature, $u_x$, $u_y$).
    12. `tex_u8_rgb_catmull_direct`: 3-channel 8-bit RGB texture with direct Catmull-Rom cubic filtering.
    13. `func_rgb_checker_uvs`: 3-channel procedural RGB checker shader probing `.coord_mode = .uv`.
- **Total Cases**: 65 test cases ($50\text{ Mono} + 15\text{ RGB} \to 95\text{ raw field } .fimg\text{ files}, \approx 12.16\text{ MB}$).

### C. Complete Feature Zoo (`test_featurezoo`)
- **Driver**: [`src/tests/test_gold_featurezoo.zig`](file:///home/lloydf/riley-raster/src/tests/test_gold_featurezoo.zig)
- **Data Source**: [`data/shapes/`](file:///home/lloydf/riley-raster/data/shapes) (Cube, Cylinder, Sphere, Plate with Hole)
- **Scene Layout**: 2 rows $\times$ 3 columns of real FE simulation meshes arranged on a $15\text{mm}$ grid to match camera aspect ratios.
- **Cameras & Optical Models** ($400 \times 250$ landscape & $250 \times 400$ portrait):
  - **Cam 0**: Orthographic face-on view (`Rotation(0,0,0)`), Box PSF.
  - **Cam 1**: Perspective tilt ($25^\circ$) with Brown-Conrady radial/tangential distortion ($k_1, k_2, p_1, p_2$).
  - **Cam 2**: Off-axis portrait tilt ($-28^\circ$) with isotropic Gaussian PSF ($\sigma = 0.65\text{ px}$).
  - **Cam 3**: Steep compound angle ($90^\circ, 25^\circ$) with Brown-Conrady distortion and Gaussian PSF.
  - **Cam 4**: Tri-axial rotation ($18^\circ, 38^\circ, 26^\circ$) with Anisotropic Gaussian PSF ($\sigma_x, \sigma_y, \theta$).
  - **Cam 5**: Reverse pitch and yaw with Brown-Conrady distortion.
- **Threading & Buffer Mode Verification Matrix** (8 architectures verified for determinism):
  1. `1grp_1geom_1rast`: 1 render group, 1 geom worker, 1 raster worker (`in_order`, `tile_local`).
  2. `1grp_4geom_4rast`: 1 render group, 4 geom workers, 4 raster workers (`in_order`, `tile_local`).
  3. `1grp_1geom_4rast_tilelocal`: 1 render group, 1 geom worker, 4 raster workers (`tile_local`).
  4. `1grp_1geom_4rast_globalsubpx`: 1 render group, 1 geom worker, 4 raster workers (`global_subpx_full`).
  5. `1grp_1geom_4rast_stripe`: 1 render group, 1 geom worker, 4 raster workers (`global_subpx_stripe`).
  6. `2grp_1geom_2rast`: 2 render groups (2 workers each), 1 geom worker, 2 raster workers (`in_order`).
  7. `4grp_1geom_1rast_inorder`: 4 render groups (1 worker each), 1 geom worker, 1 raster worker (`in_order`).
  8. `4grp_1geom_1rast_offline`: 4 render groups (1 worker each), 1 geom worker, 1 raster worker (`offline`).
- **Passes**: 12 Mono images ($6\text{ cams} \times 2\text{ frames}$) + 4 RGB images ($2\text{ cams} \times 2\text{ frames}$) verified across all 8 threading configurations.

---

## 4. Python & Cython Test Suite (`pytest`)

- **Driver**: `.venv/bin/pytest src/riley/pytests/`
- **Location**: [`src/riley/pytests/`](file:///home/lloydf/riley-raster/src/riley/pytests)

### What It Covers
The **Python Test Suite** provides 520+ automated test cases covering Python API bindings, data pipeline utilities, and Cython wrapper parity:

1. **Mesh Pipeline & Conversions** ([`test_meshconv.py`](file:///home/lloydf/riley-raster/src/riley/pytests/test_meshconv.py), [`test_mesh_conversion.py`](file:///home/lloydf/riley-raster/src/riley/pytests/test_mesh_conversion.py)):
   - Verifies finite element connectivity conventions (ABAQUS, Exodus II, Gmsh, Riley).
   - Validates quadratic-to-linear order reduction (`reduce_mesh_order`) and polygon triangulation (`triangulate_mesh`).
   - Tests automated 3D volume boundary surface extraction (`extract_surface`).
2. **File I/O & Exodus Pipeline** ([`test_meshio.py`](file:///home/lloydf/riley-raster/src/riley/pytests/test_meshio.py), [`test_exodusio.py`](file:///home/lloydf/riley-raster/src/riley/pytests/test_exodusio.py)):
   - Loads and parses multi-block Exodus II `.e` datasets, node coordinates, time-dependent displacement fields, and boundary condition sets.
   - Tests structured CSV reading and writing for geometry, connectivity, UVs, and fields.
3. **Texture & UV Tools** ([`test_textureio.py`](file:///home/lloydf/riley-raster/src/riley/pytests/test_textureio.py), [`test_uvtools.py`](file:///home/lloydf/riley-raster/src/riley/pytests/test_uvtools.py)):
   - Tests image loading/saving for TIFF, BMP, and raw arrays across 8-bit, 16-bit, and float representations.
   - Validates planar and centered UV map generation (`project_uvs_planar_centered`) for arbitrary 3D surfaces.
4. **End-to-End Demo Parity** ([`test_riley.py`](file:///home/lloydf/riley-raster/src/riley/pytests/test_riley.py)):
   - Executes standard Python demos (`demo0_quickstart` through `demo9_feature_zoo`) and validates pixel-exact parity against the corresponding Zig demo outputs.

---

## 5. Extended & Legacy Test Suites

For deep regression sweeps and benchmarking, Riley includes additional optional suites:

- **Full Gold Suite (`zig build test-gold-all`)**: Exhaustive regression suite exercising all individual legacy test cases (`small`, `simple`, `edge`, `multimesh`, `multicamera`, `hull`, `fullscreen`, `texfunc`, `ssaa`, `psf`, `sphere`).
- **Benchmark Suite (`zig build test-bench`)**: Regression suite verifying timing harness correctness, benchmark repeatability, and node/element throughput measurement across all standard benchmark meshes.
- **Min Suite (`zig build test-min`)**: Legacy SIMD-only minimal regression suite verifying sphere200 and multimesh configurations across spline LUT filter combinations.
