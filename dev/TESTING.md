# Riley Testing Architecture & Test Suites

---

## 1. Quickstart Guide

The recommended testing workflow for Riley is:

```bash
# Step 1: Run the core test suites (fast, repository-packaged references)
zig build test-verif
zig build test-basic
.venv/bin/pytest src/riley/pytests/

# Step 2: Generate reference gold data for the Full suite
zig build gen-gold-full -Doptimize=ReleaseSafe

# Step 3: Run the exhaustive Full test suite
zig build test-full -Doptimize=ReleaseSafe
```

> [!NOTE]
> Running `gen-gold-full` and `test-full` with `-Doptimize=ReleaseSafe` is strongly recommended for high throughput, generating and verifying thousands of cases in seconds while maintaining safety checks.

---

## 2. Core Repository Packaged Test Suites

Riley provides three core repository-packaged test suites intended for routine development, CI pipelines, and release verification:

| Suite Name | Execution Command | Source / Test Driver | Primary Role | Reference Data |
| :--- | :--- | :--- | :--- | :--- |
| **Verification Suite** | `zig build test-verif` | [`src/test_verif.zig`](file:///home/lloydf/riley-raster/src/test_verif.zig) | Mathematical validation against analytical Python oracles | `gold/verif/` |
| **Basic Test Suite** | `zig build test-basic` | [`src/test_basic.zig`](file:///home/lloydf/riley-raster/src/test_basic.zig) | Fast, comprehensive coverage of core rasteriser features | `gold/basic/` |
| **Python Pytest Suite** | `.venv/bin/pytest src/riley/pytests/` | [`src/riley/pytests/`](file:///home/lloydf/riley-raster/src/riley/pytests) | Python/Cython API, mesh pipeline, Exodus conversion, demo parity | Integrated / `gold/` |

### Regenerating Reference Gold for Core Suites

```bash
# Regenerate reference data for the Basic test suite
zig build gen-gold-basic

# Regenerate reference data for the Analytic Verification suite
zig build gen-gold-verif-zig
python -m gengold.gengold_verif
```

---

## 3. Test Scenes Overview

The test suites leverage standardized benchmark scenes designed to exercise specific rasteriser subsystems:

### Scene 0: Multi-Body FE Interaction & Deformation
- **Geometry**: Two interacting FE surface meshes:
  - **Sphere**: $12\text{mm}$ diameter, positioned on the left in front ($Z = 0.0\text{ mm}$).
  - **Cylinder**: $10\text{mm}$ diameter, positioned on the right behind ($Z = -10.0\text{ mm}$).
  - **Arrangement**: $20\%$ horizontal overlap of the sphere diameter ($2.4\text{mm}$) along the $X$-axis.
- **Mesh Elements**: Full support across all 5 finite element formulations (`tri3`, `tri6`, `quad4`, `quad8`, `quad9`).
- **Dynamics**: 2-frame sequence with simultaneous nonlinear spatial displacement and scalar temperature fields.
- **Camera View**: $160 \times 100\text{ px}$, $\text{SSAA} = 2$, oblique $+10^\circ$ yaw / $-10^\circ$ pitch orientation.
- **Primary Use**: [`test_full_shader.zig`](file:///home/lloydf/riley-raster/src/tests/test_full_shader.zig) and [`test_full_texture.zig`](file:///home/lloydf/riley-raster/src/tests/test_full_texture.zig).

### Scene 1: Single Cube Surface & Optical Models
- **Geometry**: Single undeformed cube surface mesh (`tri3`).
- **Shader**: Procedural sharp checkerboard shader (`.builtin = .checker`, $24 \times 24$ density) with uniform grey background ($0.5$).
- **Camera View**: $128 \times 128\text{ px}$, $95\%$ sensor frame fill factor ($5\%$ border margin), viewed obliquely from above at $+5^\circ$ yaw / $-5^\circ$ pitch orientation to inspect front, top, and side faces simultaneously.
- **Primary Use**: [`test_full_dist_psf.zig`](file:///home/lloydf/riley-raster/src/tests/test_full_dist_psf.zig) and [`test_full_ssaa_pxmap.zig`](file:///home/lloydf/riley-raster/src/tests/test_full_ssaa_pxmap.zig).

### Scene 2: Multi-Sphere Cropping & Edge Occlusion
- **Geometry**: Two interacting FE sphere surface meshes (`tri3`):
  - **Sphere 1 (Front Left)**: Positioned in the foreground at $Z = 0.0\text{ mm}$, cropped tightly by the left, top, and bottom edges of the camera sensor frame.
  - **Sphere 2 (Back Right)**: Positioned well behind the foreground sphere ($Z = -20.0\text{ mm}$), fully contained within the field of view with its top, right, and bottom edges visible while its inner edge is occluded by Sphere 1.
- **Shaders**:
  - **Sphere 1**: Procedural checkerboard shader.
  - **Sphere 2**: Direct Catmull-Rom cubic texture map.
- **Camera View**: Face-on camera view configured to crop Sphere 1 along sensor boundaries while keeping Sphere 2 uncropped.
- **Primary Use**: [`test_full_tiling.zig`](file:///home/lloydf/riley-raster/src/tests/test_full_tiling.zig).

### Scene 3: Multi-Shape FE Grid & Order Pairs
- **Geometry**: 16 finite element surface meshes arranged in 2 rows $\times$ 4 shape pairs:
  - **Shapes per Row**: 2x Cubes, 2x Spheres, 2x Cylinders, 2x Plate with hole (thickness $2.0\text{ mm}$).
  - **Pair Overlap**: Each pair overlaps $25\%$ row-wise (Y) and $75\%$ column-wise (X), with the front mesh positioned lower in front ($Z = 0.0\text{ mm}$) and the back mesh higher behind ($Z = -10.0\text{ mm}$). A thin $0.1\text{ mm}$ clearance separates adjacent pairs.
  - **Element Order Pairs**:
    - **Top Row**: Front mesh is low-order, back mesh is equivalent high-order (`tri3`/`tri6`, `quad4`/`quad8`).
    - **Bottom Row**: Front mesh is high-order, back mesh is equivalent low-order (`tri6`/`tri3`, `quad8`/`quad4`).
- **Dynamics**: 4 temporal frames (1 static baseline frame + 3 nonlinear deformation steps for all meshes).
- **Shaders**: 8 distinct 16-bit auto-scaled shaders cycled across the meshes:
  1. Nodal field interpolation (within-frame auto-scaling, u16)
  2. Procedural Checker function (u16)
  3. Direct Cubic Catmull-Rom texture sampling from u8 source (u16)
  4. Procedural Eggbox function (u16)
  5. Nodal field interpolation (over-frames auto-scaling, u16)
  6. Direct Linear texture sampling from u16 source (u16)
  7. Procedural Smooth Checker function on UV coordinates (u16)
  8. LUT-interpolated Catmull-Rom texture sampling from f64 source (u16)
- **Primary Use**: [`test_full_scene_camera_threads.zig`](file:///home/lloydf/riley-raster/src/tests/test_full_scene_camera_threads.zig).

---

## 4. Verification Oracle Suite (`test_verif`)

- **Driver**: [`src/test_verif.zig`](file:///home/lloydf/riley-raster/src/test_verif.zig)
- **Generators**: [`src/gengold/gengold_verif.py`](file:///home/lloydf/riley-raster/src/gengold/gengold_verif.py) & [`src/gen_gold_verif.zig`](file:///home/lloydf/riley-raster/src/gen_gold_verif.zig)
- **Data Location**: `gold/verif/`

The **Verification Suite** validates the mathematical foundations of the Riley rasterisation engine against closed-form analytical equations and Python oracle solvers:

1. **Newton-Raphson Shape Function Inversion**: Validates parametric coordinate $(u, v)$ convergence across linear and quadratic elements (`tri3`, `tri6`, `quad4`, `quad8`, `quad9`) and verifies analytical Jacobian derivatives.
2. **Ray-Intersection & Sub-Pixel Sampling**: Validates ray-plane and ray-quadric intersection precision across arbitrary 3D orientations.
3. **Geometric & Depth Boundaries**: Validates depth sorting ($Z$-buffering) and multi-mesh tie-breaking under exact floating-point tolerances.

---

## 5. Basic Test Suite (`test_basic`)

- **Driver**: [`src/test_basic.zig`](file:///home/lloydf/riley-raster/src/test_basic.zig)
- **Generator**: [`src/gen_gold_basic.zig`](file:///home/lloydf/riley-raster/src/gen_gold_basic.zig)
- **Data Location**: `gold/basic/`

The **Basic Suite** is the primary regression suite packaged with the repository, structured into three sub-suites:

### A. Single Element Deformation & Edge Curvature (`test_oneelem`)
- **Driver**: [`src/tests/test_gold_oneelem.zig`](file:///home/lloydf/riley-raster/src/tests/test_gold_oneelem.zig)
- **Coverage**: 10-frame continuous deformations (rotation, shear, stretch) and midside node bulging across `tri3`, `tri6`, `quad4`, `quad8`, and `quad9`, plus extreme inverted curvature edge cases.

### B. Multi-Mesh Occlusion & Shader Matrix (`test_twoshapes`)
- **Driver**: [`src/tests/test_gold_twoshapes.zig`](file:///home/lloydf/riley-raster/src/tests/test_gold_twoshapes.zig)
- **Coverage**: Multi-body occlusion between a sphere and a cube across 5 element types $\times$ 13 curated mono and RGB shaders.

### C. Complete Feature Zoo (`test_featurezoo`)
- **Driver**: [`src/tests/test_gold_featurezoo.zig`](file:///home/lloydf/riley-raster/src/tests/test_gold_featurezoo.zig)
- **Coverage**: 2 rows $\times$ 3 columns of FE simulation meshes rendered under 6 distinct camera/distortion/PSF configurations across 8 parallel thread and buffer architectures (`tile_local`, `global_subpx_full`, `global_subpx_stripe`, multi-group `in_order` and `offline`).

---

## 6. Full Test Suite (`test_full`)

- **Driver**: [`src/test_full.zig`](file:///home/lloydf/riley-raster/src/test_full.zig)
- **Generator**: [`src/gen_gold_full.zig`](file:///home/lloydf/riley-raster/src/gen_gold_full.zig)
- **Data Location**: `gold/test_full_*/` (uncommitted, generated locally)

The **Full Test Suite** is an exhaustive factorial regression suite replacing legacy benchmark and monolithic test runs. Gold reference images are generated locally on-demand and verified across 7 specialized sub-suites:

```bash
# Generate reference gold for all 7 sub-suites
zig build gen-gold-full -Doptimize=ReleaseSafe

# Run all 7 sub-suites against generated gold
zig build test-full -Doptimize=ReleaseSafe
```

### Sub-Suite 1: Full Shader Suite (`test_full_shader`)
- **Driver**: [`src/tests/test_full_shader.zig`](file:///home/lloydf/riley-raster/src/tests/test_full_shader.zig)
- **Scene**: Scene 0 (Sphere + Cylinder, 5 element types: `tri3`, `tri6`, `quad4`, `quad8`, `quad9`)
- **Coverage** (1,260 cases + 252 `tri3opt` parity checks):
  - **Nodal Shaders**: Mono & RGB scalar/vector fields, scaling strategies (`none`, `auto` within frames, `auto` over frames), normal lighting types (`none`, interpolated `.avg`, analytic `.exact`).
  - **Texture Shaders**: Mono & RGB texture sampling across `u8`, `u16`, and `f64` precisions with normal lighting options.
  - **Procedural Function Shaders**: 9 built-in functions (`constant`, `linear`, `quadratic`, `sinusoidal`, `sinusoidal_approx`, `checker`, `checker_smooth`, `lambertian_normal_z`, `eggbox`) evaluated across 4 coordinate spaces (`parametric`, `uv`, `world_reference`, `world_deformed`) with normal lighting options.
  - **`tri3opt` Parity**: Verifies that the fixed-point subpixel snapped fast triangle pipeline matches standard `tri3` floating-point reference renders within tight tolerances.

### Sub-Suite 2: Full Texture Sampling Suite (`test_full_texture`)
- **Driver**: [`src/tests/test_full_texture.zig`](file:///home/lloydf/riley-raster/src/tests/test_full_texture.zig)
- **Scene**: Scene 0 (Sphere + Cylinder, 5 element types)
- **Coverage** (600 cases):
  - **Data Types & Channels**: `u8`, `u16`, `f64` in single-channel mono and 3-channel RGB.
  - **Filter Kernels**: Direct Bilinear, Direct Catmull-Rom Cubic, and LUT-Interpolated Catmull-Rom Cubic.
  - **Edge Addressing Modes**: `clamp`, `mirror`, `repeat`, and `border`.

### Sub-Suite 3: Full Distortion & PSF Suite (`test_full_dist_psf`)
- **Driver**: [`src/tests/test_full_dist_psf.zig`](file:///home/lloydf/riley-raster/src/tests/test_full_dist_psf.zig)
- **Scene**: Scene 1 (Cube `tri3`, high-density checkerboard, grey background)
- **Coverage** (240 test cases):
  - **SSAA Levels**: $1$ (direct pixel centers) and $4$ ($16$ subpixel samples/pixel).
  - **Distortion Models (10)**: Baseline `none`, light barrel/pincushion ($k_1 = \pm 1000.0$), extreme barrel/pincushion ($k_1 = \pm 2500.0, k_2 = 1.0\times 10^7$) with Brown-Conrady and Brown-Conrady-Ext, and quadratic polynomial distortion combinations.
  - **Point Spread Functions (4)**: Baseline pixel box (`off`), separable Gaussian ($\sigma = 1.5\text{ px}$), non-separable Gaussian ($\sigma = 1.5\text{ px}$), and anisotropic Gaussian ($\sigma_x = 1.5\text{ px}, \sigma_y = 0.1\text{ px}$).
  - **Buffer Architectures (3)**: Verified for identical bit-accurate output across `tile_local`, `global_subpx_full`, and `global_subpx_stripe`.

### Sub-Suite 4: Full SSAA & Pixel Center Mapping Suite (`test_full_ssaa_pxmap`)
- **Driver**: [`src/tests/test_full_ssaa_pxmap.zig`](file:///home/lloydf/riley-raster/src/tests/test_full_ssaa_pxmap.zig)
- **Scene**: Scene 1 (Cube `tri3`)
- **Coverage** (72 cases):
  - **SSAA Levels**: $1, 2, 3, 4$ ($1, 4, 9, 16$ subpixels per pixel).
  - **Subpixel Center Mapping Engines**: `full_in_mem` (precomputed global grid), `per_tile` (on-the-fly tile evaluation), and `affine_jac` (first-order Jacobian local approximation).
  - **Distortion Models**: Brown-Conrady, Brown-Conrady-Ext, and Brown-Conrady-Polynomial.
  - **PSFs & Halos**: Pixel box and Gaussian halo filtering ($\sigma = 1.5\text{ px}$, $5\text{ px}$ halo margin).

### Sub-Suite 5: Full Hull Geometry Suite (`test_full_hull`)
- **Driver**: [`src/tests/test_full_hull.zig`](file:///home/lloydf/riley-raster/src/tests/test_full_hull.zig)
- **Scenes**: Single element scenes and continuous deformation sequences across all 5 element types (`tri3`, `tri6`, `quad4`, `quad8`, `quad9`).
- **Coverage**:
  - $128 \times 128\text{ px}$, $\text{SSAA} = 2$, procedural checkerboard shader.
  - Exact bounding hull rasterisation compared against gold references and deformation delta non-zero assertions.

### Sub-Suite 6: Full Tiling, Buffers & Threading Suite (`test_full_tiling`)
- **Driver**: [`src/tests/test_full_tiling.zig`](file:///home/lloydf/riley-raster/src/tests/test_full_tiling.zig)
- **Scene**: Scene 2 (Two overlapping/occluded spheres)
- **Coverage**:
  - **Resolutions (4)**: Non-power-of-two and prime dimensions ($31 \times 19$, $65 \times 47$, $161 \times 103$, $401 \times 251$).
  - **SSAA Levels**: $1$ and $4$.
  - **Tile Dimensions**: $16 \times 16$, $32 \times 32$, $64 \times 64$.
  - **Buffer Strategies**: `tile_local`, `global_subpx_full`, `global_subpx_stripe`.
  - **Thread Configurations**: $1$, $2$, and $4$ worker threads.
  - **Invariance Invariant**: Asserts bit-identical rendered images regardless of tile size, buffer mode, or worker count.

### Sub-Suite 7: Full Scene, Cameras & Threads Suite (`test_full_scene_camera_threads`)
- **Driver**: [`src/tests/test_full_scene_camera_threads.zig`](file:///home/lloydf/riley-raster/src/tests/test_full_scene_camera_threads.zig)
- **Scene**: Scene 3 (16 FE meshes in 2 rows $\times$ 4 shape pairs, 4 deformation frames, 8 16-bit shaders)
- **Coverage**:
  - **Cameras (8)**:
    - `cam0`: Centered face-on ($256 \times 256$, $\text{SSAA} = 1$).
    - `cam1`: Centered face-on ($256 \times 256$, $\text{SSAA} = 2$).
    - `cam2`: Cropped ($128 \times 128$, $\text{SSAA} = 2$, Anisotropic Gaussian PSF $\sigma_x = 1.2, \sigma_y = 0.2$).
    - `cam3`: Cropped ($128 \times 128$, $\text{SSAA} = 2$, Brown-Conrady Extreme distortion).
    - `cam4`: Cropped ($128 \times 128$, $\text{SSAA} = 2$, Brown-Conrady Ext + Separable Gaussian PSF $\sigma = 1.5$).
    - `cam5`: Stereo Left $+30^\circ$ yaw ($256 \times 256$, $\text{SSAA} = 2$).
    - `cam6`: Stereo Right $-30^\circ$ yaw ($256 \times 256$, $\text{SSAA} = 2$).
    - `cam7`: Oblique $+20^\circ$ yaw / $+20^\circ$ pitch ($256 \times 256$, $\text{SSAA} = 2$).
  - **Execution Configurations**: Evaluated across tile sizes ($16 \times 16, 32 \times 32, 64 \times 64$), buffer modes (`tile_local`, `global_subpx_full`, `global_subpx_stripe`), and thread counts ($1, 2, 4$ threads).
  - **Invariance Invariant**: Asserts identical bit-accurate output matching single-threaded reference gold images across all configurations.

### Automated Failure Diagnostics
When any case in the Full Test Suite encounters a regression or discrepancy exceeding tolerance:
- Failure artifacts are saved to `./fails/TEST_SUITE/TEST_CASE/`.
- Saved artifacts include:
  1. `*_actual.bmp` & `*_actual.csv`: The current render output.
  2. `*_gold.bmp` & `*_gold.csv`: The reference gold image.
  3. `*_diff.bmp`: A high-contrast visual difference map highlighting failing pixel locations.

---

## 7. Python & Cython Test Suite (`pytest`)

- **Driver**: `.venv/bin/pytest src/riley/pytests/`
- **Location**: [`src/riley/pytests/`](file:///home/lloydf/riley-raster/src/riley/pytests)

The **Python Test Suite** provides 520+ automated test cases covering:
1. **Mesh Pipeline & Conversions**: Element connectivity verification, quadratic-to-linear order reduction, surface extraction, and polygon triangulation.
2. **File I/O & Exodus Pipeline**: Multi-block Exodus II `.e` and CSV file reading/writing.
3. **Texture & UV Tools**: Image loading/saving (BMP, TIFF) and centered planar UV projection.
4. **End-to-End Demo Parity**: Verification that Python demos produce identical pixel output to the corresponding Zig demo binaries.
