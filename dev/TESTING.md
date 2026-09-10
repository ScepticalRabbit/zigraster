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

The **Full Test Suite** is an exhaustive factorial regression suite replacing legacy benchmark and monolithic test runs. Gold reference images are generated locally on-demand and verified across 4 specialized sub-suites:

```bash
# Generate reference gold for all 4 sub-suites
zig build gen-gold-full -Doptimize=ReleaseSafe

# Run all 4 sub-suites against generated gold
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
