# Riley Testing Architecture & Test Suites

This document outlines the testing strategy, test suite organization, reference data generation, and procedures for executing test suites in the Riley codebase.

---

## 1. Testing Philosophy & Overview

Riley uses a multi-tiered testing hierarchy designed to guarantee rendering precision, numerical stability, topology invariance, multi-camera calibration correctness, and regression protection across both CPU scalar and SIMD vectorised pipelines:

```
+-------------------------------------------------------------------------------+
|                               TESTING HIERARCHY                               |
+-------------------------------------------------------------------------------+
|  1. Analytic Verification Suite  | Exact mathematical ray-intersection &      |
|     (test-verif)                 | silhouette/depth oracle verification       |
+----------------------------------+--------------------------------------------+
|  2. Basic Test Suite             | Primary lightweight suite packaged with    |
|     (test-basic)                 | the repo probing all features, shaders,    |
|                                  | cameras, and element types                 |
+----------------------------------+--------------------------------------------+
|  3. Benchmark Regression Suite   | Validates consistency and accuracy of      |
|     (test-bench)                 | all standard benchmarking workloads        |
+----------------------------------+--------------------------------------------+
|  4. Full Gold Regression Suite   | Comprehensive test matrix covering every   |
|     (test-gold-all)              | filter, shader, camera, and element config |
+----------------------------------+--------------------------------------------+
|  5. Python / Cython Suite        | End-to-end Python pipeline, parity tests,  |
|     (pytest)                     | and mesh conversion tests                  |
+----------------------------------+--------------------------------------------+
```

---

## 2. Test Suites

### A. Basic Test Suite (`test-basic`)

The **Basic Suite** is the curated, repository-packaged test suite designed to provide comprehensive feature coverage with minimal storage overhead. Reference data is stored as raw $f64$ binary images in [`.fimg`](file:///home/lloydf/riley-raster/src/riley/zig/imageio.zig) format under `gold/basic/`.

The Basic Suite consists of three sub-suites:

#### 1. Single Element Distortion & Curvature (`test_oneelem`)
- **Driver**: [`src/tests/test_gold_oneelem.zig`](file:///home/lloydf/riley-raster/src/tests/test_gold_oneelem.zig)
- **Generator**: [`src/gengold/gen_gold_oneelem.zig`](file:///home/lloydf/riley-raster/src/gengold/gen_gold_oneelem.zig)
- **Data Source**: [`data/edge/`](file:///home/lloydf/riley-raster/data/edge)
- **Resolution**: $128 \times 128$, Mono (1 channel, $f64$), SSAA = 1, Background = 127.5 (50% dynamic range mid-grey).
- **Shader**: Function Checker shader (`coord_mode = .para`, $4\times 4$ frequency).
- **Coverage**:
  - 10-frame time series for continuous deformation: `distort_rot`, `distort_shear`, `distort_stretch` across all 5 element types (`tri3`, `tri6`, `quad4`, `quad8`, `quad9`).
  - 10-frame midside deformation: `distort_bulge`, `distort_tan` across quadratic elements (`tri6`, `quad8`, `quad9`).
  - Vertbulge and inverted curvature edge cases: `vertbulge`, `bulgein_rot`, `bulgeout_rot` across quadratic elements (`tri6`, `quad8`, `quad9`).
- **Total Tests / Images**: 228 images ($\approx 29.9\text{ MB}$ raw `.fimg`).

#### 2. Multi-Mesh Occlusion & Shader Matrix (`test_twoshapes`)
- **Driver**: [`src/tests/test_gold_twoshapes.zig`](file:///home/lloydf/riley-raster/src/tests/test_gold_twoshapes.zig)
- **Generator**: [`src/gengold/gen_gold_twoshapes.zig`](file:///home/lloydf/riley-raster/src/gengold/gen_gold_twoshapes.zig)
- **Data Source**: [`data/shapes/cube_surf/`](file:///home/lloydf/riley-raster/data/shapes/cube_surf) and [`data/shapes/sphere_surf/`](file:///home/lloydf/riley-raster/data/shapes/sphere_surf)
- **Resolution**: $160 \times 100$, SSAA = 2, Camera 0 (Oblique $+20^\circ$ yaw / $-20^\circ$ pitch up and to the right, auto-zoomed to 90% bounds), Background = 127.5 (50% dynamic range mid-grey).
- **Scene**: Two bodies — Sphere on the left in front ($Z=0$), Cube on the right behind ($Z=-10\text{mm}$, full shape depth offset), with a 10% overlap in X to test depth buffer sorting and boundary rasterisation.
- **Shader Matrix**: 5 element types $\times$ 9 curated shaders:
  - **Mono Shaders (6)**:
    1. `nodal_grey`: Nodal scalar temperature field with auto-scaling over frames.
    2. `tex_u8_linear`: Speckle texture with direct linear filtering.
    3. `tex_u8_cubic`: Speckle texture with LUT-interpolated Catmull-Rom cubic filtering.
    4. `tex_u8_bspline`: Speckle texture with LUT-interpolated B-spline filtering.
    5. `func_checker`: Parametric function checker shader.
    6. `func_eggbox`: World-coordinate eggbox function shader.
  - **RGB Shaders (3)**:
    7. `nodal_rgb`: 3-channel nodal vector field (temperature, $u_x$, $u_y$).
    8. `tex_rgb_cubic`: 3-channel RGB speckle texture with Catmull-Rom cubic filtering.
    9. `func_rgb_checker`: 3-channel parametric RGB checker shader.
- **Total Tests / Images**: 45 images (30 Mono, 15 RGB $\to \approx 9.6\text{ MB}$ raw `.fimg`).

#### 3. Complete Feature Zoo (`test_featurezoo`)
- **Driver**: [`src/tests/test_gold_featurezoo.zig`](file:///home/lloydf/riley-raster/src/tests/test_gold_featurezoo.zig)
- **Generator**: [`src/gengold/gen_gold_featurezoo.zig`](file:///home/lloydf/riley-raster/src/gengold/gen_gold_featurezoo.zig)
- **Data Source**: [`data/shapes/`](file:///home/lloydf/riley-raster/data/shapes) (Cube, Cylinder, Sphere, Plate with Hole).
- **Resolution**: $400 \times 250$ (and $250 \times 400$ for portrait aspect cameras).
- **Frames**: 2 frames per camera: Frame 0 (undeformed reference state) and Frame 1 (deformed state).
- **Cameras**:
  - **Cam 0**: Face-On orthographic alignment (`Rotation.init(0, 0, 0)`), SSAA = 1, Box PSF.
  - **Cam 1**: Perspective tilt ($25^\circ$) with Brown-Conrady radial/tangential distortion, SSAA = 2.
  - **Cam 2**: Off-axis portrait tilt ($-28^\circ$) with isotropic Gaussian PSF ($\sigma = 0.65\text{ px}$), SSAA = 2.
  - **Cam 3**: Steep compound angle ($90^\circ, 25^\circ$) with Brown distortion and Gaussian PSF, SSAA = 2.
  - **Cam 4**: Tri-axial rotation ($18^\circ, 38^\circ, 26^\circ$) with Anisotropic Gaussian PSF, SSAA = 2.
  - **Cam 5**: Reverse pitch and yaw with Brown distortion, SSAA = 2.
- **Threading & Buffer Mode Verification Matrix** (8 combinations tested against gold for both Mono & RGB):
  1. `1grp_1geom_1rast`: 1 render group, 1 geom worker, 1 raster worker (`in_order`, `tile_local`).
  2. `1grp_4geom_4rast`: 1 render group, 4 geom workers, 4 raster workers (`in_order`, `tile_local`).
  3. `1grp_1geom_4rast_tilelocal`: 1 render group, 1 geom worker, 4 raster workers (`tile_local`).
  4. `1grp_1geom_4rast_globalsubpx`: 1 render group, 1 geom worker, 4 raster workers (`global_subpx_full`).
  5. `1grp_1geom_4rast_stripe`: 1 render group, 1 geom worker, 4 raster workers (`global_subpx_stripe`).
  6. `2grp_1geom_2rast`: 2 render groups (2 workers each), 1 geom worker, 2 raster workers (`in_order`).
  7. `4grp_1geom_1rast_inorder`: 4 render groups (1 worker each), 1 geom worker, 1 raster worker (`in_order`).
  8. `4grp_1geom_1rast_offline`: 4 render groups (1 worker each), 1 geom worker, 1 raster worker (`offline`).
- **Passes**:
  - **Mono Pass**: All 6 cameras $\times$ 2 frames ($12\text{ images} \to 9.60\text{ MB}$) across all 8 threading configurations.
  - **RGB Pass**: 2 representative cameras (Cam 0 Face-On + Cam 1 Perspective) $\times$ 2 frames ($4\text{ images} \to 9.60\text{ MB}$) across all 8 threading configurations.
- **Total Reference Images**: 16 images ($\approx 19.2\text{ MB}$ raw `.fimg`).

---

### B. Verification Oracle Suite (`test-verif`)
- **Driver**: [`src/test_verif.zig`](file:///home/lloydf/riley-raster/src/test_verif.zig)
- **Generator**: [`src/gengold/gengold_verif.py`](file:///home/lloydf/riley-raster/src/gengold/gengold_verif.py) and [`src/gen_gold_verif.zig`](file:///home/lloydf/riley-raster/src/gen_gold_verif.zig)
- Validates geometric accuracy, Newton solver convergence, depth ordering, and edge silhouettes against pure analytical Python oracles.

### C. Full Gold Suite (`test-gold-all`)
- **Driver**: [`src/test_gold_all.zig`](file:///home/lloydf/riley-raster/src/test_gold_all.zig)
- **Generator**: [`src/gen_gold_all.zig`](file:///home/lloydf/riley-raster/src/gen_gold_all.zig)
- Comprehensive regression suite exercising all individual components (`small`, `simple`, `edge`, `multimesh`, `multicamera`, `hull`, `fullscreen`, `texfunc`, `ssaa`, `psf`, `sphere`).

---

## 3. Running Tests & Generating Reference Data

### Running Zig Tests

```bash
# Run the primary Basic test suite
zig build test-basic

# Run the analytic verification suite
zig build test-verif

# Run the full regression test suite
zig build test-gold-all

# Run the benchmark test suite
zig build test-bench
```

### Generating Gold Reference Datasets

```bash
# Generate reference gold and visual TIFFs for the Basic test suite
zig build gen-gold-basic

# Generate reference gold for the analytic verification suite
zig build gen-gold-verif

# Generate reference gold for all suites
zig build gen-gold-all
```

### Running Python / Cython Tests

```bash
# Run full Python test suite with pytest
.venv/bin/pytest src/riley/pytests/ -v
```

---

## 4. Visual Validation Artifacts

When running `zig build gen-gold-basic`, the generator produces two files for every rendered frame:
1. **`.fimg`**: The canonical double-precision float image used for exact floating-point comparisons during test runs.
2. **`.tiff`**: Auto-scaled 8-bit visualization images saved alongside the gold files for manual visual inspection of field gradients, texture filtering, and boundary antialiasing.
