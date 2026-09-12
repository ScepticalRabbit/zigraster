# Riley Developer Notes
This document collects information for developers including style guides, testing architecture, benchmark executables, and performance regression workflows.

## Style Guide
This project follows the Computer Aided Validation Laboratory style guides for Python and Zig which can be found [here](https://github.com/Computer-Aided-Validation-Laboratory/styleguides). Riley is designed around three equally important principles:

1. **Make it correct.**
2. **Make it fast.**
3. **Make it simple for users.**

## Testing Architecture & Core Packaged Suites
Riley provides a layered testing architecture designed for fast routine verification, rigorous mathematical validation, and exhaustive factorial test coverage:

### Quick Commands
```shell
# 1. Combined Verification and Basic Suites (preferred routine/CI command)
zig build test-verif-basic -Doptimize=ReleaseSafe

# 2. Analytic Verification Suite (Mathematical & Numerical Validation)
zig build test-verif -Doptimize=ReleaseSafe

# 3. Basic Test Suite (Fast Core Feature Coverage)
zig build test-basic -Doptimize=ReleaseSafe

# 4. Full Test Suite (Exhaustive Factorial System Coverage)
zig build gen-gold-full -Doptimize=ReleaseSafe  # Generate Full gold (if needed)
zig build test-full -Doptimize=ReleaseSafe

# 5. Python Integration Suite
.venv/bin/pytest src/riley/pytests/
```

### Core Suites Summary

| Suite Name | Command | Primary Role | Reference Data |
| :--- | :--- | :--- | :--- |
| **Combined Core Suite** | `zig build test-verif-basic` | Compiles and runs Verification and Basic together to avoid duplicate CI compile work | `gold/verif/`, `gold/basic/` |
| **Verification Suite** | `zig build test-verif` | Inverse solver recovery, silhouette area/centroid, depth ordering, and camera distortion oracles | `gold/verif/` |
| **Basic Suite** | `zig build test-basic` | Fast coverage across 1-element, 2-shape FE interaction, and feature zoo cases | `gold/basic/` |
| **Full Suite** | `zig build test-full` | Exhaustive sweeps over shaders, textures, PSF/distortion, SSAA, hulls, tiling, scenes, threads, and outputs | `gold/full_*/` |
| **Python Pytests** | `.venv/bin/pytest src/riley/pytests/` | Python/Cython API, mesh pipeline, Exodus conversion, and demo parity | Integrated / `gold/` |

> [!NOTE]
> For complete documentation of all 4 test scenes, 8 sub-suites, gold generation, tolerances, and design contracts, see [**`dev/TESTING.md`**](file:///home/lloydf/riley-raster/dev/TESTING.md).

## Precision and SIMD Build Matrix
The `zig build` workflow supports direct control over precision, SIMD mode, Newton solver mode, and SIMD vector width:

```shell
zig build <STEP> -Dprecision=f64 -Dsimd=on -Doptimize=ReleaseSafe
zig build <STEP> -Dprecision=f64 -Dsimd=off -Doptimize=ReleaseSafe
zig build <STEP> -Dprecision=f32 -Dsimd=on -Doptimize=ReleaseSafe
zig build <STEP> -Dprecision=f32 -Dsimd=off -Doptimize=ReleaseSafe
zig build <STEP> -Dnewton-solver=robust -Doptimize=ReleaseSafe
zig build <STEP> -Dsimd-vector-width=8 -Doptimize=ReleaseSafe
```

Suggested first-pass development checks on the main production path:

```shell
zig build test-verif -Dprecision=f64 -Dsimd=on -Doptimize=ReleaseSafe
zig build test-basic -Dprecision=f64 -Dsimd=on -Doptimize=ReleaseSafe
zig build test-full -Dprecision=f64 -Dsimd=on -Doptimize=ReleaseSafe
```

For broader matrix checks across precisions and SIMD modes:

```shell
zig build test-verif -Dprecision=f64 -Dsimd=on -Doptimize=ReleaseSafe
zig build test-basic -Dprecision=f64 -Dsimd=on -Doptimize=ReleaseSafe

zig build test-basic -Dprecision=f64 -Dsimd=off -Doptimize=ReleaseSafe
zig build test-basic -Dprecision=f32 -Dsimd=on -Doptimize=ReleaseSafe
zig build test-basic -Dprecision=f32 -Dsimd=off -Doptimize=ReleaseSafe
```

## Focused Verification Suite

The focused verification suite checks independent analytic and numerical contracts rather than broad image-output stability. It currently covers:

- inverse element-solver recovery from known parent coordinates;
- undistorted silhouette area and centroid against Python-generated analytic references;
- overlapping-rabbit depth ordering at four rear-surface separations;
- camera-distortion forward/inverse recovery from known ideal raster points.

The suite is intentionally fixed to the production `f64` configuration with SIMD enabled:

```shell
zig build test-verif -Dprecision=f64 -Dsimd=on -Doptimize=ReleaseSafe
```

The ordinary test command is Zig-only. It reads compact comparison data from
`./gold/verif/`; it does not run Python or regenerate expected results.

The depth cases place the rear rabbit at separations of one largest mesh-coordinate span,
`1/100` span, `1/1000` span, and twice the active depth-buffer tolerance. The final case is
constructed in inverse camera-depth space because Riley's depth-buffer comparison tolerance
has inverse-depth units. Each case renders the individual masks and both mesh submission
orders, then verifies the analytic front-over-rear composition pixel by pixel.

Regenerate the verification comparison data with the repository virtual environment:

```shell
zig build gen-gold-verif -Dprecision=f64 -Dsimd=on -Doptimize=ReleaseSafe
```

This first runs the Zig input generator and then
`./src/gengold/gengold_verif.py`. The Python stage independently integrates the projected
linear and quadratic element boundaries and writes only the compact analytic results beneath
`./gold/verif/`. These files remain ignored for now. If they are committed later, only the
`f64`, SIMD-enabled dataset should be added.

Changes to verification comparison data should be reviewed together with the generator and
the numerical diff. Do not regenerate comparison data as part of `test-verif`.

## Benchmark Binaries
The benchmark entry points exposed through `zig build` are:

- `bench-dicuq`
- `bench-fullraster`
- `bench-tiltraster`
- `bench-geom`
- `bench-sphere2000`
- `bench-sphere2000zoom`
- `bench-thread-geom`
- `benches`

To install benchmark binaries into `./bin/`:

```shell
zig build install-bench-fullraster -Doptimize=ReleaseFast --prefix .
zig build install-bench-tiltraster -Doptimize=ReleaseFast --prefix .
zig build install-bench-geom -Doptimize=ReleaseFast --prefix .
zig build install-bench-sphere2000 -Doptimize=ReleaseFast --prefix .
zig build install-bench-sphere2000zoom -Doptimize=ReleaseFast --prefix .
zig build install-bench-dicuq -Doptimize=ReleaseFast --prefix .
zig build install-bench-bins -Doptimize=ReleaseFast --prefix .
```

The installed binaries land in `./bin/`.

## Performance Regression Scripts
The older performance regression workflow uses the scripts in `./scripts/` with reference data under `./perf/`.

Compile the standard SIMD benchmark binaries:

```shell
python ./scripts/compile_para_simd_benchmarks.py
```

Generate local gold performance statistics:

```shell
python ./scripts/gen_gold_perf_all.py
```

Run performance regression checks against those references:

```shell
python ./scripts/test_perf_all.py
```

You can also isolate a single case:

```shell
python ./scripts/test_perf_fullraster.py
python ./scripts/test_perf_geom.py
python ./scripts/test_perf_sphere2000.py
python ./scripts/test_perf_sphere2000zoom.py
```

These scripts support perf profiles such as `1thread` and `4thread` through their `--profile` flag.

## Benchmark Experiment Orchestration
For the newer raster-performance study workflow, use:

- `scripts/compile_perf_all.py`
- `scripts/bench_perf_raster.py`

First compile the benchmark binaries used by the experiment matrix:

```shell
python ./scripts/compile_perf_all.py
```

This builds the configured `bench_tiltraster` variants into `./bin/`.

Then run the raster benchmark experiments:

```shell
python ./scripts/bench_perf_raster.py
```

By default this uses the constants defined at the top of the script, including which experiments are enabled, the run count, the active SSAA levels, and the case matrix.

Typical useful commands are:

```shell
python ./scripts/bench_perf_raster.py --runs 5
python ./scripts/bench_perf_raster.py --dry-run
python ./scripts/bench_perf_raster.py --out-root out/bench_stats_perf_manual
python ./scripts/bench_perf_raster.py --image-out-dir out/bench_images_perf_manual
```

The script writes:

- per-run experiment output under `./out/bench_stats_perf/<timestamp>/`
- rendered images under `./out/bench_images_perf/`
- a timing summary CSV such as `./out/time_bench_perf_raster_<timestamp>.csv`

The current experiment groups are driven directly by constants in the script:

- `DEFAULT_EXPERIMENT1`
- `DEFAULT_EXPERIMENT2`
- `EXPERIMENT1_SUB_SAMPLES`
- `EXPERIMENT2_SUB_SAMPLES`
- `EXPERIMENT1_CASES_BASE`

If you want to change the study matrix, edit those constants first.

## Python Parity Check
The packaged Python tests live in `src/riley/pytests/`.

Run the full packaged Python test suite with:

```shell
python -m pytest --pyargs riley.pytests -s
```

or:

```shell
python -m riley test
```

To compare the Python bindings against the Zig demo outputs specifically:

```shell
python -m pytest --pyargs riley.pytests.test_riley -s
```

To force a fresh Zig render instead of reusing cached demo BMPs:

```shell
python ./pyscripts/test_riley.py --force-zig-render
```

Run a packaged Python demo directly with:

```shell
python -m riley demo0_quickstart
python -m riley demo1_sphere200
python -m riley demo2_psf
python -m riley demo3_rabbits
python -m riley demo6_dicuq
python -m riley demo7_dic_from_exodus
python -m riley demo8_stereocal
python -m riley demo9_feature_zoo
```

Python demo output is written to `Path.cwd() / "out_riley_py" / "<demo-name>"`.

## Notes
- Plain `zig run` and `zig test` under `./src/` still use the default Riley path of `f64` with SIMD enabled.
- The public C ABI is fixed to that same production path.
- Some older benchmark helper scripts remain in `./scripts/` for historical studies. Prefer the current commands above unless you specifically need an archived workflow.
