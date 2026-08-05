# Riley Developer Notes
This document collects the heavier development workflows for regression testing, gold generation and performance benchmarking. The top-level `README.md` keeps the quick-start path and the min test suite; this file is for the deeper day-to-day developer workflows.

## Extended Regression Test Suites
Once the min suite passes, the next two regression suites are:

- `all`: broader gold-regression coverage
- `bench`: benchmark-style render regressions

> NOTE: the combined `all` and `bench` gold renders take roughly 24 GB of disk space.

Before generating gold for the larger benchmark cases, generate the bench mesh data:

```shell
python ./data/bench/gen_bench_data.py
```

This populates `./data/bench/` with the larger meshes used by the benchmark oriented test cases.

Generate gold:

```shell
zig run -O ReleaseSafe ./src/gen_gold_all.zig
zig run -O ReleaseSafe ./src/gen_gold_min.zig
```

or with the build system:

```shell
zig build gen-gold -Doptimize=ReleaseSafe
zig build gen-gold-min -Doptimize=ReleaseSafe
```

Run the broader regression suites:

```shell
zig test -O ReleaseSafe ./src/test_gold_all.zig
zig test -O ReleaseSafe ./src/test_bench.zig
```

or with the build system:

```shell
zig build test-gold-all -Doptimize=ReleaseSafe
zig build test-bench -Doptimize=ReleaseSafe
```

## Precision and SIMD Build Matrix
The `zig build` workflow supports direct control over precision, SIMD mode, Newton solver mode and SIMD vector width:

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
zig build test-min -Dprecision=f64 -Dsimd=on -Doptimize=ReleaseSafe
zig build test-gold-all -Dprecision=f64 -Dsimd=on -Doptimize=ReleaseSafe
zig build test-bench -Dprecision=f64 -Dsimd=on -Doptimize=ReleaseSafe
```

You can run these in separate terminals in parallel.

For a broader matrix:

```shell
zig build test-min -Dprecision=f64 -Dsimd=on -Doptimize=ReleaseSafe
zig build test-gold-all -Dprecision=f64 -Dsimd=on -Doptimize=ReleaseSafe
zig build test-bench -Dprecision=f64 -Dsimd=on -Doptimize=ReleaseSafe

zig build test-gold-all -Dprecision=f64 -Dsimd=off -Doptimize=ReleaseSafe
zig build test-bench -Dprecision=f64 -Dsimd=off -Doptimize=ReleaseSafe

zig build test-min -Dprecision=f32 -Dsimd=on -Doptimize=ReleaseSafe
zig build test-gold-all -Dprecision=f32 -Dsimd=on -Doptimize=ReleaseSafe
zig build test-bench -Dprecision=f32 -Dsimd=on -Doptimize=ReleaseSafe

zig build test-gold-all -Dprecision=f32 -Dsimd=off -Doptimize=ReleaseSafe
zig build test-bench -Dprecision=f32 -Dsimd=off -Doptimize=ReleaseSafe
```

The min suite requires SIMD on. Scalar min orchestration is not implemented.

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
python -m riley demo_sphere200
python -m riley demo_rabbits
python -m riley demo_dicuq
python -m riley demo_dic_from_exodus
python -m riley demo_stereocal
```

Python demo output is written to `Path.cwd() / "out-riley-py" / "<demo-name>"`.

## Notes
- Plain `zig run` and `zig test` under `./src/` still use the default Riley path of `f64` with SIMD enabled.
- The public C ABI is fixed to that same production path.
- Some older benchmark helper scripts remain in `./scripts/` for historical studies. Prefer the current commands above unless you specifically need an archived workflow.
