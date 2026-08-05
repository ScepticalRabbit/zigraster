# Riley
`Riley` is a software rasteriser written in Zig for digital image correlation uncertainty quantification (DIC UQ). It renders deformed speckle pattern images from finite element simulations and supports higher order surface elements including `tri3`, `tri6`, `quad4`, `quad8` and `quad9`. `Riley` supports nodal, texture and analytic function shaders, including higher order texture sampling and mixed scenes with multiple mesh and shader types.

We chose Zig because it gives us explicit memory control, strong compile-time specialisation and direct SIMD support. `Riley` uses `comptime` to generate specialised raster paths for geometry, shader and build-policy combinations so the hot loop avoids runtime dispatch.

## Getting Started: Zig
`Riley` currently targets Zig `0.16.0`, available [here](https://ziglang.org/download/).

The first check to run is the minimal regression suite:

```shell
zig test -O ReleaseSafe ./src/test_min.zig
```

or with the build system:

```shell
zig build test-min -Doptimize=ReleaseSafe
```

Plain `zig run` and `zig test` on files under `./src/` default to the standard Riley configuration:

- precision: `f64`
- SIMD: `on`
- Newton solver: `fast`

The build system can override these defaults:

```shell
zig build <STEP> -Dprecision=f64 -Dsimd=on -Doptimize=ReleaseSafe
zig build <STEP> -Dprecision=f64 -Dsimd=off -Doptimize=ReleaseSafe
zig build <STEP> -Dprecision=f32 -Dsimd=on -Doptimize=ReleaseSafe
zig build <STEP> -Dnewton-solver=robust -Doptimize=ReleaseSafe
zig build <STEP> -Dsimd-vector-width=8 -Doptimize=ReleaseSafe
```

Run `zig build --help` to see the available build steps.

The min suite contains two high-signal cases:

- `multimesh`: two elements of each supported type in one scene, rendered with nodal and texture shading
- `sphere200`: a sphere rendered across shader combinations and element orientations

These are the quickest correctness checks and should be your first stop before running wider gold or benchmark suites.

Only the main production-path min gold is kept in the repository by default, so the min suite is intended primarily for `f64` with SIMD enabled. The min suite also requires `.simd = .on`; the scalar min orchestration is not implemented.

For deeper test, gold generation and benchmark workflows, see [dev/README.md](./dev/README.md).

## Getting Started: Python
We provide Python bindings to the Riley C ABI through Cython, and publish a `riley-raster` package on PyPI.

Install from PyPI with:

```shell
pip install riley-raster
```

This builds Riley from Zig source on your local machine, so installation can take a minute or two depending on hardware.

For local development, clone the repository, create a virtual environment and install from source:

```shell
pip install -e .
```

Python demos live in `src/riley/pydemos/` and the Python tests live in `src/riley/pytests/`.

Run the packaged Python test suite with:

```shell
python -m pytest --pyargs riley.pytests -s
```

or through Riley's module entry point:

```shell
python -m riley test
```

The repo parity test inside `riley.pytests` compares Python demo output against Zig demo output. It runs when the repository assets are available from the current working directory and skips cleanly otherwise.

## Capability Demonstration
We include several demonstration scripts in `./src/` and Python equivalents in `./src/riley/pydemos/`.

In Zig:

```shell
zig run -O ReleaseFast ./src/demo_<CASE>.zig
```

or with the build system:

```shell
zig build demo-<CASE> -Doptimize=ReleaseFast
```

where `CASE` is one of `sphere200`, `rabbits`, `dicuq`, or `stereocal`. Zig demo output is written to `./out/demo-CASE/`.

Zig demo source on GitHub:

- [`demo_sphere200.zig`](https://github.com/Computer-Aided-Validation-Laboratory/riley-raster/blob/main/src/demo_sphere200.zig)
- [`demo_rabbits.zig`](https://github.com/Computer-Aided-Validation-Laboratory/riley-raster/blob/main/src/demo_rabbits.zig)
- [`demo_dicuq.zig`](https://github.com/Computer-Aided-Validation-Laboratory/riley-raster/blob/main/src/demo_dicuq.zig)
- [`demo_stereocal.zig`](https://github.com/Computer-Aided-Validation-Laboratory/riley-raster/blob/main/src/demo_stereocal.zig)

In Python:

```shell
python -m riley demo_<CASE>
```

Python demo output is written to `Path.cwd() / "out-riley-py" / "demo-CASE"`.

Python demo source on GitHub:

- [`demo_sphere200.py`](https://github.com/Computer-Aided-Validation-Laboratory/riley-raster/blob/main/src/riley/pydemos/demo_sphere200.py)
- [`demo_rabbits.py`](https://github.com/Computer-Aided-Validation-Laboratory/riley-raster/blob/main/src/riley/pydemos/demo_rabbits.py)
- [`demo_dicuq.py`](https://github.com/Computer-Aided-Validation-Laboratory/riley-raster/blob/main/src/riley/pydemos/demo_dicuq.py)
- [`demo_dic_from_exodus.py`](https://github.com/Computer-Aided-Validation-Laboratory/riley-raster/blob/main/src/riley/pydemos/demo_dic_from_exodus.py)
- [`demo_stereocal.py`](https://github.com/Computer-Aided-Validation-Laboratory/riley-raster/blob/main/src/riley/pydemos/demo_stereocal.py)

### Demo 1: Speckle Sphere
For this demonstration we import a sphere mesh and apply a speckle pattern texture shader. This is a representative single-mesh, single-shader case.

![fig_sphere](https://raw.githubusercontent.com/Computer-Aided-Validation-Laboratory/riley-raster/main/images/demo_sphere200.bmp)

### Demo 2: Rendering Rabbits
This demonstration renders rabbit meshes composed of all supported element types: `tri3`, `tri6`, `quad4`, `quad8` and `quad9`. It also exercises the main shader families in one scene.

![fig_rabbit_render](https://raw.githubusercontent.com/Computer-Aided-Validation-Laboratory/riley-raster/main/images/demo_rabbitrender.bmp)

### Demo 3: Digital Image Correlation Uncertainty Quantification
This case demonstrates a representative stereo DIC UQ rendering of a plate with a hole in tension. The input FE model is in `./data/FE/`.

| Camera 0 | Camera 1 |
|:---:|:---:|
| ![DIC Camera 0](https://raw.githubusercontent.com/Computer-Aided-Validation-Laboratory/riley-raster/main/images/dicuq_cam0_frame0_field0.bmp) | ![DIC Camera 1](https://raw.githubusercontent.com/Computer-Aided-Validation-Laboratory/riley-raster/main/images/dicuq_cam1_frame0_field0.bmp) |

### Demo 4: Stereo Calibration
This demonstration uses the stereo setup from the DIC UQ case and renders stereo calibration target images. The input meshes are in `./data/calplate/`.

| Camera 0 | Camera 1 |
|:---:|:---:|
| ![Cal Camera 0](https://raw.githubusercontent.com/Computer-Aided-Validation-Laboratory/riley-raster/main/images/cal_cam0_frame0_field0.bmp) | ![Cal Camera 1](https://raw.githubusercontent.com/Computer-Aided-Validation-Laboratory/riley-raster/main/images/cal_cam1_frame0_field0.bmp) |

## Project Layout
The main Zig entry point for the rendering pipeline is the `raster(...)` family in `./src/riley/zig/riley.zig`.

Useful top-level locations:

- `src/`: Zig demos, tests, benchmarks and the core Riley source
- `src/riley/zig/`: core Zig implementation
- `src/riley/pydemos/`: packaged Python demos
- `src/riley/pytests/`: packaged Python tests
- `pyscripts/`: compatibility wrappers for the packaged Python demo/test entry points
- `scripts/`: benchmark and performance orchestration scripts
- `gold/`: gold reference renders
- `out/`: Zig render and benchmark output
- `out-riley-py/`: Python render output
- `dev/README.md`: detailed developer testing and benchmark notes

For a mathematical and architectural overview, see the engrXiv preprint: [Riley: A computational framework for higher-order finite element image synthesis applied to digital image correlation uncertainty quantification](https://engrxiv.org/preprint/view/7300/version/9460).

## C Interface
`Riley` provides a C-compatible API for use from other languages. The Python bindings use this interface through Cython, but it can also be called from C or from any language with a C FFI.

The public C ABI is intentionally fixed to the production Riley build with: precision=`f64`, SIMD=`on`.

This keeps the exported ABI stable for downstream callers. The extern types and functions live in [`src/riley/zig/c-riley.zig`](./src/riley/zig/c-riley.zig).

## Citing Riley
If you have found `Riley` useful you can cite it using:

> Fletcher, L., Hirst, J., and Bielajewa, W. (2026).
> *Riley: A computational framework for higher-order finite element image synthesis applied to digital image correlation uncertainty quantification*.
> engrXiv preprint. https://engrxiv.org/preprint/view/7300

```bibtex
@article{fletcher2026riley,
  title   = {Riley: A computational framework for higher-order finite element image synthesis applied to digital image correlation uncertainty quantification},
  author  = {Fletcher, Lloyd and Hirst, Joel and Bielajewa, Wiera},
  year    = {2026},
  journal = {engrXiv},
  note    = {Preprint},
  url     = {https://engrxiv.org/preprint/view/7300}
}
```

## Contributors
- Lloyd Fletcher ([ScepticalRabbit](https://github.com/ScepticalRabbit)), UK Atomic Energy Authority
- Joel Hirst ([JoelPhys](https://github.com/JoelPhys)), UK Atomic Energy Authority
- Wiera Bielajewa ([WieraB](https://github.com/WieraB)), UK Atomic Energy Authority

## Dedication
Named in memory of Riley, and for Feebee, her sister and bondmate. Without your love and support, this project would never have happened.

![Riley](https://raw.githubusercontent.com/Computer-Aided-Validation-Laboratory/riley-raster/main/images/RileyHelping.jpg)
