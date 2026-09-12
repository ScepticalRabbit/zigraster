# Riley Focused Verification Gold

This directory contains compact analytic comparison data for `zig build test-verif`.

The current dataset is generated and supported only for Riley's production configuration:

- precision: `f64`;
- SIMD: enabled;
- Newton solver: fast.

`silhouette_metrics.csv` has no header so it can be loaded by Riley's scalar-grid CSV reader.
Rows follow the focused case order in `src/verif_2_silhouette.zig`; columns are analytic
centroid x in pixels, analytic centroid y in pixels, and analytic area in square pixels.

Regenerate the data from the repository root with:

```shell
zig build gen-gold-verif -Dprecision=f64 -Dsimd=on -Doptimize=ReleaseSafe
```

The Zig stage writes selected rendered images and projected boundary-node metadata beneath
`out/verif_oracle_inputs/`. The Python stage independently integrates those boundaries and
writes the compact comparison CSV here. Riley-rendered images are diagnostic inputs, not the
analytic authority.

The distortion oracle is generated independently with OpenCV and NumPy:

- `distortion_oracle_cases.csv` is the compact model manifest;
- `distortion_oracle_points.csv` stores ideal, distorted, and inverse results;
- `distortion_oracle_jacobians.csv` stores sparse reference Jacobians.

The deterministic field spans approximately $x\in[-0.8,0.8]$ and
$y\in[-0.6,0.6]$. It exercises individual and mixed radial, tangential,
thin-prism, and sensor-tilt coefficients; every polynomial monomial; both map
directions; and Brown--Conrady/polynomial compositions. The files were generated
with NumPy 2.5.3 and OpenCV 5.0.0.

This directory remains ignored by Git until the focused suite is approved as a committed core
test suite.
