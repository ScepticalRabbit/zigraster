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

This directory remains ignored by Git until the focused suite is approved as a committed core
test suite.
