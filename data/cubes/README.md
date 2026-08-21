# Mesh-convention cube fixtures

These files are small CSV exports of PyVale's 10 mm element-test cubes.
`connectivity.csv` deliberately retains the source Exodus convention: it is
one-based and node-major. Riley mesh-convention tests use these fixtures to
verify normalization to zero-based, row-major connectivity.

The supported fixtures are `tet4`, `tet10`, `hex8`, `hex20`, and `hex27`.
`tet14` is retained as the explicit unsupported-topology fixture.
