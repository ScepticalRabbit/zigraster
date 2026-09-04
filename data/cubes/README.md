# Mesh-convention cube fixtures

These directories contain small fixtures from PyVale's 10 mm element-test
cubes. The CSV files contain previously extracted coordinate and connectivity
tables. The original Exodus output is the authoritative source for testing
source-format conversion.

Each supported case also includes `cube_out.e`, the original MOOSE Exodus
output copied from PyVale. These files retain 21 time steps of nodal
displacement, strain and temperature results and provide authoritative Exodus
connectivity fixtures for adapter tests and future rendering tests.

The supported fixtures are `tet4`, `tet10`, `hex8`, `hex20`, and `hex27`.
`tet14` is retained as the explicit unsupported-topology fixture.
