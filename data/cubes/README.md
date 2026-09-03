# Mesh-convention cube fixtures

These CSV meshes derive from PyVale's 10 mm element-test cubes. `tet4`, `tet10`,
`hex8`, `hex20`, and `hex27` are canonical: `connectivity.csv` is zero-based,
row-major, and uses Riley local-node order.

`tet14` is the explicit unsupported-topology fixture and deliberately retains
its original one-based, node-major source layout.
