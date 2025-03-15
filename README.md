# Zig Raster
My implementation of a rasterisation renderer in Zig. I am using this project to learn Zig for scientific computing applications. A rasteriser includes enough linear algebra to explore the language for this purpose. I come from a Python background with some experience in Cython and C but wanted a new high performance compiled language with good Python interop, hence Zig.

The rasteriser is built using [Zig 0.14](https://ziglang.org/download/) and can be run using:
```shell
zig run -O ReleaseFast src/main.zig
```

This project is inspired by the rasteriser implementation on [Scratchapixel](https://www.scratchapixel.com/index.html). See their description of the rasterisation process [here](https://www.scratchapixel.com/lessons/3d-basic-rendering/rasterization-practical-implementation/overview-rasterization-algorithm.html) and their code [here](https://github.com/scratchapixel/scratchapixel-code/tree/main/rasterization-practical-implementation).

## Test Case
The test case for rendering is finite element solid mechanics simulation of a XXXXX, see image below of the vertical displacement field. I performed this simulation using [Gmsh](https://gmsh.info/) to create the mesh and [MOOSE](https://mooseframework.inl.gov/) as the physics solver. The Gmsh `.geo` and MOOSE input `.i` file can be found [here](). I skinned the 3D mesh using `pyvista` then I parsed the mesh (nodal coordinates and connectivity table) and the output displacement field to `.csv` files to be read into Zig (files are in the [data]() directory).

## Core Functionality
