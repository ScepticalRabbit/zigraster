# Design Specification: Procedural Speckle Pattern Generation for Riley

## Motivation
`Riley` is a software rasteriser specifically designed for uncertainty quantification of image-based measurement techniques such as digital image correlation (DIC). A key aspect of simulating digital image correlation is defining the random pattern of speckles on the surface of the mesh that will be tracked to determine the surface deformation. Currently, `Riley` supports texture shading where a 2D texture containing a speckle pattern can be mapped onto the surface of a 3D mesh using normalised coordinates called UVs. Defining the 2D-texture-to-3D-surface map with the required UV coordinates is a significant bottleneck to automated experimental design using `Riley`. Being able to procedurally generate a speckle pattern that does not require UVs based on user input defining the desired statistical characteristics of the pattern would significantly reduce user effort in setting up virtual DIC experiments.

## Aims & Objectives
The aim of this project is to develop a performant procedural speckle pattern generation shader for `Riley`. This speckle shader, `SpeckShader`, should support a variety of speckle pattern generation methods. This procedural speckle shader should support:

- An initial scalar implementation of a UV-free procedural speckle pattern shader followed by a performant SIMD implementation using Zig's `@Vector`
- Have user-configurable parameters that control the characteristics of the speckle pattern
- Have equivalent or better performance to a cubic texture shader based on a minimum-element full-screen raster benchmark

**NOTE:** The specification below is given based on my initial thoughts and does not include the reality of actually developing the speckle shader. I fully expect there to be significant changes as long as we maintain: 1) correctness and 2) performance.     

## Software & Research Challenges
The shader is invoked after all the visibility and in/out testing, when we need to fill and colour a sub-pixel (`Riley` uses sub-pixel anti-aliasing for pixel integration). Once these tests all pass, we know the parametric coordinates of the hit location in the element (`xi`, `eta`). Using the element shape functions, we can then use (`xi`, `eta`) to interpolate nodal attributes to this location, including world coordinates (deformed/undeformed), normals, and UVs.

- What method or acceleration structure will give the best performance for different cases? A BVH is probably not going to be the best option here because we already know our hit location from the raster process and we know where we are in 3D space from nodal attribute interpolation. A voxel-based hash grid or direct generation from our undeformed 3D world coordinates will probably work best depending on the method.
- How can we reproduce the statistics and/or characteristics of a user input pattern?

## Inputs
- Defined as a shader attached to a `MeshInput`
    - Option in the `ShaderInput` tagged union in `shaderops_common.zig` - possibly `SpeckInput` and then with shader kernel `SpeckKernel`?
    - Prototype based on the analytic function shader `FuncInput`, as this has machinery for different coordinate interpolations (UV, world deformed, world undeformed) and normals.
- User-defined speckle pattern characteristics:
    - Speckle pattern method: disk/sphere, Gaussian blob, Perlin noise, etc. (Look in the literature for some other possibilities, but this is a good minimum set.)
    - Speckle pattern randomisation/hyperparameters: depends on the chosen method above; at a minimum, a user-configurable seed or a seed generated from machine noise if null.
    - Speckle pattern characteristics: nominal speckle size in (length/pixel), contrast, brightness, black/white ratio, digitisation.

## Data Processing
### Pre-processing & Shader Setup
In the shader preparation stage, we will want to do any pre-processing needed to set up our speckle shader. The speckle shader will be easiest to define on the undeformed world coordinates so we can generate this once and then use undeformed world coordinates to index back into our pattern for deformed meshes. We should avoid as much work as possible in the raster hot loop.

### Raster Hot Loop & Pixel Fill
In the hot loop, we should be able to invoke a `fill` function from our shader kernel, which will work out the resulting colour based on the interpolated undeformed world coordinates and/or surface normals if needed.

### Rough Shading Process
Pre-processing:
1. Build undeformed-surface AABBs in 3D.
2. Create regular voxel grid.
3. Identify active voxels near the surface.
4. For each active voxel, hash-generate candidate speckles.
5. Project each candidate to the closest undeformed surface point.
6. Insert each speckle ID into all overlapping voxels.

During shading:
- Assume we have a visible hit and the element parametric coordinates (`xi`, `eta`):

1. Interpolate undeformed world coordinates
2. Find voxel (with hash?)
3. Gather candidate speckles
4. Evaluate blobs in tangent coordinates
5. Accumulate intensity
6. Pass to scratch resolve to apply Riley’s usual SSAA / PSF afterwards.

## Outputs
- Rendered image with a procedural pattern on all supported element types with meshes of varying complexity.

## Useful References
- Preprint describing `Riley` in detail: https://engrxiv.org/preprint/view/7300/version/9460
- Sur's 2D Boolean speckle generation paper: https://hal.science/hal-01664997v1, https://members.loria.fr/FSur/software/BSpeckleRender/.

## Deliverables: In Priority Order
- A scalar implementation of the procedural speckle pattern shader demonstrated on all supported element types and on meshes of different levels of complexity
- A verification test suite using speckle pattern analysis techniques to verify the pattern produced matches the user inputs (e.g. histogram, brightness/contrast, speckle size, etc)
- A render regression test suite using predefined random seeds for reproducibility
- A demonstration case showing how to use procedural speckle patterns called demo_procedural_speckles.zig.
- A SIMD implementation using Zig's `@Vector` investigating outer/inner SIMD for performance (e.g. SIMD over pixels or SIMD within the shader kernel)
- A benchmark performance analysis report as a markdown file comparing different methods of procedural speckle generation with their benefits and drawbacks
