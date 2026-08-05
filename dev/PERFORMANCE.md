# Riley Performance Guide

1. **Measure before optimising.** Use a representative benchmark and profile to establish whether the bottleneck is compute, memory bandwidth, cache misses, or synchronisation.

2. **Optimise the algorithm and data layout before SIMD or threading.** A poor memory-access pattern remains poor when executed wider or on more cores.

3. **Make the single-threaded scalar path fast first.** Progress in order: scalar correctness, scalar performance, SIMD, then multithreading.

4. **Treat cache locality as a primary design constraint.** For memory-bound work, contiguous data and fewer bytes moved usually matter more than extra arithmetic.

5. **Keep the raster hot loop small and predictable.** It should contain only work required per covered sub-pixel; move setup, validation, allocation, and uncommon cases outside it.

6. **Hoist loop invariants.** Precompute constants, configuration-derived values, strides, bounds, scale factors, and element-local data before entering the innermost loop.

7. **Use incrementing indices in hot loops.** Compute base offsets once, then advance them; do not repeatedly multiply dimensions or reconstruct array indices per lane/pixel.

8. **Prefer contiguous structure-of-arrays layouts for wide processing.** SIMD and cache prefetching work best when each operation reads adjacent values.

9. **Avoid pointer chasing in gathers.** Replace linked structures, per-pixel indirection, and scattered allocations with compact arrays, direct indexing, or an element-local buffer where possible.

10. **Load element-local attributes once.** Gather nodal fields, UVs, world coordinates, and normals into a small local shader buffer per element rather than repeatedly accessing global arrays per sub-pixel.

11. **Use `comptime` specialisation to remove hot-loop dispatch.** Element type, node count, shader type, channel count, sampler configuration, and raster policy should be selected outside the hot loop where practical.

12. **Do not make code branchless by default.** Remove unpredictable or per-lane branches, but retain cheap predictable branches when they avoid unnecessary work.

13. **Keep SIMD lanes full.** Batch enough independent pixels, sub-pixels, or elements to use the configured width; use masks for partial vectors and isolate tails.

14. **Choose SIMD direction based on data movement.** Use outer SIMD over pixels/sub-pixels when data is contiguous and independent; use inner SIMD across a small fixed dimension only when outer SIMD becomes gather- or bandwidth-bound.

15. **Avoid horizontal reductions in the hot path.** They often limit vector efficiency. Prefer lane-independent work and defer reductions where the algorithm permits.

16. **Minimise hot-loop memory writes.** Accumulate in registers or compact per-thread scratch buffers, then write sequentially. Avoid read-modify-write traffic and shared output locations.

17. **Tile work for locality and ownership.** Raster tiles should fit working data reasonably in cache, give each worker exclusive output ownership, and avoid locking during rendering.

18. **Thread only independent, sufficiently large work units.** Avoid fine-grained scheduling, shared mutable state, false sharing, and synchronisation in per-pixel work.

19. **Use approximations only with an accuracy contract.** Fast maths, LUTs, lower precision, and reduced sampling are valid only when verification shows they preserve Riley's required render and DIC behaviour.

20. **Keep an equivalent correctness path and benchmark it continuously.** Every performance change should be checked against regression renders and measured across scalar/SIMD, relevant precisions, and representative scene types.
