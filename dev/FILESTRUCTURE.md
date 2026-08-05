# Riley Code File Structure

## File organisation
- All file names should be lower case unless directly exposed as a struct
- Use io as a suffix for files that contain operations for reading and writing to disk e.g. imageio.zig
- Use ops as a suffix for collections of utility/data operation functions e.g. meshops.zig
- Use the kernel suffix for files defining structs containing only comptime known constants and inline functions e.g. geometrykernels
- If a scalar/simd split is required the top level file name becomes a thin wrapper with no implementation details then all implementation and tests is moved into _common.zig, _simd.zig, _scalar.zig files.
- File names should not have underscores unless using _common, _simd or _scalar

## Exceptions
- ABI-facing files may retain established external naming
- Generated artefacts and non-Zig files are out of scope
- Thin wrapper files should contain no implementation logic and no tests
- Existing file names should only be changed when the rename improves clarity

## Within file code organisation
1. Module header block / documentation / description
2. Imports
3. Module constants
4. Public entry-point functions
5. Public constants and public types
6. Main implementation stages, in pipeline order
   - organise code from public entry points down to deeper private internals
   - within each stage place functions in pipeline or call order
   - place stage-local types as close as practical above their first real use
   - if a struct owns methods, keep the struct and its methods together as one unit
   - stage entry function first, then deeper helpers below it
7. Generic low-level helpers
8. Tests

For files using the `ops` suffix:
- these files may not control the overall application pipeline themselves
- even so, functions should still be grouped in local pipeline or call order
- types should still be placed near where they are first used
- organise the file from higher-level public operations down to deeper private helpers

**Module Header Block**
```
// --------------------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
```
**Marking Code Blocks**
Each code block should be denoted by a 90 column comment -. As below
```
// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------
```

Not all sections may be needed depending on the file. The main rule is to keep the
reader moving from the public surface down through the implementation in call order,
with types placed close to where they are first needed.
