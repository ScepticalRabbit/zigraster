# Zig Style Guide

These rules apply to all Zig code created or modified in this repository.

- **Follow the repository naming conventions.**
    - Use the approved abbreviations listed in `./dev/ABBREVIATIONS.md`.
    - Use doubled lowercase letters for simple iteration indices.
        - Examples include `nn` for nodes, `ee` for elements, `rr` for rows,
          `cc` for columns, and `ii`, `jj`, or `kk` for generic dimensions.
        - Example:

          ```zig
          for (0..N) |nn| {
              // ...
          }
          ```
    - Single or doubled capital letters are acceptable for small comptime constants,
      such as `N`, `D`, or `NN`.
    - Use descriptive names when the meaning of a short name is not obvious.

- **Follow the standard Zig file structure.**
    - Follow the organisation described in `./dev/FILESTRUCTURE.md`.
    - Put all imports at the top of the file.
    - Put public types and the external API near the top of the file.
    - Group functions by functionality and approximate call order as the file
      progresses.
    - Put private helper functions below the public or higher-level functions that call
      them.
    - Preserve the existing file structure unless restructuring is explicitly required.

- **Do not re-export imports or imported declarations.**
    - Do not use `pub` on the same declaration as `@import`.
    - Do not directly re-export a type, function, or namespace from another file.
    - Import declarations directly from their defining source file where they are
      needed.
    - Avoid:

      ```zig
      pub const Mesh = @import("mesh.zig").Mesh;
      ```

    - Prefer:

      ```zig
      const mesh = @import("mesh.zig");
      const Mesh = mesh.Mesh;
      ```

- **Use consistent function parameter ordering.**
    - Put comptime parameters first in every function signature.
    - Order function parameters as follows:
        1. comptime parameters;
        2. allocators;
        3. input/output or stream objects;
        4. values passed by value;
        5. const pointers and const slices;
        6. mutable pointers and mutable slices used as outputs or working storage.

- **Make allocation and I/O explicit in function signatures.**
    - Any function that allocates memory must take an allocator parameter.
    - Any function that takes an allocator is assumed to allocate memory.
    - Do not hide allocation behind global state, implicit allocators, or unrelated
      objects.
    - Any function that performs I/O must take an explicit I/O or stream parameter.
    - Any function that takes an I/O or stream parameter is assumed to perform I/O.
    - Do not pass allocator or I/O parameters through functions that do not use them.

- **Prefer function-scoped arenas for temporary allocations.**
    - Name the allocator passed into an allocating function `outer_alloc`.
    - Create an arena allocator near the top of the function.
    - Use the arena allocator for allocations whose lifetime ends when the function
      returns.
    - Use `outer_alloc` only for memory that must outlive the function or be returned
      to the caller.
    - Always deinitialise the arena with `defer`.
    - Never return a pointer or slice backed by the function-scoped arena.
    - Prefer:

      ```zig
      fn buildResult(
          outer_alloc: std.mem.Allocator,
          input: []const f64,
      ) !Result {
          var arena = std.heap.ArenaAllocator.init(outer_alloc);
          defer arena.deinit();
          const alloc = arena.allocator();

          const scratch = try alloc.alloc(f64, input.len);
          const output = try outer_alloc.alloc(f64, input.len);
          errdefer outer_alloc.free(output);

          // Use scratch to construct output.

          return .{ .values = output };
      }
      ```

- **Make allocation ownership unambiguous.**
    - The allocator used to allocate returned memory must also be used by the caller to
      free it.
    - Document ownership for returned slices, pointers, and structs containing owned
      memory.
    - Use `deinit` methods for types that own multiple allocations or require structured
      cleanup.
    - Use `errdefer` immediately after successful allocations when later operations can
      fail.
    - Do not return partially initialised owning values without a clear cleanup path.
    - Do not store an allocator inside a long-lived type unless that type owns memory
      and needs the allocator for later cleanup or resizing.

- **Avoid allocation in hot loops and performance-critical paths.**
    - Do not allocate or free memory inside per-pixel, per-sample, per-element,
      per-node, or solver-iteration loops unless the allocation is unavoidable and
      justified.
    - Allocate scratch memory outside the hot loop and reuse it.
    - Pre-size dynamic containers when the required or expected capacity is known.
    - Prefer caller-provided output buffers or reusable workspace objects for
      frequently called kernels.
    - Do not create a new arena inside a hot loop.
    - When repeated independent operations need temporary memory, prefer one arena or
      scratch allocator outside the loop and reset or reuse it between iterations when
      this is safe.
    - Treat unexpected allocator activity in a hot path as a performance bug.

- **Avoid unnecessary or hidden memory traffic.**
    - Do not duplicate large slices or buffers unless ownership or mutation requires
      it.
    - Prefer slices and const pointers for borrowed data.
    - Prefer writing into caller-provided output memory when this makes ownership and
      lifetime clearer.
    - Avoid repeated growth of dynamic arrays when capacity can be estimated.
    - Do not use a general-purpose allocator locally when the caller has already
      supplied an allocator.
    - Do not use global mutable allocators.

- **Prefer `for` loops over `while` loops.**
    - Use `for` wherever the iteration can naturally be expressed using a range,
      slice, array, or collection.
    - Use `while` for convergence loops, irregular termination conditions, retry logic,
      or other cases where `while` more accurately represents the algorithm.

- **Keep function calls and expressions readable.**
    - Avoid deeply nested function calls or Zig builtins.
    - Two nested calls can be acceptable when the expression remains clear.
    - Treat three or more nested function or builtin calls as a warning that the
      expression should normally be split into intermediate variables.
    - This applies especially to chains of `@` builtins, casts, bounds operations, and
      indexing calculations.
    - Break complex expressions into named intermediate variables.
    - Give intermediate variables names that explain the purpose of each
      transformation.
    - Prefer:

      ```zig
      const raw_index = @intFromFloat(value);
      const bounded_index = @min(raw_index, max_index);
      const index: usize = @intCast(bounded_index);
      ```

    - Avoid:

      ```zig
      const index: usize = @intCast(@min(@intFromFloat(value), max_index));
      ```

    - Intermediate variables are particularly useful when they:
        - clarify type conversions;
        - separate bounds checking from casting;
        - expose numerical operations;
        - simplify debugging;
        - reduce repeated expressions;
        - prevent deeply nested function calls.
    - Do not introduce intermediate variables that merely rename a simple expression
      without improving readability.

- **Keep all Zig code within 92 columns.**
    - Count indentation and whitespace as part of the 92-column limit.
    - Wrap expressions before they exceed the limit.
    - When a function signature, function call, struct literal, array literal, or
      similar construct does not fit, convert it to multiline form.
    - Add a trailing comma after the final parameter, argument, or field in multiline
      constructs so that `zig fmt` preserves the layout.
    - Prefer:

      ```zig
      const result = processTile(
          config,
          nodes,
          elements,
          output,
      );
      ```

    - Do not leave an overlong line and expect `zig fmt` to choose a readable layout.

- **Use consistent indentation and statement layout.**
    - Use four spaces for indentation.
    - Do not use tab characters.
    - Replace any tab encountered with four spaces, then recheck the affected line
      length.
    - Use no more than one semicolon per line.
    - Keep separate statements on separate lines.
    - Format code for readability rather than minimising the number of lines.

- **Use comments for intent, not narration.**
    - Use comments to explain:
        - non-obvious decisions;
        - numerical or physical assumptions;
        - invariants;
        - ownership constraints;
        - behaviour not represented by the type system.
    - Do not add comments that merely restate the code.
    - Preserve existing comments unless they are incorrect or obsolete.
    - Document public declarations when behaviour, units, ownership, or error
      conditions are not obvious.

- **Keep debugging output minimal and targeted.**
    - Print information for only one or two representative failing cases.
    - Prefer filtering by a specific pixel, tile, element, sample, or iteration.
    - Do not print for every pixel, sub-pixel, sample, node, or solver iteration.
    - Remove temporary debugging output once the problem is resolved.
    - Do not add noisy logging to performance-critical loops.

- **Run `zig fmt` after editing Zig files.**
    - Run `zig fmt` on every edited Zig file.
    - Do not run repository-wide formatting unless explicitly requested.
    - Inspect the diff after formatting.
    - Avoid unrelated formatting changes, renaming, cleanup, or refactoring.

- **Keep generated executables out of the source tree.**
    - When using `zig build-exe`, write the executable to `./bin/`.
    - Do not emit executables into the repository root or source directories.
    - Confirm that generated binaries are not accidentally included in the final diff.

- **Make the smallest coherent change required for the task.**
    - Do not change public APIs unless required.
    - Do not reorganise unrelated code.
    - Do not introduce abstractions for a single simple use.
    - Do not change numerical types, tolerances, algorithms, or memory ownership as
      incidental cleanup.
    - Preserve the style of surrounding code where it does not conflict with this
      guide.

- **Validate every code change before completion.**
    - Run `zig fmt` on edited files.
    - Compile the affected target.
    - Run relevant focused tests.
    - Inspect the final diff.
    - Check for tab characters and lines exceeding 92 columns.
    - Confirm that no generated executables or artefacts were added accidentally.
    - Do not claim that formatting, compilation, or tests passed unless they were
      actually run successfully.
    - Clearly state any checks that could not be run.
