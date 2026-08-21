const std = @import("std");

const riley_version = std.SemanticVersion{
    .major = 2026,
    .minor = 7,
    .patch = 1,
};

const RunEntry = struct {
    step_name: []const u8,
    description: []const u8,
    source_path: []const u8,
};

const TestEntry = struct {
    step_name: []const u8,
    description: []const u8,
    source_path: []const u8,
};

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const precision = b.option([]const u8, "precision", "Floating point precision: f64 or f32") orelse
        "f64";
    const simd = b.option([]const u8, "simd", "SIMD mode: on or off") orelse "on";
    const newton_solver = b.option(
        []const u8,
        "newton-solver",
        "Newton solver mode: fast or robust",
    ) orelse "fast";
    const simd_vector_width = b.option(
        u32,
        "simd-vector-width",
        "SIMD vector width (0 to use default for precision)",
    ) orelse 0;
    const speckle_boundary_blur = b.option(
        bool,
        "speckle-boundary-blur",
        "Enable smooth procedural speckle boundaries",
    ) orelse false;
    const speckle_neighbor_count = b.option(
        u8,
        "speckle-neighbor-count",
        "Procedural speckle candidate cell count: 9, 4, or 1",
    ) orelse 9;
    const speckle_evaluator = b.option(
        []const u8,
        "speckle-evaluator",
        "Procedural speckle evaluator: cell-hash, list-naive, or list-indexed",
    ) orelse "cell-hash";
    const speckle_shape = b.option(
        []const u8,
        "speckle-shape",
        "Procedural speckle shape: disk or gaussian",
    ) orelse "gaussian";
    validatePrecision(precision);
    validateSimd(simd);
    validateNewtonSolver(newton_solver);
    validateSpeckleNeighborCount(speckle_neighbor_count);
    validateSpeckleEvaluator(speckle_evaluator);
    validateSpeckleShape(speckle_shape);

    const build_options_module = createBuildOptionsModule(
        b,
        precision,
        simd,
        newton_solver,
        simd_vector_width,
        speckle_boundary_blur,
        speckle_neighbor_count,
        speckle_evaluator,
        speckle_shape,
    );
    const shared_lib = addRileySharedLibrary(
        b,
        target,
        optimize,
        build_options_module,
    );
    b.installArtifact(shared_lib);
    _ = b.addInstallHeaderFile(
        b.path("src/riley/cyth/riley.h"),
        "riley.h",
    );

    const tests = [_]TestEntry{
        .{
            .step_name = "test-min",
            .description = "Run the MIN test suite",
            .source_path = "src/test_min.zig",
        },
        .{
            .step_name = "test-gold-all",
            .description = "Run the ALL gold regression test suite",
            .source_path = "src/test_gold_all.zig",
        },
        .{
            .step_name = "test-bench",
            .description = "Run the benchmark regression test suite",
            .source_path = "src/test_bench.zig",
        },
    };

    for (tests) |entry| {
        const test_step = b.step(entry.step_name, entry.description);
        const test_run = addTestRunStep(
            b,
            .ReleaseSafe,
            entry,
            precision,
            simd,
            newton_solver,
            simd_vector_width,
            speckle_boundary_blur,
            speckle_neighbor_count,
            speckle_evaluator,
            speckle_shape,
        );
        test_step.dependOn(&test_run.step);
    }

    const demos = [_]RunEntry{
        .{
            .step_name = "demo-sphere200",
            .description = "Run the sphere200 demo",
            .source_path = "src/demo_sphere200.zig",
        },
        .{
            .step_name = "demo-rabbits",
            .description = "Run the rabbits demo",
            .source_path = "src/demo_rabbits.zig",
        },
        .{
            .step_name = "demo-rabbits-rgb",
            .description = "Run the rabbits RGB demo",
            .source_path = "src/demo_rabbits_rgb.zig",
        },
        .{
            .step_name = "demo-rabbits-fields",
            .description = "Run the rabbits fields demo",
            .source_path = "src/demo_rabbits_fields.zig",
        },
        .{
            .step_name = "demo-dicuq",
            .description = "Run the DIC UQ demo",
            .source_path = "src/demo_dicuq.zig",
        },
        .{
            .step_name = "demo-stereocal",
            .description = "Run the stereo calibration demo",
            .source_path = "src/demo_stereocal.zig",
        },
    };

    const demos_step = b.step("demos", "Run all demo entrypoints");
    for (demos) |entry| {
        const run_step = b.step(entry.step_name, entry.description);
        const run_artifact = addRunStep(
            b,
            target,
            optimize,
            build_options_module,
            entry,
        );
        run_step.dependOn(&run_artifact.step);
        demos_step.dependOn(&run_artifact.step);
    }

    const generators = [_]RunEntry{
        .{
            .step_name = "gen-gold-all",
            .description = "Generate the ALL gold datasets",
            .source_path = "src/gen_gold_all.zig",
        },
        .{
            .step_name = "gen-gold-min",
            .description = "Generate the MIN gold datasets",
            .source_path = "src/gen_gold_min.zig",
        },
    };

    const gold_step = b.step("gen-gold", "Run all gold generation entrypoints");
    for (generators) |entry| {
        const run_step = b.step(entry.step_name, entry.description);
        const run_artifact = addRunStep(
            b,
            target,
            optimize,
            build_options_module,
            entry,
        );
        run_step.dependOn(&run_artifact.step);
        if (std.mem.eql(u8, entry.step_name, "gen-gold-all")) {
            gold_step.dependOn(&run_artifact.step);
        }
    }

    const benches = [_]RunEntry{
        .{
            .step_name = "bench-dicuq",
            .description = "Run the DIC UQ benchmark",
            .source_path = "src/bench_dicuq.zig",
        },
        .{
            .step_name = "bench-fullraster",
            .description = "Run the fullraster benchmark",
            .source_path = "src/bench_fullraster.zig",
        },
        .{
            .step_name = "bench-tiltraster",
            .description = "Run the tiltraster benchmark",
            .source_path = "src/bench_tiltraster.zig",
        },
        .{
            .step_name = "bench-geom",
            .description = "Run the geom benchmark",
            .source_path = "src/bench_geom.zig",
        },
        .{
            .step_name = "bench-sphere2000",
            .description = "Run the sphere2000 benchmark",
            .source_path = "src/bench_sphere2000.zig",
        },
        .{
            .step_name = "bench-sphere2000zoom",
            .description = "Run the sphere2000zoom benchmark",
            .source_path = "src/bench_sphere2000zoom.zig",
        },
        .{
            .step_name = "bench-thread-geom",
            .description = "Run the threaded geom benchmark",
            .source_path = "src/bench_thread_geom.zig",
        },
    };

    const bench_runs_step = b.step("benches", "Run all benchmark entrypoints");
    const bench_bins_step = b.step(
        "install-bench-bins",
        "Install benchmark binaries under the selected prefix bin directory",
    );
    for (benches) |entry| {
        const run_step = b.step(entry.step_name, entry.description);
        const run_artifact = addRunStep(
            b,
            target,
            optimize,
            build_options_module,
            entry,
        );
        run_step.dependOn(&run_artifact.step);
        bench_runs_step.dependOn(&run_artifact.step);

        const install_step = b.step(
            b.fmt("install-{s}", .{entry.step_name}),
            b.fmt("Install the {s} benchmark executable", .{entry.step_name}),
        );
        const install_artifact = addBenchInstallStep(
            b,
            target,
            optimize,
            build_options_module,
            entry,
            precision,
            simd,
            simd_vector_width,
        );
        install_step.dependOn(&install_artifact.step);
        bench_bins_step.dependOn(&install_artifact.step);
    }
}

fn addRileySharedLibrary(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    build_options_module: *std.Build.Module,
) *std.Build.Step.Compile {
    const shared_lib = b.addLibrary(.{
        .linkage = .dynamic,
        .name = "riley",
        .root_module = createRootModule(
            b,
            target,
            optimize,
            build_options_module,
            "src/riley/zig/c-riley.zig",
            true,
            .library,
        ),
        .version = riley_version,
    });
    return shared_lib;
}

fn addTestRunStep(
    b: *std.Build,
    optimize: std.builtin.OptimizeMode,
    entry: TestEntry,
    precision: []const u8,
    simd: []const u8,
    newton_solver: []const u8,
    simd_vector_width: u32,
    speckle_boundary_blur: bool,
    speckle_neighbor_count: u8,
    speckle_evaluator: []const u8,
    speckle_shape: []const u8,
) *std.Build.Step.Run {
    const run_step = b.addSystemCommand(&.{
        "sh",
        "-c",
        \\set -eu
        \\step_name="$1"
        \\src="$2"
        \\precision="$3"
        \\simd="$4"
        \\newton_solver="$5"
        \\simd_vector_width="$6"
        \\speckle_boundary_blur="$7"
        \\speckle_neighbor_count="$8"
        \\speckle_evaluator="$9"
        \\speckle_shape="${10}"
        \\zigexe="${11}"
        \\opt="${12}"
        \\cache_root=".zig-cache/riley-test"
        \\mkdir -p "$cache_root"
        \\src_hash="$(
        \\    find src -type f -print0 |
        \\    sort -z |
        \\    xargs -0 sha256sum |
        \\    sha256sum |
        \\    cut -d' ' -f1
        \\)"
        \\tree_dir="${cache_root}/${step_name}_${precision}_${simd}_${newton_solver}_${simd_vector_width}_${speckle_boundary_blur}_${speckle_neighbor_count}_${speckle_evaluator}_${speckle_shape}_${opt}_${src_hash}"
        \\if [ ! -d "$tree_dir" ]; then
        \\    lock_dir="${tree_dir}.lock"
        \\    while ! mkdir "$lock_dir" 2>/dev/null; do
        \\        sleep 0.1
        \\    done
        \\    cleanup() {
        \\        rmdir "$lock_dir"
        \\    }
        \\    trap cleanup EXIT
        \\    if [ ! -d "$tree_dir" ]; then
        \\        mkdir -p "$tree_dir"
        \\        cp -a src "$tree_dir/src"
        \\        src_file="$tree_dir/$src"
        \\        src_orig="${src_file}.orig"
        \\        mv "$src_file" "$src_orig"
        \\        {
        \\            printf 'pub const build_options = struct {\n'
        \\            printf '    pub const precision = "%s";\n' "$precision"
        \\            printf '    pub const simd = "%s";\n' "$simd"
        \\            printf '    pub const newton_solver = "%s";\n' "$newton_solver"
        \\            printf '    pub const simd_vector_width: comptime_int = %s;\n' "$simd_vector_width"
        \\            printf '    pub const speckle_boundary_blur = %s;\n' "$speckle_boundary_blur"
        \\            printf '    pub const speckle_neighbor_count: comptime_int = %s;\n' "$speckle_neighbor_count"
        \\            printf '    pub const speckle_evaluator = "%s";\n' "$speckle_evaluator"
        \\            printf '    pub const speckle_shape = "%s";\n' "$speckle_shape"
        \\            printf '};\n\n'
        \\            cat "$src_orig"
        \\        } > "$src_file"
        \\    fi
        \\fi
        \\"$zigexe" test -lc -O "$opt" "$tree_dir/$src"
        ,
        "--",
        entry.step_name,
        entry.source_path,
        precision,
        simd,
        newton_solver,
        b.fmt("{d}", .{simd_vector_width}),
        if (speckle_boundary_blur) "true" else "false",
        b.fmt("{d}", .{speckle_neighbor_count}),
        speckle_evaluator,
        speckle_shape,
        b.graph.zig_exe,
        @tagName(optimize),
    });
    return run_step;
}

fn addRunStep(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    build_options_module: *std.Build.Module,
    entry: RunEntry,
) *std.Build.Step.Run {
    const executable = b.addExecutable(.{
        .name = entry.step_name,
        .root_module = createRootModule(
            b,
            target,
            optimize,
            build_options_module,
            entry.source_path,
            false,
            .executable,
        ),
    });
    return b.addRunArtifact(executable);
}

fn addBenchInstallStep(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    build_options_module: *std.Build.Module,
    entry: RunEntry,
    precision: []const u8,
    simd: []const u8,
    simd_vector_width: u32,
) *std.Build.Step.InstallArtifact {
    const basename = std.fs.path.basename(entry.source_path);
    const binary_name = benchmarkBinaryName(
        b,
        basename[0 .. basename.len - ".zig".len],
        precision,
        simd,
        simd_vector_width,
    );
    const executable = b.addExecutable(.{
        .name = binary_name,
        .root_module = createRootModule(
            b,
            target,
            optimize,
            build_options_module,
            entry.source_path,
            false,
            .executable,
        ),
    });
    return b.addInstallArtifact(executable, .{
        .dest_sub_path = binary_name,
    });
}

fn createBuildOptionsModule(
    b: *std.Build,
    precision: []const u8,
    simd: []const u8,
    newton_solver: []const u8,
    simd_vector_width: u32,
    speckle_boundary_blur: bool,
    speckle_neighbor_count: u8,
    speckle_evaluator: []const u8,
    speckle_shape: []const u8,
) *std.Build.Module {
    const options = b.addOptions();
    options.addOption([]const u8, "precision", precision);
    options.addOption([]const u8, "simd", simd);
    options.addOption([]const u8, "newton_solver", newton_solver);
    options.addOption(u32, "simd_vector_width", simd_vector_width);
    options.addOption(bool, "speckle_boundary_blur", speckle_boundary_blur);
    options.addOption(u8, "speckle_neighbor_count", speckle_neighbor_count);
    options.addOption([]const u8, "speckle_evaluator", speckle_evaluator);
    options.addOption([]const u8, "speckle_shape", speckle_shape);
    return options.createModule();
}

fn createRootModule(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    build_options_module: *std.Build.Module,
    source_path: []const u8,
    link_libc: bool,
    wrapper_kind: WrapperKind,
) *std.Build.Module {
    const wrapper_text = wrapperSourceText(
        b,
        wrapper_kind,
        source_path,
    );
    const wrapper_files = b.addWriteFiles();
    const wrapper_source = wrapper_files.add(
        b.fmt("{s}.wrapper.zig", .{source_path}),
        wrapper_text,
    );
    const imports = buildWrapperImports(
        b,
        target,
        optimize,
        build_options_module,
        source_path,
        link_libc,
        wrapper_kind,
        wrapper_text,
    );
    return b.createModule(.{
        .root_source_file = wrapper_source,
        .target = target,
        .optimize = optimize,
        .link_libc = link_libc,
        .imports = imports,
    });
}

const WrapperKind = enum {
    executable,
    test_module,
    library,
};

fn wrapperSourceText(
    b: *std.Build,
    wrapper_kind: WrapperKind,
    source_path: []const u8,
) []const u8 {
    return switch (wrapper_kind) {
        .executable =>
        \\const entry_source = @import("entry_source");
        \\pub const build_options = @import("build_options");
        \\pub const main = entry_source.main;
        \\comptime {
        \\    _ = entry_source;
        \\}
        \\
        ,
        .test_module => blk: {
            const source_text = std.Io.Dir.cwd().readFileAlloc(
                b.graph.io,
                source_path,
                b.allocator,
                .limited(16 * 1024 * 1024),
            ) catch @panic("Failed to read test source file.");
            break :blk std.fmt.allocPrint(
                b.allocator,
                \\pub const build_options = @import("build_options");
                \\
                \\{s}
            ,
                .{source_text},
            ) catch @panic("Failed to write test wrapper source.");
        },
        .library =>
        \\const entry_source = @import("entry_source");
        \\pub const build_options = @import("build_options");
        \\comptime {
        \\    _ = entry_source;
        \\}
        \\
        ,
    };
}

fn buildWrapperImports(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    build_options_module: *std.Build.Module,
    source_path: []const u8,
    link_libc: bool,
    wrapper_kind: WrapperKind,
    _: []const u8,
) []const std.Build.Module.Import {
    var imports: std.ArrayList(std.Build.Module.Import) = .empty;
    imports.append(b.allocator, .{
        .name = "build_options",
        .module = build_options_module,
    }) catch @panic("OOM building imports.");

    if (wrapper_kind != .test_module) {
        imports.append(b.allocator, .{
            .name = "entry_source",
            .module = b.createModule(.{
                .root_source_file = b.path(source_path),
                .target = target,
                .optimize = optimize,
                .link_libc = link_libc,
            }),
        }) catch @panic("OOM building entry source import.");
        return imports.items;
    }
    return imports.items;
}

fn validateSpeckleShape(shape: []const u8) void {
    if (std.mem.eql(u8, shape, "disk") or
        std.mem.eql(u8, shape, "gaussian")) return;
    @panic("Supported -Dspeckle-shape values are disk and gaussian.");
}

fn validateSpeckleEvaluator(evaluator: []const u8) void {
    if (std.mem.eql(u8, evaluator, "cell-hash") or
        std.mem.eql(u8, evaluator, "list-naive") or
        std.mem.eql(u8, evaluator, "list-indexed")) return;
    @panic("Supported -Dspeckle-evaluator values are cell-hash, list-naive, and list-indexed.");
}

fn validateSpeckleNeighborCount(count: u8) void {
    if (count == 9 or count == 4 or count == 1) return;
    @panic("Supported -Dspeckle-neighbor-count values are 9, 4, and 1.");
}

fn validatePrecision(precision: []const u8) void {
    if (std.mem.eql(u8, precision, "f32") or
        std.mem.eql(u8, precision, "f64"))
    {
        return;
    }
    @panic("Supported -Dprecision values are f32 and f64.");
}

fn validateSimd(simd: []const u8) void {
    if (std.mem.eql(u8, simd, "on") or
        std.mem.eql(u8, simd, "off"))
    {
        return;
    }
    @panic("Supported -Dsimd values are on and off.");
}

fn validateNewtonSolver(newton_solver: []const u8) void {
    if (std.mem.eql(u8, newton_solver, "fast") or
        std.mem.eql(u8, newton_solver, "robust"))
    {
        return;
    }
    @panic("Supported -Dnewton-solver values are fast and robust.");
}

fn benchmarkBinaryName(
    b: *std.Build,
    base_name: []const u8,
    precision: []const u8,
    simd: []const u8,
    simd_vector_width: u32,
) []const u8 {
    const simd_tag = if (std.mem.eql(u8, simd, "on")) "simd" else "scalar";
    const default_width: u32 = if (std.mem.eql(u8, precision, "f32")) 16 else 8;

    if (simd_vector_width == 0 or simd_vector_width == default_width) {
        return b.fmt(
            "{s}_{s}_{s}",
            .{ base_name, precision, simd_tag },
        );
    } else {
        return b.fmt(
            "{s}_{s}_{s}_v{d}",
            .{
                base_name,
                precision,
                simd_tag,
                simd_vector_width,
            },
        );
    }
}
