// --------------------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");

const buildconfig = @import("../riley/zig/buildconfig.zig");
const shaderops = @import("../riley/zig/shaderops_common.zig");

const F = buildconfig.F;

pub const DemoSpec = struct {
    command_name: []const u8,
    output_default: []const u8,
    pixels_num_default: [2]u32,
    dimensions: Dimensions = .exposed_and_reported,
    comparison: ?Comparison = null,
    mask_report_label: []const u8,

    pub const Dimensions = enum {
        hidden,
        exposed_and_reported,
    };

    pub const Comparison = struct {
        texture_command: []const u8,
        procedural_command: []const u8,
    };
};

pub const DemoArgs = struct {
    params: shaderops.Speckle2DParams = defaultParams(),
    out_dir: []const u8,
    pixels_num: [2]u32,
};

pub fn defaultParams() shaderops.Speckle2DParams {
    return if (buildconfig.speckle_direct_fixed)
        .{ .radius_jitter = 0.0 }
    else
        .{};
}

pub fn parseDemoArgs(raw_args: anytype, comptime spec: DemoSpec) !?DemoArgs {
    var args = DemoArgs{
        .out_dir = spec.output_default,
        .pixels_num = spec.pixels_num_default,
    };
    var arg_idx: usize = 1;
    while (arg_idx < raw_args.len) {
        const arg = std.mem.span(raw_args[arg_idx]);
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printUsage(spec);
            return null;
        }
        if (arg_idx + 1 >= raw_args.len) {
            std.debug.print("Missing value for {s}\n\n", .{arg});
            printUsage(spec);
            return error.MissingArgumentValue;
        }

        const value = std.mem.span(raw_args[arg_idx + 1]);
        if (std.mem.eql(u8, arg, "--size")) {
            args.params.radius_mean = try std.fmt.parseFloat(F, value);
        } else if (std.mem.eql(u8, arg, "--occupancy")) {
            args.params.occupancy = try std.fmt.parseFloat(F, value);
        } else if (std.mem.eql(u8, arg, "--cells-u")) {
            args.params.cells_per_uv[0] = try std.fmt.parseFloat(F, value);
        } else if (std.mem.eql(u8, arg, "--cells-v")) {
            args.params.cells_per_uv[1] = try std.fmt.parseFloat(F, value);
        } else if (std.mem.eql(u8, arg, "--jitter")) {
            args.params.radius_jitter = try std.fmt.parseFloat(F, value);
        } else if (std.mem.eql(u8, arg, "--softness")) {
            args.params.edge_softness = try std.fmt.parseFloat(F, value);
        } else if (std.mem.eql(u8, arg, "--threshold")) {
            args.params.perlin_coverage_threshold = try std.fmt.parseFloat(F, value);
        } else if (std.mem.eql(u8, arg, "--transition")) {
            args.params.perlin_coverage_transition_width = try std.fmt.parseFloat(F, value);
        } else if (std.mem.eql(u8, arg, "--seed")) {
            args.params.seed = try std.fmt.parseInt(u32, value, 0);
        } else if (spec.dimensions == .exposed_and_reported and std.mem.eql(u8, arg, "--width")) {
            args.pixels_num[0] = try parsePositiveU32(value);
        } else if (spec.dimensions == .exposed_and_reported and std.mem.eql(u8, arg, "--height")) {
            args.pixels_num[1] = try parsePositiveU32(value);
        } else if (std.mem.eql(u8, arg, "--output")) {
            args.out_dir = value;
        } else {
            std.debug.print("Unknown option: {s}\n\n", .{arg});
            printUsage(spec);
            return error.UnknownArgument;
        }
        arg_idx += 2;
    }
    return args;
}

pub fn printProceduralConfig(
    params: shaderops.Speckle2DParams,
    pixels_num: [2]u32,
    comptime spec: DemoSpec,
) void {
    if (spec.comparison) |comparison| {
        std.debug.print(
            "  texture baseline: zig build {s} -Dsimd=off\n",
            .{comparison.texture_command},
        );
    }
    if (spec.dimensions == .exposed_and_reported) {
        std.debug.print(
            "  image dimensions: {d} x {d} pixels\n",
            .{ pixels_num[0], pixels_num[1] },
        );
    }

    const perlin_note = if (buildconfig.speckle_shape == .perlin)
        " (not used by Perlin)"
    else
        "";
    const effective_boundary_blur = if (buildconfig.speckle_shape == .disk and
        buildconfig.speckle_boundary_blur)
        params.edge_softness
    else
        0.0;

    std.debug.print(
        "  evaluator (compile-time): {s}\n",
        .{buildconfig.speckle_evaluator_name},
    );
    std.debug.print("  shape (compile-time): {s}\n", .{@tagName(buildconfig.speckle_shape)});
    std.debug.print(
        "  neighbor count (compile-time): {d}{s}\n",
        .{ buildconfig.speckle_neighbor_count, perlin_note },
    );
    std.debug.print(
        "  effective boundary blur: {d} cell units\n",
        .{effective_boundary_blur},
    );
    std.debug.print("  seed: {d} (0x{x})\n", .{ params.seed, params.seed });
    std.debug.print(
        "  cells per UV: {d} x {d}\n",
        .{ params.cells_per_uv[0], params.cells_per_uv[1] },
    );

    switch (comptime buildconfig.speckle_shape) {
        .perlin => {
            std.debug.print(
                "  coverage threshold: {d}\n",
                .{params.perlin_coverage_threshold},
            );
            std.debug.print(
                "  coverage transition width: {d}\n",
                .{params.perlin_coverage_transition_width},
            );
        },
        .disk, .gaussian => {
            if (comptime buildconfig.speckle_direct_fixed) {
                std.debug.print("  fixed radius: {d} cell units\n", .{params.radius_mean});
                std.debug.print("  radius jitter: zero (required by direct-fixed)\n", .{});
            } else {
                std.debug.print("  nominal radius: {d} cell units\n", .{params.radius_mean});
                std.debug.print("  radius jitter: {d} cell units\n", .{params.radius_jitter});
            }
            std.debug.print("  occupancy: {d}\n", .{params.occupancy});
        },
    }

    if (comptime buildconfig.speckle_classified_indexed) {
        const samples: F = @floatFromInt(buildconfig.speckle_mask_samples_per_cell);
        const classification_dims = [2]F{
            @ceil(params.cells_per_uv[0] * samples),
            @ceil(params.cells_per_uv[1] * samples),
        };
        std.debug.print("  classifier: precomputed static 2-bit classification\n", .{});
        std.debug.print(
            "  classification resolution: {d} x {d} microcells ({d} samples/cell)\n",
            .{
                classification_dims[0],
                classification_dims[1],
                buildconfig.speckle_mask_samples_per_cell,
            },
        );
    }

    if (comptime buildconfig.speckle_evaluator == .mask_1bit or
        buildconfig.speckle_evaluator == .mask_u8)
    {
        const samples: F = @floatFromInt(buildconfig.speckle_mask_samples_per_cell);
        const mask_dims = [2]F{
            @ceil(params.cells_per_uv[0] * samples) + 1.0,
            @ceil(params.cells_per_uv[1] * samples) + 1.0,
        };
        const storage = if (comptime buildconfig.speckle_evaluator == .mask_1bit)
            "packed 1-bit coverage"
        else
            "8-bit coverage";
        std.debug.print("  {s}: {s}\n", .{ spec.mask_report_label, storage });
        std.debug.print(
            "  mask resolution: {d} x {d} texels ({d} samples/cell)\n",
            .{ mask_dims[0], mask_dims[1], buildconfig.speckle_mask_samples_per_cell },
        );
    }
}

fn parsePositiveU32(value: []const u8) !u32 {
    const parsed = try std.fmt.parseInt(u32, value, 10);
    if (parsed == 0) return error.InvalidPixelDimension;
    return parsed;
}

fn printUsage(comptime spec: DemoSpec) void {
    std.debug.print(
        "Usage:\n  zig build {s} -Dsimd=off -- [options]\n\n",
        .{spec.command_name},
    );
    if (spec.comparison) |comparison| {
        std.debug.print(
            "Direct comparison:\n  Texture:    zig build {s} -Dsimd=off\n  Procedural: zig build {s} -Dsimd=off\n\n",
            .{ comparison.texture_command, comparison.procedural_command },
        );
    }
    std.debug.print(
        \\Options:
        \\  --size <value>        Mean radius in cell units (disk/Gaussian)
        \\  --occupancy <value>   Active-cell probability (disk/Gaussian)
        \\  --cells-u <value>     Procedural cell count across U
        \\  --cells-v <value>     Procedural cell count across V
        \\  --jitter <value>      Radius variation in cell units (disk/Gaussian)
        \\  --softness <value>    Boundary-blur width in cell units (disk only)
        \\  --threshold <value>   Coverage threshold (Perlin only)
        \\  --transition <value>  Coverage transition width (Perlin only)
        \\  --seed <integer>      Deterministic unsigned 32-bit seed
        \\
    , .{});
    if (spec.dimensions == .exposed_and_reported) {
        std.debug.print(
            "  --width <integer>     Image width in pixels (default: {d})\n" ++
                "  --height <integer>    Image height in pixels (default: {d})\n",
            .{ spec.pixels_num_default[0], spec.pixels_num_default[1] },
        );
    }
    std.debug.print(
        \\  --output <path>       Output directory
        \\  --help                Show this help
        \\
        \\Disk/Gaussian constraints:
        \\  jitter <= size
        \\  size + jitter + effective boundary blur <= 1
        \\
    , .{});
}
