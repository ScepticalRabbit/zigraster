// --------------------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");

const orch = @import("dev_support/orchestration.zig");
const camera = @import("riley/zig/camera.zig");
const iio = @import("riley/zig/imageio.zig");
const meshio = @import("riley/zig/meshio.zig");
const meshpipeline = @import("riley/zig/meshpipeline.zig");
const rastcfg = @import("riley/zig/rasterconfig.zig");
const riley = @import("riley/zig/riley.zig");
const shaderops = @import("riley/zig/shaderops_common.zig");
const uvio = @import("riley/zig/uvio.zig");
const buildconfig = @import("riley/zig/buildconfig.zig");

const F = buildconfig.F;
const data_dir = "data/min/tri6_sphere200/";
const out_dir_def = "./out/demo-procedural-speckles";
const pixel_num = [2]u32{ 720, 450 };

// --------------------------------------------------------------------------------------
// Public Entry-Point Func
// --------------------------------------------------------------------------------------

pub fn main(init: std.process.Init) !void {
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const allocator = arena.allocator();

    const args = (try parseDemoArgs(init.minimal.args.vector)) orelse return;
    try args.params.validate();

    const config = rastcfg.RasterConfig{
        .total_threads = 4,
        .max_geom_workers_per_job = 4,
        .max_raster_workers_per_job = 4,
        .save_strategy = .disk,
        .image_save_mode = .grey,
        .image_save_opts = &[_]iio.ImageSaveOpts{
            .{ .format = .bmp, .bits = 8, .scaling = .none },
        },
        .background_value = 255.0,
        .report = .off,
    };
    var threaded_io = riley.getThreadedIo(
        allocator,
        init.minimal,
        config.total_threads,
    );
    defer threaded_io.deinit();
    const io = threaded_io.io();

    std.debug.print("Procedural speckle prototype\n", .{});
    printProceduralConfig(args.params);
    std.debug.print("Loading UV-mapped sphere and preparing three deformation frames...\n", .{});

    const sim_data = try meshio.loadSimData(
        allocator,
        io,
        data_dir ++ "coords.csv",
        data_dir ++ "connect.csv",
        null,
        null,
    );
    const uvs = try uvio.loadUVMap(allocator, io, data_dir ++ "uvs.csv");
    const disp = try makeDemoDisp(allocator, &sim_data.coords);
    const mesh_input = meshpipeline.MeshInput{
        .mesh_type = .tri6,
        .coords = sim_data.coords,
        .connect = sim_data.connect,
        .disp = disp,
        .shader = .{ .func = .{
            .uvs = uvs.array,
            .coord_mode = .uv,
            .builtin = .speckle,
            .params = args.params.toFuncShaderParams(),
            .bits = 8,
            .scaling = .auto,
            .normal_type = .none,
        } },
    };

    const camera_prep = try orch.initCameraForCoords(
        allocator,
        &sim_data.coords,
        pixel_num,
        1.28,
    );
    const camera_input = camera.CameraInput{
        .pixels_num = camera_prep.pixels_num,
        .pixels_size = camera_prep.pixels_size,
        .pos_world = camera_prep.pos_world,
        .rot_world = camera_prep.rot_world,
        .roi_cent_world = camera_prep.roi_cent_world,
        .focal_length = camera_prep.focal_length,
        .sub_sample = 2,
        .distortion = camera_prep.distortion,
    };
    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = config.total_threads },
    };

    std.debug.print("Rendering reference, stretch, and twist frames...\n", .{});
    const images = try riley.raster(
        allocator,
        &render_groups,
        &[_]camera.CameraInput{camera_input},
        &[_]meshpipeline.MeshInput{mesh_input},
        config,
        args.out_dir,
    );
    if (images) |img_const| {
        var img = img_const;
        allocator.free(img.slice);
        img.deinit(allocator);
    }

    std.debug.print("Demo complete. Images saved under {s}/\n", .{args.out_dir});
}

// --------------------------------------------------------------------------------------
// Argument Handling
// --------------------------------------------------------------------------------------

const DemoArgs = struct {
    params: shaderops.Speckle2DParams = .{},
    out_dir: []const u8 = out_dir_def,
};

fn printProceduralConfig(params: shaderops.Speckle2DParams) void {
    const evaluator_name = switch (comptime buildconfig.speckle_evaluator) {
        .cell_hash => "cell-hash",
        .list_naive => "list-naive",
        .list_indexed => "list-indexed",
        .mask_1bit => "mask-1bit",
        .mask_u8 => "mask-u8",
    };
    const perlin_note = if (comptime buildconfig.speckle_shape == .perlin)
        " (not used by Perlin)"
    else
        "";
    const effective_boundary_blur = if (comptime buildconfig.speckle_shape == .disk and
        buildconfig.speckle_boundary_blur)
        params.edge_softness
    else
        0.0;

    std.debug.print("  evaluator (compile-time): {s}\n", .{evaluator_name});
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
            std.debug.print("  nominal radius: {d} cell units\n", .{params.radius_mean});
            std.debug.print("  radius jitter: {d} cell units\n", .{params.radius_jitter});
            std.debug.print("  occupancy: {d}\n", .{params.occupancy});
        },
    }

    if (comptime buildconfig.speckle_evaluator == .mask_1bit or
        buildconfig.speckle_evaluator == .mask_u8)
    {
        const samples: F = @floatFromInt(buildconfig.speckle_mask_samples_per_cell);
        const mask_dims = [2]F{
            @ceil(params.cells_per_uv[0] * samples) + 1.0,
            @ceil(params.cells_per_uv[1] * samples) + 1.0,
        };
        const allocation = if (comptime buildconfig.speckle_evaluator == .mask_1bit)
            "packed 1-bit coverage"
        else
            "8-bit coverage";
        std.debug.print("  generated mask allocation: {s}\n", .{allocation});
        std.debug.print(
            "  mask resolution: {d} x {d} texels ({d} samples/cell)\n",
            .{ mask_dims[0], mask_dims[1], buildconfig.speckle_mask_samples_per_cell },
        );
    }
}

fn parseDemoArgs(raw_args: anytype) !?DemoArgs {
    var args = DemoArgs{};
    var arg_idx: usize = 1;
    while (arg_idx < raw_args.len) {
        const arg = std.mem.span(raw_args[arg_idx]);
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printUsage();
            return null;
        }
        if (arg_idx + 1 >= raw_args.len) {
            std.debug.print("Missing value for {s}\n\n", .{arg});
            printUsage();
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
        } else if (std.mem.eql(u8, arg, "--output")) {
            args.out_dir = value;
        } else {
            std.debug.print("Unknown option: {s}\n\n", .{arg});
            printUsage();
            return error.UnknownArgument;
        }
        arg_idx += 2;
    }
    return args;
}

fn printUsage() void {
    std.debug.print(
        \\Usage:
        \\  zig build demo-procedural-speckles -Dsimd=off -- [options]
        \\
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
        \\  --output <path>       Output directory
        \\  --help                Show this help
        \\
        \\Disk/Gaussian constraints:
        \\  jitter <= size
        \\  size + jitter + effective boundary blur <= 1
        \\
    , .{});
}

// --------------------------------------------------------------------------------------
// Deformation Preparation
// --------------------------------------------------------------------------------------

fn makeDemoDisp(
    allocator: std.mem.Allocator,
    coords: *const meshio.Coords,
) !meshio.Field {
    const node_num = coords.mat.rows_num;
    var disp = try meshio.Field.initAlloc(allocator, 3, node_num, 3);

    var center = [3]F{ 0.0, 0.0, 0.0 };
    for (0..node_num) |node_idx| {
        center[0] += coords.x(node_idx);
        center[1] += coords.y(node_idx);
        center[2] += coords.z(node_idx);
    }
    const node_num_f: F = @floatFromInt(node_num);
    for (0..3) |axis| center[axis] /= node_num_f;

    var z_extent: F = 0.0;
    for (0..node_num) |node_idx| {
        z_extent = @max(z_extent, @abs(coords.z(node_idx) - center[2]));
    }
    z_extent = @max(z_extent, 1.0e-12);

    for (0..node_num) |node_idx| {
        const rel_x = coords.x(node_idx) - center[0];
        const rel_y = coords.y(node_idx) - center[1];
        const rel_z = coords.z(node_idx) - center[2];

        disp.array.set(&.{ 1, node_idx, 0 }, 0.18 * rel_x);
        disp.array.set(&.{ 1, node_idx, 1 }, -0.08 * rel_y);
        disp.array.set(&.{ 1, node_idx, 2 }, 0.04 * rel_z);

        const twist = 0.16 * rel_z / z_extent;
        disp.array.set(&.{ 2, node_idx, 0 }, -twist * rel_y);
        disp.array.set(&.{ 2, node_idx, 1 }, twist * rel_x);
        disp.array.set(&.{ 2, node_idx, 2 }, 0.12 * rel_z);
    }
    return disp;
}
