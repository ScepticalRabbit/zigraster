// --------------------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");

const buildconfig = @import("riley/zig/buildconfig.zig");
const camera = @import("riley/zig/camera.zig");
const cameraops = @import("riley/zig/cameraops.zig");
const iio = @import("riley/zig/imageio.zig");
const meshio = @import("riley/zig/meshio.zig");
const meshpipeline = @import("riley/zig/meshpipeline.zig");
const rastcfg = @import("riley/zig/rasterconfig.zig");
const riley = @import("riley/zig/riley.zig");
const Rotation = @import("riley/zig/rotation.zig").Rotation;
const sceneops = @import("riley/zig/sceneops.zig");
const shaderops = @import("riley/zig/shaderops_common.zig");
const uvio = @import("riley/zig/uvio.zig");

const F = buildconfig.F;
const data_dir = "data/rabbits/riley_tri6/";
const out_dir_def = "out/demo-procedural-rabbit";
const pixel_num = [2]u32{ 800, 500 };
const pixel_size = [2]F{ @floatCast(5.3e-6), @floatCast(5.3e-6) };
const focal_leng: F = @floatCast(50.0e-3);

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
        .save_strategy = .disk,
        .total_threads = 4,
        .max_raster_workers_per_job = 4,
        .image_save_opts = &[_]iio.ImageSaveOpts{
            .{ .format = .bmp, .bits = 8, .scaling = .auto },
        },
        .report = .bench,
    };
    var threaded_io = riley.getThreadedIo(
        allocator,
        init.minimal,
        config.total_threads,
    );
    defer threaded_io.deinit();
    const io = threaded_io.io();

    std.debug.print("Procedural rabbit demo\n", .{});
    printProceduralConfig(args.params, args.pixels_num);

    const sim_data = try meshio.loadSimData(
        allocator,
        io,
        data_dir ++ "coords.csv",
        data_dir ++ "connectivity.csv",
        null,
        null,
    );
    const uvs = try uvio.loadUVMap(allocator, io, data_dir ++ "uvs.csv");
    const mesh_input = meshpipeline.MeshInput{
        .mesh_type = .tri6,
        .coords = sim_data.coords,
        .connect = sim_data.connect,
        .disp = null,
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

    const rot = Rotation.init(0.0, std.math.pi, 0.0);
    const roi_pos = sceneops.boundsCenter(&sim_data.coords);
    const cam_pos = cameraops.posFillFrameFromRot(
        &sim_data.coords,
        args.pixels_num,
        pixel_size,
        focal_leng,
        rot,
        1.01,
    );
    const camera_prep = try camera.CameraPrepared.init(
        allocator,
        .{
            .pixels_num = args.pixels_num,
            .pixels_size = pixel_size,
            .pos_world = cam_pos,
            .rot_world = rot,
            .roi_cent_world = roi_pos,
            .focal_length = focal_leng,
            .sub_sample = 2,
        },
    );
    const camera_input = camera.CameraInput{
        .pixels_num = camera_prep.pixels_num,
        .pixels_size = camera_prep.pixels_size,
        .pos_world = camera_prep.pos_world,
        .rot_world = camera_prep.rot_world,
        .roi_cent_world = camera_prep.roi_cent_world,
        .focal_length = camera_prep.focal_length,
        .sub_sample = camera_prep.sub_sample,
        .distortion = camera_prep.distortion,
    };
    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = config.total_threads },
    };

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

    std.debug.print("Rabbit image saved under {s}/\n", .{args.out_dir});
}

// --------------------------------------------------------------------------------------
// Argument Handling
// --------------------------------------------------------------------------------------

const DemoArgs = struct {
    params: shaderops.Speckle2DParams = if (buildconfig.speckle_direct_fixed)
        .{ .radius_jitter = 0.0 }
    else
        .{},
    out_dir: []const u8 = out_dir_def,
    pixels_num: [2]u32 = pixel_num,
};

fn printProceduralConfig(params: shaderops.Speckle2DParams, pixels_num: [2]u32) void {
    const evaluator_name = switch (comptime buildconfig.speckle_evaluator) {
        .cell_hash => "cell-hash",
        .list_naive => "list-naive",
        .list_indexed => "list-indexed",
        .classified_indexed => "classified-indexed",
        .direct_fixed => "direct-fixed",
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

    std.debug.print("  image dimensions: {d} x {d} pixels\n", .{ pixels_num[0], pixels_num[1] });
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
            if (comptime buildconfig.speckle_evaluator == .direct_fixed) {
                std.debug.print("  fixed radius: {d} cell units\n", .{params.radius_mean});
                std.debug.print("  radius jitter: zero (required by direct-fixed)\n", .{});
            } else {
                std.debug.print("  nominal radius: {d} cell units\n", .{params.radius_mean});
                std.debug.print("  radius jitter: {d} cell units\n", .{params.radius_jitter});
            }
            std.debug.print("  occupancy: {d}\n", .{params.occupancy});
        },
    }

    if (comptime buildconfig.speckle_evaluator == .classified_indexed) {
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
        std.debug.print("  mask storage (compile-time): {s}\n", .{storage});
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
        } else if (std.mem.eql(u8, arg, "--width")) {
            args.pixels_num[0] = try parsePositiveU32(value);
        } else if (std.mem.eql(u8, arg, "--height")) {
            args.pixels_num[1] = try parsePositiveU32(value);
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

fn parsePositiveU32(value: []const u8) !u32 {
    const parsed = try std.fmt.parseInt(u32, value, 10);
    if (parsed == 0) return error.InvalidPixelDimension;
    return parsed;
}

fn printUsage() void {
    std.debug.print(
        \\Usage:
        \\  zig build demo-procedural-rabbit -Dsimd=off -- [options]
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
        \\  --width <integer>     Image width in pixels (default: 800)
        \\  --height <integer>    Image height in pixels (default: 500)
        \\  --output <path>       Output directory
        \\  --help                Show this help
        \\
        \\Disk/Gaussian constraints:
        \\  jitter <= size
        \\  size + jitter + effective boundary blur <= 1
        \\
    , .{});
}
