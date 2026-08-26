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
const data_dir = "data/min/tri6_sphere200/";
const out_dir_def = "./out/demo-procedural-sphere200";
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

    std.debug.print("Procedural sphere200 comparison demo\n", .{});
    std.debug.print("  texture baseline: zig build demo-sphere200 -Dsimd=off\n", .{});
    std.debug.print("  seed: {d} (0x{x})\n", .{ args.params.seed, args.params.seed });
    std.debug.print(
        "  cells per UV: {d} x {d}\n",
        .{ args.params.cells_per_uv[0], args.params.cells_per_uv[1] },
    );
    std.debug.print("  nominal radius/size: {d} cell units\n", .{args.params.radius_mean});
    std.debug.print("  occupancy: {d}\n", .{args.params.occupancy});
    std.debug.print("  texture allocation: none\n", .{});

    const sim_data = try meshio.loadSimData(
        allocator,
        io,
        data_dir ++ "coords.csv",
        data_dir ++ "connect.csv",
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

    const rot = Rotation.init(0.0, 0.0, 0.0);
    const roi_pos = sceneops.boundsCenter(&sim_data.coords);
    const cam_pos = cameraops.posFillFrameFromRot(
        &sim_data.coords,
        pixel_num,
        pixel_size,
        focal_leng,
        rot,
        1.0,
    );
    const camera_prep = try camera.CameraPrepared.init(
        allocator,
        .{
            .pixels_num = pixel_num,
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

    std.debug.print("Comparison image saved under {s}/\n", .{args.out_dir});
}

// --------------------------------------------------------------------------------------
// Argument Handling
// --------------------------------------------------------------------------------------

const DemoArgs = struct {
    params: shaderops.Speckle2DParams = .{},
    out_dir: []const u8 = out_dir_def,
};

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
        \\  zig build demo-procedural-sphere200 -Dsimd=off -- [options]
        \\
        \\Direct comparison:
        \\  Texture:    zig build demo-sphere200 -Dsimd=off
        \\  Procedural: zig build demo-procedural-sphere200 -Dsimd=off
        \\
        \\Options:
        \\  --size <value>       Mean speckle radius in cell units
        \\  --occupancy <value>  Active-cell probability in [0, 1]
        \\  --cells-u <value>    Procedural cell count across U
        \\  --cells-v <value>    Procedural cell count across V
        \\  --jitter <value>     Symmetric radius variation in cell units
        \\  --softness <value>   Edge-transition width in cell units
        \\  --seed <integer>     Deterministic unsigned 32-bit seed
        \\  --output <path>      Output directory
        \\  --help               Show this help
        \\
        \\Constraints:
        \\  jitter <= size
        \\  size + jitter + softness <= 1
        \\
    , .{});
}
