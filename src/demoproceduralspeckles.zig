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
const pdemo = @import("dev_support/proceduraldemo.zig");
const camera = @import("riley/zig/camera.zig");
const iio = @import("riley/zig/imageio.zig");
const meshio = @import("riley/zig/meshio.zig");
const meshpipeline = @import("riley/zig/meshpipeline.zig");
const rastcfg = @import("riley/zig/rasterconfig.zig");
const riley = @import("riley/zig/riley.zig");
const uvio = @import("riley/zig/uvio.zig");
const buildconfig = @import("riley/zig/buildconfig.zig");

const F = buildconfig.F;
const data_dir = "data/min/tri6_sphere200/";
const demo_spec = pdemo.DemoSpec{
    .command_name = "demo-procedural-speckles",
    .output_default = "./out/demo-procedural-speckles",
    .pixels_num_default = .{ 720, 450 },
    .dimensions = .hidden,
    .mask_report_label = "generated mask allocation",
};

// --------------------------------------------------------------------------------------
// Public Entry-Point Func
// --------------------------------------------------------------------------------------

pub fn main(init: std.process.Init) !void {
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const allocator = arena.allocator();

    const args = (try pdemo.parseDemoArgs(init.minimal.args.vector, demo_spec)) orelse return;
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
    pdemo.printProceduralConfig(args.params, args.pixels_num, demo_spec);
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
        args.pixels_num,
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
