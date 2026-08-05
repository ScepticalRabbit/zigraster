// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const std = @import("std");
const buildconfig = @import("riley/zig/buildconfig.zig");
const common = @import("dev_support/benchcommon.zig");
const orch = @import("dev_support/orchestration.zig");
const tcfg = @import("dev_support/testconfig.zig");
const riley = @import("riley/zig/riley.zig");
const rastcfg = @import("riley/zig/rasterconfig.zig");
const gk = @import("riley/zig/geometrykernels.zig");
const mo = @import("riley/zig/meshpipeline.zig");
const iio = @import("riley/zig/imageio.zig");
const cam = @import("riley/zig/camera.zig");
const cameraops = @import("riley/zig/cameraops.zig");
const newton = @import("riley/zig/newton.zig");
const rops = @import("riley/zig/rasterops.zig");
const vsd = @import("riley/zig/vecsimd.zig");
const csvio = @import("riley/zig/csvio.zig");
const Rotation = @import("riley/zig/rotation.zig").Rotation;
const sceneops = @import("riley/zig/sceneops.zig");

const F = buildconfig.F;
const CameraPrepared = cam.CameraPrepared;
const CameraInput = cam.CameraInput;

fn loadRepresentativePixelsByStatus(
    allocator: std.mem.Allocator,
    io: std.Io,
    solver_dir: []const u8,
    solver_status_match: []const u8,
    max_pixels: usize,
) ![]const [2]usize {
    if (max_pixels == 0) {
        return allocator.alloc([2]usize, 0);
    }

    const csv_path = try std.fs.path.join(allocator, &[_][]const u8{
        solver_dir,
        "diag_cam0_frame0_solver.csv",
    });
    defer allocator.free(csv_path);

    var pixels = std.ArrayList([2]usize).empty;
    errdefer pixels.deinit(allocator);

    var lines = try csvio.readCsvToList(allocator, io, csv_path);
    defer csvio.freeCsvLines(allocator, &lines);

    for (lines.items, 0..) |line, line_idx| {
        if (line_idx == 0) continue;
        var field_iter = std.mem.splitScalar(u8, line, ',');
        const subpx_x_str = field_iter.next() orelse continue;
        const subpx_y_str = field_iter.next() orelse continue;
        _ = field_iter.next() orelse continue;
        _ = field_iter.next() orelse continue;
        const solver_status = field_iter.next() orelse continue;
        if (!std.mem.eql(u8, solver_status, solver_status_match)) continue;

        const subpx_x = try std.fmt.parseUnsigned(usize, subpx_x_str, 10);
        const subpx_y = try std.fmt.parseUnsigned(usize, subpx_y_str, 10);
        try pixels.append(allocator, .{ subpx_x, subpx_y });
        if (pixels.items.len >= max_pixels) break;
    }

    return try pixels.toOwnedSlice(allocator);
}

fn tracePixelSet(
    writer: anytype,
    heading: []const u8,
    pixels: []const [2]usize,
    camera: *const CameraPrepared,
    mesh_input: *const mo.MeshInput,
) !void {
    try writer.print("{s}: {any}\n\n", .{ heading, pixels });
    if (pixels.len == 0) {
        try writer.writeAll("No matching pixels found.\n\n");
        return;
    }

    const elem_coords = rops.gatherElemNodeCoords(
        8,
        &mesh_input.coords,
        &mesh_input.connect,
        0,
    );
    const coords_world = vsd.Vec3SIMD(8, F){
        .x = elem_coords.x,
        .y = elem_coords.y,
        .z = elem_coords.z,
    };

    const x_scale = camera.image_dist *
        @as(F, @floatFromInt(camera.pixels_num[0])) / camera.image_dims[0];
    const y_scale = camera.image_dist *
        @as(F, @floatFromInt(camera.pixels_num[1])) / camera.image_dims[1];
    var coords_clip = vsd.mat44Mul(8, F, camera.world_to_cam_mat, coords_world);
    coords_clip.x *= @splat(x_scale);
    coords_clip.y *= @splat(-y_scale);
    coords_clip.z = -coords_clip.z;
    const node_x: [8]F = coords_clip.x;
    const node_y: [8]F = coords_clip.y;
    const node_w: [8]F = coords_clip.z;

    const offsets = camera.calcRasterOffsets();
    const seed = gk.Quad89Kernel(8).initSeed(.centroid, null);
    const sub_sample_f: F = @floatFromInt(camera.sub_sample);

    for (pixels) |debug_pixel| {
        const subpx_x = debug_pixel[0];
        const subpx_y = debug_pixel[1];
        const target_x = (@as(F, @floatFromInt(subpx_x)) + 0.5) /
            sub_sample_f - offsets.x_off;
        const target_y = (@as(F, @floatFromInt(subpx_y)) + 0.5) /
            sub_sample_f - offsets.y_off;
        try newton.reportSolve(
            8,
            writer,
            subpx_x,
            subpx_y,
            target_x,
            target_y,
            node_x[0..],
            node_y[0..],
            node_w[0..],
            seed.xi,
            seed.eta,
        );
    }
}

fn traceRepresentativePixels(
    allocator: std.mem.Allocator,
    io: std.Io,
    out_dir: []const u8,
    solver_dir: []const u8,
    camera: *const CameraPrepared,
    mesh_input: *const mo.MeshInput,
) !void {
    const fail_pixels = try loadRepresentativePixelsByStatus(
        allocator,
        io,
        solver_dir,
        "fail_iter_lim",
        8,
    );
    defer allocator.free(fail_pixels);

    const converged_pixels = try loadRepresentativePixelsByStatus(
        allocator,
        io,
        solver_dir,
        "converged_resid",
        8,
    );
    defer allocator.free(converged_pixels);

    const cwd_dir = std.Io.Dir.cwd();
    var out_dir_handle = try cwd_dir.openDir(io, out_dir, .{});
    defer out_dir_handle.close(io);

    const console_file = try out_dir_handle.createFile(
        io,
        "console_out.txt",
        .{},
    );
    defer console_file.close(io);

    var write_buf: [4096]u8 = undefined;
    var file_writer = console_file.writer(io, &write_buf);
    const writer = &file_writer.interface;

    try tracePixelSet(
        writer,
        "Representative failed pixels",
        fail_pixels,
        camera,
        mesh_input,
    );
    try tracePixelSet(
        writer,
        "Representative converged_resid pixels",
        converged_pixels,
        camera,
        mesh_input,
    );

    try file_writer.flush();
}

pub fn main(init: std.process.Init) !void {
    @setEvalBranchQuota(buildconfig.comptime_eval_branch_quota);
    const allocator = init.gpa;
    const io = init.io;

    const texture_grey = try iio.loadImage(
        u8,
        1,
        allocator,
        io,
        "texture/speckle.bmp",
        .bmp,
    );
    defer texture_grey.deinit(allocator);

    const texture_rgb = try iio.loadImage(
        u8,
        3,
        allocator,
        io,
        "texture/speckle_rgb.bmp",
        .bmp,
    );
    defer texture_rgb.deinit(allocator);

    const data_dir = "data/bench/quad8_fullraster";
    const render_defaults = common.BenchRenderDefaults{
        .pixels_num = .{ 800, 500 },
        .sub_sample = 1,
        .focal_leng = 50.0e-3,
        .pixels_size = .{ 5.3e-6, 5.3e-6 },
        .fov_scale = 1.0,
        .rot = Rotation.init(0, 0, 0),
    };
    const f32_solver_dir = "out/check_newton/quad8_nodal_rgb_centroid_guess";
    const out_dir = if (F == f32)
        f32_solver_dir
    else if (F == f64)
        "out/check_newton_f64/quad8_nodal_rgb_centroid_guess"
    else
        @compileError("Only f32 and f64 precision are supported.");

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    const mesh_input = try common.loadBenchmarkMeshInput(
        u8,
        aa,
        io,
        .quad8,
        .nodal_rgb,
        null,
        null,
        data_dir,
        texture_grey,
        texture_rgb,
    );

    const roi_pos = sceneops.boundsCenter(&mesh_input.coords);
    const cam_pos = cameraops.posFillFrameFromRot(
        &mesh_input.coords,
        render_defaults.pixels_num,
        render_defaults.pixels_size,
        render_defaults.focal_leng,
        render_defaults.rot,
        render_defaults.fov_scale,
    );
    const camera = try CameraPrepared.init(
        aa,
        .{
            .pixels_num = render_defaults.pixels_num,
            .pixels_size = render_defaults.pixels_size,
            .pos_world = cam_pos,
            .rot_world = render_defaults.rot,
            .roi_cent_world = roi_pos,
            .focal_length = render_defaults.focal_leng,
            .sub_sample = render_defaults.sub_sample,
            .distortion = render_defaults.distortion,
        },
    );
    defer camera.deinit(aa);

    const camera_input = CameraInput{
        .pixels_num = camera.pixels_num,
        .pixels_size = camera.pixels_size,
        .pos_world = camera.pos_world,
        .rot_world = camera.rot_world,
        .roi_cent_world = camera.roi_cent_world,
        .focal_length = camera.focal_length,
        .sub_sample = camera.sub_sample,
        .distortion = camera.distortion,
    };

    var config = tcfg.getRasterConfig(.testing);
    if (F == f32 or F == f64) {
        config.newton_seed_mode = .centroid;
    }
    config.total_threads = 1;
    config.max_geom_workers_per_job = 1;
    config.max_raster_workers_per_job = 1;
    config.max_geom_jobs_in_flight_per_group = 1;
    config.frame_batch_size_per_group = 1;
    config.save_strategy = .disk;
    config.report = .full_stats;
    config.image_save_opts = &[_]iio.ImageSaveOpts{
        .{ .format = .fimg, .bits = null, .scaling = .none, .channels = 3 },
        .{ .format = .bmp, .bits = 8, .scaling = .auto, .channels = 3 },
    };
    config.full_stats_opts = .{
        .formats = &[_]iio.ImageSaveOpts{
            .{ .format = .bmp, .bits = 8, .scaling = .auto },
            .{ .format = .csv, .bits = null, .scaling = .none },
        },
        .save_solver_csv = true,
        .save_iter_map = true,
        .save_xi_map = true,
        .save_eta_map = true,
        .save_conv_map = true,
        .save_jac_det_map = true,
        .save_tile_timing_map = true,
        .save_tile_density_map = true,
        .save_tile_occupancy_map = true,
        .save_depth_map = true,
        .save_earlyout_map = true,
        .save_pixel_occupancy_map = true,
        .save_normals_map = false,
    };

    var out_dir_handle = try orch.openDirEnsured(io, out_dir);
    out_dir_handle.close(io);

    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = @max(@as(u16, 1), config.total_threads) },
    };

    const images = try riley.raster(
        aa,
        &render_groups,
        &[_]CameraInput{camera_input},
        &[_]mo.MeshInput{mesh_input},
        config,
        out_dir,
    );
    if (images) |img| {
        aa.free(img.slice);
        var img_mut = img;
        img_mut.deinit(aa);
    }

    const trace_source_dir = if (F == f32) out_dir else f32_solver_dir;
    try traceRepresentativePixels(
        aa,
        io,
        out_dir,
        trace_source_dir,
        &camera,
        &mesh_input,
    );

    std.debug.print(
        "Saved quad8 nodal RGB full_stats diagnostics to {s}\n",
        .{out_dir},
    );
}
