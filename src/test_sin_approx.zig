// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const std = @import("std");

const benchcommon = @import("dev_support/benchcommon.zig");
const orch = @import("dev_support/orchestration.zig");
const tcfg = @import("dev_support/testconfig.zig");
const CameraInput = @import("riley/zig/camera.zig").CameraInput;
const CameraPrepared = @import("riley/zig/camera.zig").CameraPrepared;
const Rotation = @import("riley/zig/rotation.zig").Rotation;
const buildconfig = @import("riley/zig/buildconfig.zig");
const cameraops = @import("riley/zig/cameraops.zig");
const iio = @import("riley/zig/imageio.zig");
const meshio = @import("riley/zig/meshio.zig");
const mo = @import("riley/zig/meshpipeline.zig");
const sceneops = @import("riley/zig/sceneops.zig");
const NDArray = @import("riley/zig/ndarray.zig").NDArray;
const riley = @import("riley/zig/riley.zig");
const shaderops = @import("riley/zig/shaderops.zig");

const F = buildconfig.F;

const data_dir = "data/bench/tri3_fullraster";
const out_dir_root = "out/test_sin_approx";
const pixels_num = [_]u32{ 800, 500 };
const pixels_size = [_]F{ 5.3e-6, 5.3e-6 };
const focal_length: F = 50.0e-3;
const sub_sample: u32 = 2;
const wave_oscillations: F = 10.0;
const compare_abs_tol: F = if (F == f32) 1e-4 else 1e-11;

const ImageDiffStats = struct {
    max_abs: F,
    mean_abs: F,
    rmse: F,
};

fn sinusoidalParams(
    builtin: shaderops.FuncShaderBuiltin,
) shaderops.FuncShaderParams {
    const wave_num = 2.0 * std.math.pi * wave_oscillations;
    return switch (builtin) {
        .sinusoidal => .{
            .settings = .{
                .sinusoidal = .{
                    .wave_num_scalar = .{ wave_num, wave_num },
                    .wave_num_rgb = .{ wave_num, wave_num, wave_num },
                },
            },
        },
        .sinusoidal_approx => .{
            .settings = .{
                .sinusoidal_approx = .{
                    .wave_num_scalar = .{ wave_num, wave_num },
                    .wave_num_rgb = .{ wave_num, wave_num, wave_num },
                },
            },
        },
        else => unreachable,
    };
}

fn loadFuncMeshInput(
    allocator: std.mem.Allocator,
    io: std.Io,
    builtin: shaderops.FuncShaderBuiltin,
) !mo.MeshInput {
    const coord_path = try std.fs.path.join(
        allocator,
        &[_][]const u8{ data_dir, "coords.csv" },
    );
    const connect_path = try std.fs.path.join(
        allocator,
        &[_][]const u8{ data_dir, "connect.csv" },
    );

    const sim_data = try meshio.loadSimData(
        allocator,
        io,
        coord_path,
        connect_path,
        null,
        null,
    );

    return .{
        .mesh_type = .tri3,
        .coords = sim_data.coords,
        .connect = sim_data.connect,
        .disp = null,
        .shader = .{
            .func = .{
                .uvs = null,
                .coord_mode = .para,
                .builtin = builtin,
                .params = sinusoidalParams(builtin),
                .bits = 8,
                .scaling = .none,
                .normal_type = .none,
            },
        },
    };
}

fn renderSinImage(
    allocator: std.mem.Allocator,
    io: std.Io,
    builtin: shaderops.FuncShaderBuiltin,
) !NDArray(F) {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    const mesh_input = try loadFuncMeshInput(aa, io, builtin);
    const roi_pos = sceneops.boundsCenter(&mesh_input.coords);
    const cam_pos = cameraops.posFillFrameFromRot(
        &mesh_input.coords,
        pixels_num,
        pixels_size,
        focal_length,
        Rotation.init(0, 0, 0),
        1.0,
    );
    const camera = try CameraPrepared.init(
        aa,
        .{
            .pixels_num = pixels_num,
            .pixels_size = pixels_size,
            .pos_world = cam_pos,
            .rot_world = Rotation.init(0, 0, 0),
            .roi_cent_world = roi_pos,
            .focal_length = focal_length,
            .sub_sample = sub_sample,
            .distortion = .none,
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

    var config = tcfg.getRasterConfig(.bench);
    config.save_strategy = .memory;
    config.image_save_opts = &[_]iio.ImageSaveOpts{};

    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = @max(@as(u16, 1), config.total_threads) },
    };
    const result = (try riley.raster(
        aa,
        &render_groups,
        &[_]CameraInput{camera_input},
        &[_]mo.MeshInput{mesh_input},
        config,
        null,
    )) orelse return error.NoResult;
    defer aa.free(result.slice);

    return try benchcommon.extractFirstFrameImage(allocator, &result);
}

fn calcImageDiffStats(
    lhs: *const NDArray(F),
    rhs: *const NDArray(F),
) !ImageDiffStats {
    if (lhs.dims.len != rhs.dims.len) return error.ShapeMismatch;
    for (lhs.dims, rhs.dims) |lhs_dim, rhs_dim| {
        if (lhs_dim != rhs_dim) return error.ShapeMismatch;
    }
    if (lhs.slice.len != rhs.slice.len) return error.ShapeMismatch;

    var max_abs: F = 0.0;
    var sum_abs: F = 0.0;
    var sum_sq: F = 0.0;

    for (lhs.slice, rhs.slice) |lhs_val, rhs_val| {
        const abs_diff = @abs(lhs_val - rhs_val);
        max_abs = @max(max_abs, abs_diff);
        sum_abs += abs_diff;
        sum_sq += abs_diff * abs_diff;
    }

    const count_f = @as(F, @floatFromInt(lhs.slice.len));
    return .{
        .max_abs = max_abs,
        .mean_abs = sum_abs / count_f,
        .rmse = @sqrt(sum_sq / count_f),
    };
}

fn makeAbsDiffImage(
    allocator: std.mem.Allocator,
    lhs: *const NDArray(F),
    rhs: *const NDArray(F),
) !NDArray(F) {
    if (lhs.dims.len != rhs.dims.len) return error.ShapeMismatch;

    var diff = try NDArray(F).initFlat(allocator, lhs.dims);
    for (0..lhs.slice.len) |ii| {
        diff.slice[ii] = @abs(lhs.slice[ii] - rhs.slice[ii]);
    }
    return diff;
}

fn saveGreyBMP(
    io: std.Io,
    out_dir: std.Io.Dir,
    file_stem: []const u8,
    image: *const NDArray(F),
) !void {
    try iio.saveImage(
        io,
        out_dir,
        file_stem,
        image,
        0,
        .{
            .format = .bmp,
            .bits = 8,
            .scaling = .auto,
            .channels = 1,
        },
    );
}

test "sinusoidal approx matches builtin fullraster render" {
    const allocator = std.heap.page_allocator;
    const io = std.testing.io;

    var builtin_image = try renderSinImage(
        allocator,
        io,
        .sinusoidal,
    );
    defer {
        allocator.free(builtin_image.slice);
        builtin_image.deinit(allocator);
    }

    var approx_image = try renderSinImage(
        allocator,
        io,
        .sinusoidal_approx,
    );
    defer {
        allocator.free(approx_image.slice);
        approx_image.deinit(allocator);
    }

    const stats = try calcImageDiffStats(
        &builtin_image,
        &approx_image,
    );

    var diff_image = try makeAbsDiffImage(
        allocator,
        &builtin_image,
        &approx_image,
    );
    defer {
        allocator.free(diff_image.slice);
        diff_image.deinit(allocator);
    }

    var out_dir = try orch.openDirEnsured(io, out_dir_root);
    defer out_dir.close(io);

    try saveGreyBMP(io, out_dir, "sinusoidal_builtin", &builtin_image);
    try saveGreyBMP(io, out_dir, "sinusoidal_approx", &approx_image);
    try saveGreyBMP(io, out_dir, "sinusoidal_abs_diff", &diff_image);

    std.debug.print(
        "sin approx diff: max={e:.6}, mean={e:.6}, rmse={e:.6}\n",
        .{ stats.max_abs, stats.mean_abs, stats.rmse },
    );

    try std.testing.expect(stats.max_abs <= compare_abs_tol);
}
