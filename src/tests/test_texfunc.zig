// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const std = @import("std");
const Timestamp = std.Io.Clock.Timestamp;
const common = @import("../dev_support/tests.zig");
const policy = @import("../dev_support/testpolicy.zig");
const orch = @import("../dev_support/orchestration.zig");
const tcfg = @import("../dev_support/testconfig.zig");
const gk = @import("../riley/zig/geometrykernels.zig");
const meshio = @import("../riley/zig/meshio.zig");
const mo = @import("../riley/zig/meshpipeline.zig");
const shaderops = @import("../riley/zig/shaderops.zig");
const uvio = @import("../riley/zig/uvio.zig");
const CameraInput = @import("../riley/zig/camera.zig").CameraInput;
const iio = @import("../riley/zig/imageio.zig");
const riley = @import("../riley/zig/riley.zig");
const buildconfig = @import("../riley/zig/buildconfig.zig");
const F = buildconfig.F;

const gold_root = policy.goldRoot(.texfunc);
const data_root = "data/min";
const test_type = "sphere200";
const CoordMode = enum { uv, param };

const SphereCasePrepared = struct {
    coords: meshio.Coords,
    connect: meshio.Connect,
    uvs: uvio.UVMap,
    camera_input: CameraInput,
};

fn loadSphereCase(
    allocator: std.mem.Allocator,
    io: std.Io,
    mesh_type: gk.MeshType,
) !SphereCasePrepared {
    const data_dir = try std.fmt.allocPrint(
        allocator,
        "{s}/{s}_{s}",
        .{ data_root, @tagName(mesh_type), test_type },
    );
    const coord_path = try std.fs.path.join(
        allocator,
        &[_][]const u8{ data_dir, "coords.csv" },
    );
    const connect_path = try std.fs.path.join(
        allocator,
        &[_][]const u8{ data_dir, "connect.csv" },
    );
    const uv_path = try std.fs.path.join(
        allocator,
        &[_][]const u8{ data_dir, "uvs.csv" },
    );

    const sim_data = try meshio.loadSimData(
        allocator,
        io,
        coord_path,
        connect_path,
        null,
        null,
    );
    const uvs = try uvio.loadUVMap(allocator, io, uv_path);
    const camera = try orch.initCameraForCoords(
        allocator,
        &sim_data.coords,
        .{ 640, 400 },
        1.0,
    );

    return .{
        .coords = sim_data.coords,
        .connect = sim_data.connect,
        .uvs = uvs,
        .camera_input = CameraInput{
            .pixels_num = camera.pixels_num,
            .pixels_size = camera.pixels_size,
            .pos_world = camera.pos_world,
            .rot_world = camera.rot_world,
            .roi_cent_world = camera.roi_cent_world,
            .focal_length = camera.focal_length,
            .sub_sample = camera.sub_sample,
            .distortion = camera.distortion,
        },
    };
}

fn runTexFuncCase(
    allocator: std.mem.Allocator,
    io: std.Io,
    mesh_type: gk.MeshType,
    builtin: shaderops.FuncShaderBuiltin,
    coord_mode: CoordMode,
    is_rgb: bool,
) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    const prepared = try loadSphereCase(
        aa,
        io,
        mesh_type,
    );

    const coord_name = if (coord_mode == .uv) "uv" else "param";
    const uvs = if (coord_mode == .uv) prepared.uvs.array else null;
    const normal_type: shaderops.NormalType =
        if (builtin == .lambertian_normal_z) .avg else .none;

    const case_dir_name = if (is_rgb)
        try std.fmt.allocPrint(
            aa,
            "{s}_{s}_texfunc_rgb_{s}_{s}",
            .{
                test_type,
                @tagName(mesh_type),
                coord_name,
                @tagName(builtin),
            },
        )
    else
        try std.fmt.allocPrint(
            aa,
            "{s}_{s}_texfunc_{s}_{s}",
            .{
                test_type,
                @tagName(mesh_type),
                coord_name,
                @tagName(builtin),
            },
        );
    const gold_dir = try std.fmt.allocPrint(aa, "{s}/{s}", .{ gold_root, case_dir_name });

    const mesh_input = mo.MeshInput{
        .mesh_type = mesh_type,
        .coords = prepared.coords,
        .connect = prepared.connect,
        .disp = null,
        .shader = if (is_rgb)
            .{
                .func_rgb = .{
                    .uvs = uvs,
                    .coord_mode = if (coord_mode == .uv) .uv else .para,
                    .builtin = builtin,
                    .normal_type = normal_type,
                },
            }
        else
            .{
                .func = .{
                    .uvs = uvs,
                    .coord_mode = if (coord_mode == .uv) .uv else .para,
                    .builtin = builtin,
                    .normal_type = normal_type,
                },
            },
    };

    if (tcfg.TEST_CASE_VERBOSE) {
        std.debug.print("Testing {s} ... ", .{case_dir_name});
    }

    var config = tcfg.getRasterConfig(.testing);
    config.save_strategy = .memory;
    config.image_save_opts = &[_]iio.ImageSaveOpts{
        .{ .format = .csv, .bits = null, .scaling = .none },
    };

    const camera_input: CameraInput = prepared.camera_input;
    const time_start = Timestamp.now(io, .awake);
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
    const time_end = Timestamp.now(io, .awake);
    const duration_ms = @as(F, @floatFromInt(time_start.durationTo(time_end).raw.nanoseconds)) / 1e6;

    const frames_num = if (result.dims.len == 5)
        result.dims[1]
    else
        result.dims[0];
    var first_err: ?anyerror = null;
    for (0..frames_num) |frame_idx| {
        if (is_rgb) {
            for (0..3) |channel_idx| {
                const gold_path = try common.findGoldPath(
                    aa,
                    io,
                    gold_dir,
                    0,
                    frame_idx,
                    channel_idx,
                    false,
                );

                common.compareNDArrayToGold(
                    aa,
                    io,
                    &result,
                    0,
                    frame_idx,
                    channel_idx,
                    1,
                    gold_path,
                    tcfg.REL_TOL,
                    tcfg.ABS_TOL,
                ) catch |err| {
                    if (first_err == null) first_err = err;
                    const fail_dir_name = try std.fmt.allocPrint(
                        aa,
                        "all_{s}{s}",
                        .{ case_dir_name, common.impl_suffix },
                    );
                    try common.saveComparisonArtifactsFromResult(
                        aa,
                        io,
                        common.default_fails_root,
                        fail_dir_name,
                        &result,
                        0,
                        frame_idx,
                        channel_idx,
                        gold_path,
                        1,
                    );
                };
            }
        } else {
            const gold_path = try common.findGoldPath(
                aa,
                io,
                gold_dir,
                0,
                frame_idx,
                0,
                false,
            );

            common.compareNDArrayToGold(
                aa,
                io,
                &result,
                0,
                frame_idx,
                0,
                1,
                gold_path,
                tcfg.REL_TOL,
                tcfg.ABS_TOL,
            ) catch |err| {
                if (first_err == null) first_err = err;
                const fail_dir_name = try std.fmt.allocPrint(
                    aa,
                    "all_{s}{s}",
                    .{ case_dir_name, common.impl_suffix },
                );
                try common.saveComparisonArtifactsFromResult(
                    aa,
                    io,
                    common.default_fails_root,
                    fail_dir_name,
                    &result,
                    0,
                    frame_idx,
                    0,
                    gold_path,
                    1,
                );
            };
        }
    }

    if (first_err) |err| return err;

    if (tcfg.TEST_CASE_VERBOSE) {
        std.debug.print("MATCHED ({d:.2} ms)\n", .{duration_ms});
    }
}

test "TexFunc Suite" {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const simd_on = buildconfig.config.simd == .on;
    std.debug.print("Running TexFunc Gold Tests with .simd = .{s}...\n", .{
        if (simd_on) "on" else "off",
    });
    const suite_start = Timestamp.now(std.testing.io, .awake);

    const io = std.testing.io;
    const mesh_types = [_]gk.MeshType{
        .tri3,
        .tri6,
        .quad4ibi,
        .quad4newton,
        .quad8,
        .quad9,
    };
    const builtins = [_]shaderops.FuncShaderBuiltin{
        .constant,
        .linear,
        .quadratic,
        .sinusoidal,
        .checker_smooth,
        .lambertian_normal_z,
    };
    const coord_modes = [_]CoordMode{ .uv, .param };
    const rgb_modes = [_]bool{ false, true };

    for (mesh_types) |mesh_type| {
        for (builtins) |builtin| {
            for (coord_modes) |coord_mode| {
                for (rgb_modes) |is_rgb| {
                    try runTexFuncCase(
                        allocator,
                        io,
                        mesh_type,
                        builtin,
                        coord_mode,
                        is_rgb,
                    );
                }
            }
        }
    }

    const suite_end = Timestamp.now(io, .awake);
    const suite_ms = @as(
        F,
        @floatFromInt(suite_start.durationTo(suite_end).raw.nanoseconds),
    ) / 1e6;
    std.debug.print("TexFunc Gold Test Suite took {d:.3} ms\n", .{suite_ms});
}
