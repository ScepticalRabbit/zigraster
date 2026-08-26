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
const CameraInput = @import("../riley/zig/camera.zig").CameraInput;
const iio = @import("../riley/zig/imageio.zig");
const rastcfg = @import("../riley/zig/rasterconfig.zig");
const rotation = @import("../riley/zig/rotation.zig");
const riley = @import("../riley/zig/riley.zig");
const vec = @import("../riley/zig/vecstack.zig");
const buildconfig = @import("../riley/zig/buildconfig.zig");
const F = buildconfig.F;

const gold_root = policy.goldRoot(.projected_tex);
const data_root = "data/min";
const test_type = "sphere200";

const SphereCasePrepared = struct {
    coords: meshio.Coords,
    connect: meshio.Connect,
    camera_ref: CameraInput,
    camera_stereo: CameraInput,
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

    const sim_data = try meshio.loadSimData(
        allocator,
        io,
        coord_path,
        connect_path,
        null,
        null,
    );

    const camera_ref_prep = try orch.initCameraForCoords(
        allocator,
        &sim_data.coords,
        .{ 640, 400 },
        1.0,
    );

    const camera_ref = CameraInput{
        .pixels_num = camera_ref_prep.pixels_num,
        .pixels_size = camera_ref_prep.pixels_size,
        .pos_world = camera_ref_prep.pos_world,
        .rot_world = camera_ref_prep.rot_world,
        .roi_cent_world = camera_ref_prep.roi_cent_world,
        .focal_length = camera_ref_prep.focal_length,
        .sub_sample = camera_ref_prep.sub_sample,
        .distortion = camera_ref_prep.distortion,
    };

    const stereo_angle: F = 15.0 * std.math.pi / 180.0;
    const cam_dist: F = camera_ref_prep.pos_world.get(2);
    const pos_stereo = vec.initVec3(
        F,
        cam_dist * @sin(stereo_angle),
        0.0,
        cam_dist * @cos(stereo_angle),
    );
    const rot_stereo = rotation.Rotation.init(0.0, stereo_angle, 0.0);

    const camera_stereo = CameraInput{
        .pixels_num = camera_ref_prep.pixels_num,
        .pixels_size = camera_ref_prep.pixels_size,
        .pos_world = pos_stereo,
        .rot_world = rot_stereo,
        .roi_cent_world = camera_ref_prep.roi_cent_world,
        .focal_length = camera_ref_prep.focal_length,
        .sub_sample = camera_ref_prep.sub_sample,
        .distortion = camera_ref_prep.distortion,
    };

    return .{
        .coords = sim_data.coords,
        .connect = sim_data.connect,
        .camera_ref = camera_ref,
        .camera_stereo = camera_stereo,
    };
}

fn runProjectedTexCase(
    allocator: std.mem.Allocator,
    io: std.Io,
    mesh_type: gk.MeshType,
    is_stereo: bool,
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

    const cam_name = if (is_stereo) "stereo" else "ref";
    const target_cam = if (is_stereo)
        prepared.camera_stereo
    else
        prepared.camera_ref;

    const case_dir_name = if (is_rgb)
        try std.fmt.allocPrint(
            aa,
            "{s}_{s}_projtex_rgb_{s}",
            .{ test_type, @tagName(mesh_type), cam_name },
        )
    else
        try std.fmt.allocPrint(
            aa,
            "{s}_{s}_projtex_grey_{s}",
            .{ test_type, @tagName(mesh_type), cam_name },
        );
    const gold_dir = try std.fmt.allocPrint(aa, "{s}/{s}", .{ gold_root, case_dir_name });

    const tex_grey = try iio.loadImage(
        u8,
        1,
        aa,
        io,
        "texture/speckle.bmp",
        .bmp,
    );
    const tex_rgb = try iio.loadImage(
        u8,
        3,
        aa,
        io,
        "texture/speckle_rgb.bmp",
        .bmp,
    );

    const mesh_input = mo.MeshInput{
        .mesh_type = mesh_type,
        .coords = prepared.coords,
        .connect = prepared.connect,
        .disp = null,
        .shader = if (is_rgb)
            .{
                .projected_tex_rgb_u8 = .{
                    .ref_camera = prepared.camera_ref,
                    .ref_coords = null,
                    .tex = tex_rgb,
                },
            }
        else
            .{
                .projected_tex_u8 = .{
                    .ref_camera = prepared.camera_ref,
                    .ref_coords = null,
                    .tex = tex_grey,
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

    const time_start = Timestamp.now(io, .awake);
    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = @max(@as(u16, 1), config.total_threads) },
    };
    const result = (try riley.raster(
        aa,
        &render_groups,
        &[_]CameraInput{target_cam},
        &[_]mo.MeshInput{mesh_input},
        config,
        null,
    )) orelse return error.NoResult;
    defer aa.free(result.slice);
    const time_end = Timestamp.now(io, .awake);
    const duration_ms = @as(
        F,
        @floatFromInt(time_start.durationTo(time_end).raw.nanoseconds),
    ) / 1e6;

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

    if (tcfg.TEST_CASE_VERBOSE) {
        if (first_err) |err| {
            std.debug.print("FAIL ({s})\n", .{@errorName(err)});
        } else {
            std.debug.print("PASS ({d:.2} ms)\n", .{duration_ms});
        }
    }
    if (first_err) |err| return err;
}

test "ProjectedTex Tri3 Ref Grey" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    try runProjectedTexCase(allocator, io, .tri3, false, false);
}

test "ProjectedTex Tri3 Stereo Grey" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    try runProjectedTexCase(allocator, io, .tri3, true, false);
}

test "ProjectedTex Tri3 Ref RGB" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    try runProjectedTexCase(allocator, io, .tri3, false, true);
}

test "ProjectedTex Tri3 Stereo RGB" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    try runProjectedTexCase(allocator, io, .tri3, true, true);
}

test "ProjectedTex Tri6 Ref Grey" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    try runProjectedTexCase(allocator, io, .tri6, false, false);
}

test "ProjectedTex Tri6 Stereo Grey" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    try runProjectedTexCase(allocator, io, .tri6, true, false);
}

test "ProjectedTex Tri6 Ref RGB" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    try runProjectedTexCase(allocator, io, .tri6, false, true);
}

test "ProjectedTex Tri6 Stereo RGB" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    try runProjectedTexCase(allocator, io, .tri6, true, true);
}

test "ProjectedTex Quad4Newton Ref Grey" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    try runProjectedTexCase(allocator, io, .quad4newton, false, false);
}

test "ProjectedTex Quad4Newton Stereo Grey" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    try runProjectedTexCase(allocator, io, .quad4newton, true, false);
}

test "ProjectedTex Quad4Newton Ref RGB" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    try runProjectedTexCase(allocator, io, .quad4newton, false, true);
}

test "ProjectedTex Quad4Newton Stereo RGB" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    try runProjectedTexCase(allocator, io, .quad4newton, true, true);
}

test "ProjectedTex Quad4Ibi Ref Grey" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    try runProjectedTexCase(allocator, io, .quad4ibi, false, false);
}

test "ProjectedTex Quad4Ibi Stereo Grey" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    try runProjectedTexCase(allocator, io, .quad4ibi, true, false);
}

test "ProjectedTex Quad4Ibi Ref RGB" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    try runProjectedTexCase(allocator, io, .quad4ibi, false, true);
}

test "ProjectedTex Quad4Ibi Stereo RGB" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    try runProjectedTexCase(allocator, io, .quad4ibi, true, true);
}

test "ProjectedTex Quad8 Ref Grey" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    try runProjectedTexCase(allocator, io, .quad8, false, false);
}

test "ProjectedTex Quad8 Stereo Grey" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    try runProjectedTexCase(allocator, io, .quad8, true, false);
}

test "ProjectedTex Quad8 Ref RGB" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    try runProjectedTexCase(allocator, io, .quad8, false, true);
}

test "ProjectedTex Quad8 Stereo RGB" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    try runProjectedTexCase(allocator, io, .quad8, true, true);
}

test "ProjectedTex Quad9 Ref Grey" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    try runProjectedTexCase(allocator, io, .quad9, false, false);
}

test "ProjectedTex Quad9 Stereo Grey" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    try runProjectedTexCase(allocator, io, .quad9, true, false);
}

test "ProjectedTex Quad9 Ref RGB" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    try runProjectedTexCase(allocator, io, .quad9, false, true);
}

test "ProjectedTex Quad9 Stereo RGB" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    try runProjectedTexCase(allocator, io, .quad9, true, true);
}
