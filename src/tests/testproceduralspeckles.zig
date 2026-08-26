// --------------------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");

const orch = @import("../dev_support/orchestration.zig");
const camera = @import("../riley/zig/camera.zig");
const iio = @import("../riley/zig/imageio.zig");
const meshio = @import("../riley/zig/meshio.zig");
const meshpipeline = @import("../riley/zig/meshpipeline.zig");
const rastcfg = @import("../riley/zig/rasterconfig.zig");
const riley = @import("../riley/zig/riley.zig");
const shaderops = @import("../riley/zig/shaderops_common.zig");
const uvio = @import("../riley/zig/uvio.zig");
const buildconfig = @import("../riley/zig/buildconfig.zig");

const F = buildconfig.F;

// --------------------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------------------

test "procedural speckle renders deterministically without a texture" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;

    const data_dir = "data/min/tri3_sphere200/";
    const sim_data = try meshio.loadSimData(
        allocator,
        io,
        data_dir ++ "coords.csv",
        data_dir ++ "connect.csv",
        null,
        null,
    );
    const uvs = try uvio.loadUVMap(allocator, io, data_dir ++ "uvs.csv");
    const camera_prep = try orch.initCameraForCoords(
        allocator,
        &sim_data.coords,
        .{ 48, 30 },
        1.0,
    );

    const camera_input = camera.CameraInput{
        .pixels_num = camera_prep.pixels_num,
        .pixels_size = camera_prep.pixels_size,
        .pos_world = camera_prep.pos_world,
        .rot_world = camera_prep.rot_world,
        .roi_cent_world = camera_prep.roi_cent_world,
        .focal_length = camera_prep.focal_length,
        .sub_sample = 1,
        .distortion = camera_prep.distortion,
    };
    const params = shaderops.Speckle2DParams{
        .seed = 12345,
        .cells_per_uv = .{ 24.0, 20.0 },
        .occupancy = 0.8,
        .radius_mean = 0.42,
        .radius_jitter = 0.06,
        .edge_softness = 0.03,
    };
    const mesh_input = meshpipeline.MeshInput{
        .mesh_type = .tri3,
        .coords = sim_data.coords,
        .connect = sim_data.connect,
        .disp = null,
        .shader = .{ .func = .{
            .uvs = uvs.array,
            .coord_mode = .uv,
            .builtin = .speckle,
            .params = params.toFuncShaderParams(),
            .bits = 8,
            .scaling = .auto,
            .normal_type = .none,
        } },
    };
    const config = rastcfg.RasterConfig{
        .save_strategy = .memory,
        .image_save_mode = .grey,
        .image_save_opts = &[_]iio.ImageSaveOpts{},
        .background_value = 255.0,
        .report = .off,
    };
    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = 1 },
    };

    var first = (try riley.raster(
        allocator,
        &render_groups,
        &[_]camera.CameraInput{camera_input},
        &[_]meshpipeline.MeshInput{mesh_input},
        config,
        null,
    )).?;
    defer {
        allocator.free(first.slice);
        first.deinit(allocator);
    }

    var second = (try riley.raster(
        allocator,
        &render_groups,
        &[_]camera.CameraInput{camera_input},
        &[_]meshpipeline.MeshInput{mesh_input},
        config,
        null,
    )).?;
    defer {
        allocator.free(second.slice);
        second.deinit(allocator);
    }

    try std.testing.expectEqualSlices(F, first.slice, second.slice);

    var min_value = std.math.inf(F);
    var max_value = -std.math.inf(F);
    for (first.slice) |value| {
        min_value = @min(min_value, value);
        max_value = @max(max_value, value);
    }
    try std.testing.expect(min_value < max_value);
    try std.testing.expect(min_value >= 0.0);
    try std.testing.expect(max_value <= 255.0);
}
