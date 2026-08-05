// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const std = @import("std");
const buildconfig = @import("../riley/zig/buildconfig.zig");
const policy = @import("testpolicy.zig");
const F = buildconfig.F;
const benchcommon = @import("benchcommon.zig");
const orch = @import("orchestration.zig");
const tcfg = @import("testconfig.zig");
const cam = @import("../riley/zig/camera.zig");
const gk = @import("../riley/zig/geometrykernels.zig");
const iio = @import("../riley/zig/imageio.zig");
const mo = @import("../riley/zig/meshpipeline.zig");
const meshio = @import("../riley/zig/meshio.zig");
const sceneops = @import("../riley/zig/sceneops.zig");
const NDArray = @import("../riley/zig/ndarray.zig").NDArray;
const texops = @import("../riley/zig/textureops.zig");
const riley = @import("../riley/zig/riley.zig");

pub const gold_root = policy.goldRoot(.ssaa);
pub const pixel_num = [_]u32{ 640, 400 };
pub const fov_scale: F = 0.75;
pub const ssaa_values = [_]u8{4};
pub const mesh_types = [_]gk.MeshType{ .tri3, .tri6, .quad4ibi, .quad8, .quad9 };
pub const DistortionCase = enum {
    none,
    brown,
    brownext,
};
pub const distortion_cases = [_]DistortionCase{ .none, .brown, .brownext };
pub const samp_cfg: texops.TextureSampleConfig = .{
    .sample = .cubic_catmull_rom,
    .mode = .lut_lerp,
};
pub const distortion_brown = cam.DistortionModel{
    .brown_conrady = .{
        .k1 = -0.18,
        .k2 = 0.02,
        .k3 = -0.004,
        .p1 = 0.0012,
        .p2 = -0.0018,
    },
};
pub const distortion_brownext = cam.DistortionModel{
    .brown_conrady_ext = .{
        .k1 = -0.18,
        .k2 = 0.02,
        .k3 = -0.004,
        .k4 = 0.01,
        .k5 = -0.002,
        .k6 = 0.0004,
        .p1 = 0.0012,
        .p2 = -0.0018,
    },
};

fn translateCoords(coords: *meshio.Coords, translation: [3]F) void {
    for (0..coords.mat.rows_num) |nn| {
        coords.mat.set(nn, 0, coords.mat.get(nn, 0) + translation[0]);
        coords.mat.set(nn, 1, coords.mat.get(nn, 1) + translation[1]);
        coords.mat.set(nn, 2, coords.mat.get(nn, 2) + translation[2]);
    }
}

pub fn caseName(
    allocator: std.mem.Allocator,
    mesh_type: gk.MeshType,
    ssaa: u8,
    distortion_case: DistortionCase,
) ![]const u8 {
    return std.fmt.allocPrint(
        allocator,
        "sphere200multicull_{s}_ssaa{d}_{s}",
        .{ @tagName(mesh_type), ssaa, @tagName(distortion_case) },
    );
}

pub fn getDistortionModel(distortion_case: DistortionCase) cam.DistortionModel {
    return switch (distortion_case) {
        .none => .none,
        .brown => distortion_brown,
        .brownext => distortion_brownext,
    };
}

pub fn goldSubpixelCenterMap(
    distortion_case: DistortionCase,
    subpixel_center_map: @import("../riley/zig/camera.zig").SubPixelCenterMap,
) @import("../riley/zig/camera.zig").SubPixelCenterMap {
    return switch (subpixel_center_map) {
        .affine_jac => switch (distortion_case) {
            .brown, .brownext => .affine_jac,
            .none => .full_in_mem,
        },
        else => .full_in_mem,
    };
}

pub fn buildSphere200MultiCullMeshInputs(
    allocator: std.mem.Allocator,
    io: std.Io,
    mesh_type: gk.MeshType,
    texture_grey: texops.Tex(u8, 1),
    texture_rgb: texops.Tex(u8, 3),
) ![]mo.MeshInput {
    const data_dir = try std.fmt.allocPrint(
        allocator,
        "data/min/{s}_sphere200",
        .{@tagName(mesh_type)},
    );
    const base_mesh_input = try benchcommon.loadBenchmarkMeshInput(
        u8,
        allocator,
        io,
        mesh_type,
        .tex8_grey,
        samp_cfg,
        null,
        data_dir,
        texture_grey,
        texture_rgb,
    );
    const mesh_inputs = try allocator.alloc(mo.MeshInput, 2);

    const left_coords = try orch.copyCoords(allocator, base_mesh_input.coords);
    const right_coords = try orch.copyCoords(allocator, base_mesh_input.coords);

    const bounds = sceneops.boundsForCoords(&left_coords);
    const diameter = bounds.extent[0];
    const overlap_x = 0.7 * diameter;
    var right_coords_mut = right_coords;
    translateCoords(&right_coords_mut, .{ overlap_x, 0.0, -20.0 * diameter });

    mesh_inputs[0] = base_mesh_input;
    mesh_inputs[0].coords = left_coords;
    mesh_inputs[1] = base_mesh_input;
    mesh_inputs[1].coords = right_coords_mut;
    return mesh_inputs;
}

pub fn renderCase(
    outer_alloc: std.mem.Allocator,
    io: std.Io,
    mesh_type: gk.MeshType,
    ssaa: u8,
    distortion_case: DistortionCase,
    subpixel_center_map: @import("../riley/zig/camera.zig").SubPixelCenterMap,
) !NDArray(F) {
    var arena = std.heap.ArenaAllocator.init(outer_alloc);
    defer arena.deinit();
    const aa = arena.allocator();

    const texture_grey = try iio.loadImage(
        u8,
        1,
        aa,
        io,
        "texture/speckle.bmp",
        .bmp,
    );
    const texture_rgb = try iio.loadImage(
        u8,
        3,
        aa,
        io,
        "texture/speckle_rgb.bmp",
        .bmp,
    );
    const mesh_inputs = try buildSphere200MultiCullMeshInputs(
        aa,
        io,
        mesh_type,
        texture_grey,
        texture_rgb,
    );
    const camera = try orch.initCameraForMeshes(
        aa,
        mesh_inputs,
        pixel_num,
        fov_scale,
    );
    defer camera.deinit(aa);

    var camera_input = cam.CameraInput{
        .pixels_num = camera.pixels_num,
        .pixels_size = camera.pixels_size,
        .pos_world = camera.pos_world,
        .rot_world = camera.rot_world,
        .roi_cent_world = camera.roi_cent_world,
        .focal_length = camera.focal_length,
        .sub_sample = camera.sub_sample,
        .distortion = camera.distortion,
        .subpixel_center_map = subpixel_center_map,
    };
    camera_input.sub_sample = ssaa;
    camera_input.distortion = getDistortionModel(distortion_case);

    var config = tcfg.getRasterConfig(.gold);
    config.save_strategy = .memory;
    config.image_save_opts = &[_]iio.ImageSaveOpts{
        .{ .format = .csv, .bits = null, .scaling = .none },
    };

    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = @max(@as(u16, 1), config.total_threads) },
    };
    const result = (try riley.raster(
        outer_alloc,
        &render_groups,
        &[_]cam.CameraInput{camera_input},
        mesh_inputs,
        config,
        null,
    )) orelse return error.NoResult;
    defer {
        outer_alloc.free(result.slice);
        var result_mut = result;
        result_mut.deinit(outer_alloc);
    }

    return try benchcommon.extractFirstFrameImage(outer_alloc, &result);
}
