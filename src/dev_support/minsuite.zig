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
const F = buildconfig.F;
const common = @import("benchcommon.zig");
const orch = @import("orchestration.zig");
const riley = @import("../riley/zig/riley.zig");
const iio = @import("../riley/zig/imageio.zig");
const meshio = @import("../riley/zig/meshio.zig");
const mo = @import("../riley/zig/meshpipeline.zig");
const sceneops = @import("../riley/zig/sceneops.zig");
const gk = @import("../riley/zig/geometrykernels.zig");
const texops = @import("../riley/zig/textureops.zig");
const CameraInput = @import("../riley/zig/camera.zig").CameraInput;
const tcfg = @import("testconfig.zig");
const rastcfg = @import("../riley/zig/rasterconfig.zig");
const policy = @import("testpolicy.zig");

fn translateCoords(coords: *meshio.Coords, translation: [3]F) void {
    for (0..coords.mat.rows_num) |nn| {
        coords.mat.set(nn, 0, coords.mat.get(nn, 0) + translation[0]);
        coords.mat.set(nn, 1, coords.mat.get(nn, 1) + translation[1]);
        coords.mat.set(nn, 2, coords.mat.get(nn, 2) + translation[2]);
    }
}

pub fn calcMinCaseName(
    allocator: std.mem.Allocator,
    etype: gk.MeshType,
    shader_type: common.ShaderType,
    samp_cfg: texops.TextureSampleConfig,
) ![]const u8 {
    return common.calcCaseName(
        allocator,
        etype,
        shader_type,
        samp_cfg,
        null,
        1.0,
    );
}

fn buildSphere200MultiCullMeshInputs(
    allocator: std.mem.Allocator,
    io: std.Io,
    etype: gk.MeshType,
    shader_type: common.ShaderType,
    samp_cfg: texops.TextureSampleConfig,
    data_dir: []const u8,
    texture_grey: texops.Tex(u8, 1),
    texture_rgb: texops.Tex(u8, 3),
) ![]mo.MeshInput {
    const base_mesh_input = try common.loadBenchmarkMeshInput(
        u8,
        allocator,
        io,
        etype,
        shader_type,
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

pub fn runSphere200MultiCullQuiet(
    outer_alloc: std.mem.Allocator,
    io: std.Io,
    etype: gk.MeshType,
    shader_type: common.ShaderType,
    samp_cfg: texops.TextureSampleConfig,
    data_dir: []const u8,
    pixel_num: [2]u32,
    texture_grey: texops.Tex(u8, 1),
    texture_rgb: texops.Tex(u8, 3),
    config: rastcfg.RasterConfig,
    out_dir_base: []const u8,
    fov_scale: F,
) !common.BenchResult {
    var arena = std.heap.ArenaAllocator.init(outer_alloc);
    defer arena.deinit();
    const aa = arena.allocator();

    const mesh_inputs = try buildSphere200MultiCullMeshInputs(
        aa,
        io,
        etype,
        shader_type,
        samp_cfg,
        data_dir,
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

    var config_run = config;
    config_run.report = .off;

    if (out_dir_base.len > 0) {
        var out_dir = try orch.openDirEnsured(io, out_dir_base);
        out_dir.close(io);
    }

    const case_name = try calcMinCaseName(
        aa,
        etype,
        shader_type,
        samp_cfg,
    );
    const out_path = if (out_dir_base.len > 0)
        try std.fs.path.join(aa, &[_][]const u8{
            out_dir_base,
            case_name,
        })
    else
        null;

    const e2e_start = std.Io.Clock.Timestamp.now(io, .awake);
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
    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = @max(@as(u16, 1), config_run.total_threads) },
    };
    var image_arr = try riley.raster(
        outer_alloc,
        &render_groups,
        &[_]CameraInput{camera_input},
        mesh_inputs,
        config_run,
        out_path,
    );
    const e2e_end = std.Io.Clock.Timestamp.now(io, .awake);

    const e2e_ms = @as(F, @floatFromInt(
        e2e_start.durationTo(e2e_end).raw.nanoseconds,
    )) / 1e6;
    const fps = 1000.0 / e2e_ms;

    const return_image = (config.save_strategy == .memory or config.save_strategy == .both);

    const image_final = if (return_image) blk: {
        var images = image_arr orelse return error.NoResult;
        image_arr = null;
        defer {
            outer_alloc.free(images.slice);
            images.deinit(outer_alloc);
        }
        break :blk try common.extractFirstFrameImage(outer_alloc, &images);
    } else null;

    if (image_arr) |images| {
        outer_alloc.free(images.slice);
        var images_mut = images;
        images_mut.deinit(outer_alloc);
    }

    return .{
        .e2e_ms = e2e_ms,
        .geom_ms = 0.0,
        .raster_ms = 0.0,
        .cam_ms = 0.0,
        .resolve_ms = 0.0,
        .fps = fps,
        .total_elems = 0,
        .vis_elems = 0,
        .total_px = @as(u64, camera_input.pixels_num[0]) *
            @as(u64, camera_input.pixels_num[1]),
        .shaded_px = 0,
        .metrics = .{
            .raster_tpx_mpx_s = 0.0,
            .frame_tpx_mpx_s = 0.0,
            .e2e_tpx_mpx_s = 0.0,
            .msubpx_sec = 0.0,
            .mshades_sec = 0.0,
            .msubshades_sec = 0.0,
            .melems_sec = 0.0,
            .mnodes_sec = 0.0,
            .mops_sec = 0.0,
        },
        .pipeline_times = .{},
        .image = image_final,
    };
}
