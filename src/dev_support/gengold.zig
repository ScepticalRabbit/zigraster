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

const orch = @import("orchestration.zig");
const gk = @import("../riley/zig/geometrykernels.zig");
const meshio = @import("../riley/zig/meshio.zig");
const mo = @import("../riley/zig/meshpipeline.zig");
const shaderops = @import("../riley/zig/shaderops.zig");
const MeshType = gk.MeshType;
const MeshInput = mo.MeshInput;
const CameraPrepared = @import("../riley/zig/camera.zig").CameraPrepared;
const CameraInput = @import("../riley/zig/camera.zig").CameraInput;
const rastcfg = @import("../riley/zig/rasterconfig.zig");
const riley = @import("../riley/zig/riley.zig");
const RasterConfig = rastcfg.RasterConfig;
const iio = @import("../riley/zig/imageio.zig");
const texops = @import("../riley/zig/textureops.zig");
const testcommon = @import("tests.zig");

pub fn renderAndSave(
    outer_alloc: std.mem.Allocator,
    io: std.Io,
    camera: *const CameraPrepared,
    mt: MeshType,
    coords: meshio.Coords,
    connect: meshio.Connect,
    disp: ?meshio.Field,
    sh: shaderops.ShaderInput,
    dir: []const u8,
    add_disp: bool,
    config: RasterConfig,
) !void {
    var out_dir = try orch.openDirEnsured(io, dir);
    out_dir.close(io);

    const mesh_input = MeshInput{
        .mesh_type = mt,
        .coords = coords,
        .connect = connect,
        .disp = if (add_disp) disp else null,
        .shader = sh,
    };

    const meshes = &[_]MeshInput{mesh_input};
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
        .{ .io = io, .workers = @max(@as(u16, 1), config.total_threads) },
    };
    const images = try riley.raster(
        outer_alloc,
        &render_groups,
        &[_]CameraInput{camera_input},
        meshes,
        config,
        dir,
    );
    if (images) |img| {
        outer_alloc.free(img.slice);
        img.deinit(outer_alloc);
    }
}

pub fn runGenerationExt(
    outer_alloc: std.mem.Allocator,
    io: std.Io,
    test_type: []const u8,
    mesh_types: []const MeshType,
    fov_scale: F,
    texture: texops.Tex(u8, 1),
    pixel_num: [2]u32,
    samp_cfgs: []const texops.TextureSampleConfig,
    gold_dir_root: []const u8,
    data_dir_root: []const u8,
    config: RasterConfig,
) !void {
    for (mesh_types) |mt| {
        try testcommon.runSingleMeshSuiteDriver(
            .generate,
            outer_alloc,
            io,
            test_type,
            mt,
            fov_scale,
            texture,
            pixel_num,
            samp_cfgs,
            gold_dir_root,
            data_dir_root,
            config,
            0,
            0,
            .both,
            false,
        );
    }
}

pub fn runMultimeshGeneration(
    outer_alloc: std.mem.Allocator,
    io: std.Io,
    config: RasterConfig,
) !void {
    try runMultimeshGenerationExt(
        outer_alloc,
        io,
        config,
        policy.goldRoot(.multimesh),
        &orch.default_multimesh_dir_paths,
        .{ 1200, 800 },
    );
}

pub fn runMultimeshGenerationExt(
    outer_alloc: std.mem.Allocator,
    io: std.Io,
    config: RasterConfig,
    out_dir_root: []const u8,
    dir_paths: []const []const u8,
    pixel_num: [2]u32,
) !void {
    var arena = std.heap.ArenaAllocator.init(outer_alloc);
    defer arena.deinit();
    const aa = arena.allocator();

    const shader_modes = [_]enum { nodal, texture }{ .nodal, .texture };

    for (shader_modes) |mode| {
        _ = arena.reset(.free_all);
        const gold_dir = if (mode == .nodal)
            try std.fmt.allocPrint(aa, "{s}/allelem_nodal", .{out_dir_root})
        else
            try std.fmt.allocPrint(aa, "{s}/allelem_tex_cubic_lut_lerp", .{out_dir_root});

        const mesh_inputs = try orch.buildMultimeshInputs(
            aa,
            io,
            dir_paths,
            if (mode == .nodal) .nodal else .texture,
        );

        const fov_scale_factor: F = 1.1;
        const camera = try orch.initCameraForMeshes(
            aa,
            mesh_inputs,
            pixel_num,
            fov_scale_factor,
        );
        defer camera.deinit(aa);

        std.debug.print("Generating Multimesh Gold Data for {s}...\n", .{gold_dir});
        var out_dir = try orch.openDirEnsured(io, gold_dir);
        out_dir.close(io);
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
            .{ .io = io, .workers = @max(@as(u16, 1), config.total_threads) },
        };
        const images = try riley.raster(
            aa,
            &render_groups,
            &[_]CameraInput{camera_input},
            mesh_inputs,
            config,
            gold_dir,
        );
        if (images) |img| {
            aa.free(img.slice);
            img.deinit(aa);
        }
    }
}

pub fn runMultimeshMixedGeneration(
    allocator: std.mem.Allocator,
    io: std.Io,
    config: RasterConfig,
) !void {
    const gold_root = comptime policy.goldRoot(.multimesh);
    try runMultimeshMixedGenerationExt(
        allocator,
        io,
        config,
        gold_root ++ "/allelem_allshade",
        &orch.default_multimesh_dir_paths,
        .{ 1600, 800 },
    );
}

pub fn runMultimeshMixedGenerationExt(
    allocator: std.mem.Allocator,
    io: std.Io,
    config: RasterConfig,
    gold_dir: []const u8,
    dir_paths: []const []const u8,
    pixel_num: [2]u32,
) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    const texture = try iio.loadImage(
        u8,
        1,
        aa,
        io,
        "texture/speckle-simple.tiff",
        .tiff,
    );

    const mesh_inputs = try orch.buildMixedMeshInputs(
        aa,
        io,
        dir_paths,
        texture,
    );

    const fov_scale_factor: F = 1.2;
    const camera = try orch.initCameraForMeshes(
        aa,
        mesh_inputs,
        pixel_num,
        fov_scale_factor,
    );
    defer camera.deinit(aa);

    std.debug.print("Generating Multimesh Gold Data for {s}...\n", .{gold_dir});
    var out_dir = try orch.openDirEnsured(io, gold_dir);
    out_dir.close(io);
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
        .{ .io = io, .workers = @max(@as(u16, 1), config.total_threads) },
    };
    const images = try riley.raster(
        aa,
        &render_groups,
        &[_]CameraInput{camera_input},
        mesh_inputs,
        config,
        gold_dir,
    );
    if (images) |img| {
        aa.free(img.slice);
        img.deinit(aa);
    }
}

pub fn runMultimeshMixedRGBGeneration(
    allocator: std.mem.Allocator,
    io: std.Io,
    config: RasterConfig,
) !void {
    const gold_root = comptime policy.goldRoot(.multimesh);
    try runMultimeshMixedRGBGenerationExt(
        allocator,
        io,
        config,
        gold_root ++ "/allelem_allshade_rgb",
        &orch.default_multimesh_dir_paths,
        .{ 1200, 800 },
    );
}

pub fn runMultimeshMixedRGBGenerationExt(
    allocator: std.mem.Allocator,
    io: std.Io,
    config: RasterConfig,
    gold_dir: []const u8,
    dir_paths: []const []const u8,
    pixel_num: [2]u32,
) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    const texture = try iio.loadImage(
        u8,
        3,
        aa,
        io,
        "texture/speckle_rgb.bmp",
        .bmp,
    );

    const mesh_inputs = try orch.buildMixedRgbMeshInputs(
        aa,
        io,
        dir_paths,
        texture,
    );

    const fov_scale_factor: F = 1.1;
    const camera_rgb = try orch.initCameraForMeshes(
        aa,
        mesh_inputs,
        pixel_num,
        fov_scale_factor,
    );
    defer camera_rgb.deinit(aa);

    var config_rgb = config;
    if (config_rgb.image_save_opts.len == 0) {
        config_rgb.image_save_opts = &[_]iio.ImageSaveOpts{
            .{ .format = .bmp, .bits = 8, .scaling = .auto, .channels = 3 },
            .{ .format = .fimg, .bits = null, .scaling = .none, .channels = 3 },
        };
    } else {
        // If save_opts are provided, ensure they use 3 channels for RGB
        const opts_rgb = try aa.alloc(iio.ImageSaveOpts, config_rgb.image_save_opts.len);
        for (config_rgb.image_save_opts, 0..) |opt, ii| {
            opts_rgb[ii] = opt;
            opts_rgb[ii].channels = 3;
        }
        config_rgb.image_save_opts = opts_rgb;
    }

    std.debug.print("Generating Multimesh Gold Data for {s}...\n", .{gold_dir});
    var out_dir = try orch.openDirEnsured(io, gold_dir);
    out_dir.close(io);
    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = @max(@as(u16, 1), config_rgb.total_threads) },
    };
    const camera_input = CameraInput{
        .pixels_num = camera_rgb.pixels_num,
        .pixels_size = camera_rgb.pixels_size,
        .pos_world = camera_rgb.pos_world,
        .rot_world = camera_rgb.rot_world,
        .roi_cent_world = camera_rgb.roi_cent_world,
        .focal_length = camera_rgb.focal_length,
        .sub_sample = camera_rgb.sub_sample,
        .distortion = camera_rgb.distortion,
    };
    _ = try riley.raster(
        aa,
        &render_groups,
        &[_]CameraInput{camera_input},
        mesh_inputs,
        config_rgb,
        gold_dir,
    );
}

pub fn generateDistortEdgeGold(
    allocator: std.mem.Allocator,
    io: std.Io,
    gold_dir_root: []const u8,
    data_dir_root: []const u8,
    pixel_num: [2]u32,
    config: RasterConfig,
) !void {
    const midside_mesh_types = [_]gk.MeshType{ .tri6, .quad8, .quad9 };
    const full_mesh_types = [_]gk.MeshType{
        .tri3,
        .tri6,
        .quad4ibi,
        .quad4newton,
        .quad8,
        .quad9,
    };
    const distortion_cases = [_]struct {
        name: []const u8,
        mesh_types: []const gk.MeshType,
    }{
        .{ .name = "distort_bulge", .mesh_types = &midside_mesh_types },
        .{ .name = "distort_tan", .mesh_types = &midside_mesh_types },
        .{ .name = "distort_stretch", .mesh_types = &full_mesh_types },
        .{ .name = "distort_shear", .mesh_types = &full_mesh_types },
        .{ .name = "distort_rot", .mesh_types = &full_mesh_types },
    };

    for (distortion_cases) |distortion_case| {
        for (distortion_case.mesh_types) |mesh_type| {
            try testcommon.runEdgeTexFuncConstantSuiteDriver(
                .generate,
                allocator,
                io,
                distortion_case.name,
                mesh_type,
                gold_dir_root,
                data_dir_root,
                pixel_num,
                null,
                config,
            );
        }
    }
}

pub fn generateDistortEdgeGoldForHullMode(
    allocator: std.mem.Allocator,
    io: std.Io,
    gold_dir_root: []const u8,
    data_dir_root: []const u8,
    pixel_num: [2]u32,
    config: RasterConfig,
    hull_mode: rastcfg.HullMode,
) !void {
    const midside_mesh_types = [_]gk.MeshType{ .tri6, .quad8, .quad9 };
    const full_mesh_types = [_]gk.MeshType{
        .tri3,
        .tri6,
        .quad4ibi,
        .quad4newton,
        .quad8,
        .quad9,
    };
    const distortion_cases = [_]struct {
        name: []const u8,
        mesh_types: []const gk.MeshType,
    }{
        .{ .name = "distort_bulge", .mesh_types = &midside_mesh_types },
        .{ .name = "distort_tan", .mesh_types = &midside_mesh_types },
        .{ .name = "distort_stretch", .mesh_types = &full_mesh_types },
        .{ .name = "distort_shear", .mesh_types = &full_mesh_types },
        .{ .name = "distort_rot", .mesh_types = &full_mesh_types },
    };

    for (distortion_cases) |distortion_case| {
        for (distortion_case.mesh_types) |mesh_type| {
            try testcommon.runEdgeTexFuncConstantSuiteDriver(
                .generate,
                allocator,
                io,
                distortion_case.name,
                mesh_type,
                gold_dir_root,
                data_dir_root,
                pixel_num,
                hull_mode,
                config,
            );
        }
    }
}
