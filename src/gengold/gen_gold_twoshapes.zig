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
const camera = @import("../riley/zig/camera.zig");
const cameraops = @import("../riley/zig/cameraops.zig");
const gk = @import("../riley/zig/geometrykernels.zig");
const iio = @import("../riley/zig/imageio.zig");
const meshio = @import("../riley/zig/meshio.zig");
const mo = @import("../riley/zig/meshpipeline.zig");
const orch = @import("../dev_support/orchestration.zig");
const policy = @import("../dev_support/testpolicy.zig");
const rastcfg = @import("../riley/zig/rasterconfig.zig");
const riley = @import("../riley/zig/riley.zig");
const Rotation = @import("../riley/zig/rotation.zig").Rotation;
const sceneops = @import("../riley/zig/sceneops.zig");
const shaderops = @import("../riley/zig/shaderops_common.zig");
const tcfg = @import("../dev_support/testconfig.zig");
const texops = @import("../riley/zig/textureops.zig");
const uvio = @import("../riley/zig/uvio.zig");

const F = buildconfig.F;
const CameraInput = camera.CameraInput;
const MeshInput = mo.MeshInput;

const pixel_num_twoshapes = [_]u32{ 160, 100 };
const pixel_size_twoshapes = [2]F{ @floatCast(5.3e-6), @floatCast(5.3e-6) };
const focal_length_twoshapes: F = @floatCast(50.0e-3);

pub const TwoShapesShaderKind = enum {
    nodal_grey,
    tex_u8_linear,
    tex_u8_cubic,
    tex_u8_bspline,
    func_checker,
    func_eggbox,
    nodal_rgb,
    tex_rgb_cubic,
    func_rgb_checker,

    pub fn isRgb(self: TwoShapesShaderKind) bool {
        return switch (self) {
            .nodal_rgb, .tex_rgb_cubic, .func_rgb_checker => true,
            else => false,
        };
    }
};

fn sliceFieldToSingleFrame(
    allocator: std.mem.Allocator,
    field: meshio.Field,
) !meshio.Field {
    const node_n = field.getCoordN();
    const chan_n = field.getFieldsN();
    var result = try meshio.Field.initAlloc(allocator, 1, node_n, chan_n);
    for (0..node_n) |nn| {
        for (0..chan_n) |cc| {
            result.array.set(
                &.{ 0, nn, cc },
                field.array.get(&.{ 0, nn, cc }),
            );
        }
    }
    return result;
}

fn buildRgbField(
    allocator: std.mem.Allocator,
    temp: meshio.Field,
    disp: meshio.Field,
) !meshio.Field {
    const time_n = temp.getTimeN();
    const node_n = temp.getCoordN();
    var field = try meshio.Field.initAlloc(allocator, time_n, node_n, 3);
    for (0..time_n) |tt| {
        for (0..node_n) |nn| {
            field.array.set(&.{ tt, nn, 0 }, temp.array.get(&.{ tt, nn, 0 }));
            field.array.set(&.{ tt, nn, 1 }, disp.array.get(&.{ tt, nn, 0 }));
            field.array.set(&.{ tt, nn, 2 }, disp.array.get(&.{ tt, nn, 1 }));
        }
    }
    return field;
}

fn loadShapeMesh(
    allocator: std.mem.Allocator,
    io: std.Io,
    shape_dir_name: []const u8,
    mesh_type: gk.MeshType,
    shader_kind: TwoShapesShaderKind,
    texture_grey: texops.Tex(u8, 1),
    texture_rgb: texops.Tex(u8, 3),
) !MeshInput {
    const elem_str = @tagName(mesh_type);
    const dir = try std.fmt.allocPrint(
        allocator,
        "data/shapes/{s}/{s}/",
        .{ shape_dir_name, elem_str },
    );
    const temp_files = &[_][]const u8{
        try std.fmt.allocPrint(allocator, "{s}temperature.csv", .{dir}),
    };
    const disp_files = &[_][]const u8{
        try std.fmt.allocPrint(allocator, "{s}disp_x.csv", .{dir}),
        try std.fmt.allocPrint(allocator, "{s}disp_y.csv", .{dir}),
        try std.fmt.allocPrint(allocator, "{s}disp_z.csv", .{dir}),
    };
    const sim = try meshio.loadSimData(
        allocator,
        io,
        try std.fmt.allocPrint(allocator, "{s}coords.csv", .{dir}),
        try std.fmt.allocPrint(allocator, "{s}connect.csv", .{dir}),
        temp_files,
        disp_files,
    );
    const uvs = try uvio.loadUVMap(
        allocator,
        io,
        try std.fmt.allocPrint(allocator, "{s}uvs.csv", .{dir}),
    );
    const raw_temp = sim.field orelse return error.MissingTemperature;
    const raw_disp = sim.disp orelse return error.MissingDisplacement;

    const temp = try sliceFieldToSingleFrame(allocator, raw_temp);
    const disp = try sliceFieldToSingleFrame(allocator, raw_disp);

    const shader: shaderops.ShaderInput = switch (shader_kind) {
        .nodal_grey => .{ .nodal = .{
            .field = temp,
            .bits = 8,
            .scaling = .auto,
            .scale_over = .over_frames,
            .normal_type = .none,
        } },
        .tex_u8_linear => .{ .tex_u8 = .{
            .uvs = uvs.array,
            .tex = texture_grey,
            .samp_cfg = .{ .sample = .linear, .mode = .direct },
            .bits = 8,
            .scaling = .auto,
            .normal_type = .none,
        } },
        .tex_u8_cubic => .{ .tex_u8 = .{
            .uvs = uvs.array,
            .tex = texture_grey,
            .samp_cfg = .{ .sample = .cubic_catmull_rom, .mode = .lut_lerp },
            .bits = 8,
            .scaling = .auto,
            .normal_type = .none,
        } },
        .tex_u8_bspline => .{ .tex_u8 = .{
            .uvs = uvs.array,
            .tex = texture_grey,
            .samp_cfg = .{ .sample = .cubic_bspline, .mode = .lut_lerp },
            .bits = 8,
            .scaling = .auto,
            .normal_type = .none,
        } },
        .func_checker => .{ .func = .{
            .uvs = uvs.array,
            .coord_mode = .para,
            .builtin = .checker,
            .params = .{
                .coord_scale = .{ 4.0, 4.0 },
                .settings = .{ .checker = .{} },
            },
            .bits = 8,
            .scaling = .auto,
            .normal_type = .none,
        } },
        .func_eggbox => .{ .func = .{
            .coord_mode = .world_reference,
            .builtin = .eggbox,
            .params = .{
                .coord_scale = .{ 1.0, 1.0 },
                .settings = .{ .eggbox = .{ .pitch = .{ 0.005, 0.005 } } },
            },
            .bits = 8,
            .scaling = .auto,
            .normal_type = .none,
        } },
        .nodal_rgb => .{ .nodal = .{
            .field = try buildRgbField(allocator, temp, disp),
            .bits = 8,
            .scaling = .auto,
            .scale_over = .over_frames,
            .normal_type = .none,
        } },
        .tex_rgb_cubic => .{ .tex_rgb_u8 = .{
            .uvs = uvs.array,
            .tex = texture_rgb,
            .samp_cfg = .{ .sample = .cubic_catmull_rom, .mode = .lut_lerp },
            .bits = 8,
            .scaling = .auto,
            .normal_type = .none,
        } },
        .func_rgb_checker => .{ .func_rgb = .{
            .uvs = uvs.array,
            .coord_mode = .para,
            .builtin = .checker,
            .params = .{
                .coord_scale = .{ 4.0, 4.0 },
                .settings = .{ .checker = .{} },
            },
            .bits = 8,
            .scaling = .auto,
            .normal_type = .none,
        } },
    };

    return .{
        .mesh_type = mesh_type,
        .coords = sim.coords,
        .connect = sim.connect,
        .disp = disp,
        .shader = shader,
    };
}

pub fn buildTwoShapesScene(
    allocator: std.mem.Allocator,
    io: std.Io,
    mesh_type: gk.MeshType,
    shader_kind: TwoShapesShaderKind,
    texture_grey: texops.Tex(u8, 1),
    texture_rgb: texops.Tex(u8, 3),
) ![]MeshInput {
    var meshes = try allocator.alloc(MeshInput, 2);
    meshes[0] = try loadShapeMesh(
        allocator,
        io,
        "cube_surf",
        mesh_type,
        shader_kind,
        texture_grey,
        texture_rgb,
    );
    meshes[1] = try loadShapeMesh(
        allocator,
        io,
        "sphere_surf",
        mesh_type,
        shader_kind,
        texture_grey,
        texture_rgb,
    );

    // Sphere on left in front (z = 0.0), Cube on right further behind (z = -0.010)
    // Overlap in X is 10% of 10mm (1.0mm)
    const cube_center = [3]F{ 0.0045, 0.0, -0.010 };
    const sphere_center = [3]F{ -0.0045, 0.0, 0.0 };

    sceneops.centerMeshGroupAt(
        meshes,
        sceneops.meshGroupSingle(0),
        cube_center,
    );
    sceneops.centerMeshGroupAt(
        meshes,
        sceneops.meshGroupSingle(1),
        sphere_center,
    );

    return meshes;
}

pub fn generateTwoShapesCase(
    allocator: std.mem.Allocator,
    io: std.Io,
    mesh_type: gk.MeshType,
    shader_kind: TwoShapesShaderKind,
    texture_grey: texops.Tex(u8, 1),
    texture_rgb: texops.Tex(u8, 3),
    gold_dir_root: []const u8,
    config: rastcfg.RasterConfig,
) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    const meshes = try buildTwoShapesScene(
        aa,
        io,
        mesh_type,
        shader_kind,
        texture_grey,
        texture_rgb,
    );

    const case_dir_name = try std.fmt.allocPrint(
        aa,
        "twoshapes_{s}_{s}",
        .{ @tagName(mesh_type), @tagName(shader_kind) },
    );

    const gold_dir = try std.fmt.allocPrint(
        aa,
        "{s}/{s}",
        .{ gold_dir_root, case_dir_name },
    );

    var out_dir = try orch.openDirEnsured(io, gold_dir);
    out_dir.close(io);

    const target = sceneops.boundsCenterOverMeshes(meshes);
    const rot = Rotation.init(
        0,
        std.math.degreesToRadians(20.0),
        std.math.degreesToRadians(-20.0),
    );
    const pos = cameraops.posFillFrameFromRotOverMeshesAndTarg(
        meshes,
        target,
        pixel_num_twoshapes,
        pixel_size_twoshapes,
        focal_length_twoshapes,
        rot,
        0.9,
    );

    const camera_input = CameraInput{
        .pixels_num = pixel_num_twoshapes,
        .pixels_size = pixel_size_twoshapes,
        .pos_world = pos,
        .rot_world = rot,
        .roi_cent_world = target,
        .focal_length = focal_length_twoshapes,
        .sub_sample = 2,
        .distortion = .none,
    };

    var run_config = config;
    run_config.save_strategy = .disk;
    run_config.background_value = 127.5;

    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = @max(@as(u16, 1), run_config.total_threads) },
    };

    const result = try riley.raster(
        aa,
        &render_groups,
        &[_]CameraInput{camera_input},
        meshes,
        run_config,
        gold_dir,
    );

    if (result) |images| {
        aa.free(images.slice);
        var images_mut = images;
        images_mut.deinit(aa);
    }
}

pub fn generateAllTwoShapesCases(
    allocator: std.mem.Allocator,
    io: std.Io,
    texture_grey: texops.Tex(u8, 1),
    texture_rgb: texops.Tex(u8, 3),
    gold_dir_root: []const u8,
    config: rastcfg.RasterConfig,
) !void {
    const mesh_types = [_]gk.MeshType{
        .tri3,
        .tri6,
        .quad4,
        .quad8,
        .quad9,
    };
    const shader_kinds = std.enums.values(TwoShapesShaderKind);

    for (mesh_types) |mesh_type| {
        for (shader_kinds) |shader_kind| {
            try generateTwoShapesCase(
                allocator,
                io,
                mesh_type,
                shader_kind,
                texture_grey,
                texture_rgb,
                gold_dir_root,
                config,
            );
        }
    }
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const aa = arena.allocator();

    const texture_grey = try iio.loadImage(
        u8,
        1,
        aa,
        io,
        "texture/speck128_mono_u8.bmp",
        .bmp,
    );
    const texture_rgb = try iio.loadImage(
        u8,
        3,
        aa,
        io,
        "texture/speck128_rgb_u8.bmp",
        .bmp,
    );

    var config = tcfg.getRasterConfig(.gold_gen);
    config.save_strategy = .disk;
    config.image_save_opts = &[_]iio.ImageSaveOpts{
        .{ .format = .fimg, .bits = null, .scaling = .none },
        .{ .format = .tiff, .bits = 8, .scaling = .auto },
    };

    std.debug.print("Generating Basic Suite: twoshapes cases...\n", .{});
    try generateAllTwoShapesCases(
        aa,
        io,
        texture_grey,
        texture_rgb,
        policy.goldRoot(.basic),
        config,
    );
    std.debug.print("Done twoshapes.\n", .{});
}
