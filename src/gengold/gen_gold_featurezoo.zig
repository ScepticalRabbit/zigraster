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

const pixel_size_zoo = [2]F{ @floatCast(5.3e-6), @floatCast(5.3e-6) };
const focal_length_zoo: F = @floatCast(50.0e-3);

const Case = struct {
    shape: []const u8,
    elem: []const u8,
    mesh_type: gk.MeshType,
};

const cases = [_]Case{
    .{ .shape = "cube_surf", .elem = "quad9", .mesh_type = .quad9 },
    .{ .shape = "cube_surf", .elem = "tri6", .mesh_type = .tri6 },
    .{ .shape = "cylinder_surf", .elem = "quad8", .mesh_type = .quad8 },
    .{ .shape = "cylinder_surf", .elem = "tri6", .mesh_type = .tri6 },
    .{ .shape = "platewithhole_surf", .elem = "quad4", .mesh_type = .quad4 },
    .{ .shape = "platewithhole_surf", .elem = "tri3", .mesh_type = .tri3 },
};

const mesh_centers = [cases.len][3]F{
    .{ -0.0125, 0.0125, 0.0 },
    .{ 0.0125, 0.0125, 0.0 },
    .{ -0.0125, 0.0, 0.0 },
    .{ 0.0125, 0.0, 0.0 },
    .{ -0.0125, -0.0125, 0.0 },
    .{ 0.0125, -0.0125, 0.0 },
};

fn sliceFieldToTwoFrames(
    allocator: std.mem.Allocator,
    field: meshio.Field,
) !meshio.Field {
    const time_n = field.getTimeN();
    const node_n = field.getCoordN();
    const chan_n = field.getFieldsN();
    const last_t = if (time_n > 1) time_n - 1 else 0;
    var result = try meshio.Field.initAlloc(allocator, 2, node_n, chan_n);
    for (0..node_n) |nn| {
        for (0..chan_n) |cc| {
            result.array.set(
                &.{ 0, nn, cc },
                field.array.get(&.{ 0, nn, cc }),
            );
            result.array.set(
                &.{ 1, nn, cc },
                field.array.get(&.{ last_t, nn, cc }),
            );
        }
    }
    return result;
}

fn buildRgbFieldTwoFrames(
    allocator: std.mem.Allocator,
    temp: meshio.Field,
    disp: meshio.Field,
) !meshio.Field {
    const node_n = temp.getCoordN();
    const time_temp = temp.getTimeN();
    const time_disp = disp.getTimeN();
    const last_temp = if (time_temp > 1) time_temp - 1 else 0;
    const last_disp = if (time_disp > 1) time_disp - 1 else 0;

    var field = try meshio.Field.initAlloc(allocator, 2, node_n, 3);
    for (0..node_n) |nn| {
        // Frame 0: static
        field.array.set(&.{ 0, nn, 0 }, temp.array.get(&.{ 0, nn, 0 }));
        field.array.set(&.{ 0, nn, 1 }, disp.array.get(&.{ 0, nn, 0 }));
        field.array.set(&.{ 0, nn, 2 }, disp.array.get(&.{ 0, nn, 1 }));

        // Frame 1: deformed
        field.array.set(&.{ 1, nn, 0 }, temp.array.get(&.{ last_temp, nn, 0 }));
        field.array.set(&.{ 1, nn, 1 }, disp.array.get(&.{ last_disp, nn, 0 }));
        field.array.set(&.{ 1, nn, 2 }, disp.array.get(&.{ last_disp, nn, 1 }));
    }
    return field;
}

fn textureShader(
    comptime T: type,
    comptime C: usize,
    texture: texops.Tex(T, C),
    uvs: uvio.UVMap,
    comptime bits: u8,
    linear: bool,
    normal_type: shaderops.NormalType,
) shaderops.ShaderInput {
    const samp_cfg = texops.TexSampConfig{
        .sample = if (linear) .linear else .cubic_catmull_rom,
        .mode = if (linear) .direct else .lut_lerp,
    };
    const input = shaderops.TexInput(T, C){
        .uvs = uvs.array,
        .tex = texture,
        .samp_cfg = samp_cfg,
        .bits = bits,
        .scaling = .auto,
        .normal_type = normal_type,
    };
    if (C == 1 and T == u8) return .{ .tex_u8 = input };
    if (C == 3 and T == u8) return .{ .tex_rgb_u8 = input };
    unreachable;
}

fn loadZooMesh(
    comptime T: type,
    comptime C: usize,
    comptime bits: u8,
    allocator: std.mem.Allocator,
    io: std.Io,
    case: Case,
    case_index: usize,
    texture: texops.Tex(T, C),
) !MeshInput {
    const dir = try std.fmt.allocPrint(
        allocator,
        "data/shapes/{s}/{s}/",
        .{ case.shape, case.elem },
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

    const temp = try sliceFieldToTwoFrames(allocator, raw_temp);
    const disp = try sliceFieldToTwoFrames(allocator, raw_disp);

    const normal_type: shaderops.NormalType = switch (case_index % 3) {
        0 => .none,
        1 => .exact,
        else => .avg,
    };

    const shader: shaderops.ShaderInput = switch (case_index) {
        0, 2 => textureShader(
            T,
            C,
            texture,
            uvs,
            bits,
            case_index != 0,
            normal_type,
        ),
        1, 5 => blk: {
            const field = if (C == 1)
                temp
            else
                try buildRgbFieldTwoFrames(allocator, temp, disp);
            break :blk .{ .nodal = .{
                .field = field,
                .bits = bits,
                .scaling = .auto,
                .scale_over = if (case_index == 1)
                    .over_frames
                else
                    .within_frames,
                .normal_type = normal_type,
            } };
        },
        3, 4 => blk: {
            const params = if (case_index == 3)
                shaderops.FuncShaderParams{
                    .coord_scale = .{ 1000.0, 1000.0 },
                    .settings = .{ .checker = .{} },
                }
            else
                shaderops.FuncShaderParams{
                    .coord_scale = .{ 1.0, 1.0 },
                    .settings = .{ .eggbox = .{
                        .pitch = .{ 0.005, 0.005 },
                    } },
                };
            const input = shaderops.FuncInput{
                .coord_mode = if (case_index == 3)
                    .world_reference
                else
                    .world_deformed,
                .builtin = if (case_index == 3) .checker else .eggbox,
                .params = params,
                .bits = bits,
                .scaling = .auto,
                .normal_type = normal_type,
            };
            break :blk if (C == 1)
                .{ .func = input }
            else
                .{ .func_rgb = input };
        },
        else => unreachable,
    };

    return .{
        .mesh_type = case.mesh_type,
        .coords = sim.coords,
        .connect = sim.connect,
        .disp = if (case_index % 2 == 1) disp else null,
        .shader = shader,
    };
}

pub fn buildZooScene(
    comptime T: type,
    comptime C: usize,
    comptime bits: u8,
    allocator: std.mem.Allocator,
    io: std.Io,
    texture: texops.Tex(T, C),
) ![]MeshInput {
    var meshes = std.ArrayList(MeshInput).empty;
    var groups = std.ArrayList(sceneops.MeshGroup).empty;
    for (cases, 0..) |case, index| {
        try meshes.append(
            allocator,
            try loadZooMesh(T, C, bits, allocator, io, case, index, texture),
        );
        try groups.append(allocator, sceneops.meshGroupSingle(index));
    }
    const plate = &meshes.items[5];
    for (0..plate.coords.mat.rows_num) |node| {
        const xx = plate.coords.mat.get(node, 0);
        const yy = plate.coords.mat.get(node, 1);
        plate.coords.mat.set(node, 0, -yy);
        plate.coords.mat.set(node, 1, xx);
    }
    for (groups.items, mesh_centers) |group, center| {
        sceneops.centerMeshGroupAt(meshes.items, group, center);
    }
    return try meshes.toOwnedSlice(allocator);
}

fn makeZooCamera(
    meshes: []MeshInput,
    pixels_num: [2]u32,
    rot: Rotation,
    sub_sample: u32,
    distortion: camera.DistortionModel,
    psf: camera.PointSpreadFunc,
) camera.CameraInput {
    const target = sceneops.boundsCenterOverMeshes(meshes);
    return .{
        .pixels_num = pixels_num,
        .pixels_size = pixel_size_zoo,
        .pos_world = cameraops.posFillFrameFromRotOverMeshesAndTarg(
            meshes,
            target,
            pixels_num,
            pixel_size_zoo,
            focal_length_zoo,
            rot,
            1.1,
        ),
        .rot_world = rot,
        .roi_cent_world = target,
        .focal_length = focal_length_zoo,
        .sub_sample = sub_sample,
        .distortion = distortion,
        .psf = psf,
    };
}

pub fn buildAllZooCameras(meshes: []MeshInput) [6]camera.CameraInput {
    const deg = std.math.degreesToRadians;
    const brown: camera.DistortionModel = .{ .brown_conrady = .{
        .k1 = -0.12,
        .k2 = 0.035,
        .p1 = 0.0002,
        .p2 = -0.0001,
    } };
    const gaussian: camera.PointSpreadFunc = .{ .gaussian = .{
        .sigma_px = 0.65,
        .supp_rad_px = 2.0,
    } };

    return .{
        // Cam 0: Face On
        makeZooCamera(
            meshes,
            .{ 400, 250 },
            Rotation.init(0, 0, 0),
            1,
            .none,
            .{ .pixel_box = .{} },
        ),
        // Cam 1: Perspective Tilt + Brown Distortion
        makeZooCamera(
            meshes,
            .{ 400, 250 },
            Rotation.init(0, deg(25.0), 0),
            2,
            brown,
            .{ .pixel_box = .{} },
        ),
        // Cam 2: Off-axis Tilt + Gaussian PSF (Portrait)
        makeZooCamera(
            meshes,
            .{ 250, 400 },
            Rotation.init(0, deg(-28.0), 0),
            2,
            .none,
            gaussian,
        ),
        // Cam 3: Steep Angle + Brown + Gaussian PSF
        makeZooCamera(
            meshes,
            .{ 400, 250 },
            Rotation.init(deg(90.0), deg(25.0), 0),
            2,
            brown,
            gaussian,
        ),
        // Cam 4: Multi-Axis Compound Rotation + Anisotropic Gaussian PSF
        makeZooCamera(
            meshes,
            .{ 400, 250 },
            Rotation.init(deg(18.0), deg(38.0), deg(26.0)),
            2,
            .none,
            .{ .anisotropic_gaussian = .{
                .sigma_x_px = 0.55,
                .sigma_y_px = 0.9,
                .theta_rad = deg(25.0),
                .supp_rad_px = 2.5,
            } },
        ),
        // Cam 5: Reverse Pitch & Yaw + Brown Distortion
        makeZooCamera(
            meshes,
            .{ 400, 250 },
            Rotation.init(deg(-90.0), deg(-20.0), deg(5.0)),
            2,
            brown,
            .{ .pixel_box = .{} },
        ),
    };
}

pub fn generateZooMonoGold(
    allocator: std.mem.Allocator,
    io: std.Io,
    texture_grey: texops.Tex(u8, 1),
    gold_dir_root: []const u8,
    config: rastcfg.RasterConfig,
) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    const meshes = try buildZooScene(u8, 1, 8, aa, io, texture_grey);
    const cameras = buildAllZooCameras(meshes);

    const gold_dir = try std.fmt.allocPrint(
        aa,
        "{s}/featurezoo_mono",
        .{gold_dir_root},
    );

    var out_dir = try orch.openDirEnsured(io, gold_dir);
    out_dir.close(io);

    var run_config = config;
    run_config.save_strategy = .disk;
    run_config.background_value = 127.5;

    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = @max(@as(u16, 1), run_config.total_threads) },
    };

    const result = try riley.raster(
        aa,
        &render_groups,
        &cameras,
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

pub fn generateZooRgbGold(
    allocator: std.mem.Allocator,
    io: std.Io,
    texture_rgb: texops.Tex(u8, 3),
    gold_dir_root: []const u8,
    config: rastcfg.RasterConfig,
) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    const meshes = try buildZooScene(u8, 3, 8, aa, io, texture_rgb);
    const all_cameras = buildAllZooCameras(meshes);
    const rgb_cameras = [_]CameraInput{ all_cameras[0], all_cameras[1] };

    const gold_dir = try std.fmt.allocPrint(
        aa,
        "{s}/featurezoo_rgb",
        .{gold_dir_root},
    );

    var out_dir = try orch.openDirEnsured(io, gold_dir);
    out_dir.close(io);

    var run_config = config;
    run_config.save_strategy = .disk;
    run_config.background_value = 127.5;

    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = @max(@as(u16, 1), run_config.total_threads) },
    };

    const result = try riley.raster(
        aa,
        &render_groups,
        &rgb_cameras,
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

pub fn generateAllFeatureZooCases(
    allocator: std.mem.Allocator,
    io: std.Io,
    texture_grey: texops.Tex(u8, 1),
    texture_rgb: texops.Tex(u8, 3),
    gold_dir_root: []const u8,
    config: rastcfg.RasterConfig,
) !void {
    try generateZooMonoGold(allocator, io, texture_grey, gold_dir_root, config);
    try generateZooRgbGold(allocator, io, texture_rgb, gold_dir_root, config);
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

    std.debug.print("Generating Basic Suite: featurezoo cases...\n", .{});
    try generateAllFeatureZooCases(
        aa,
        io,
        texture_grey,
        texture_rgb,
        policy.goldRoot(.basic),
        config,
    );
    std.debug.print("Done featurezoo.\n", .{});
}
