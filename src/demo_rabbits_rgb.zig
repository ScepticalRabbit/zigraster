// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const std = @import("std");

const orch = @import("dev_support/orchestration.zig");
const cammod = @import("riley/zig/camera.zig");
const cameraops = @import("riley/zig/cameraops.zig");
const gk = @import("riley/zig/geometrykernels.zig");
const iio = @import("riley/zig/imageio.zig");
const meshio = @import("riley/zig/meshio.zig");
const mo = @import("riley/zig/meshpipeline.zig");
const rastcfg = @import("riley/zig/rasterconfig.zig");
const riley = @import("riley/zig/riley.zig");
const Rotation = @import("riley/zig/rotation.zig").Rotation;
const sceneops = @import("riley/zig/sceneops.zig");
const shaderops = @import("riley/zig/shaderops_common.zig");
const texops = @import("riley/zig/textureops.zig");
const uvio = @import("riley/zig/uvio.zig");
const buildconfig = @import("riley/zig/buildconfig.zig");

const CameraInput = cammod.CameraInput;
const CameraPrepared = cammod.CameraPrepared;
const MeshInput = mo.MeshInput;
const FuncShaderBuiltin = shaderops.FuncShaderBuiltin;
const FuncShaderParams = shaderops.FuncShaderParams;
const F = buildconfig.F;

const rabbit_mesh_types = [_]gk.MeshType{
    .tri3,
    .tri6,
    .quad4ibi,
    .quad8,
    .quad9,
};

const out_dir_root = "./out/demo-rabbits-rgb";
const pixel_num = [_]u32{ 1600, 800 };
const fov_scale: F = @floatCast(1.01);
const overlap_frac_xy = [2]F{ 0.85, 0.8 };
const checker_squares_per_axis: F = 36.0;
const background_value: F = 0.5 * @as(F, std.math.maxInt(u8));

const ShaderMode = enum {
    tex,
    nodal,
    func,
};

fn shaderModeForMeshIndex(mesh_idx: usize) ShaderMode {
    return switch (@mod(mesh_idx, 3)) {
        0 => .tex,
        1 => .nodal,
        else => .func,
    };
}

fn buildRabbitDir(
    allocator: std.mem.Allocator,
    rabbit_name: []const u8,
    mesh_type: gk.MeshType,
) ![]const u8 {
    return try std.fmt.allocPrint(
        allocator,
        "data/rabbits/{s}_{s}",
        .{ rabbit_name, orch.meshDataName(mesh_type) },
    );
}

fn loadStaticMesh(
    allocator: std.mem.Allocator,
    io: std.Io,
    data_dir: []const u8,
) !meshio.SimData {
    const coords_path = try std.fmt.allocPrint(
        allocator,
        "{s}/coords.csv",
        .{data_dir},
    );
    const connect_path = try std.fmt.allocPrint(
        allocator,
        "{s}/connectivity.csv",
        .{data_dir},
    );
    return try meshio.loadSimData(
        allocator,
        io,
        coords_path,
        connect_path,
        null,
        null,
    );
}

fn loadRabbitUvMap(
    allocator: std.mem.Allocator,
    io: std.Io,
    data_dir: []const u8,
) !uvio.UVMap {
    const uv_path = try std.fmt.allocPrint(
        allocator,
        "{s}/uvs.csv",
        .{data_dir},
    );
    return try uvio.loadUVMap(allocator, io, uv_path);
}

fn buildUvRgbField(
    allocator: std.mem.Allocator,
    uvs: uvio.UVMap,
) !meshio.Field {
    const node_num = uvs.array.dims[0];
    var field = try meshio.Field.initAlloc(allocator, 1, node_num, 3);

    for (0..node_num) |nn| {
        const uu = uvs.array.get(&[_]usize{ nn, 0 });
        const vv = uvs.array.get(&[_]usize{ nn, 1 });
        field.array.set(&[_]usize{ 0, nn, 0 }, uu);
        field.array.set(&[_]usize{ 0, nn, 1 }, vv);
        field.array.set(&[_]usize{ 0, nn, 2 }, 0.5 * (uu + vv));
    }

    return field;
}

fn makeRgbMeshInput(
    allocator: std.mem.Allocator,
    io: std.Io,
    rabbit_name: []const u8,
    mesh_type: gk.MeshType,
    shader_mode: ShaderMode,
    texture: texops.Tex(u8, 3),
) !MeshInput {
    const data_dir = try buildRabbitDir(allocator, rabbit_name, mesh_type);
    const sim_data = try loadStaticMesh(allocator, io, data_dir);
    const uvs = try loadRabbitUvMap(allocator, io, data_dir);

    const shader: shaderops.ShaderInput = switch (shader_mode) {
        .tex => .{ .tex_rgb_u8 = .{
            .uvs = uvs.array,
            .tex = texture,
            .samp_cfg = .{
                .sample = .cubic_catmull_rom,
                .mode = .lut_lerp,
            },
            .bits = 8,
            .scaling = .none,
            .normal_type = .none,
        } },
        .nodal => .{ .nodal = .{
            .field = try buildUvRgbField(allocator, uvs),
            .bits = 8,
            .scaling = .auto,
            .scale_over = .over_frames,
            .normal_type = .none,
        } },
        .func => .{ .func_rgb = .{
            .uvs = uvs.array,
            .coord_mode = .uv,
            .builtin = FuncShaderBuiltin.checker,
            .params = FuncShaderParams{
                .coord_scale = .{
                    checker_squares_per_axis,
                    checker_squares_per_axis,
                },
            },
            .bits = 8,
            .scaling = .auto,
            .normal_type = .none,
        } },
    };

    return .{
        .mesh_type = mesh_type,
        .coords = try sceneops.duplicateCoords(allocator, sim_data.coords),
        .connect = sim_data.connect,
        .disp = null,
        .shader = shader,
    };
}

fn buildRabbitPairScene(
    allocator: std.mem.Allocator,
    io: std.Io,
    texture: texops.Tex(u8, 3),
) ![]MeshInput {
    var mesh_list = std.ArrayList(MeshInput).empty;
    var group_list = std.ArrayList(sceneops.MeshGroup).empty;

    for (rabbit_mesh_types) |mesh_type| {
        const pair_start = mesh_list.items.len;
        const front_mode = shaderModeForMeshIndex(pair_start);
        const back_mode = shaderModeForMeshIndex(pair_start + 1);
        try mesh_list.append(allocator, try makeRgbMeshInput(
            allocator,
            io,
            "riley",
            mesh_type,
            front_mode,
            texture,
        ));
        try mesh_list.append(allocator, try makeRgbMeshInput(
            allocator,
            io,
            "feebs",
            mesh_type,
            back_mode,
            texture,
        ));

        const front_group = sceneops.meshGroupSingle(pair_start);
        const back_group = sceneops.meshGroupSingle(pair_start + 1);
        sceneops.overlapMeshGroupBounds(
            mesh_list.items,
            front_group,
            back_group,
            .{
                .overlap_frac = .{
                    overlap_frac_xy[0],
                    overlap_frac_xy[1],
                    0.0,
                },
                .enabled_axes = .{ true, true, false },
                .direction = .{ .positive, .negative, .current },
            },
        );

        try group_list.append(allocator, sceneops.meshGroupSpan(pair_start, 2));
    }

    sceneops.arrangeMeshGroupsGrid(
        mesh_list.items,
        group_list.items,
        .{
            .gap = .{ 0.18, 0.28, 0.0 },
            .max_divs = .{ 3, 2, 1 },
        },
    );
    return try mesh_list.toOwnedSlice(allocator);
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var arena = std.heap.ArenaAllocator.init(init.gpa);
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

    const mesh_inputs = try buildRabbitPairScene(aa, io, texture);
    const rot = Rotation.init(0.0, std.math.pi, 0.0);
    const roi_pos = sceneops.boundsCenterOverMeshes(mesh_inputs);
    const cam_pos = cameraops.posFillFrameFromRotOverMeshes(
        mesh_inputs,
        pixel_num,
        orch.default_pixel_size,
        orch.default_focal_length,
        rot,
        fov_scale,
    );
    const camera = try CameraPrepared.init(
        aa,
        .{
            .pixels_num = pixel_num,
            .pixels_size = orch.default_pixel_size,
            .pos_world = cam_pos,
            .rot_world = rot,
            .roi_cent_world = roi_pos,
            .focal_length = orch.default_focal_length,
            .sub_sample = 2,
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
    const config = rastcfg.RasterConfig{
        .save_strategy = .disk,
        .image_save_mode = .rgb,
        .background_value = background_value,
        .image_save_opts = &[_]iio.ImageSaveOpts{
            .{ .format = .bmp, .bits = 8, .scaling = .none },
        },
    };
    const render_groups = [_]riley.RenderGroupSpec{
        .{
            .io = io,
            .workers = @max(@as(u16, 1), config.total_threads),
        },
    };

    const images = try riley.raster(
        aa,
        &render_groups,
        &[_]CameraInput{camera_input},
        mesh_inputs,
        config,
        out_dir_root,
    );
    if (images) |img| {
        aa.free(img.slice);
        img.deinit(aa);
    }
}
