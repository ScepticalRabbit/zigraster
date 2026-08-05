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

const NDArray = @import("../riley/zig/ndarray.zig").NDArray;
const iio = @import("../riley/zig/imageio.zig");
const meshio = @import("../riley/zig/meshio.zig");
const mo = @import("../riley/zig/meshpipeline.zig");
const sceneops = @import("../riley/zig/sceneops.zig");
const gk = @import("../riley/zig/geometrykernels.zig");
const texops = @import("../riley/zig/textureops.zig");
const uvio = @import("../riley/zig/uvio.zig");
const CameraPrepared = @import("../riley/zig/camera.zig").CameraPrepared;
const cameraops = @import("../riley/zig/cameraops.zig");
const Rotation = @import("../riley/zig/rotation.zig").Rotation;
const policy = @import("testpolicy.zig");

pub const default_multimesh_mesh_types = [_]gk.MeshType{
    .tri3,
    .tri6,
    .quad4ibi,
    .quad8,
    .quad9,
};

pub const default_multimesh_dir_paths = [_][]const u8{
    "data/simple/tri3_twoelems/",
    "data/simple/tri6_twoelems/",
    "data/simple/quad4_twoelems/",
    "data/simple/quad8_twoelems/",
    "data/simple/quad9_twoelems/",
};

pub const default_pixel_size = [_]F{ 5.3e-6, 5.3e-6 };
pub const default_focal_length: F = 50.0e-3;

pub fn defaultRotation() Rotation {
    return Rotation.init(0, 0, 0);
}

pub fn loadData(
    outer_alloc: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
) !meshio.SimData {
    const coords_path = try std.fmt.allocPrint(
        outer_alloc,
        "{s}/coords.csv",
        .{path},
    );
    const connect_path = try std.fmt.allocPrint(
        outer_alloc,
        "{s}/connectivity.csv",
        .{path},
    );
    const field_paths = [_][]const u8{
        try std.fmt.allocPrint(outer_alloc, "{s}/field_disp_x.csv", .{path}),
        try std.fmt.allocPrint(outer_alloc, "{s}/field_disp_y.csv", .{path}),
        try std.fmt.allocPrint(outer_alloc, "{s}/field_disp_z.csv", .{path}),
    };
    return try meshio.loadSimData(
        outer_alloc,
        io,
        coords_path,
        connect_path,
        field_paths[0..],
        null,
    );
}

pub fn meshDataName(mesh_type: gk.MeshType) []const u8 {
    return policy.meshName(.fixture_case, mesh_type);
}

pub fn testTypeSuffix(test_type: []const u8) []const u8 {
    if (std.mem.eql(u8, test_type, "full")) return "fullscreen";
    if (std.mem.eql(u8, test_type, "twoelems")) return "twoelems";
    if (std.mem.eql(u8, test_type, "single")) return "single";
    return test_type;
}

pub const SingleMeshPrepared = struct {
    sim_data: meshio.SimData,
    uvs: uvio.UVMap,
    camera: CameraPrepared,
};

fn initCameraForSimDataAllFrames(
    allocator: std.mem.Allocator,
    sim_data: *const meshio.SimData,
    mesh_type: gk.MeshType,
    pixel_num: [2]u32,
    fov_scale: F,
) !CameraPrepared {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    const time_steps = if (sim_data.field) |field| field.getTimeN() else 1;
    var mesh_inputs = try aa.alloc(mo.MeshInput, time_steps);

    for (0..time_steps) |tt| {
        var frame_coords = try meshio.Coords.initAlloc(aa, sim_data.coords.mat.rows_num);

        for (0..sim_data.coords.mat.rows_num) |nn| {
            frame_coords.mat.set(nn, 0, sim_data.coords.x(nn));
            frame_coords.mat.set(nn, 1, sim_data.coords.y(nn));
            frame_coords.mat.set(nn, 2, sim_data.coords.z(nn));

            if (sim_data.field) |field| {
                frame_coords.mat.set(
                    nn,
                    0,
                    frame_coords.x(nn) + field.array.get(&[_]usize{ tt, nn, 0 }),
                );
                frame_coords.mat.set(
                    nn,
                    1,
                    frame_coords.y(nn) + field.array.get(&[_]usize{ tt, nn, 1 }),
                );
                frame_coords.mat.set(
                    nn,
                    2,
                    frame_coords.z(nn) + field.array.get(&[_]usize{ tt, nn, 2 }),
                );
            }
        }

        mesh_inputs[tt] = .{
            .mesh_type = mesh_type,
            .coords = frame_coords,
            .connect = sim_data.connect,
            .disp = null,
            .shader = .{
                .func = .{
                    .uvs = null,
                    .coord_mode = .para,
                    .builtin = .constant,
                    .normal_type = .none,
                },
            },
        };
    }

    return try initCameraForMeshes(
        allocator,
        mesh_inputs,
        pixel_num,
        fov_scale,
    );
}

pub fn prepareSingleMeshCase(
    allocator: std.mem.Allocator,
    io: std.Io,
    test_type: []const u8,
    mesh_type: gk.MeshType,
    pixel_num: [2]u32,
    fov_scale: F,
    data_dir_root: []const u8,
) !SingleMeshPrepared {
    const suffix = testTypeSuffix(test_type);
    const data_name = meshDataName(mesh_type);
    const case_name = try std.fmt.allocPrint(
        allocator,
        "{s}_{s}",
        .{ data_name, suffix },
    );
    const data_path = try std.fmt.allocPrint(
        allocator,
        "{s}/{s}",
        .{ data_dir_root, case_name },
    );

    const sim_data = try loadData(allocator, io, data_path);
    const uv_path = try std.fmt.allocPrint(allocator, "{s}/uvs.csv", .{data_path});
    const uvs = try uvio.loadUVMap(allocator, io, uv_path);
    const camera = if (std.mem.startsWith(u8, test_type, "distort_"))
        try initCameraForSimDataAllFrames(
            allocator,
            &sim_data,
            mesh_type,
            pixel_num,
            fov_scale,
        )
    else
        try initCameraForCoords(
            allocator,
            &sim_data.coords,
            pixel_num,
            fov_scale,
        );

    return .{
        .sim_data = sim_data,
        .uvs = uvs,
        .camera = camera,
    };
}

pub fn initCameraForCoords(
    allocator: std.mem.Allocator,
    coords: *const meshio.Coords,
    pixel_num: [2]u32,
    fov_scale: F,
) !CameraPrepared {
    return try initCameraForCoordsWithRotation(
        allocator,
        coords,
        pixel_num,
        fov_scale,
        defaultRotation(),
    );
}

pub fn initCameraForCoordsWithRotation(
    allocator: std.mem.Allocator,
    coords: *const meshio.Coords,
    pixel_num: [2]u32,
    fov_scale: F,
    rot: Rotation,
) !CameraPrepared {
    const cam_pos = cameraops.posFillFrameFromRot(
        coords,
        pixel_num,
        default_pixel_size,
        default_focal_length,
        rot,
        fov_scale,
    );
    return try CameraPrepared.init(
        allocator,
        .{
            .pixels_num = pixel_num,
            .pixels_size = default_pixel_size,
            .pos_world = cam_pos,
            .rot_world = rot,
            .roi_cent_world = sceneops.boundsCenter(coords),
            .focal_length = default_focal_length,
            .sub_sample = 2,
        },
    );
}

pub fn initStereoCamerasForCoords(
    allocator: std.mem.Allocator,
    coords: *const meshio.Coords,
    pixel_num: [2]u32,
    fov_scale: F,
    half_angle_deg: F,
) ![2]CameraPrepared {
    const half_angle_rad = half_angle_deg * std.math.pi / 180.0;
    return .{
        try initCameraForCoordsWithRotation(
            allocator,
            coords,
            pixel_num,
            fov_scale,
            Rotation.init(0.0, -half_angle_rad, 0.0),
        ),
        try initCameraForCoordsWithRotation(
            allocator,
            coords,
            pixel_num,
            fov_scale,
            Rotation.init(0.0, half_angle_rad, 0.0),
        ),
    };
}

pub fn initCameraForMeshes(
    allocator: std.mem.Allocator,
    mesh_inputs: []mo.MeshInput,
    pixel_num: [2]u32,
    fov_scale: F,
) !CameraPrepared {
    const rot = defaultRotation();
    const roi_pos = sceneops.boundsCenterOverMeshes(mesh_inputs);
    const cam_pos = cameraops.posFillFrameFromRotOverMeshes(
        mesh_inputs,
        pixel_num,
        default_pixel_size,
        default_focal_length,
        rot,
        fov_scale,
    );
    return try CameraPrepared.init(
        allocator,
        .{
            .pixels_num = pixel_num,
            .pixels_size = default_pixel_size,
            .pos_world = cam_pos,
            .rot_world = rot,
            .roi_cent_world = roi_pos,
            .focal_length = default_focal_length,
            .sub_sample = 2,
        },
    );
}

pub const MultimeshShaderMode = enum { nodal, texture };

pub fn buildMultimeshInputs(
    allocator: std.mem.Allocator,
    io: std.Io,
    dir_paths: []const []const u8,
    shader_mode: MultimeshShaderMode,
) ![]mo.MeshInput {
    const sim_datas = try meshio.loadMultiSimData(allocator, io, dir_paths, .{});
    const mesh_inputs = switch (shader_mode) {
        .nodal => try mo.meshInputFromSimDataSlice(
            allocator,
            io,
            sim_datas,
            &default_multimesh_mesh_types,
            .nodal,
            null,
            null,
            null,
        ),
        .texture => try mo.meshInputFromSimDataSlice(
            allocator,
            io,
            sim_datas,
            &default_multimesh_mesh_types,
            .tex,
            dir_paths,
            "texture/speckle-simple.tiff",
            null,
        ),
    };
    sceneops.arrangeMeshesGrid(mesh_inputs, .{
        .gap = .{ 0.1, 0.1, 0.0 },
        .max_divs = .{ 3, 2, 1 },
    });
    return mesh_inputs;
}

pub fn copyCoords(
    allocator: std.mem.Allocator,
    coords: meshio.Coords,
) !meshio.Coords {
    return sceneops.duplicateCoords(allocator, coords);
}

fn buildGradientRgbField(
    allocator: std.mem.Allocator,
    coords: meshio.Coords,
    field: meshio.Field,
) !meshio.Field {
    const num_coords = coords.mat.rows_num;
    var rgb_field_arr = try NDArray(F).initFlat(
        allocator,
        &[_]usize{ field.array.dims[0], num_coords, 3 },
    );

    var min_x: F = std.math.inf(F);
    var max_x: F = -std.math.inf(F);
    for (0..num_coords) |nn| {
        const x_val = coords.x(nn);
        if (x_val < min_x) min_x = x_val;
        if (x_val > max_x) max_x = x_val;
    }
    const range_x = max_x - min_x;

    for (0..field.array.dims[0]) |tt| {
        for (0..num_coords) |nn| {
            const x_val = coords.x(nn);
            const t = if (range_x > 0) (x_val - min_x) / range_x else 0.5;

            var rr: F = 0;
            var gg: F = 0;
            var bb: F = 0;

            if (t < 0.5) {
                const t_scaled = t * 2.0;
                rr = 1.0 - t_scaled;
                gg = t_scaled;
            } else {
                const t_scaled = (t - 0.5) * 2.0;
                gg = 1.0 - t_scaled;
                bb = t_scaled;
            }

            rgb_field_arr.set(&[_]usize{ tt, nn, 0 }, rr);
            rgb_field_arr.set(&[_]usize{ tt, nn, 1 }, gg);
            rgb_field_arr.set(&[_]usize{ tt, nn, 2 }, bb);
        }
    }

    return .{
        .array = rgb_field_arr,
        .array_mem = rgb_field_arr.slice,
    };
}

pub fn buildMixedMeshInputs(
    allocator: std.mem.Allocator,
    io: std.Io,
    dir_paths: []const []const u8,
    texture: texops.Tex(u8, 1),
) ![]mo.MeshInput {
    const sim_datas = try meshio.loadMultiSimData(allocator, io, dir_paths, .{});
    var mesh_inputs = try allocator.alloc(mo.MeshInput, 10);

    for (0..default_multimesh_mesh_types.len) |ii| {
        mesh_inputs[ii] = .{
            .mesh_type = default_multimesh_mesh_types[ii],
            .coords = try copyCoords(allocator, sim_datas[ii].coords),
            .connect = sim_datas[ii].connect,
            .disp = sim_datas[ii].field,
            .shader = .{ .nodal = .{
                .field = sim_datas[ii].field.?,
                .bits = 8,
                .scaling = .auto,
                .scale_over = .within_frames,
            } },
        };
    }

    for (0..default_multimesh_mesh_types.len) |ii| {
        const uv_path = try std.fmt.allocPrint(
            allocator,
            "{s}uvs.csv",
            .{dir_paths[ii]},
        );
        const uvs = try uvio.loadUVMap(allocator, io, uv_path);

        mesh_inputs[ii + default_multimesh_mesh_types.len] = .{
            .mesh_type = default_multimesh_mesh_types[ii],
            .coords = try copyCoords(allocator, sim_datas[ii].coords),
            .connect = sim_datas[ii].connect,
            .disp = sim_datas[ii].field,
            .shader = .{ .tex_u8 = .{
                .uvs = uvs.array,
                .tex = texture,
                .samp_cfg = .{
                    .sample = .cubic_catmull_rom,
                    .mode = .lut_lerp,
                },
                .bits = 8,
                .scaling = .none,
            } },
        };
    }

    sceneops.arrangeMeshesGrid(mesh_inputs, .{
        .gap = .{ 0.15, 0.15, 0.0 },
        .max_divs = .{ 5, 2, 1 },
    });
    return mesh_inputs;
}

pub fn buildMixedRgbMeshInputs(
    allocator: std.mem.Allocator,
    io: std.Io,
    dir_paths: []const []const u8,
    texture: texops.Tex(u8, 3),
) ![]mo.MeshInput {
    const sim_datas = try meshio.loadMultiSimData(allocator, io, dir_paths, .{});
    var mesh_inputs = try allocator.alloc(mo.MeshInput, 10);

    for (0..default_multimesh_mesh_types.len) |ii| {
        const uv_path = try std.fmt.allocPrint(
            allocator,
            "{s}uvs.csv",
            .{dir_paths[ii]},
        );
        const uvs = try uvio.loadUVMap(allocator, io, uv_path);

        mesh_inputs[ii] = .{
            .mesh_type = default_multimesh_mesh_types[ii],
            .coords = try copyCoords(allocator, sim_datas[ii].coords),
            .connect = sim_datas[ii].connect,
            .disp = sim_datas[ii].field,
            .shader = .{ .tex_rgb_u8 = .{
                .uvs = uvs.array,
                .tex = texture,
                .samp_cfg = .{
                    .sample = .cubic_catmull_rom,
                    .mode = .lut_lerp,
                },
                .bits = 8,
                .scaling = .none,
            } },
        };
    }

    for (0..default_multimesh_mesh_types.len) |ii| {
        const rgb_field = try buildGradientRgbField(
            allocator,
            sim_datas[ii].coords,
            sim_datas[ii].field.?,
        );

        mesh_inputs[ii + default_multimesh_mesh_types.len] = .{
            .mesh_type = default_multimesh_mesh_types[ii],
            .coords = try copyCoords(allocator, sim_datas[ii].coords),
            .connect = sim_datas[ii].connect,
            .disp = sim_datas[ii].field,
            .shader = .{ .nodal = .{
                .field = rgb_field,
                .bits = 8,
                .scaling = .auto,
                .scale_over = .within_frames,
            } },
        };
    }

    sceneops.arrangeMeshesGrid(mesh_inputs, .{
        .gap = .{ 0.15, 0.15, 0.0 },
        .max_divs = .{ 5, 2, 1 },
    });
    return mesh_inputs;
}

pub fn openDirEnsured(io: std.Io, dir_path: []const u8) !std.Io.Dir {
    const cwd = std.Io.Dir.cwd();
    var parts_iter = std.mem.splitScalar(u8, dir_path, '/');
    var path_buf: [256]u8 = undefined;
    var path_len: usize = 0;

    while (parts_iter.next()) |part| {
        if (path_len > 0) {
            path_buf[path_len] = '/';
            path_len += 1;
        }
        std.mem.copyForwards(u8, path_buf[path_len..], part);
        path_len += part.len;
        cwd.createDir(io, path_buf[0..path_len], .default_dir) catch |err| {
            if (err != error.PathAlreadyExists) return err;
        };
    }

    return try cwd.openDir(io, dir_path, .{});
}
