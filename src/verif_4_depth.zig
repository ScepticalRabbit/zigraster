// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const std = @import("std");
const buildconfig = @import("riley/zig/buildconfig.zig");
const F = buildconfig.F;

const benchcommon = @import("dev_support/benchcommon.zig");
const orch = @import("dev_support/orchestration.zig");
const tcfg = @import("dev_support/testconfig.zig");
const cammod = @import("riley/zig/camera.zig");
const cameraops = @import("riley/zig/cameraops.zig");
const gk = @import("riley/zig/geometrykernels.zig");
const iio = @import("riley/zig/imageio.zig");
const mo = @import("riley/zig/meshpipeline.zig");
const meshio = @import("riley/zig/meshio.zig");
const riley = @import("riley/zig/riley.zig");
const Rotation = @import("riley/zig/rotation.zig").Rotation;
const NDArray = @import("riley/zig/ndarray.zig").NDArray;
const sceneops = @import("riley/zig/sceneops.zig");

const CameraPrepared = cammod.CameraPrepared;
const MeshInput = mo.MeshInput;

const OVERLAP_X: F = 0.85;
const OVERLAP_Y: F = 0.8;
pub const BEHIND_FACT: F = 1.05;

const pixel_num = [_]u32{ 1200, 600 };
const fov_scale: F = 1.01;
const out_root = "verif/verif_4";

const mesh_types = [_]gk.MeshType{
    .tri3,
    .tri6,
    .quad4ibi,
    .quad8,
    .quad9,
};

const DataCase = struct {
    case_name: []const u8,
    front_mesh_type: gk.MeshType,
    back_mesh_type: gk.MeshType,
    front_data_dir: []const u8,
    back_data_dir: []const u8,
    front_connect_name: []const u8,
    back_connect_name: []const u8,
    rot: Rotation,
};

fn pairedBackMeshType(front_mesh_type: gk.MeshType) gk.MeshType {
    return switch (front_mesh_type) {
        .tri3 => .tri6,
        .tri6 => .tri3,
        .quad4ibi => .quad8,
        .quad4newton => .quad8,
        .quad8, .quad9 => .quad4ibi,
    };
}

fn translateCoords(coords: *meshio.Coords, translation: [3]F) void {
    for (0..coords.mat.rows_num) |nn| {
        coords.mat.set(nn, 0, coords.mat.get(nn, 0) + translation[0]);
        coords.mat.set(nn, 1, coords.mat.get(nn, 1) + translation[1]);
        coords.mat.set(nn, 2, coords.mat.get(nn, 2) + translation[2]);
    }
}

fn loadStaticMesh(
    allocator: std.mem.Allocator,
    io: std.Io,
    data_dir: []const u8,
    connect_name: []const u8,
) !meshio.SimData {
    const coords_path = try std.fmt.allocPrint(
        allocator,
        "{s}/coords.csv",
        .{data_dir},
    );
    const connect_path = try std.fmt.allocPrint(
        allocator,
        "{s}/{s}",
        .{ data_dir, connect_name },
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

fn saveImageArtifacts(
    io: std.Io,
    out_dir: std.Io.Dir,
    base_name: []const u8,
    image: *const NDArray(F),
) !void {
    try iio.saveImage(
        io,
        out_dir,
        base_name,
        image,
        0,
        .{
            .format = .csv,
            .bits = null,
            .scaling = .none,
            .channels = image.dims[0],
        },
    );
    try iio.saveImage(
        io,
        out_dir,
        base_name,
        image,
        0,
        .{
            .format = .bmp,
            .bits = 8,
            .scaling = .{ .fixed = .{ 0.0, 1.0 } },
            .channels = image.dims[0],
        },
    );
}

fn calcDiffImage(
    allocator: std.mem.Allocator,
    both_image: *const NDArray(F),
    frontonly_image: *const NDArray(F),
) !NDArray(F) {
    var diff = try NDArray(F).initFlat(allocator, both_image.dims);
    for (0..both_image.slice.len) |ii| {
        diff.slice[ii] = both_image.slice[ii] - frontonly_image.slice[ii];
    }
    return diff;
}

fn renderSingle(
    allocator: std.mem.Allocator,
    io: std.Io,
    camera_input: cammod.CameraInput,
    meshes: []const MeshInput,
) !NDArray(F) {
    var config = tcfg.getRasterConfig(.preview);
    config.save_strategy = .memory;
    config.image_save_opts = &[_]iio.ImageSaveOpts{
        .{ .format = .csv, .bits = null, .scaling = .none },
    };

    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = @max(@as(u16, 1), config.total_threads) },
    };
    const result = (try riley.raster(
        allocator,
        &render_groups,
        &[_]cammod.CameraInput{camera_input},
        meshes,
        config,
        null,
    )) orelse return error.NoResult;
    defer {
        allocator.free(result.slice);
        var res_mut = result;
        res_mut.deinit(allocator);
    }

    return try benchcommon.extractFirstFrameImage(allocator, &result);
}

fn buildCaseSpec(
    case_name: []const u8,
    mesh_type: gk.MeshType,
) !DataCase {
    const is_rabbit = std.mem.eql(u8, case_name, "rabbit");
    const back_mesh_type = pairedBackMeshType(mesh_type);
    if (is_rabbit) {
        return .{
            .case_name = case_name,
            .front_mesh_type = mesh_type,
            .back_mesh_type = mesh_type,
            .front_data_dir = try std.fmt.allocPrint(
                std.heap.page_allocator,
                "data/rabbits/riley_{s}",
                .{orch.meshDataName(mesh_type)},
            ),
            .back_data_dir = try std.fmt.allocPrint(
                std.heap.page_allocator,
                "data/rabbits/feebs_{s}",
                .{orch.meshDataName(mesh_type)},
            ),
            .front_connect_name = "connectivity.csv",
            .back_connect_name = "connectivity.csv",
            .rot = Rotation.init(0.0, std.math.pi, 0.0),
        };
    }

    // Sphere case - need to handle quad4 variants differently than rabbit/simple
    const front_data_name = if (mesh_type == .quad4ibi)
        "quad4ibi"
    else
        orch.meshDataName(mesh_type);
    const back_data_name = if (back_mesh_type == .quad4ibi)
        "quad4ibi"
    else
        orch.meshDataName(back_mesh_type);
    return .{
        .case_name = case_name,
        .front_mesh_type = mesh_type,
        .back_mesh_type = back_mesh_type,
        .front_data_dir = try std.fmt.allocPrint(
            std.heap.page_allocator,
            "data/bench/{s}_sphere200",
            .{front_data_name},
        ),
        .back_data_dir = try std.fmt.allocPrint(
            std.heap.page_allocator,
            "data/bench/{s}_sphere200",
            .{back_data_name},
        ),
        .front_connect_name = "connect.csv",
        .back_connect_name = "connect.csv",
        .rot = Rotation.init(0.0, 0.0, 0.0),
    };
}

fn runCase(
    allocator: std.mem.Allocator,
    io: std.Io,
    case_spec: DataCase,
) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    const front_sim_data = try loadStaticMesh(
        aa,
        io,
        case_spec.front_data_dir,
        case_spec.front_connect_name,
    );
    const back_sim_data = try loadStaticMesh(
        aa,
        io,
        case_spec.back_data_dir,
        case_spec.back_connect_name,
    );

    const front_coords = try orch.copyCoords(aa, front_sim_data.coords);
    const back_coords = try orch.copyCoords(aa, back_sim_data.coords);

    const bounds = sceneops.boundsForCoords(&front_coords);
    const width = bounds.extent[0];
    const height = bounds.extent[1];
    const x_sep = width * (1.0 - OVERLAP_X);
    const y_sep = height * (1.0 - OVERLAP_Y);

    var front_coords_mut = front_coords;
    var back_coords_mut = back_coords;
    translateCoords(
        &front_coords_mut,
        .{ 0.5 * x_sep, -0.5 * y_sep, 0.0 },
    );
    translateCoords(
        &back_coords_mut,
        .{ -0.5 * x_sep, 0.5 * y_sep, 0.0 },
    );

    var front_mesh = MeshInput{
        .mesh_type = case_spec.front_mesh_type,
        .coords = front_coords_mut,
        .connect = front_sim_data.connect,
        .disp = null,
        .shader = .{ .func = .{
            .uvs = null,
            .coord_mode = .para,
            .builtin = .constant,
            .params = .{
                .output_scale = 0.0,
                .output_offset = 1.0,
            },
            .bits = 8,
            .scaling = .none,
            .normal_type = .none,
        } },
    };
    var back_mesh = MeshInput{
        .mesh_type = case_spec.back_mesh_type,
        .coords = back_coords_mut,
        .connect = back_sim_data.connect,
        .disp = null,
        .shader = .{ .func = .{
            .uvs = null,
            .coord_mode = .para,
            .builtin = .constant,
            .params = .{
                .output_scale = 0.0,
                .output_offset = 0.5,
            },
            .bits = 8,
            .scaling = .none,
            .normal_type = .none,
        } },
    };

    const temp_meshes = [_]MeshInput{ front_mesh, back_mesh };
    const roi_pos = sceneops.boundsCenterOverMeshes(&temp_meshes);
    const cam_pos = cameraops.posFillFrameFromRotOverMeshes(
        &temp_meshes,
        pixel_num,
        orch.default_pixel_size,
        orch.default_focal_length,
        case_spec.rot,
        fov_scale,
    );
    const camera = try CameraPrepared.init(
        aa,
        .{
            .pixels_num = pixel_num,
            .pixels_size = orch.default_pixel_size,
            .pos_world = cam_pos,
            .rot_world = case_spec.rot,
            .roi_cent_world = roi_pos,
            .focal_length = orch.default_focal_length,
            .sub_sample = 1,
        },
    );
    defer camera.deinit(aa);
    const camera_input = cammod.CameraInput{
        .pixels_num = camera.pixels_num,
        .pixels_size = camera.pixels_size,
        .pos_world = camera.pos_world,
        .rot_world = camera.rot_world,
        .roi_cent_world = camera.roi_cent_world,
        .focal_length = camera.focal_length,
        .sub_sample = camera.sub_sample,
        .distortion = camera.distortion,
    };

    const front_centroid = sceneops.boundsForCoords(&front_mesh.coords).center;
    const cam_axis = [3]F{
        camera_input.pos_world.slice[0] - camera_input.roi_cent_world.slice[0],
        camera_input.pos_world.slice[1] - camera_input.roi_cent_world.slice[1],
        camera_input.pos_world.slice[2] - camera_input.roi_cent_world.slice[2],
    };
    const cam_axis_norm = @sqrt(
        cam_axis[0] * cam_axis[0] +
            cam_axis[1] * cam_axis[1] +
            cam_axis[2] * cam_axis[2],
    );
    const cam_axis_unit = [3]F{
        cam_axis[0] / cam_axis_norm,
        cam_axis[1] / cam_axis_norm,
        cam_axis[2] / cam_axis_norm,
    };
    const front_dist =
        (camera_input.pos_world.slice[0] - front_centroid[0]) *
        cam_axis_unit[0] +
        (camera_input.pos_world.slice[1] - front_centroid[1]) *
            cam_axis_unit[1] +
        (camera_input.pos_world.slice[2] - front_centroid[2]) *
            cam_axis_unit[2];
    const behind_extra = (BEHIND_FACT - 1.0) * front_dist;
    translateCoords(&back_mesh.coords, .{
        -cam_axis_unit[0] * behind_extra,
        -cam_axis_unit[1] * behind_extra,
        -cam_axis_unit[2] * behind_extra,
    });

    const both_image = try renderSingle(
        allocator,
        io,
        camera_input,
        &[_]MeshInput{ front_mesh, back_mesh },
    );
    defer {
        allocator.free(both_image.slice);
        var both_mut = both_image;
        both_mut.deinit(allocator);
    }
    const frontonly_image = try renderSingle(
        allocator,
        io,
        camera_input,
        &[_]MeshInput{front_mesh},
    );
    defer {
        allocator.free(frontonly_image.slice);
        var front_mut = frontonly_image;
        front_mut.deinit(allocator);
    }
    const diff_image = try calcDiffImage(allocator, &both_image, &frontonly_image);
    defer {
        allocator.free(diff_image.slice);
        var diff_mut = diff_image;
        diff_mut.deinit(allocator);
    }

    const out_dir_path = try std.fmt.allocPrint(
        aa,
        "{s}/d_{s}_{s}",
        .{
            out_root,
            orch.meshDataName(case_spec.front_mesh_type),
            case_spec.case_name,
        },
    );
    var out_dir = try orch.openDirEnsured(io, out_dir_path);
    defer out_dir.close(io);

    try saveImageArtifacts(io, out_dir, "both", &both_image);
    try saveImageArtifacts(io, out_dir, "frontonly", &frontonly_image);
    try saveImageArtifacts(io, out_dir, "diff", &diff_image);

    std.debug.print(
        "Rendered d_{s}_{s}\n",
        .{ orch.meshDataName(case_spec.front_mesh_type), case_spec.case_name },
    );
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var root_dir = try orch.openDirEnsured(io, out_root);
    defer root_dir.close(io);

    const case_names = [_][]const u8{ "rabbit", "sphere" };
    for (case_names) |case_name| {
        for (mesh_types) |mesh_type| {
            const case_spec = try buildCaseSpec(case_name, mesh_type);
            try runCase(allocator, io, case_spec);
        }
    }

    std.debug.print("Done.\n", .{});
}
