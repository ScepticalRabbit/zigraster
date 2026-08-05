// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const std = @import("std");
const policy = @import("../dev_support/testpolicy.zig");
const orch = @import("../dev_support/orchestration.zig");
const tcfg = @import("../dev_support/testconfig.zig");
const gk = @import("../riley/zig/geometrykernels.zig");
const meshio = @import("../riley/zig/meshio.zig");
const mo = @import("../riley/zig/meshpipeline.zig");
const uvio = @import("../riley/zig/uvio.zig");
const CameraInput = @import("../riley/zig/camera.zig").CameraInput;
const iio = @import("../riley/zig/imageio.zig");
const rastcfg = @import("../riley/zig/rasterconfig.zig");
const shaderops = @import("../riley/zig/shaderops.zig");
const riley = @import("../riley/zig/riley.zig");

const data_root = "data/min";
const test_type = "sphere200";
const CoordMode = enum { uv, param };

const SphereCasePrepared = struct {
    coords: meshio.Coords,
    connect: meshio.Connect,
    uvs: uvio.UVMap,
    camera_input: CameraInput,
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
    const uv_path = try std.fs.path.join(
        allocator,
        &[_][]const u8{ data_dir, "uvs.csv" },
    );

    const sim_data = try meshio.loadSimData(
        allocator,
        io,
        coord_path,
        connect_path,
        null,
        null,
    );
    const uvs = try uvio.loadUVMap(allocator, io, uv_path);
    const camera = try orch.initCameraForCoords(
        allocator,
        &sim_data.coords,
        .{ 640, 400 },
        1.0,
    );

    return .{
        .coords = sim_data.coords,
        .connect = sim_data.connect,
        .uvs = uvs,
        .camera_input = CameraInput{
            .pixels_num = camera.pixels_num,
            .pixels_size = camera.pixels_size,
            .pos_world = camera.pos_world,
            .rot_world = camera.rot_world,
            .roi_cent_world = camera.roi_cent_world,
            .focal_length = camera.focal_length,
            .sub_sample = camera.sub_sample,
            .distortion = camera.distortion,
        },
    };
}

fn renderCase(
    allocator: std.mem.Allocator,
    io: std.Io,
    mesh_input: mo.MeshInput,
    camera_input: CameraInput,
    out_dir_path: []const u8,
    config: rastcfg.RasterConfig,
) !void {
    var out_dir = try orch.openDirEnsured(io, out_dir_path);
    out_dir.close(io);

    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = @max(@as(u16, 1), config.total_threads) },
    };
    const images = try riley.raster(
        allocator,
        &render_groups,
        &[_]CameraInput{camera_input},
        &[_]mo.MeshInput{mesh_input},
        config,
        out_dir_path,
    );
    if (images) |img| {
        allocator.free(img.slice);
        img.deinit(allocator);
    }
}

pub fn main(init: std.process.Init) !void {
    try mainWithOutputRoot(init, policy.goldRoot(.texfunc));
}

pub fn mainWithOutputRoot(
    init: std.process.Init,
    output_root: []const u8,
) !void {
    const io = init.io;
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const allocator = arena.allocator();

    const mesh_types = [_]gk.MeshType{
        .tri3,
        .tri6,
        .quad4ibi,
        .quad4newton,
        .quad8,
        .quad9,
    };
    const builtins = [_]shaderops.FuncShaderBuiltin{
        .constant,
        .linear,
        .quadratic,
        .sinusoidal,
        .checker_smooth,
        .lambertian_normal_z,
    };
    const coord_modes = [_]CoordMode{ .uv, .param };
    var config = tcfg.getRasterConfig(.gold);
    config.save_strategy = .disk;
    config.image_save_opts = &[_]iio.ImageSaveOpts{
        .{ .format = .fimg, .bits = null, .scaling = .none },
        .{ .format = .bmp, .bits = 8, .scaling = .auto },
    };

    for (mesh_types) |mesh_type| {
        _ = arena.reset(.free_all);
        const prepared = try loadSphereCase(
            allocator,
            io,
            mesh_type,
        );

        for (coord_modes) |coord_mode| {
            const uvs = if (coord_mode == .uv) prepared.uvs.array else null;
            const coord_name = if (coord_mode == .uv) "uv" else "param";

            for (builtins) |builtin| {
                const normal_type: shaderops.NormalType =
                    if (builtin == .lambertian_normal_z) .avg else .none;

                const scalar_dir = try std.fmt.allocPrint(
                    allocator,
                    "{s}/{s}_{s}_texfunc_{s}_{s}",
                    .{
                        output_root,
                        test_type,
                        @tagName(mesh_type),
                        coord_name,
                        @tagName(builtin),
                    },
                );
                try renderCase(
                    allocator,
                    io,
                    .{
                        .mesh_type = mesh_type,
                        .coords = prepared.coords,
                        .connect = prepared.connect,
                        .disp = null,
                        .shader = .{
                            .func = .{
                                .uvs = uvs,
                                .coord_mode = if (coord_mode == .uv) .uv else .para,
                                .builtin = builtin,
                                .normal_type = normal_type,
                            },
                        },
                    },
                    prepared.camera_input,
                    scalar_dir,
                    config,
                );

                const rgb_dir = try std.fmt.allocPrint(
                    allocator,
                    "{s}/{s}_{s}_texfunc_rgb_{s}_{s}",
                    .{
                        output_root,
                        test_type,
                        @tagName(mesh_type),
                        coord_name,
                        @tagName(builtin),
                    },
                );
                try renderCase(
                    allocator,
                    io,
                    .{
                        .mesh_type = mesh_type,
                        .coords = prepared.coords,
                        .connect = prepared.connect,
                        .disp = null,
                        .shader = .{
                            .func_rgb = .{
                                .uvs = uvs,
                                .coord_mode = if (coord_mode == .uv) .uv else .para,
                                .builtin = builtin,
                                .normal_type = normal_type,
                            },
                        },
                    },
                    prepared.camera_input,
                    rgb_dir,
                    config,
                );
            }
        }
    }
}
