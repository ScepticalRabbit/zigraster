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
const riley = @import("riley/zig/riley.zig");
const rastcfg = @import("riley/zig/rasterconfig.zig");
const meshio = @import("riley/zig/meshio.zig");
const uvio = @import("riley/zig/uvio.zig");
const mo = @import("riley/zig/meshpipeline.zig");
const gk = @import("riley/zig/geometrykernels.zig");
const iio = @import("riley/zig/imageio.zig");
const camera_mod = @import("riley/zig/camera.zig");
const cameraops = @import("riley/zig/cameraops.zig");
const Rotation = @import("riley/zig/rotation.zig").Rotation;
const sceneops = @import("riley/zig/sceneops.zig");

const MeshInput = mo.MeshInput;
const CameraPrepared = camera_mod.CameraPrepared;

const rabbit_mesh_types = [_]gk.MeshType{
    .tri3,
    .tri6,
    .quad4ibi,
    .quad8,
    .quad9,
};

const rabbit_dir_paths = [_][]const u8{
    "data/rabbits/rabbit_tri3/",
    "data/rabbits/rabbit_tri6/",
    "data/rabbits/rabbit_quad4/",
    "data/rabbits/rabbit_quad8/",
    "data/rabbits/rabbit_quad9/",
};

fn buildUvGreyField(
    allocator: std.mem.Allocator,
    uvs: uvio.UVMap,
) !meshio.Field {
    const node_num = uvs.array.dims[0];
    var field = try meshio.Field.initAlloc(allocator, 1, node_num, 1);

    // Collapse UVs to a scalar grey field so we can render a BMP while still
    // deriving the nodal data directly from the rabbit UV map.
    for (0..node_num) |nn| {
        const u = uvs.array.get(&[_]usize{ nn, 0 });
        const v = uvs.array.get(&[_]usize{ nn, 1 });
        field.array.set(&[_]usize{ 0, nn, 0 }, 0.5 * (u + v));
    }

    return field;
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const aa = arena.allocator();

    const out_dir_root = "out/rabbits";
    const pixel_num = [_]u32{ 3000, 1500 };
    const fov_scale: f64 = 1.01;

    std.debug.print("Loading rabbit meshes...\n", .{});
    const sim_datas = try meshio.loadMultiSimData(
        aa,
        io,
        &rabbit_dir_paths,
        .{
            .field_files = null,
            .disp_files = null,
        },
    );

    std.debug.print("Loading rabbit UV maps...\n", .{});
    var uv_maps = try aa.alloc(uvio.UVMap, rabbit_dir_paths.len);
    for (rabbit_dir_paths, 0..) |dir_path, ii| {
        const uv_path = try std.fmt.allocPrint(aa, "{s}uvs.csv", .{dir_path});
        uv_maps[ii] = try uvio.loadUVMap(aa, io, uv_path);
    }

    std.debug.print("Loading texture...\n", .{});
    const texture = try iio.loadImage(
        u8,
        1,
        aa,
        io,
        "texture/speckle-simple.tiff",
        .tiff,
    );

    std.debug.print("Preparing rabbit mesh inputs...\n", .{});
    var mesh_inputs = try aa.alloc(MeshInput, rabbit_mesh_types.len * 3);

    // Bottom row: texfunc
    for (0..rabbit_mesh_types.len) |ii| {
        mesh_inputs[ii] = .{
            .mesh_type = rabbit_mesh_types[ii],
            .coords = try orch.copyCoords(aa, sim_datas[ii].coords),
            .connect = sim_datas[ii].connect,
            .disp = null,
            .shader = .{ .func = .{
                .uvs = uv_maps[ii].array,
                .coord_mode = .uv,
                .builtin = .sinusoidal,
                .bits = 8,
                .scaling = .auto,
                .normal_type = .none,
            } },
        };
    }

    // Middle row: nodal interpolation using a scalar field derived from UVs.
    for (0..rabbit_mesh_types.len) |ii| {
        mesh_inputs[ii + rabbit_mesh_types.len] = .{
            .mesh_type = rabbit_mesh_types[ii],
            .coords = try orch.copyCoords(aa, sim_datas[ii].coords),
            .connect = sim_datas[ii].connect,
            .disp = null,
            .shader = .{ .nodal = .{
                .field = try buildUvGreyField(aa, uv_maps[ii]),
                .bits = 8,
                .scaling = .auto,
                .scale_over = .over_frames,
                .normal_type = .none,
            } },
        };
    }

    // Top row: greyscale texture shader.
    for (0..rabbit_mesh_types.len) |ii| {
        mesh_inputs[ii + rabbit_mesh_types.len * 2] = .{
            .mesh_type = rabbit_mesh_types[ii],
            .coords = try orch.copyCoords(aa, sim_datas[ii].coords),
            .connect = sim_datas[ii].connect,
            .disp = null,
            .shader = .{ .tex_u8 = .{
                .uvs = uv_maps[ii].array,
                .tex = texture,
                .samp_cfg = .{
                    .sample = .cubic_catmull_rom,
                    .mode = .lut_lerp,
                },
                .bits = 8,
                .scaling = .none,
                .normal_type = .none,
            } },
        };
    }

    sceneops.arrangeMeshesGrid(mesh_inputs, .{
        .gap = .{ 0.12, 0.12, 0.0 },
        .max_divs = .{ 5, 3, 1 },
    });

    std.debug.print("Setting up camera...\n", .{});
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

    const config = rastcfg.RasterConfig{
        .save_strategy = .disk,
        .image_save_opts = &[_]iio.ImageSaveOpts{
            .{ .format = .bmp, .bits = 8, .scaling = .auto },
        },
    };

    std.debug.print("Rendering rabbits to {s}/...\n", .{out_dir_root});
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
        &[_]@TypeOf(camera_input){camera_input},
        mesh_inputs,
        config,
        out_dir_root,
    );

    if (images) |img| {
        aa.free(img.slice);
        img.deinit(aa);
    }

    std.debug.print("Done.\n", .{});
}
