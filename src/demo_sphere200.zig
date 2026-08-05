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
const riley = @import("riley/zig/riley.zig");
const RasterConfig = riley.RasterConfig;
const meshio = @import("riley/zig/meshio.zig");
const uvio = @import("riley/zig/uvio.zig");
const iio = @import("riley/zig/imageio.zig");
const mo = @import("riley/zig/meshpipeline.zig");
const MeshInput = mo.MeshInput;
const gk = @import("riley/zig/geometrykernels.zig");
const MeshType = gk.MeshType;
const camera_mod = @import("riley/zig/camera.zig");
const cameraops = @import("riley/zig/cameraops.zig");
const sceneops = @import("riley/zig/sceneops.zig");
const CameraInput = camera_mod.CameraInput;
const Rotation = @import("riley/zig/rotation.zig").Rotation;
const CameraPrepared = camera_mod.CameraPrepared;
const MatSlice = @import("riley/zig/matslice.zig").MatSlice;
const F = buildconfig.F;

pub fn main(init: std.process.Init) !void {
    const outer_alloc = init.gpa;

    var arena = std.heap.ArenaAllocator.init(outer_alloc);
    defer arena.deinit();
    const aa = arena.allocator();

    // 1. Setup Rasteriser Configuration
    const config = RasterConfig{
        .save_strategy = .disk,
        .total_threads = 4,
        .max_raster_workers_per_job = 4,
        .image_save_opts = &[_]iio.ImageSaveOpts{
            .{ .format = .bmp, .bits = 8, .scaling = .auto },
        },
        .report = .bench,
    };
    var threaded_io = riley.getThreadedIo(
        aa,
        init.minimal,
        config.total_threads,
    );
    defer threaded_io.deinit();
    const io = threaded_io.io();

    const data_dir = "data/min/tri6_sphere200/";
    const out_dir_root = "./out/demo-sphere200";
    const pixel_num = [_]u32{ 800, 500 };

    // 2. Load Simulation Data
    std.debug.print("Loading sphere simulation data from {s}...\n", .{data_dir});
    const coord_path = data_dir ++ "coords.csv";
    const conn_path = data_dir ++ "connect.csv";
    // For this demo, we don't need the field data as we are using texture shading
    const sim_data = try meshio.loadSimData(aa, io, coord_path, conn_path, null, null);

    // 3. Load UV map for the texture
    std.debug.print("Loading UV map...\n", .{});
    const uv_path = data_dir ++ "uvs.csv";
    const uvs = try uvio.loadUVMap(aa, io, uv_path);

    // 4. Load Texture for shading
    std.debug.print("Loading speckle texture...\n", .{});
    const texture = try iio.loadImage(
        u8,
        1,
        aa,
        io,
        "texture/speckle.bmp",
        .bmp,
    );

    // 5. Prepare Mesh Input
    std.debug.print("Preparing mesh input...\n", .{});
    const mesh_input = MeshInput{
        .mesh_type = .tri6,
        .coords = sim_data.coords,
        .connect = sim_data.connect,
        .disp = null,
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

    // 6. Setup Camera
    // Position camera to frame the sphere
    std.debug.print("Setting up camera...\n", .{});
    const pixel_size = [_]F{
        @floatCast(5.3e-6),
        @floatCast(5.3e-6),
    };
    const focal_leng: F = @floatCast(50.0e-3);
    const rot = Rotation.init(0, 0, 0);
    const fov_scale_factor: F = 1.0;

    const roi_pos = sceneops.boundsCenter(&sim_data.coords);
    const cam_pos = cameraops.posFillFrameFromRot(
        &sim_data.coords,
        pixel_num,
        pixel_size,
        focal_leng,
        rot,
        fov_scale_factor,
    );
    const camera = try CameraPrepared.init(
        aa,
        .{
            .pixels_num = pixel_num,
            .pixels_size = pixel_size,
            .pos_world = cam_pos,
            .rot_world = rot,
            .roi_cent_world = roi_pos,
            .focal_length = focal_leng,
            .sub_sample = 2,
        },
    );
    defer camera.deinit(aa);

    // 7. Run the Rasteriser
    std.debug.print("Rendering sphere to {s}/...\n", .{out_dir_root});
    const meshes = [_]MeshInput{mesh_input};
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
        &meshes,
        config,
        out_dir_root,
    );

    if (images) |img| {
        aa.free(img.slice);
        img.deinit(aa);
    }

    std.debug.print("Demo complete. Images saved to {s}/\n", .{out_dir_root});
}
