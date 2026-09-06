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
const meshio = @import("riley/zig/meshio.zig");
const uvio = @import("riley/zig/uvio.zig");
const iio = @import("riley/zig/imageio.zig");
const mo = @import("riley/zig/meshpipeline.zig");
const camera = @import("riley/zig/camera.zig");
const cameraops = @import("riley/zig/cameraops.zig");
const sceneops = @import("riley/zig/sceneops.zig");
const Rotation = @import("riley/zig/rotation.zig").Rotation;

const F = buildconfig.F;
const raster_threads: u16 = 8;

pub fn main(init: std.process.Init) !void {
    const outer_alloc = init.gpa;
    var arena = std.heap.ArenaAllocator.init(outer_alloc);
    defer arena.deinit();
    const aa = arena.allocator();

    const config_base = riley.RasterConfig{
        .save_strategy = .disk,
        .total_threads = raster_threads,
        .max_raster_workers_per_job = raster_threads,
        .image_save_opts = &[_]iio.ImageSaveOpts{
            .{ .format = .bmp, .bits = 8, .scaling = .auto },
        },
        .report = .bench,
    };
    var threaded_io = riley.getThreadedIo(
        aa,
        init.minimal,
        config_base.total_threads,
    );
    defer threaded_io.deinit();
    const io = threaded_io.io();

    const data_dir = "data/min/tri6_sphere200/";
    const out_dir_root = "./out/demo2_psf";
    const pixel_num = [_]u32{ 800, 500 };

    std.debug.print(
        "Loading sphere simulation data from {s} with {d} raster threads...\n",
        .{ data_dir, raster_threads },
    );
    const sim_data = try meshio.loadSimData(
        aa,
        io,
        data_dir ++ "coords.csv",
        data_dir ++ "connect.csv",
        null,
        null,
    );
    const uvs = try uvio.loadUVMap(aa, io, data_dir ++ "uvs.csv");
    const texture = try iio.loadImage(
        u8,
        1,
        aa,
        io,
        "texture/speckle.bmp",
        .bmp,
    );
    const mesh = mo.MeshInput{
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

    const pixel_size = [_]F{ @floatCast(5.3e-6), @floatCast(5.3e-6) };
    const focal_length: F = @floatCast(50.0e-3);
    const rotation = Rotation.init(0, 0, 0);
    const roi_cent_world = sceneops.boundsCenter(&sim_data.coords);
    const pos_world = cameraops.posFillFrameFromRot(
        &sim_data.coords,
        pixel_num,
        pixel_size,
        focal_length,
        rotation,
        1.0,
    );
    const camera_input = camera.CameraInput{
        .pixels_num = pixel_num,
        .pixels_size = pixel_size,
        .pos_world = pos_world,
        .rot_world = rotation,
        .roi_cent_world = roi_cent_world,
        .focal_length = focal_length,
        .sub_sample = 2,
        .psf = .{ .gaussian = .{
            .sigma_px = 1.0,
            .supp_rad_px = 3.0,
            .separable = .yes,
        } },
    };
    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = config_base.total_threads },
    };
    const modes = [_]riley.BufferMode{
        .global_subpx_full,
        .global_subpx_stripe,
    };

    for (modes) |mode| {
        var config = config_base;
        config.buffer_mode = mode;
        const out_dir = try std.fs.path.join(
            aa,
            &[_][]const u8{ out_dir_root, @tagName(mode) },
        );
        std.debug.print("Rendering PSF sphere with {s}...\n", .{@tagName(mode)});
        if (try riley.raster(
            aa,
            &render_groups,
            &[_]camera.CameraInput{camera_input},
            &[_]mo.MeshInput{mesh},
            config,
            out_dir,
        )) |image| {
            aa.free(image.slice);
            var image_mut = image;
            image_mut.deinit(aa);
        }
    }

    std.debug.print("Demo complete. Images saved to {s}/\n", .{out_dir_root});
}
