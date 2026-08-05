// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const std = @import("std");
const gengold = @import("dev_support/gengold.zig");
const tcfg = @import("dev_support/testconfig.zig");
const riley = @import("riley/zig/riley.zig");
const mo = @import("riley/zig/meshpipeline.zig");
const gk = @import("riley/zig/geometrykernels.zig");
const iio = @import("riley/zig/imageio.zig");
const texops = @import("riley/zig/textureops.zig");

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var arena = std.heap.ArenaAllocator.init(init.gpa);
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

    const mesh_types = [_]gk.MeshType{ .tri3, .tri6, .quad4ibi, .quad8, .quad9 };
    const samp_cfgs = [_]texops.TextureSampleConfig{
        .{ .sample = .linear, .mode = .direct },
        .{ .sample = .cubic_catmull_rom, .mode = .direct },
        .{ .sample = .cubic_catmull_rom, .mode = .lut },
        .{ .sample = .cubic_catmull_rom, .mode = .lut_lerp },
        .{ .sample = .quintic_bspline, .mode = .direct },
        .{ .sample = .quintic_bspline, .mode = .lut },
        .{ .sample = .quintic_bspline, .mode = .lut_lerp },
    };

    const pixel_num = [_]u32{ 160, 100 };

    const out_dir_root = "out/small";
    const data_dir = "data/small";

    var config = tcfg.getRasterConfig(.preview);
    config.save_strategy = .disk;
    config.image_save_opts = &[_]iio.ImageSaveOpts{
        .{ .format = .bmp, .bits = 8, .scaling = .auto },
        .{ .format = .csv, .bits = null, .scaling = .none },
    };
    config.report = .full_stats;
    config.full_stats_opts = .{
        .formats = &[_]iio.ImageSaveOpts{
            .{ .format = .bmp, .bits = 8, .scaling = .auto },
            .{ .format = .csv, .bits = null, .scaling = .none },
        },
        .save_iter_map = true,
        .save_tile_timing_map = true,
        .save_tile_density_map = true,
        .save_tile_occupancy_map = true,
        .save_depth_map = true,
        .save_earlyout_map = true,
        .save_pixel_occupancy_map = true,
    };

    std.debug.print("Rendering Small Data to {s}/...\n", .{out_dir_root});
    try gengold.runGenerationExt(
        aa,
        io,
        "single",
        &mesh_types,
        1.1,
        texture,
        pixel_num,
        &samp_cfgs,
        out_dir_root,
        data_dir,
        config,
    );

    std.debug.print("Done.\n", .{});
}
