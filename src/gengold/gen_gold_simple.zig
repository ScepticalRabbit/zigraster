// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const std = @import("std");
const gengold = @import("../dev_support/gengold.zig");
const policy = @import("../dev_support/testpolicy.zig");
const tcfg = @import("../dev_support/testconfig.zig");
const riley = @import("../riley/zig/riley.zig");
const mo = @import("../riley/zig/meshpipeline.zig");
const gk = @import("../riley/zig/geometrykernels.zig");
const iio = @import("../riley/zig/imageio.zig");
const texops = @import("../riley/zig/textureops.zig");

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

    const mesh_types = [_]gk.MeshType{
        .tri3,
        .tri6,
        .quad4ibi,
        .quad4newton,
        .quad8,
        .quad9,
    };

    const samp_cfgs = [_]texops.TextureSampleConfig{
        .{ .sample = .nearest, .mode = .direct },
        .{ .sample = .linear, .mode = .direct },
        .{ .sample = .cubic_catmull_rom, .mode = .lut_lerp },
        .{ .sample = .cubic_mitchell_netravali, .mode = .lut_lerp },
        .{ .sample = .lanczos3, .mode = .lut_lerp },
        .{ .sample = .cubic_bspline, .mode = .lut_lerp },
        .{ .sample = .quintic_bspline, .mode = .lut_lerp },
    };
    const pixel_num = [_]u32{ 640, 400 };
    var config = tcfg.getRasterConfig(.gold);
    config.save_strategy = .disk;
    config.image_save_opts = &[_]iio.ImageSaveOpts{
        .{ .format = .fimg, .bits = null, .scaling = .none },
        .{ .format = .bmp, .bits = 8, .scaling = .auto },
    };

    std.debug.print("Generating Simple Gold Data (Two Elements only)...\n", .{});
    try gengold.runGenerationExt(
        aa,
        io,
        "twoelems",
        &mesh_types,
        1.1,
        texture,
        pixel_num,
        &samp_cfgs,
        policy.goldRoot(.simple),
        "data/simple",
        config,
    );

    std.debug.print("Done.\n", .{});
}
