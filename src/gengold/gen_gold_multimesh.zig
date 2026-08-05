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
const tcfg = @import("../dev_support/testconfig.zig");
const riley = @import("../riley/zig/riley.zig");
const iio = @import("../riley/zig/imageio.zig");

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const aa = arena.allocator();

    var config = tcfg.getRasterConfig(.gold);
    config.save_strategy = .disk;
    config.image_save_opts = &[_]iio.ImageSaveOpts{
        .{ .format = .fimg, .bits = null, .scaling = .none },
        .{ .format = .bmp, .bits = 8, .scaling = .auto },
    };

    std.debug.print("Generating Multimesh Gold Data...\n", .{});

    try gengold.runMultimeshGeneration(aa, io, config);
    try gengold.runMultimeshMixedGeneration(aa, io, config);
    try gengold.runMultimeshMixedRGBGeneration(aa, io, config);

    std.debug.print("Done.\n", .{});
}
