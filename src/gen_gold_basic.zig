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
const gen_zoo = @import("gengold/gen_gold_featurezoo.zig");
const gen_oneelem = @import("gengold/gen_gold_oneelem.zig");
const gen_twoshapes = @import("gengold/gen_gold_twoshapes.zig");
const iio = @import("riley/zig/imageio.zig");
const policy = @import("dev_support/testpolicy.zig");
const tcfg = @import("dev_support/testconfig.zig");

pub fn main(init: std.process.Init) !void {
    @setEvalBranchQuota(buildconfig.comptime_eval_branch_quota);
    const io = init.io;
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const aa = arena.allocator();

    const texture_grey = try iio.loadImage(
        u8,
        1,
        aa,
        io,
        "texture/speckle_mono.tiff",
        .tiff,
    );
    const texture_rgb = try iio.loadImage(
        u8,
        3,
        aa,
        io,
        "texture/speckle_rgb.bmp",
        .bmp,
    );

    var config = tcfg.getRasterConfig(.gold_gen);
    config.save_strategy = .disk;
    config.image_save_opts = &[_]iio.ImageSaveOpts{
        .{ .format = .fimg, .bits = null, .scaling = .none },
        .{ .format = .tiff, .bits = 8, .scaling = .auto },
    };

    const gold_dir = policy.goldRoot(.basic);

    std.debug.print("Generating BASIC Gold Suite to {s}...\n\n", .{gold_dir});

    std.debug.print("--- 1/3: OneElem Cases ---\n", .{});
    try gen_oneelem.generateAllOneElemCases(
        aa,
        io,
        gold_dir,
        "data/edge",
        config,
    );

    std.debug.print("\n--- 2/3: TwoShapes Cases ---\n", .{});
    try gen_twoshapes.generateAllTwoShapesCases(
        aa,
        io,
        texture_grey,
        texture_rgb,
        gold_dir,
        config,
    );

    std.debug.print("\n--- 3/3: FeatureZoo Cases ---\n", .{});
    try gen_zoo.generateAllFeatureZooCases(
        aa,
        io,
        texture_grey,
        texture_rgb,
        gold_dir,
        config,
    );

    std.debug.print("\nBASIC Gold Suite Generation Complete.\n", .{});
}
