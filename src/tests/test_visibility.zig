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
const minsuite = @import("../dev_support/minsuite.zig");
const tcfg = @import("../dev_support/testconfig.zig");
const iio = @import("../riley/zig/imageio.zig");

test "MIN multi-cull render is unchanged by a halo-only bypass" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const texture_grey = try iio.loadImage(
        u8,
        1,
        allocator,
        io,
        "texture/speckle.bmp",
        .bmp,
    );
    defer texture_grey.deinit(allocator);
    const texture_rgb = try iio.loadImage(
        u8,
        3,
        allocator,
        io,
        "texture/speckle_rgb.bmp",
        .bmp,
    );
    defer texture_rgb.deinit(allocator);

    var config_base = tcfg.getRasterConfig(.bench);
    config_base.save_strategy = .memory;
    config_base.image_save_opts = &[_]iio.ImageSaveOpts{};
    var config_halo = config_base;
    config_halo.raster_halo_px_override = 3;

    var result_base = try minsuite.runSphere200MultiCullQuiet(
        allocator,
        io,
        .tri3,
        .nodal_grey,
        .{ .sample = .nearest, .mode = .direct },
        "data/min/tri3_sphere200",
        .{ 160, 100 },
        texture_grey,
        texture_rgb,
        config_base,
        "",
        0.75,
    );
    defer result_base.deinit(allocator);
    var result_halo = try minsuite.runSphere200MultiCullQuiet(
        allocator,
        io,
        .tri3,
        .nodal_grey,
        .{ .sample = .nearest, .mode = .direct },
        "data/min/tri3_sphere200",
        .{ 160, 100 },
        texture_grey,
        texture_rgb,
        config_halo,
        "",
        0.75,
    );
    defer result_halo.deinit(allocator);

    const image_base = result_base.image orelse return error.NoResult;
    const image_halo = result_halo.image orelse return error.NoResult;
    try std.testing.expectEqualSlices(usize, image_base.dims, image_halo.dims);
    try std.testing.expectEqualSlices(F, image_base.slice, image_halo.slice);
}
