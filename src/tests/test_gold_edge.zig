// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const std = @import("std");
const common = @import("../dev_support/tests.zig");
const policy = @import("../dev_support/testpolicy.zig");
const orch = @import("../dev_support/orchestration.zig");
const tcfg = @import("../dev_support/testconfig.zig");
const buildconfig = @import("../riley/zig/buildconfig.zig");
const F = buildconfig.F;
const gk = @import("../riley/zig/geometrykernels.zig");
const iio = @import("../riley/zig/imageio.zig");
const texops = @import("../riley/zig/textureops.zig");

const SHADER_FILTER: common.ShaderFilter = .both;

pub fn main() !void {
    std.debug.print(
        "Please use 'zig test -O ReleaseSafe src/test_gold_edge.zig' " ++
            "to run this test suite.\n",
        .{},
    );
}

test "Gold Edge Suite" {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const io = std.testing.io;

    const texture = blk: {
        break :blk try iio.loadImage(
            u8,
            1,
            allocator,
            io,
            "texture/speckle-simple.tiff",
            .tiff,
        );
    };
    defer texture.deinit(allocator);

    const mesh_types = [_]gk.MeshType{ .tri6, .quad8, .quad9 };
    const distort_mesh_types = [_]gk.MeshType{
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
    const pixel_num = [_]u32{ 320, 200 };
    const pixel_num_distort_midside = [_]u32{ 800, 500 };

    const start_time = std.Io.Clock.Timestamp.now(io, .awake);

    const simd_on = buildconfig.config.simd == .on;
    std.debug.print("Running Gold Edge Tests with .simd = .{s}...\n", .{
        if (simd_on) "on" else "off",
    });

    try common.runMeshTypesSuite(
        allocator,
        io,
        &[_][]const u8{ "bulgein_rot", "bulgeout_rot", "vertbulge" },
        &mesh_types,
        1.1,
        texture,
        pixel_num,
        &samp_cfgs,
        policy.goldRoot(.edge),
        "data/edge",
        tcfg.REL_TOL,
        tcfg.ABS_TOL,
        SHADER_FILTER,
        false,
    );

    for (distort_mesh_types) |mt| {
        try common.runDistortEdgeTexFuncTest(
            allocator,
            io,
            mt,
            policy.goldRoot(.edge),
            "data/edge",
            pixel_num_distort_midside,
        );
    }

    const end_time = std.Io.Clock.Timestamp.now(io, .awake);
    const duration_ms = @as(
        F,
        @floatFromInt(start_time.durationTo(end_time).raw.nanoseconds),
    ) / 1e6;
    std.debug.print("Gold Edge Test Suite took {d:.3} ms\n", .{duration_ms});
}
