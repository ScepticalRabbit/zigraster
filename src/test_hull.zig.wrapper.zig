pub const build_options = struct {
    pub const precision = "f64";
    pub const simd = "on";
};
// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const std = @import("std");
const common = @import("dev_support/tests.zig");
const policy = @import("dev_support/testpolicy.zig");
const buildconfig = @import("riley/zig/buildconfig.zig");
const F = buildconfig.F;
const gk = @import("riley/zig/geometrykernels.zig");
const rastcfg = @import("riley/zig/rasterconfig.zig");

pub fn main() !void {
    std.debug.print(
        "Please use 'zig test -O ReleaseSafe src/test_hull.zig' " ++
            "to run this test suite.\n",
        .{},
    );
}

test "Gold Hull Suite" {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const io = std.testing.io;

    const distort_mesh_types = [_]gk.MeshType{
        .tri3,
        .tri6,
        .quad4ibi,
        .quad4newton,
        .quad8,
        .quad9,
    };
    const hull_modes = [_]rastcfg.HullMode{
        .off,
        .on_no_fallback,
        .on_convex_fallback,
    };
    const pixel_num = [_]u32{ 800, 500 };
    const midside_mesh_types = [_]gk.MeshType{ .tri6, .quad8, .quad9 };

    const start_time = std.Io.Clock.Timestamp.now(io, .awake);
    const simd_on = buildconfig.config.simd == .on;
    std.debug.print("Running Gold Hull Tests with .simd = .{s}...\n", .{
        if (simd_on) "on" else "off",
    });

    for (hull_modes) |hull_mode| {
        for (distort_mesh_types) |mesh_type| {
            try common.runDistortEdgeTexFuncTestForHullMode(
                allocator,
                io,
                mesh_type,
                policy.goldRoot(.hull),
                "data/edge",
                pixel_num,
                hull_mode,
            );
        }

        for (midside_mesh_types) |mesh_type| {
            try common.runEdgeTexFuncConstantCaseForHullMode(
                allocator,
                io,
                "vertbulge",
                mesh_type,
                policy.goldRoot(.hull),
                "data/edge",
                pixel_num,
                hull_mode,
            );
        }
    }

    const end_time = std.Io.Clock.Timestamp.now(io, .awake);
    const duration_ms = @as(
        F,
        @floatFromInt(start_time.durationTo(end_time).raw.nanoseconds),
    ) / 1e6;
    std.debug.print("Gold Hull Test Suite took {d:.3} ms\n", .{duration_ms});
}
