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
const common = @import("../dev_support/tests.zig");
const tcfg = @import("../dev_support/testconfig.zig");

test "Gold Multimesh Suite" {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const io = std.testing.io;

    const start_time = std.Io.Clock.Timestamp.now(io, .awake);

    const simd_on = @import("../riley/zig/buildconfig.zig").config.simd == .on;
    std.debug.print("Running Gold Multimesh Tests with .simd = .{s}...\n", .{
        if (simd_on) "on" else "off",
    });

    try common.runMultimeshTest(allocator, io, tcfg.REL_TOL, tcfg.ABS_TOL);
    try common.runMultimeshMixedTest(allocator, io, tcfg.REL_TOL, tcfg.ABS_TOL);
    try common.runMultimeshMixedRGBTest(allocator, io, tcfg.REL_TOL, tcfg.ABS_TOL);

    const end_time = std.Io.Clock.Timestamp.now(io, .awake);
    const duration_ms = @as(
        F,
        @floatFromInt(start_time.durationTo(end_time).raw.nanoseconds),
    ) / 1e6;
    std.debug.print("Multi-Mesh Test Suite took {d:.3} ms\n", .{duration_ms});
}
