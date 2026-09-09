// --------------------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const buildconfig = @import("riley/zig/buildconfig.zig");

pub const solver = @import("tests/test_verif_solver.zig");
pub const silhouette = @import("tests/test_verif_silhouette.zig");
pub const depth = @import("tests/test_verif_depth.zig");
pub const distortion = @import("tests/test_verif_distortion.zig");

comptime {
    if (buildconfig.F != f64 or buildconfig.config.simd != .on) {
        @compileError("test_verif.zig requires precision=f64 and simd=on");
    }
}

fn runCase(
    comptime name: []const u8,
    allocator: std.mem.Allocator,
    io: std.Io,
    comptime run: fn (std.mem.Allocator, std.Io) anyerror!void,
) !void {
    std.debug.print("Running verification case: {s}...\n", .{name});
    const start = std.Io.Clock.Timestamp.now(io, .awake);
    try run(allocator, io);
    const end = std.Io.Clock.Timestamp.now(io, .awake);
    const elapsed_s = @as(f64, @floatFromInt(start.durationTo(end).raw.nanoseconds)) / 1.0e9;
    std.debug.print("Verification case {s} took {d:.3} seconds.\n", .{ name, elapsed_s });
}

test "focused analytic verification suite" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const start = std.Io.Clock.Timestamp.now(io, .awake);

    std.debug.print("\nRunning verification tests.\n\n", .{});
    try runCase("solver", allocator, io, solver.run);
    try runCase("silhouette", allocator, io, silhouette.run);
    try runCase("depth buffer", allocator, io, depth.run);
    try runCase("camera distortion", allocator, io, distortion.run);

    const end = std.Io.Clock.Timestamp.now(io, .awake);
    const elapsed_s = @as(f64, @floatFromInt(start.durationTo(end).raw.nanoseconds)) / 1.0e9;
    std.debug.print("\nVerification tests took {d:.3} seconds.\n", .{elapsed_s});
}
