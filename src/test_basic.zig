// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const std = @import("std");

pub const oneelem = @import("tests/test_gold_oneelem.zig");
pub const twoshapes = @import("tests/test_gold_twoshapes.zig");
pub const featurezoo = @import("tests/test_gold_featurezoo.zig");

fn runSuite(
    comptime name: []const u8,
    allocator: std.mem.Allocator,
    io: std.Io,
    comptime run: fn (std.mem.Allocator, std.Io) anyerror!void,
) !void {
    std.debug.print("Running {s} suite...\n", .{name});
    const start = std.Io.Clock.Timestamp.now(io, .awake);
    try run(allocator, io);
    const end = std.Io.Clock.Timestamp.now(io, .awake);
    const elapsed_s = @as(
        f64,
        @floatFromInt(start.durationTo(end).raw.nanoseconds),
    ) / 1.0e9;
    std.debug.print("{s} suite took {d:.3} seconds.\n", .{ name, elapsed_s });
}

test "basic test suite" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const start = std.Io.Clock.Timestamp.now(io, .awake);

    std.debug.print("\nRunning basic test suite.\n\n", .{});
    try runSuite("oneelem", allocator, io, oneelem.run);
    try runSuite("twoshapes", allocator, io, twoshapes.run);
    try runSuite("featurezoo", allocator, io, featurezoo.run);

    const end = std.Io.Clock.Timestamp.now(io, .awake);
    const elapsed_s = @as(
        f64,
        @floatFromInt(start.durationTo(end).raw.nanoseconds),
    ) / 1.0e9;
    std.debug.print(
        "\nBasic test suite complete. Took {d:.3} seconds.\n\n",
        .{elapsed_s},
    );
}
