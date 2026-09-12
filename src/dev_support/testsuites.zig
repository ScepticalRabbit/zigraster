// --------------------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");

pub fn durationToSeconds(start: std.Io.Clock.Timestamp, end: std.Io.Clock.Timestamp) f64 {
    const elapsed_ns = start.durationTo(end).raw.nanoseconds;
    return @as(f64, @floatFromInt(elapsed_ns)) / 1.0e9;
}

pub fn runSuite(
    comptime name: []const u8,
    allocator: std.mem.Allocator,
    io: std.Io,
    comptime run: fn (std.mem.Allocator, std.Io) anyerror!void,
) !void {
    std.debug.print("Running {s} suite...\n", .{name});
    const start = std.Io.Clock.Timestamp.now(io, .awake);
    try run(allocator, io);
    const end = std.Io.Clock.Timestamp.now(io, .awake);
    const elapsed_s = durationToSeconds(start, end);
    std.debug.print("{s} suite took {d:.3} seconds.\n", .{ name, elapsed_s });
}

pub fn runCase(
    comptime name: []const u8,
    allocator: std.mem.Allocator,
    io: std.Io,
    comptime run: fn (std.mem.Allocator, std.Io) anyerror!void,
) !void {
    std.debug.print("Running verification case: {s}...\n", .{name});
    const start = std.Io.Clock.Timestamp.now(io, .awake);
    try run(allocator, io);
    const end = std.Io.Clock.Timestamp.now(io, .awake);
    const elapsed_s = durationToSeconds(start, end);
    std.debug.print("Verification case {s} took {d:.3} seconds.\n", .{ name, elapsed_s });
}
