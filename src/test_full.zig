// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const std = @import("std");

pub const input_verif_suite = @import("tests/test_full_input_verif.zig");
pub const shader_suite = @import("tests/test_full_shader.zig");
pub const texture_suite = @import("tests/test_full_texture.zig");
pub const dist_psf_suite = @import("tests/test_full_dist_psf.zig");
pub const ssaa_pxmap_suite = @import("tests/test_full_ssaa_pxmap.zig");
pub const hull_suite = @import("tests/test_full_hull.zig");
pub const tiling_suite = @import("tests/test_full_tiling.zig");
pub const scene_camera_threads_suite = @import(
    "tests/test_full_scene_camera_threads.zig",
);
pub const image_output_suite = @import("tests/test_full_image_output.zig");

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

test "full test suite" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const start = std.Io.Clock.Timestamp.now(io, .awake);

    std.debug.print("\nRunning full test suite.\n\n", .{});
    try runSuite("input_verif", allocator, io, input_verif_suite.run);
    try runSuite("shader", allocator, io, shader_suite.run);
    try runSuite("texture", allocator, io, texture_suite.run);
    try runSuite("dist_psf", allocator, io, dist_psf_suite.run);
    try runSuite("ssaa_pxmap", allocator, io, ssaa_pxmap_suite.run);
    try runSuite("hull", allocator, io, hull_suite.run);
    try runSuite("tiling", allocator, io, tiling_suite.run);
    try runSuite(
        "scene_camera_threads",
        allocator,
        io,
        scene_camera_threads_suite.run,
    );
    try runSuite(
        "image_output",
        allocator,
        io,
        image_output_suite.run,
    );

    const end = std.Io.Clock.Timestamp.now(io, .awake);
    const elapsed_s = @as(
        f64,
        @floatFromInt(start.durationTo(end).raw.nanoseconds),
    ) / 1.0e9;
    std.debug.print(
        "\nFull test suite complete. Took {d:.3} seconds.\n\n",
        .{elapsed_s},
    );
}
