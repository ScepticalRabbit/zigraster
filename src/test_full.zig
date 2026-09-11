// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const std = @import("std");
const testsuites = @import("dev_support/testsuites.zig");

const input_verif_suite = @import("tests/test_full_input_verif.zig");
const shader_suite = @import("tests/test_full_shader.zig");
const texture_suite = @import("tests/test_full_texture.zig");
const dist_psf_suite = @import("tests/test_full_dist_psf.zig");
const ssaa_pxmap_suite = @import("tests/test_full_ssaa_pxmap.zig");
const hull_suite = @import("tests/test_full_hull.zig");
const tiling_suite = @import("tests/test_full_tiling.zig");
const scene_camera_threads_suite = @import(
    "tests/test_full_scene_camera_threads.zig",
);
const image_output_suite = @import("tests/test_full_image_output.zig");

test "full test suite" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const start = std.Io.Clock.Timestamp.now(io, .awake);

    std.debug.print("\nRunning full test suite.\n\n", .{});
    try testsuites.runSuite("input_verif", allocator, io, input_verif_suite.run);
    try testsuites.runSuite("shader", allocator, io, shader_suite.run);
    try testsuites.runSuite("texture", allocator, io, texture_suite.run);
    try testsuites.runSuite("dist_psf", allocator, io, dist_psf_suite.run);
    try testsuites.runSuite("ssaa_pxmap", allocator, io, ssaa_pxmap_suite.run);
    try testsuites.runSuite("hull", allocator, io, hull_suite.run);
    try testsuites.runSuite("tiling", allocator, io, tiling_suite.run);
    try testsuites.runSuite(
        "scene_camera_threads",
        allocator,
        io,
        scene_camera_threads_suite.run,
    );
    try testsuites.runSuite(
        "image_output",
        allocator,
        io,
        image_output_suite.run,
    );

    const end = std.Io.Clock.Timestamp.now(io, .awake);
    const elapsed_s = testsuites.durationToSeconds(start, end);
    std.debug.print(
        "\nFull test suite complete. Took {d:.3} seconds.\n\n",
        .{elapsed_s},
    );
}
