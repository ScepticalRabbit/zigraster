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
const testsuites = @import("dev_support/testsuites.zig");

const solver = @import("tests/test_verif_solver.zig");
const silhouette = @import("tests/test_verif_silhouette.zig");
const depth = @import("tests/test_verif_depth.zig");
const distortion = @import("tests/test_verif_distortion.zig");

comptime {
    if (buildconfig.F != f64 or buildconfig.config.simd != .on) {
        @compileError("test_verif.zig requires precision=f64 and simd=on");
    }
}

test "focused analytic verification suite" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const start = std.Io.Clock.Timestamp.now(io, .awake);

    std.debug.print("\nRunning verification tests.\n\n", .{});
    try testsuites.runCase("solver", allocator, io, solver.run);
    try testsuites.runCase("silhouette", allocator, io, silhouette.run);
    try testsuites.runCase("depth buffer", allocator, io, depth.run);
    try testsuites.runCase("camera distortion", allocator, io, distortion.run);

    const end = std.Io.Clock.Timestamp.now(io, .awake);
    const elapsed_s = testsuites.durationToSeconds(start, end);
    std.debug.print("\nVerification tests took {d:.3} seconds.\n", .{elapsed_s});
}
