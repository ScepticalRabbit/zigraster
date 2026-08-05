// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const std = @import("std");

// NOTE: pub const needed here for refAllDecls to work!
pub const small = @import("tests/test_gold_small.zig");
pub const simple = @import("tests/test_gold_simple.zig");
pub const edge = @import("tests/test_gold_edge.zig");
pub const multimesh = @import("tests/test_gold_multimesh.zig");
pub const multicamera = @import("tests/test_gold_multicamera.zig");
pub const hull = @import("tests/test_hull.zig");
pub const nodal_normals = @import("tests/test_nodal_normals.zig");
pub const texfunc = @import("tests/test_texfunc.zig");
pub const texfloat = @import("tests/test_texfloat.zig");
pub const ssaa = @import("tests/test_gold_ssaa.zig");
pub const psf = @import("tests/test_gold_psf.zig");
pub const visibility = @import("tests/test_visibility.zig");

pub fn main(init: std.process.Init) !void {
    _ = init;
    std.debug.print("Running ALL Gold Test Suites...\n", .{});

    std.debug.print(
        "Please use 'zig test -O ReleaseSafe src/test_gold_all.zig' " ++
            "to run all tests.\n",
        .{},
    );
}

test {
    std.testing.refAllDecls(@This());
}
