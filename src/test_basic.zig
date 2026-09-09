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

pub fn main(init: std.process.Init) !void {
    _ = init;
    std.debug.print("Running BASIC Gold Test Suite...\n", .{});
    std.debug.print(
        "Please use 'zig build test-basic' to run all basic tests.\n",
        .{},
    );
}

test {
    std.testing.refAllDecls(@This());
}
