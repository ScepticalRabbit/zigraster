// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const std = @import("std");

pub const projected_tex = @import("tests/test_gold_projected_tex.zig");

pub fn main(init: std.process.Init) !void {
    _ = init;
    std.debug.print("Running ProjectedTex Gold Test Suite...\n", .{});
}

test {
    std.testing.refAllDecls(@This());
}
