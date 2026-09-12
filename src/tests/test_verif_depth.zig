// --------------------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const depth_verif = @import("../verif_4_depth.zig");

pub fn run(allocator: std.mem.Allocator, io: std.Io) !void {
    try depth_verif.runFocusedTests(allocator, io);
}
