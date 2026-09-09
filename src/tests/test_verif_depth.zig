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

test "verification depth buffer preserves the front overlapping rabbit" {
    try depth_verif.runFocusedTests(std.testing.allocator, std.testing.io);
}
