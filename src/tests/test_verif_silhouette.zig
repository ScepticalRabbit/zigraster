// --------------------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const silhouette_verif = @import("../verif_2_silhouette.zig");

test "verification silhouettes match analytic area and centroid" {
    try silhouette_verif.runFocusedTests(std.testing.allocator, std.testing.io);
}
