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
const silhouette = @import("verif_2_silhouette.zig");

pub fn main(init: std.process.Init) !void {
    if (buildconfig.F != f64 or buildconfig.config.simd != .on) {
        return error.VerifGoldRequiresF64Simd;
    }
    try silhouette.generateFocusedGoldInputs(init.gpa, init.io);
}
