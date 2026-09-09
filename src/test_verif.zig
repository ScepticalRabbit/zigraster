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

pub const solver = @import("tests/test_verif_solver.zig");
pub const silhouette = @import("tests/test_verif_silhouette.zig");
pub const depth = @import("tests/test_verif_depth.zig");
pub const distortion = @import("tests/test_verif_distortion.zig");

comptime {
    if (buildconfig.F != f64 or buildconfig.config.simd != .on) {
        @compileError("test_verif.zig requires precision=f64 and simd=on");
    }
}

test {
    std.testing.refAllDecls(@This());
}
