// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const std = @import("std");
const gen_projected_tex = @import("gengold/gen_gold_projected_tex.zig");

pub fn main(init: std.process.Init) !void {
    try gen_projected_tex.main(init);
}
