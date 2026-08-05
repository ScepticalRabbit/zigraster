// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const std = @import("std");
const gen = @import("gen_gold_texfunc.zig");

pub fn main(init: std.process.Init) !void {
    try gen.mainWithOutputRoot(init, "out-texfunc");
}
