// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const std = @import("std");
const suite = @import("../dev_support/psfsuite.zig");

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    std.debug.print("Generating PSF cases to {s}/...\n", .{suite.gold_root});
    try suite.saveAllGoldCases(init.gpa, io);
    std.debug.print("Done.\n", .{});
}
