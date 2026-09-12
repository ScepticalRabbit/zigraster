// --------------------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const cam = @import("../riley/zig/camera.zig");
const tcfg = @import("../dev_support/testconfig.zig");
const vconst = @import("../dev_support/verifconstants.zig");
const dist_verif = @import("../verif_5_dist_roundtrip.zig");

pub fn run(allocator: std.mem.Allocator, io: std.Io) !void {
    _ = io;
    const camera_base = vconst.distort_cases[0].camera_input;
    const width = @as(f64, @floatFromInt(camera_base.pixels_num[0]));
    const height = @as(f64, @floatFromInt(camera_base.pixels_num[1]));
    const points = [_][2]f64{
        .{ 0.5 * width, 0.5 * height },
        .{ 0.05 * width, 0.05 * height },
        .{ 0.95 * width, 0.05 * height },
        .{ 0.95 * width, 0.95 * height },
        .{ 0.05 * width, 0.95 * height },
        .{ 0.23 * width, 0.67 * height },
        .{ 0.78 * width, 0.31 * height },
    };

    for (vconst.camera_distortion_cases) |distortion_case| {
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const local_alloc = arena.allocator();
        const camera_input = vconst.cameraInputWithDistortion(camera_base, distortion_case);
        const camera = try cam.CameraPrepared.init(local_alloc, camera_input);

        for (points, 0..) |point, point_idx| {
            const record = dist_verif.evalPixelSample(
                &camera,
                camera_input,
                .{
                    .ideal_x = point[0],
                    .ideal_y = point[1],
                    .row_idx = point_idx,
                    .col_idx = 0,
                },
            );
            try std.testing.expect(record.converged);
            try std.testing.expect(record.in_bounds);
            try std.testing.expect(
                record.err_dist <= tcfg.DISTORTION_ROUNDTRIP_ABS_PX,
            );
            try std.testing.expect(
                record.observed_reproj_err <= tcfg.DISTORTION_ROUNDTRIP_ABS_PX,
            );
        }
    }
}
