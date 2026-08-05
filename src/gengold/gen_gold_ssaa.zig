// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const std = @import("std");
const iio = @import("../riley/zig/imageio.zig");
const suite = @import("../dev_support/ssaasuite.zig");
const orch = @import("../dev_support/orchestration.zig");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var root_dir = try orch.openDirEnsured(io, suite.gold_root);
    defer root_dir.close(io);

    for (suite.mesh_types) |mesh_type| {
        for (suite.ssaa_values) |ssaa| {
            for (suite.distortion_cases) |distortion_case| {
                const case_name_base = try suite.caseName(
                    allocator,
                    mesh_type,
                    ssaa,
                    distortion_case,
                );
                defer allocator.free(case_name_base);
                const gold_maps = [_]@import("../riley/zig/camera.zig").SubPixelCenterMap{
                    .full_in_mem,
                    .affine_jac,
                };
                for (gold_maps) |gold_map| {
                    if (gold_map == .affine_jac and distortion_case == .none) continue;

                    const case_name = if (gold_map == .full_in_mem)
                        case_name_base
                    else
                        try std.fmt.allocPrint(
                            allocator,
                            "{s}_{s}",
                            .{ case_name_base, @tagName(gold_map) },
                        );
                    defer if (gold_map != .full_in_mem) allocator.free(case_name);
                    const out_dir_path = try std.fmt.allocPrint(
                        allocator,
                        "{s}/{s}",
                        .{ suite.gold_root, case_name },
                    );
                    defer allocator.free(out_dir_path);
                    var out_dir = try orch.openDirEnsured(io, out_dir_path);
                    defer out_dir.close(io);

                    const image = try suite.renderCase(
                        allocator,
                        io,
                        mesh_type,
                        ssaa,
                        distortion_case,
                        gold_map,
                    );
                    defer {
                        allocator.free(image.slice);
                        var image_mut = image;
                        image_mut.deinit(allocator);
                    }

                    try iio.saveImage(
                        io,
                        out_dir,
                        "cam0_frame0_field0",
                        &image,
                        0,
                        .{
                            .format = .fimg,
                            .bits = null,
                            .scaling = .none,
                            .channels = 1,
                        },
                    );
                }
            }
        }
    }

    std.debug.print("Done.\n", .{});
}
