// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const std = @import("std");
const camera = @import("../riley/zig/camera.zig");
const fullfixtures = @import("../dev_support/fullfixtures.zig");
const iio = @import("../riley/zig/imageio.zig");
const mo = @import("../riley/zig/meshpipeline.zig");
const orch = @import("../dev_support/orchestration.zig");
const policy = @import("../dev_support/testpolicy.zig");
const rastcfg = @import("../riley/zig/rasterconfig.zig");
const riley = @import("../riley/zig/riley.zig");
const tcfg = @import("../dev_support/testconfig.zig");

const CameraInput = camera.CameraInput;
const MeshInput = mo.MeshInput;

pub const resolutions = [_][2]u32{
    .{ 31, 19 },
    .{ 65, 47 },
    .{ 161, 103 },
    .{ 401, 251 },
};

pub const ssaa_values = [_]u8{ 1, 4 };

pub fn formatTilingGoldDirName(
    allocator: std.mem.Allocator,
    pixel_num: [2]u32,
    ssaa: u8,
) ![]const u8 {
    return std.fmt.allocPrint(
        allocator,
        "scene2_tri3_res_{d}x{d}_ssaa_{d}",
        .{ pixel_num[0], pixel_num[1], ssaa },
    );
}

pub fn generate(allocator: std.mem.Allocator, io: std.Io) !void {
    var config = tcfg.getRasterConfig(.gold_gen);
    config.save_strategy = .disk;
    config.image_save_mode = .grey;
    config.background_value = 127.5;
    config.image_save_opts = &[_]iio.ImageSaveOpts{
        .{ .format = .fimg, .bits = null, .scaling = .none },
        .{ .format = .bmp, .bits = 8, .scaling = .none },
    };

    const gold_dir_root = policy.goldRoot(.full_tiling);

    var textures = try fullfixtures.FullTextures.init(allocator, io);
    defer textures.deinit(allocator);

    var prep = try fullfixtures.prepareScene2(allocator, io, .tri3);
    defer prep.deinit(allocator);

    const meshes = fullfixtures.buildScene2Meshes(&prep, &textures);

    for (resolutions) |pixel_num| {
        for (ssaa_values) |ssaa| {
            var arena = std.heap.ArenaAllocator.init(allocator);
            defer arena.deinit();
            const aa = arena.allocator();

            const cam_inp = fullfixtures.createScene2Camera(pixel_num, ssaa);
            const case_dir_name = try formatTilingGoldDirName(aa, pixel_num, ssaa);
            const gold_dir = try std.fmt.allocPrint(
                aa,
                "{s}/{s}",
                .{ gold_dir_root, case_dir_name },
            );

            var out_dir = try orch.openDirEnsured(io, gold_dir);
            out_dir.close(io);

            const render_groups = [_]riley.RenderGroupSpec{
                .{ .io = io, .workers = @max(@as(u16, 1), config.total_threads) },
            };

            _ = try riley.raster(
                aa,
                &render_groups,
                &[_]CameraInput{cam_inp},
                &meshes,
                config,
                gold_dir,
            );
        }
    }
}
