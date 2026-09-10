// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const std = @import("std");
const buildconfig = @import("../riley/zig/buildconfig.zig");
const camera = @import("../riley/zig/camera.zig");
const common = @import("gen_gold_full_common.zig");
const gk = @import("../riley/zig/geometrykernels.zig");
const iio = @import("../riley/zig/imageio.zig");
const mo = @import("../riley/zig/meshpipeline.zig");
const orch = @import("../dev_support/orchestration.zig");
const policy = @import("../dev_support/testpolicy.zig");
const rastcfg = @import("../riley/zig/rasterconfig.zig");
const riley = @import("../riley/zig/riley.zig");
const tcfg = @import("../dev_support/testconfig.zig");

const F = buildconfig.F;
const CameraInput = camera.CameraInput;
const MeshInput = mo.MeshInput;
const Timestamp = std.Io.Clock.Timestamp;

pub fn generate(allocator: std.mem.Allocator, io: std.Io) !void {
    var config = tcfg.getRasterConfig(.gold_gen);
    config.save_strategy = .disk;
    config.render_mode = .in_order;
    config.total_threads = 1;
    config.image_save_mode = .grey;
    config.image_save_opts = &[_]iio.ImageSaveOpts{
        .{ .format = .fimg, .bits = null, .scaling = .none },
        .{ .format = .bmp, .bits = 16, .scaling = .auto },
    };

    const gold_dir_root = policy.goldRoot(.full_scene_camera_threads);

    var textures = try common.FullTextures.init(allocator, io);
    defer textures.deinit(allocator);

    var prep = try common.prepareScene3(allocator, io);
    defer prep.deinit(allocator);

    const meshes = common.buildScene3Meshes(&prep, &textures);
    const cameras = common.createScene3Cameras();

    for (cameras, 0..) |cam_inp, cc| {
        const cam_dir_name = try std.fmt.allocPrint(
            allocator,
            "{s}/cam{d}",
            .{ gold_dir_root, cc },
        );
        defer allocator.free(cam_dir_name);

        var out_dir = try orch.openDirEnsured(io, cam_dir_name);
        out_dir.close(io);

        const render_groups = [_]riley.RenderGroupSpec{
            .{ .io = io, .workers = 1 },
        };

        _ = try riley.raster(
            allocator,
            &render_groups,
            &[_]CameraInput{cam_inp},
            &meshes,
            config,
            cam_dir_name,
        );
    }
}
