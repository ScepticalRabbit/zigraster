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

pub const ImageOutputBaseCase = struct {
    tag: []const u8,
    is_rgb: bool,
    is_u16: bool,
};

pub const base_cases = [_]ImageOutputBaseCase{
    .{ .tag = "mono_u8", .is_rgb = false, .is_u16 = false },
    .{ .tag = "mono_u16", .is_rgb = false, .is_u16 = true },
    .{ .tag = "rgb_u8", .is_rgb = true, .is_u16 = false },
    .{ .tag = "rgb_u16", .is_rgb = true, .is_u16 = true },
};

pub fn generate(allocator: std.mem.Allocator, io: std.Io) !void {
    var config = tcfg.getRasterConfig(.gold_gen);
    config.save_strategy = .disk;
    config.background_value = 127.5;
    config.render_mode = .in_order;
    config.total_threads = 1;

    const gold_dir_root = policy.goldRoot(.full_image_output);

    var textures = try common.FullTextures.init(allocator, io);
    defer textures.deinit(allocator);

    var prep = try common.prepareScene2(allocator, io, .tri3);
    defer prep.deinit(allocator);

    for (base_cases) |bc| {
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const aa = arena.allocator();

        const meshes = common.buildScene2ImageOutputMeshes(
            &prep,
            &textures,
            bc.is_rgb,
            bc.is_u16,
        );
        const cam_inp = common.createScene2ImageOutputCamera(&meshes);

        var run_config = config;
        run_config.image_save_mode = if (bc.is_rgb) .rgb else .grey;
        const bits: ?u8 = if (bc.is_u16) 16 else 8;
        run_config.image_save_opts = &[_]iio.ImageSaveOpts{
            .{ .format = .fimg, .bits = null, .scaling = .none },
            .{ .format = .bmp, .bits = bits, .scaling = .auto },
        };

        const gold_dir = try std.fmt.allocPrint(
            aa,
            "{s}/{s}",
            .{ gold_dir_root, bc.tag },
        );

        var out_dir = try orch.openDirEnsured(io, gold_dir);
        out_dir.close(io);

        const render_groups = [_]riley.RenderGroupSpec{
            .{ .io = io, .workers = 1 },
        };

        _ = try riley.raster(
            aa,
            &render_groups,
            &[_]CameraInput{cam_inp},
            &meshes,
            run_config,
            gold_dir,
        );
    }
}
