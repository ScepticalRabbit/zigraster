// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const std = @import("std");
const buildconfig = @import("riley/zig/buildconfig.zig");
const gen_dist_psf = @import("gengold/gen_gold_full_dist_psf.zig");
const gen_hull = @import("gengold/gen_gold_full_hull.zig");
const gen_sct = @import("gengold/gen_gold_full_scene_camera_threads.zig");
const gen_shader = @import("gengold/gen_gold_full_shader.zig");
const gen_ssaa_pxmap = @import("gengold/gen_gold_full_ssaa_pxmap.zig");
const gen_texture = @import("gengold/gen_gold_full_texture.zig");
const gen_tiling = @import("gengold/gen_gold_full_tiling.zig");
const iio = @import("riley/zig/imageio.zig");
const policy = @import("dev_support/testpolicy.zig");
const tcfg = @import("dev_support/testconfig.zig");

pub fn main(init: std.process.Init) !void {
    @setEvalBranchQuota(buildconfig.comptime_eval_branch_quota);
    const io = init.io;
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const aa = arena.allocator();

    var config = tcfg.getRasterConfig(.gold_gen);
    config.save_strategy = .disk;
    config.image_save_opts = &[_]iio.ImageSaveOpts{
        .{ .format = .fimg, .bits = null, .scaling = .none },
        .{ .format = .bmp, .bits = 8, .scaling = .auto },
    };

    const shader_gold_dir = policy.goldRoot(.full_shader);
    const texture_gold_dir = policy.goldRoot(.full_texture);
    const dist_psf_gold_dir = policy.goldRoot(.full_dist_psf);
    const ssaa_pxmap_gold_dir = policy.goldRoot(.full_ssaa_pxmap);
    const hull_gold_dir = policy.goldRoot(.full_hull);
    const tiling_gold_dir = policy.goldRoot(.full_tiling);
    const sct_gold_dir = policy.goldRoot(.full_scene_camera_threads);

    std.debug.print("\nGenerating FULL Gold Suite...\n\n", .{});

    const start_time = std.Io.Clock.Timestamp.now(io, .awake);

    std.debug.print("--- 1/6: Full Shader Cases ({s}) ---\n", .{shader_gold_dir});
    const start_shader = std.Io.Clock.Timestamp.now(io, .awake);
    try gen_shader.generateAllFullShaderCases(
        aa,
        io,
        shader_gold_dir,
        config,
    );
    const end_shader = std.Io.Clock.Timestamp.now(io, .awake);
    const shader_elapsed = @as(
        f64,
        @floatFromInt(start_shader.durationTo(end_shader).raw.nanoseconds),
    ) / 1.0e9;
    std.debug.print(
        "Shader gold generation took {d:.3} seconds.\n\n",
        .{shader_elapsed},
    );

    std.debug.print(
        "--- 2/6: Full Texture Cases ({s}) ---\n",
        .{texture_gold_dir},
    );
    const start_texture = std.Io.Clock.Timestamp.now(io, .awake);
    try gen_texture.generateAllFullTextureCases(
        aa,
        io,
        texture_gold_dir,
        config,
    );
    const end_texture = std.Io.Clock.Timestamp.now(io, .awake);
    const texture_elapsed = @as(
        f64,
        @floatFromInt(start_texture.durationTo(end_texture).raw.nanoseconds),
    ) / 1.0e9;
    std.debug.print(
        "Texture gold generation took {d:.3} seconds.\n\n",
        .{texture_elapsed},
    );

    std.debug.print(
        "--- 3/6: Full Distortion + PSF Cases ({s}) ---\n",
        .{dist_psf_gold_dir},
    );
    const start_dist_psf = std.Io.Clock.Timestamp.now(io, .awake);
    try gen_dist_psf.generateAllFullDistPsfCases(
        aa,
        io,
        dist_psf_gold_dir,
        config,
    );
    const end_dist_psf = std.Io.Clock.Timestamp.now(io, .awake);
    const dist_psf_elapsed = @as(
        f64,
        @floatFromInt(start_dist_psf.durationTo(end_dist_psf).raw.nanoseconds),
    ) / 1.0e9;
    std.debug.print(
        "Distortion + PSF gold generation took {d:.3} seconds.\n\n",
        .{dist_psf_elapsed},
    );

    std.debug.print(
        "--- 4/6: Full SSAA + Pixel Map Cases ({s}) ---\n",
        .{ssaa_pxmap_gold_dir},
    );
    const start_ssaa_pxmap = std.Io.Clock.Timestamp.now(io, .awake);
    try gen_ssaa_pxmap.generateAllFullSsaaPxmapCases(
        aa,
        io,
        ssaa_pxmap_gold_dir,
        config,
    );
    const end_ssaa_pxmap = std.Io.Clock.Timestamp.now(io, .awake);
    const ssaa_pxmap_elapsed = @as(
        f64,
        @floatFromInt(start_ssaa_pxmap.durationTo(end_ssaa_pxmap).raw.nanoseconds),
    ) / 1.0e9;
    std.debug.print(
        "SSAA + Pixel Map gold generation took {d:.3} seconds.\n\n",
        .{ssaa_pxmap_elapsed},
    );

    std.debug.print(
        "--- 5/6: Full Hull Cases ({s}) ---\n",
        .{hull_gold_dir},
    );
    const start_hull = std.Io.Clock.Timestamp.now(io, .awake);
    try gen_hull.generate(aa, io);
    const end_hull = std.Io.Clock.Timestamp.now(io, .awake);
    const hull_elapsed = @as(
        f64,
        @floatFromInt(start_hull.durationTo(end_hull).raw.nanoseconds),
    ) / 1.0e9;
    std.debug.print(
        "Hull gold generation took {d:.3} seconds.\n\n",
        .{hull_elapsed},
    );

    std.debug.print(
        "--- 6/7: Full Tiling Cases ({s}) ---\n",
        .{tiling_gold_dir},
    );
    const start_tiling = std.Io.Clock.Timestamp.now(io, .awake);
    try gen_tiling.generate(aa, io);
    const end_tiling = std.Io.Clock.Timestamp.now(io, .awake);
    const tiling_elapsed = @as(
        f64,
        @floatFromInt(start_tiling.durationTo(end_tiling).raw.nanoseconds),
    ) / 1.0e9;
    std.debug.print(
        "Tiling gold generation took {d:.3} seconds.\n\n",
        .{tiling_elapsed},
    );

    std.debug.print(
        "--- 7/7: Full Scene Camera Threads Cases ({s}) ---\n",
        .{sct_gold_dir},
    );
    const start_sct = std.Io.Clock.Timestamp.now(io, .awake);
    try gen_sct.generate(aa, io);
    const end_sct = std.Io.Clock.Timestamp.now(io, .awake);
    const sct_elapsed = @as(
        f64,
        @floatFromInt(start_sct.durationTo(end_sct).raw.nanoseconds),
    ) / 1.0e9;
    std.debug.print(
        "Scene Camera Threads gold generation took {d:.3} seconds.\n\n",
        .{sct_elapsed},
    );

    const end_time = std.Io.Clock.Timestamp.now(io, .awake);
    const total_elapsed = @as(
        f64,
        @floatFromInt(start_time.durationTo(end_time).raw.nanoseconds),
    ) / 1.0e9;
    std.debug.print(
        "FULL Gold Suite Generation Complete. Took {d:.3} seconds.\n\n",
        .{total_elapsed},
    );
}
