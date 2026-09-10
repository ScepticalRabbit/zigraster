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
const common_full = @import("../gengold/gen_gold_full_common.zig");
const common_test = @import("../dev_support/tests.zig");
const gengold_dist_psf = @import("../gengold/gen_gold_full_dist_psf.zig");
const mo = @import("../riley/zig/meshpipeline.zig");
const policy = @import("../dev_support/testpolicy.zig");
const rastcfg = @import("../riley/zig/rasterconfig.zig");
const riley = @import("../riley/zig/riley.zig");
const tcfg = @import("../dev_support/testconfig.zig");

const F = buildconfig.F;
const CameraInput = camera.CameraInput;
const MeshInput = mo.MeshInput;
const Timestamp = std.Io.Clock.Timestamp;

pub const FULL_REL_TOL: F = if (F == f32) 1.0e-3 else 1.0e-5;
pub const FULL_ABS_TOL: F = if (F == f32) 1.0e-3 else 1.0e-5;

pub const BufferModeCase = struct {
    tag: []const u8,
    mode: rastcfg.BufferMode,
};

pub const buffer_mode_cases = [_]BufferModeCase{
    .{ .tag = "tile_local", .mode = .tile_local },
    .{ .tag = "global_subpx", .mode = .global_subpx_full },
    .{ .tag = "global_stripe", .mode = .global_subpx_stripe },
};

pub fn runFullDistPsfCaseTest(
    allocator: std.mem.Allocator,
    io: std.Io,
    prep: *const common_full.Scene1Prepared,
    ssaa: u32,
    dist_case: gengold_dist_psf.DistCase,
    psf_case: gengold_dist_psf.PsfCase,
    buf_case: BufferModeCase,
    gold_dir_root: []const u8,
    config: rastcfg.RasterConfig,
) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    const case_name = try gengold_dist_psf.formatDistPsfCaseName(
        aa,
        ssaa,
        dist_case.tag,
        psf_case.tag,
    );
    const gold_dir = try std.fmt.allocPrint(
        aa,
        "{s}/{s}",
        .{ gold_dir_root, case_name },
    );

    const mesh = gengold_dist_psf.buildScene1Mesh(prep);
    const meshes = [_]MeshInput{mesh};

    var camera_input = prep.camera_input;
    camera_input.sub_sample = ssaa;
    camera_input.distortion = dist_case.distortion;
    camera_input.psf = psf_case.psf;

    var run_config = config;
    run_config.save_strategy = .memory;
    run_config.buffer_mode = buf_case.mode;
    run_config.background_value = common_full.grey_background_scene1;

    const start_time = Timestamp.now(io, .awake);
    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = @max(@as(u16, 1), run_config.total_threads) },
    };

    const result = try riley.raster(
        aa,
        &render_groups,
        &[_]CameraInput{camera_input},
        &meshes,
        run_config,
        null,
    );

    var render_result = result orelse return error.NoResult;
    defer aa.free(render_result.slice);

    const end_time = Timestamp.now(io, .awake);
    const duration_ms = @as(
        F,
        @floatFromInt(start_time.durationTo(end_time).raw.nanoseconds),
    ) / 1.0e6;

    const gold_path = try common_test.findGoldPath(
        aa,
        io,
        gold_dir,
        0,
        0,
        0,
        false,
    );

    common_test.compareNDArrayToGold(
        aa,
        io,
        &render_result,
        0,
        0,
        0,
        1,
        gold_path,
        FULL_REL_TOL,
        FULL_ABS_TOL,
    ) catch |err| {
        const fail_dir_name = try std.fmt.allocPrint(
            aa,
            "test_full_dist_psf/{s}_{s}",
            .{ case_name, buf_case.tag },
        );
        try common_test.saveComparisonArtifactsFromResult(
            aa,
            io,
            common_test.default_fails_root,
            fail_dir_name,
            &render_result,
            0,
            0,
            0,
            gold_path,
            1,
        );
        if (tcfg.TEST_CASE_VERBOSE) {
            std.debug.print(
                "FAIL {s} ({s}) ({d:.2} ms)\n",
                .{ case_name, buf_case.tag, duration_ms },
            );
        }
        return err;
    };

    // Verify that distortion and PSF cases differ from the base (none/box) reference
    const is_base_case = std.mem.eql(u8, dist_case.tag, "dist_none") and
        std.mem.eql(u8, psf_case.tag, "psf_box");
    if (!is_base_case) {
        const base_case_name = try gengold_dist_psf.formatDistPsfCaseName(
            aa,
            ssaa,
            "dist_none",
            "psf_box",
        );
        const base_gold_dir = try std.fmt.allocPrint(
            aa,
            "{s}/{s}",
            .{ gold_dir_root, base_case_name },
        );
        const base_gold_path = try common_test.findGoldPath(
            aa,
            io,
            base_gold_dir,
            0,
            0,
            0,
            false,
        );
        var base_gold = if (std.mem.endsWith(u8, base_gold_path, ".fimg"))
            try @import("../riley/zig/imageio.zig").loadFIMG(aa, io, base_gold_path)
        else
            try @import("../riley/zig/csvio.zig").loadScalarCsv2D(aa, io, base_gold_path);
        defer {
            aa.free(base_gold.slice);
            base_gold.deinit(aa);
        }

        const rows_n = render_result.dims[3];
        const cols_n = render_result.dims[4];
        var max_base_diff: F = 0.0;
        for (0..rows_n) |rr| {
            for (0..cols_n) |cc| {
                const v_curr = render_result.get(&[_]usize{ 0, 0, 0, rr, cc });
                const v_base = if (base_gold.dims.len == 3)
                    base_gold.get(&[_]usize{ 0, rr, cc })
                else
                    base_gold.get(&[_]usize{ rr, cc });
                const diff = @abs(v_curr - v_base);
                if (diff > max_base_diff) {
                    max_base_diff = diff;
                }
            }
        }
        if (max_base_diff < 1.0e-3) {
            return error.DistortionOrPsfHadNoEffect;
        }
    }

    if (tcfg.TEST_CASE_VERBOSE) {
        std.debug.print(
            "PASS {s} ({s}) ({d:.2} ms)\n",
            .{ case_name, buf_case.tag, duration_ms },
        );
    }
}

pub fn run(allocator: std.mem.Allocator, io: std.Io) !void {
    const config = tcfg.getRasterConfig(.testing);
    const gold_dir_root = policy.goldRoot(.full_dist_psf);

    var prep = try common_full.prepareScene1(allocator, io);
    defer prep.deinit(allocator);

    for (gengold_dist_psf.ssaa_levels) |ssaa| {
        for (gengold_dist_psf.dist_cases) |dist_case| {
            for (gengold_dist_psf.psf_cases) |psf_case| {
                for (buffer_mode_cases) |buf_case| {
                    try runFullDistPsfCaseTest(
                        allocator,
                        io,
                        &prep,
                        ssaa,
                        dist_case,
                        psf_case,
                        buf_case,
                        gold_dir_root,
                        config,
                    );
                }
            }
        }
    }
}

test "full_dist_psf" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    try run(allocator, io);
}
