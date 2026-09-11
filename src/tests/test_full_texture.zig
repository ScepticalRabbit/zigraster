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
const common_full = @import("../dev_support/fullfixtures.zig");
const common_test = @import("../dev_support/tests.zig");
const fullcase_tex = @import("fullcase_texture.zig");
const gk = @import("../riley/zig/geometrykernels.zig");
const mo = @import("../riley/zig/meshpipeline.zig");
const policy = @import("../dev_support/testpolicy.zig");
const rastcfg = @import("../riley/zig/rasterconfig.zig");
const riley = @import("../riley/zig/riley.zig");
const tcfg = @import("../dev_support/testconfig.zig");
const texops = @import("../riley/zig/textureops.zig");

const F = buildconfig.F;
const CameraInput = camera.CameraInput;
const MeshInput = mo.MeshInput;
const Timestamp = std.Io.Clock.Timestamp;
const FullTexSamplingCase = fullcase_tex.FullTexSamplingCase;
const all_tex_samp_configs = fullcase_tex.all_tex_samp_configs;

pub fn runFullTexCaseTest(
    allocator: std.mem.Allocator,
    io: std.Io,
    mesh_type: gk.MeshType,
    prep: *const common_full.Scene0Prepared,
    textures: *const common_full.FullTextures,
    case: FullTexSamplingCase,
    gold_dir_root: []const u8,
    config: rastcfg.RasterConfig,
) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    const case_dir_name = try case.formatDirName(aa, mesh_type);
    const gold_dir = try std.fmt.allocPrint(
        aa,
        "{s}/{s}",
        .{ gold_dir_root, case_dir_name },
    );

    const meshes = fullcase_tex.buildTexSamplingCaseMeshes(
        mesh_type,
        prep,
        textures,
        case,
    );

    var run_config = config;
    run_config.save_strategy = .memory;

    const start_time = Timestamp.now(io, .awake);
    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = @max(@as(u16, 1), run_config.total_threads) },
    };

    const result = try riley.raster(
        aa,
        &render_groups,
        &[_]CameraInput{prep.camera_input},
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

    const frames_num = if (render_result.dims.len == 5)
        render_result.dims[1]
    else
        render_result.dims[0];
    const is_rgb = case.is_rgb;
    const channels_num: usize = if (is_rgb) 3 else 1;

    for (0..frames_num) |ff| {
        for (0..channels_num) |ch| {
            const gold_path = try common_test.findGoldPath(
                aa,
                io,
                gold_dir,
                0,
                ff,
                ch,
                false,
            );

            common_test.compareNDArrayToGold(
                aa,
                io,
                &render_result,
                0,
                ff,
                ch,
                1,
                gold_path,
                tcfg.FULL_GOLD_REL_TOL,
                tcfg.FULL_GOLD_ABS_TOL,
            ) catch |err| {
                const fail_dir_name = try std.fmt.allocPrint(
                    aa,
                    "full_texture/{s}",
                    .{case_dir_name},
                );
                try common_test.saveComparisonArtifactsFromResult(
                    aa,
                    io,
                    common_test.default_fails_root,
                    fail_dir_name,
                    &render_result,
                    0,
                    ff,
                    ch,
                    gold_path,
                    1,
                );
                if (tcfg.TEST_CASE_VERBOSE) {
                    std.debug.print(
                        "FAIL {s} frame {d} ch {d} ({d:.2} ms)\n",
                        .{ case_dir_name, ff, ch, duration_ms },
                    );
                }
                return err;
            };
        }
    }

    if (frames_num >= 2) {
        var max_frame_diff: F = 0.0;
        const rows_n = render_result.dims[3];
        const cols_n = render_result.dims[4];
        for (0..channels_num) |ch| {
            for (0..rows_n) |rr| {
                for (0..cols_n) |cc| {
                    const v0 = render_result.get(&[_]usize{ 0, 0, ch, rr, cc });
                    const v1 = render_result.get(&[_]usize{ 0, 1, ch, rr, cc });
                    const diff = @abs(v0 - v1);
                    if (diff > max_frame_diff) {
                        max_frame_diff = diff;
                    }
                }
            }
        }
        if (max_frame_diff < 1.0e-3) {
            return error.DeformationFramesIdentical;
        }
    }

    if (tcfg.TEST_CASE_VERBOSE) {
        std.debug.print(
            "PASS {s} ({d:.2} ms, {d} frames)\n",
            .{ case_dir_name, duration_ms, frames_num },
        );
    }
}

pub fn run(allocator: std.mem.Allocator, io: std.Io) !void {
    const config = tcfg.getRasterConfig(.testing);
    const gold_dir_root = policy.goldRoot(.full_texture);

    var textures = try common_full.FullTextures.init(allocator, io);
    defer textures.deinit(allocator);

    const mesh_types = [_]gk.MeshType{
        .tri3,
        .tri6,
        .quad4,
        .quad8,
        .quad9,
    };
    const bool_values = [_]bool{ false, true };
    const dtypes = [_]@typeInfo(FullTexSamplingCase).@"struct".fields[1].type{
        .u8,
        .u16,
        .f64,
    };

    for (mesh_types) |mesh_type| {
        var prep = try common_full.prepareScene0(allocator, io, mesh_type);
        defer prep.deinit(allocator);

        for (bool_values) |is_rgb| {
            for (dtypes) |dtype| {
                for (all_tex_samp_configs) |samp_cfg| {
                    try runFullTexCaseTest(
                        allocator,
                        io,
                        mesh_type,
                        &prep,
                        &textures,
                        .{
                            .is_rgb = is_rgb,
                            .dtype = dtype,
                            .samp_cfg = samp_cfg,
                        },
                        gold_dir_root,
                        config,
                    );
                }
            }
        }
    }

    try runTexSampConfigContractTests();
}

fn runTexSampConfigContractTests() !void {
    const invalid_configs = [_]texops.TexSampConfig{
        .{ .sample = .nearest, .mode = .lut },
        .{ .sample = .nearest, .mode = .lut_lerp },
        .{ .sample = .linear, .mode = .lut },
        .{ .sample = .linear, .mode = .lut_lerp },
    };

    for (invalid_configs) |cfg_invalid| {
        try std.testing.expect(!cfg_invalid.isValid());
        const sanitized = cfg_invalid.sanitize();
        try std.testing.expect(sanitized.isValid());
        try std.testing.expectEqual(texops.TexSampMode.direct, sanitized.mode);
        try std.testing.expectEqual(cfg_invalid.sample, sanitized.sample);
    }

    const valid_configs = [_]texops.TexSampConfig{
        .{ .sample = .nearest, .mode = .direct },
        .{ .sample = .linear, .mode = .direct },
        .{ .sample = .cubic_catmull_rom, .mode = .direct },
        .{ .sample = .cubic_catmull_rom, .mode = .lut },
        .{ .sample = .cubic_catmull_rom, .mode = .lut_lerp },
        .{ .sample = .lanczos3, .mode = .lut },
    };

    for (valid_configs) |cfg_valid| {
        try std.testing.expect(cfg_valid.isValid());
        const sanitized = cfg_valid.sanitize();
        try std.testing.expectEqual(cfg_valid.mode, sanitized.mode);
        try std.testing.expectEqual(cfg_valid.sample, sanitized.sample);
    }
}
