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
const ndarray = @import("../riley/zig/ndarray.zig");
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
    try runUVBoundaryDirectSamplingTests(allocator);
    try runSmallTextureFilterSupportTests(allocator);
    try runUVBoundaryRasterPipelineTests(allocator, io);
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

fn runUVBoundaryDirectSamplingTests(allocator: std.mem.Allocator) !void {
    @setEvalBranchQuota(buildconfig.comptime_eval_branch_quota);

    var tex_u8 = try texops.Tex(u8, 1).init(allocator, 4, 4);
    defer tex_u8.deinit(allocator);
    for (0..4) |rr| {
        for (0..4) |cc| {
            const pixel_val = @as(u8, @intCast((rr * 4 + cc + 1) * 15));
            tex_u8.setVal(0, rr, cc, pixel_val);
        }
    }

    var tex_f64 = try texops.Tex(F, 3).init(allocator, 4, 4);
    defer tex_f64.deinit(allocator);
    for (0..4) |rr| {
        for (0..4) |cc| {
            const base_val = @as(F, @floatFromInt(rr * 4 + cc + 1)) / 16.0;
            tex_f64.setVal(0, rr, cc, base_val);
            tex_f64.setVal(1, rr, cc, base_val * 0.5);
            tex_f64.setVal(2, rr, cc, 1.0 - base_val);
        }
    }

    const exact_uvs = [_][2]F{
        .{ 0.0, 0.0 },
        .{ 1.0, 0.0 },
        .{ 0.0, 1.0 },
        .{ 1.0, 1.0 },
    };

    const near_uvs = [_][2]F{
        .{ 1.0e-6, 1.0e-6 },
        .{ 1.0 - 1.0e-6, 1.0 - 1.0e-6 },
        .{ 0.001, 0.999 },
        .{ 0.999, 0.001 },
    };

    const edge_support_uvs = [_][2]F{
        .{ 0.02, 0.02 },
        .{ 0.98, 0.98 },
        .{ 0.02, 0.98 },
        .{ 0.98, 0.02 },
    };

    const out_of_range_uvs = [_][2]F{
        .{ -0.5, 0.5 },
        .{ 1.5, 0.5 },
        .{ 0.5, -2.0 },
        .{ 0.5, 3.0 },
        .{ -10.0, -10.0 },
        .{ 10.0, 10.0 },
    };

    inline for (all_tex_samp_configs) |samp_cfg| {
        for (exact_uvs) |uv_coord| {
            const sample_mono = texops.sampScal(
                1,
                samp_cfg,
                tex_u8,
                uv_coord[0],
                uv_coord[1],
            );
            try std.testing.expect(std.math.isFinite(sample_mono[0]));
            try std.testing.expect(sample_mono[0] >= 0.0 and sample_mono[0] <= 255.0);

            const sample_rgb = texops.sampScal(
                3,
                samp_cfg,
                tex_f64,
                uv_coord[0],
                uv_coord[1],
            );
            for (0..3) |ch| {
                try std.testing.expect(std.math.isFinite(sample_rgb[ch]));
            }
        }

        for (near_uvs) |uv_coord| {
            const sample_mono = texops.sampScal(
                1,
                samp_cfg,
                tex_u8,
                uv_coord[0],
                uv_coord[1],
            );
            try std.testing.expect(std.math.isFinite(sample_mono[0]));

            const sample_rgb = texops.sampScal(
                3,
                samp_cfg,
                tex_f64,
                uv_coord[0],
                uv_coord[1],
            );
            for (0..3) |ch| {
                try std.testing.expect(std.math.isFinite(sample_rgb[ch]));
            }
        }

        for (edge_support_uvs) |uv_coord| {
            const sample_mono = texops.sampScal(
                1,
                samp_cfg,
                tex_u8,
                uv_coord[0],
                uv_coord[1],
            );
            try std.testing.expect(std.math.isFinite(sample_mono[0]));

            const sample_rgb = texops.sampScal(
                3,
                samp_cfg,
                tex_f64,
                uv_coord[0],
                uv_coord[1],
            );
            for (0..3) |ch| {
                try std.testing.expect(std.math.isFinite(sample_rgb[ch]));
            }
        }

        for (out_of_range_uvs) |uv_coord| {
            const sample_mono = texops.sampScal(
                1,
                samp_cfg,
                tex_u8,
                uv_coord[0],
                uv_coord[1],
            );
            try std.testing.expect(std.math.isFinite(sample_mono[0]));
            try std.testing.expect(
                sample_mono[0] >= 0.0 and sample_mono[0] <= 255.0,
            );

            const sample_rgb = texops.sampScal(
                3,
                samp_cfg,
                tex_f64,
                uv_coord[0],
                uv_coord[1],
            );
            for (0..3) |ch| {
                try std.testing.expect(std.math.isFinite(sample_rgb[ch]));
            }
        }

        const sample_far_top_left = texops.sampScal(
            1,
            samp_cfg,
            tex_u8,
            -10.0,
            -10.0,
        );
        try std.testing.expectApproxEqAbs(
            @as(F, 15.0),
            sample_far_top_left[0],
            1.0e-3,
        );

        const sample_far_bottom_right = texops.sampScal(
            1,
            samp_cfg,
            tex_u8,
            10.0,
            10.0,
        );
        try std.testing.expectApproxEqAbs(
            @as(F, 240.0),
            sample_far_bottom_right[0],
            1.0e-3,
        );
    }
}

fn runSmallTextureFilterSupportTests(allocator: std.mem.Allocator) !void {
    @setEvalBranchQuota(buildconfig.comptime_eval_branch_quota);

    var tex_1x1 = try texops.Tex(u8, 1).init(allocator, 1, 1);
    defer tex_1x1.deinit(allocator);
    tex_1x1.setVal(0, 0, 0, 180);

    var tex_2x2 = try texops.Tex(u16, 1).init(allocator, 2, 2);
    defer tex_2x2.deinit(allocator);
    tex_2x2.setVal(0, 0, 0, 1000);
    tex_2x2.setVal(0, 0, 1, 2000);
    tex_2x2.setVal(0, 1, 0, 3000);
    tex_2x2.setVal(0, 1, 1, 4000);

    var tex_3x3 = try texops.Tex(F, 1).init(allocator, 3, 3);
    defer tex_3x3.deinit(allocator);
    for (0..3) |rr| {
        for (0..3) |cc| {
            tex_3x3.setVal(0, rr, cc, @as(F, @floatFromInt(rr * 3 + cc + 1)) * 0.1);
        }
    }

    const uv_eval_points = [_][2]F{
        .{ -0.5, -0.5 },
        .{ 0.0, 0.0 },
        .{ 0.25, 0.75 },
        .{ 0.5, 0.5 },
        .{ 1.0, 1.0 },
        .{ 1.5, 1.5 },
    };

    inline for (all_tex_samp_configs) |samp_cfg| {
        for (uv_eval_points) |uv_coord| {
            const sample_1x1 = texops.sampScal(
                1,
                samp_cfg,
                tex_1x1,
                uv_coord[0],
                uv_coord[1],
            );
            try std.testing.expect(std.math.isFinite(sample_1x1[0]));
            try std.testing.expectApproxEqAbs(@as(F, 180.0), sample_1x1[0], 1.0e-3);

            const sample_2x2 = texops.sampScal(
                1,
                samp_cfg,
                tex_2x2,
                uv_coord[0],
                uv_coord[1],
            );
            try std.testing.expect(std.math.isFinite(sample_2x2[0]));

            const sample_3x3 = texops.sampScal(
                1,
                samp_cfg,
                tex_3x3,
                uv_coord[0],
                uv_coord[1],
            );
            try std.testing.expect(std.math.isFinite(sample_3x3[0]));
        }
    }
}

fn runUVBoundaryRasterPipelineTests(
    allocator: std.mem.Allocator,
    io: std.Io,
) !void {
    var prep = try common_full.prepareScene0(allocator, io, .quad4);
    defer prep.deinit(allocator);

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    const nodes_num = prep.sphere_uvs.array.dims[0];
    var out_of_range_uv_buff = try aa.alloc(F, nodes_num * 2);
    for (0..nodes_num) |node_idx| {
        const u_base = prep.sphere_uvs.getU(node_idx);
        const v_base = prep.sphere_uvs.getV(node_idx);
        out_of_range_uv_buff[node_idx * 2 + 0] = u_base * 2.0 - 0.5;
        out_of_range_uv_buff[node_idx * 2 + 1] = v_base * 2.0 - 0.5;
    }
    const out_of_range_uvs = try ndarray.NDArray(F).init(
        aa,
        out_of_range_uv_buff,
        &[_]usize{ nodes_num, 2 },
    );

    var tex_small = try texops.Tex(u8, 1).init(aa, 2, 2);
    tex_small.setVal(0, 0, 0, 50);
    tex_small.setVal(0, 0, 1, 120);
    tex_small.setVal(0, 1, 0, 180);
    tex_small.setVal(0, 1, 1, 240);

    const test_configs = [_]texops.TexSampConfig{
        .{ .sample = .cubic_catmull_rom, .mode = .direct },
        .{ .sample = .lanczos3, .mode = .lut },
        .{ .sample = .quintic_bspline, .mode = .lut_lerp },
    };

    for (test_configs) |samp_cfg| {
        const meshes = [_]MeshInput{
            .{
                .mesh_type = .quad4,
                .coords = prep.sphere_coords,
                .connect = prep.sphere_connect,
                .disp = prep.sphere_disp,
                .shader = .{
                    .tex_u8 = .{
                        .uvs = out_of_range_uvs,
                        .tex = tex_small,
                        .samp_cfg = samp_cfg,
                        .normal_type = .none,
                    },
                },
            },
        };

        var run_config = tcfg.getRasterConfig(.testing);
        run_config.save_strategy = .memory;

        const render_groups = [_]riley.RenderGroupSpec{
            .{ .io = io, .workers = 1 },
        };

        const result = try riley.raster(
            aa,
            &render_groups,
            &[_]CameraInput{prep.camera_input},
            &meshes,
            run_config,
            null,
        );

        const render_result = result orelse return error.NoResult;
        defer aa.free(render_result.slice);

        var has_nonzero: bool = false;
        for (render_result.slice) |pixel_val| {
            try std.testing.expect(!std.math.isNan(pixel_val));
            try std.testing.expect(!std.math.isInf(pixel_val));
            if (pixel_val > 0.0) {
                has_nonzero = true;
            }
        }
        try std.testing.expect(has_nonzero);
    }
}

