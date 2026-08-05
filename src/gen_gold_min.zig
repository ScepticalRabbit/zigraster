// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const std = @import("std");
const common = @import("dev_support/benchcommon.zig");
const gengold = @import("dev_support/gengold.zig");
const minsuite = @import("dev_support/minsuite.zig");
const orch = @import("dev_support/orchestration.zig");
const tcfg = @import("dev_support/testconfig.zig");
const riley = @import("riley/zig/riley.zig");
const mo = @import("riley/zig/meshpipeline.zig");
const gk = @import("riley/zig/geometrykernels.zig");
const iio = @import("riley/zig/imageio.zig");
const texops = @import("riley/zig/textureops.zig");
const buildconfig = @import("riley/zig/buildconfig.zig");
const Rotation = @import("riley/zig/rotation.zig").Rotation;
const policy = @import("dev_support/testpolicy.zig");

comptime {
    if (buildconfig.config.simd != .on) {
        @compileError(
            "src/gen_gold_min.zig requires .simd = .on. " ++
                "MIN scalar gold generation is not implemented.",
        );
    }
}

pub fn main(init: std.process.Init) !void {
    @setEvalBranchQuota(buildconfig.comptime_eval_branch_quota);
    const allocator = init.gpa;
    const io = init.io;

    const texture_grey = try iio.loadImage(
        u8,
        1,
        allocator,
        io,
        "texture/speckle.bmp",
        .bmp,
    );
    defer texture_grey.deinit(allocator);

    const texture_rgb = try iio.loadImage(
        u8,
        3,
        allocator,
        io,
        "texture/speckle_rgb.bmp",
        .bmp,
    );
    defer texture_rgb.deinit(allocator);

    const out_dir = comptime policy.goldRoot(.min);
    const pixel_num_sphere = [_]u32{ 160, 100 };
    const pixel_num_multi = [_]u32{ 640, 400 };
    const render_defaults_sphere = common.BenchRenderDefaults{
        .pixels_num = pixel_num_sphere,
        .sub_sample = 2,
        .focal_leng = 50.0e-3,
        .pixels_size = .{ 5.3e-6, 5.3e-6 },
        .fov_scale = 1.0,
        .rot = Rotation.init(0, 0, 0),
    };

    const mesh_types = comptime std.enums.values(gk.MeshType);
    const shader_types = [_]common.ShaderType{
        .nodal_grey,
        .nodal_rgb,
        .tex8_grey,
        .tex8_rgb,
    };
    const samp_cfgs = [_]texops.TextureSampleConfig{
        .{ .sample = .nearest, .mode = .direct },
        .{ .sample = .linear, .mode = .direct },
        .{ .sample = .cubic_catmull_rom, .mode = .direct },
        .{ .sample = .cubic_catmull_rom, .mode = .lut_lerp },
        .{ .sample = .cubic_mitchell_netravali, .mode = .lut_lerp },
        .{ .sample = .lanczos3, .mode = .lut_lerp },
        .{ .sample = .cubic_bspline, .mode = .lut_lerp },
        .{ .sample = .quintic_bspline, .mode = .direct },
        .{ .sample = .quintic_bspline, .mode = .lut_lerp },
    };

    var config = tcfg.getRasterConfig(.gold);
    config.save_strategy = .disk;
    config.image_save_opts = &[_]iio.ImageSaveOpts{
        .{ .format = .fimg, .bits = null, .scaling = .none },
        .{ .format = .bmp, .bits = 8, .scaling = .auto },
    };

    std.debug.print("Generating MIN Gold Data (sphere200/base)...\n", .{});
    for (mesh_types) |mt| {
        for (shader_types) |st| {
            for (samp_cfgs) |sc| {
                const mesh_name = policy.meshName(.benchmark_data, mt);
                const data_dir = try std.fmt.allocPrint(
                    allocator,
                    "data/min/{s}_sphere200",
                    .{mesh_name},
                );
                defer allocator.free(data_dir);

                const is_rgb = (st == .tex8_rgb or st == .nodal_rgb);
                const is_allowed_rgb = (st == .nodal_rgb) or
                    (st == .tex8_rgb and
                        sc.sample == .cubic_catmull_rom and
                        sc.mode == .lut_lerp);

                if (is_rgb and !is_allowed_rgb) continue;

                if (common.shouldRun(
                    .{ .run = .all, .skip_quad4ibi_sphere = true },
                    mt,
                    st,
                    sc,
                    data_dir,
                )) {
                    const num_ch: u8 = if (is_rgb) 3 else 1;
                    var opts_with_ch = [_]iio.ImageSaveOpts{
                        config.image_save_opts[0],
                        config.image_save_opts[1],
                    };
                    opts_with_ch[0].channels = num_ch;
                    opts_with_ch[1].channels = num_ch;

                    var r_config = tcfg.getRasterConfig(.bench);
                    r_config.save_strategy = .disk;
                    r_config.image_save_opts = &opts_with_ch;
                    const case_name = try minsuite.calcMinCaseName(
                        allocator,
                        mt,
                        st,
                        sc,
                    );
                    defer allocator.free(case_name);
                    const case_out_dir = try std.fs.path.join(
                        allocator,
                        &[_][]const u8{
                            out_dir,
                            "sphere200",
                            "base",
                            case_name,
                        },
                    );
                    defer allocator.free(case_out_dir);

                    var result = try common.runBenchmarkQuietWithImageOut(
                        u8,
                        allocator,
                        io,
                        mt,
                        st,
                        sc,
                        null,
                        data_dir,
                        render_defaults_sphere,
                        texture_grey,
                        texture_rgb,
                        r_config,
                        "",
                        case_out_dir,
                    );
                    result.deinit(allocator);
                }
            }
        }
    }

    std.debug.print("Generating MIN Gold Data (sphere200multicull)...\n", .{});
    for (mesh_types) |mt| {
        for (shader_types) |st| {
            for (samp_cfgs) |sc| {
                const mesh_name = policy.meshName(.benchmark_data, mt);
                const data_dir = try std.fmt.allocPrint(
                    allocator,
                    "data/min/{s}_sphere200",
                    .{mesh_name},
                );
                defer allocator.free(data_dir);

                const is_rgb = (st == .tex8_rgb or st == .nodal_rgb);
                const is_allowed_rgb = (st == .nodal_rgb) or
                    (st == .tex8_rgb and
                        sc.sample == .cubic_catmull_rom and
                        sc.mode == .lut_lerp);

                if (is_rgb and !is_allowed_rgb) continue;

                if (common.shouldRun(
                    .{ .run = .all, .skip_quad4ibi_sphere = true },
                    mt,
                    st,
                    sc,
                    data_dir,
                )) {
                    const num_ch: u8 = if (is_rgb) 3 else 1;
                    var opts_with_ch = [_]iio.ImageSaveOpts{
                        config.image_save_opts[0],
                        config.image_save_opts[1],
                    };
                    opts_with_ch[0].channels = num_ch;
                    opts_with_ch[1].channels = num_ch;

                    var result = try minsuite.runSphere200MultiCullQuiet(
                        allocator,
                        io,
                        mt,
                        st,
                        sc,
                        data_dir,
                        pixel_num_sphere,
                        texture_grey,
                        texture_rgb,
                        .{
                            .save_strategy = .disk,
                            .image_save_opts = &opts_with_ch,
                        },
                        out_dir ++ "/sphere200multicull",
                        0.75,
                    );
                    result.deinit(allocator);
                }
            }
        }
    }

    std.debug.print("Generating MIN Gold Data (multimesh)...\n", .{});
    const multi_dir_paths = [_][]const u8{
        "data/min/tri3_twoelems/",
        "data/min/tri6_twoelems/",
        "data/min/quad4_twoelems/",
        "data/min/quad8_twoelems/",
        "data/min/quad9_twoelems/",
    };

    try gengold.runMultimeshGenerationExt(
        allocator,
        io,
        config,
        out_dir ++ "/multimesh/base",
        &multi_dir_paths,
        pixel_num_multi,
    );
    try gengold.runMultimeshMixedGenerationExt(
        allocator,
        io,
        config,
        out_dir ++ "/multimesh/allelem_allshade",
        &multi_dir_paths,
        pixel_num_multi,
    );
    try gengold.runMultimeshMixedRGBGenerationExt(
        allocator,
        io,
        config,
        out_dir ++ "/multimesh/allelem_allshade_rgb",
        &multi_dir_paths,
        pixel_num_multi,
    );

    std.debug.print("Done. MIN gold references established.\n", .{});
}
