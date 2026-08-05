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
const common = @import("dev_support/benchcommon.zig");
const minsuite = @import("dev_support/minsuite.zig");
const policy = @import("dev_support/testpolicy.zig");
const tcfg = @import("dev_support/testconfig.zig");
const tests = @import("dev_support/tests.zig");
const mo = @import("riley/zig/meshpipeline.zig");
const gk = @import("riley/zig/geometrykernels.zig");
const iio = @import("riley/zig/imageio.zig");
const texops = @import("riley/zig/textureops.zig");
const Rotation = @import("riley/zig/rotation.zig").Rotation;

const simd_on = buildconfig.config.simd == .on;

comptime {
    if (!simd_on) {
        @compileError(
            "src/test_min.zig requires .simd = .on. " ++
                "MIN scalar gold/test orchestration is not implemented.",
        );
    }
}

test "MIN Suite: sphere200 and multimesh" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

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

    const gold_dir = comptime policy.goldRoot(.min);
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
    var total_fails: usize = 0;

    if (simd_on) {
        std.debug.print("\nRunning MIN Suite sphere200/base tests...\n", .{});
        for (mesh_types) |mt| {
            for (shader_types) |st| {
                for (samp_cfgs) |sc| {
                    const folder_name = policy.meshName(
                        .benchmark_data,
                        mt,
                    );
                    const data_dir = try std.fmt.allocPrint(
                        allocator,
                        "data/min/{s}_sphere200",
                        .{folder_name},
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
                        var r_config = tcfg.getRasterConfig(.bench);
                        r_config.save_strategy = .memory;
                        r_config.image_save_opts = &[_]iio.ImageSaveOpts{};

                        const case_name = try minsuite.calcMinCaseName(
                            allocator,
                            mt,
                            st,
                            sc,
                        );
                        defer allocator.free(case_name);
                        var result = try common.runBenchmarkQuiet(
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
                        );
                        defer result.deinit(allocator);

                        const gold_case_dir = try std.fs.path.join(
                            allocator,
                            &[_][]const u8{
                                gold_dir,
                                "sphere200",
                                "base",
                                case_name,
                            },
                        );
                        defer allocator.free(gold_case_dir);
                        const gold_fname = try tests.findGoldPath(
                            allocator,
                            io,
                            gold_case_dir,
                            0,
                            0,
                            0,
                            is_rgb,
                        );
                        defer allocator.free(gold_fname);

                        const channels: usize = if (is_rgb) 3 else 1;
                        tests.compareNDArrayToGold(
                            allocator,
                            io,
                            &result.image.?,
                            0,
                            0,
                            0,
                            channels,
                            gold_fname,
                            tcfg.REL_TOL,
                            tcfg.ABS_TOL,
                        ) catch |err| {
                            try tests.saveComparisonArtifactsFromResult(
                                allocator,
                                io,
                                "fails",
                                case_name,
                                &result.image.?,
                                0,
                                0,
                                0,
                                gold_fname,
                                channels,
                            );
                            if (err == error.PixelMismatch) {
                                total_fails += 1;
                                continue;
                            }
                            return err;
                        };
                    }
                }
            }
        }

        std.debug.print("Running MIN Suite sphere200multicull tests...\n", .{});
        for (mesh_types) |mt| {
            for (shader_types) |st| {
                for (samp_cfgs) |sc| {
                    const folder_name = policy.meshName(
                        .benchmark_data,
                        mt,
                    );
                    const data_dir = try std.fmt.allocPrint(
                        allocator,
                        "data/min/{s}_sphere200",
                        .{folder_name},
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
                        var r_config = tcfg.getRasterConfig(.bench);
                        r_config.save_strategy = .memory;
                        r_config.image_save_opts = &[_]iio.ImageSaveOpts{};

                        const case_name = try minsuite.calcMinCaseName(
                            allocator,
                            mt,
                            st,
                            sc,
                        );
                        defer allocator.free(case_name);
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
                            r_config,
                            "",
                            0.75,
                        );
                        defer result.deinit(allocator);

                        const gold_case_dir = try std.fs.path.join(
                            allocator,
                            &[_][]const u8{
                                gold_dir,
                                "sphere200multicull",
                                case_name,
                            },
                        );
                        defer allocator.free(gold_case_dir);
                        const gold_fname = try tests.findGoldPath(
                            allocator,
                            io,
                            gold_case_dir,
                            0,
                            0,
                            0,
                            is_rgb,
                        );
                        defer allocator.free(gold_fname);

                        const channels: usize = if (is_rgb) 3 else 1;
                        tests.compareNDArrayToGold(
                            allocator,
                            io,
                            &result.image.?,
                            0,
                            0,
                            0,
                            channels,
                            gold_fname,
                            tcfg.REL_TOL,
                            tcfg.ABS_TOL,
                        ) catch |err| {
                            try tests.saveComparisonArtifactsFromResult(
                                allocator,
                                io,
                                "fails",
                                case_name,
                                &result.image.?,
                                0,
                                0,
                                0,
                                gold_fname,
                                channels,
                            );
                            if (err == error.PixelMismatch) {
                                total_fails += 1;
                                continue;
                            }
                            return err;
                        };
                    }
                }
            }
        }
    } else {
        std.debug.print(
            "\nSkipping MIN Suite sphere200 tests with .simd = .off...\n",
            .{},
        );
    }

    std.debug.print("Running MIN Suite multimesh tests...\n", .{});
    const multi_dir_paths = [_][]const u8{
        "data/min/tri3_twoelems/",
        "data/min/tri6_twoelems/",
        "data/min/quad4_twoelems/",
        "data/min/quad8_twoelems/",
        "data/min/quad9_twoelems/",
    };

    {
        tests.runMultimeshTestExt(
            allocator,
            io,
            gold_dir ++ "/multimesh/base",
            &multi_dir_paths,
            pixel_num_multi,
            tcfg.REL_TOL,
            tcfg.ABS_TOL,
        ) catch |err| {
            total_fails += 1;
            if (err != error.PixelMismatch) {
                return err;
            }
        };
    }

    {
        tests.runMultimeshMixedTestExt(
            allocator,
            io,
            gold_dir ++ "/multimesh/allelem_allshade",
            &multi_dir_paths,
            pixel_num_multi,
            tcfg.REL_TOL,
            tcfg.ABS_TOL,
        ) catch |err| {
            total_fails += 1;
            if (err != error.PixelMismatch) {
                return err;
            }
        };
    }

    {
        tests.runMultimeshMixedRGBTestExt(
            allocator,
            io,
            gold_dir ++ "/multimesh/allelem_allshade_rgb",
            &multi_dir_paths,
            pixel_num_multi,
            tcfg.REL_TOL,
            tcfg.ABS_TOL,
        ) catch |err| {
            total_fails += 1;
            if (err != error.PixelMismatch) {
                return err;
            }
        };
    }

    if (total_fails != 0) {
        std.debug.print(
            "MIN Suite found {d} failing cases.\n",
            .{total_fails},
        );
        return error.TestUnexpectedResult;
    }

    std.debug.print("MIN Suite tests passed.\n", .{});
}
