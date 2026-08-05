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
const policy = @import("dev_support/testpolicy.zig");
const testcommon = @import("dev_support/tests.zig");
const tcfg = @import("dev_support/testconfig.zig");
const buildconfig = @import("riley/zig/buildconfig.zig");
const F = buildconfig.F;
const cfg = buildconfig.config;
const gk = @import("riley/zig/geometrykernels.zig");
const mo = @import("riley/zig/meshpipeline.zig");
const iio = @import("riley/zig/imageio.zig");
const texops = @import("riley/zig/textureops.zig");

const config = common.BenchConfig{ .run = .all };
const simd_on = cfg.simd == .on;
const impl_suffix = if (simd_on) "_simd" else "_scalar";

test "Unified Benchmark Tests" {
    const outer_alloc = std.heap.page_allocator;

    const io = std.testing.io;

    const texture_grey = try iio.loadImage(
        u8,
        1,
        outer_alloc,
        io,
        "texture/speckle.bmp",
        .bmp,
    );
    defer texture_grey.deinit(outer_alloc);
    const texture_rgb = try iio.loadImage(
        u8,
        3,
        outer_alloc,
        io,
        "texture/speckle_rgb.bmp",
        .bmp,
    );
    defer texture_rgb.deinit(outer_alloc);

    const cases = [_]struct {
        name: []const u8,
        data_name: []const u8,
        gold_dir: []const u8,
        out_dir: []const u8,
        is_sphere: bool = false,
        fov_scale: F = 1.0,
        sub_sample: u32 = 2,
        skip_quad4ibi_sphere: bool = false,
    }{
        .{
            .name = "fullraster",
            .data_name = "fullraster",
            .gold_dir = policy.goldRoot(.fullscreen),
            .out_dir = "out/fullraster",
        },
        .{
            .name = "fullraster_ssaa1",
            .data_name = "fullraster",
            .gold_dir = policy.goldRoot(.fullscreen_ssaa1),
            .out_dir = "out/fullraster_ssaa1",
            .sub_sample = 1,
        },
        .{
            .name = "geom",
            .data_name = "geom",
            .gold_dir = policy.goldRoot(.fullscreen),
            .out_dir = "out/geom",
        },
        .{
            .name = "sphere2000",
            .data_name = "sphere2000",
            .gold_dir = policy.goldRoot(.sphere2000),
            .out_dir = "out/sphere2000",
            .is_sphere = true,
            .skip_quad4ibi_sphere = true,
        },
        .{
            .name = "sphere2000_ssaa1",
            .data_name = "sphere2000",
            .gold_dir = policy.goldRoot(.sphere2000_ssaa1),
            .out_dir = "out/sphere2000_ssaa1",
            .is_sphere = true,
            .sub_sample = 1,
            .skip_quad4ibi_sphere = true,
        },
        .{
            .name = "sphere2000zoom",
            .data_name = "sphere2000",
            .gold_dir = policy.goldRoot(.sphere2000zoom),
            .out_dir = "out/sphere2000zoom",
            .is_sphere = true,
            .fov_scale = 0.5,
            .skip_quad4ibi_sphere = true,
        },
    };

    const pixel_num = [_]u32{ 800, 500 };
    const render_defaults_base = common.BenchRenderDefaults{
        .pixels_num = pixel_num,
        .sub_sample = 2,
        .focal_leng = 50.0e-3,
        .pixels_size = .{ 5.3e-6, 5.3e-6 },
        .fov_scale = 1.0,
        .rot = @import("riley/zig/rotation.zig").Rotation.init(0, 0, 0),
    };

    const mesh_types = std.enums.values(gk.MeshType);
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
    var printed_tri3opt_geom_warning = false;

    std.debug.print("Running Unified Benchmark Tests...\n", .{});

    var arena = std.heap.ArenaAllocator.init(outer_alloc);
    defer arena.deinit();
    const aa = arena.allocator();

    for (cases) |cc| {
        std.debug.print("\n--- Testing benchmark: {s} ---\n", .{cc.name});

        for (mesh_types) |mt| {
            for (shader_types) |st| {
                for (samp_cfgs) |sc| {
                    _ = arena.reset(.free_all);

                    if (policy.shouldSkipBenchGeomTest(mt, cc.name)) {
                        if (!printed_tri3opt_geom_warning) {
                            std.debug.print(
                                "WARNING: {s}\n",
                                .{policy.benchGeomSkipWarning()},
                            );
                            printed_tri3opt_geom_warning = true;
                        }
                        continue;
                    }

                    const folder_name = policy.meshName(
                        .benchmark_data,
                        mt,
                    );
                    const data_dir = try std.fmt.allocPrint(
                        aa,
                        "data/bench/{s}_{s}",
                        .{ folder_name, cc.data_name },
                    );

                    const run_config = if (cc.is_sphere)
                        common.BenchConfig{
                            .run = .all,
                            .skip_quad4ibi_sphere = cc.skip_quad4ibi_sphere,
                        }
                    else
                        config;

                    if (common.shouldRun(run_config, mt, st, sc, data_dir)) {
                        var r_config = tcfg.getRasterConfig(.bench);
                        // Bench tests compare the returned in-memory image directly
                        // against gold, so avoid the disk-save path here.
                        r_config.save_strategy = .memory;
                        r_config.image_save_opts = &[_]iio.ImageSaveOpts{};

                        const case_name = try common.calcCaseName(
                            aa,
                            if (cc.is_sphere)
                                policy.sphereGoldCaseMeshType(mt)
                            else
                                mt,
                            st,
                            sc,
                            null,
                            cc.fov_scale,
                        );

                        std.debug.print("Testing {s}/{s} ... ", .{ cc.name, case_name });

                        // 1. Run benchmark
                        var result = try common.runBenchmarkQuiet(
                            u8,
                            outer_alloc,
                            io,
                            mt,
                            st,
                            sc,
                            null,
                            data_dir,
                            .{
                                .pixels_num = render_defaults_base.pixels_num,
                                .sub_sample = cc.sub_sample,
                                .focal_leng = render_defaults_base.focal_leng,
                                .pixels_size = render_defaults_base.pixels_size,
                                .fov_scale = cc.fov_scale,
                                .rot = render_defaults_base.rot,
                            },
                            texture_grey,
                            texture_rgb,
                            r_config,
                            cc.out_dir,
                        );

                        // 2. Map filenames
                        const is_rgb = (st == .nodal_rgb or st == .tex8_rgb);
                        const channels: usize = if (is_rgb) 3 else 1;

                        const gold_dir_case = try std.fs.path.join(aa, &[_][]const u8{ cc.gold_dir, case_name });
                        const gold_path = try testcommon.findGoldPath(
                            aa,
                            io,
                            gold_dir_case,
                            0,
                            0,
                            0,
                            is_rgb,
                        );

                        // 3. Compare in-memory result to gold
                        testcommon.compareNDArrayToGold(
                            outer_alloc,
                            io,
                            &result.image.?,
                            0,
                            0,
                            0,
                            channels,
                            gold_path,
                            tcfg.REL_TOL,
                            tcfg.ABS_TOL,
                        ) catch |err| {
                            if (err == error.PixelMismatch) {
                                std.debug.print("MISMATCH!\n", .{});
                                const fail_dir_name = try std.fmt.allocPrint(
                                    aa,
                                    "bench_{s}_{s}{s}",
                                    .{ cc.name, case_name, impl_suffix },
                                );
                                try testcommon.saveComparisonArtifactsFromResult(
                                    aa,
                                    io,
                                    "fails",
                                    fail_dir_name,
                                    &result.image.?,
                                    0,
                                    0,
                                    0,
                                    gold_path,
                                    channels,
                                );
                                total_fails += 1;
                                result.deinit(outer_alloc);
                                continue;
                            }
                            result.deinit(outer_alloc);
                            return err;
                        };

                        std.debug.print("MATCHED\n", .{});
                        result.deinit(outer_alloc);
                    }
                }
            }
        }
    }

    if (total_fails == 0) {
        std.debug.print("\nALL BENCHMARK TESTS PASSED!\n", .{});
    } else {
        std.debug.print("\n{d} TESTS FAILED!\n", .{total_fails});
        try std.testing.expect(total_fails == 0);
    }
}
