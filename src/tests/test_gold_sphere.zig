// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const std = @import("std");
const common = @import("../dev_support/benchcommon.zig");
const orch = @import("../dev_support/orchestration.zig");
const policy = @import("../dev_support/testpolicy.zig");
const testcommon = @import("../dev_support/tests.zig");
const tcfg = @import("../dev_support/testconfig.zig");
const buildconfig = @import("../riley/zig/buildconfig.zig");
const cfg = buildconfig.config;
const gk = @import("../riley/zig/geometrykernels.zig");
const iio = @import("../riley/zig/imageio.zig");
const texops = @import("../riley/zig/textureops.zig");
const Rotation = @import("../riley/zig/rotation.zig").Rotation;

const config = common.BenchConfig{ .run = .all };
const simd_on = cfg.simd == .on;

test "Sphere Gold Tests" {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

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

    const fails_root = "fails";
    const impl_suffix = if (simd_on) "_simd" else "_scalar";

    const cases = [_]struct { ds: []const u8, gold: []const u8, out: []const u8 }{
        .{
            .ds = "sphere2000",
            .gold = policy.goldRoot(.sphere2000),
            .out = "out-sphere2000",
        },
    };

    const pixel_num = [_]u32{ 800, 500 };
    const render_defaults = common.BenchRenderDefaults{
        .pixels_num = pixel_num,
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
        .{ .sample = .linear, .mode = .direct },
        .{ .sample = .cubic_catmull_rom, .mode = .direct },
        .{ .sample = .cubic_catmull_rom, .mode = .lut_lerp },
        .{ .sample = .quintic_bspline, .mode = .direct },
        .{ .sample = .quintic_bspline, .mode = .lut_lerp },
    };

    var total_fails: usize = 0;

    std.debug.print("Running Sphere Gold Tests with .simd = .{s}...\n", .{
        if (simd_on) "on" else "off",
    });

    inline for (cases) |c| {
        std.debug.print("\n--- Testing dataset: {s} ---\n", .{c.ds});

        inline for (mesh_types) |mt| {
            inline for (shader_types) |st| {
                inline for (samp_cfgs) |sc| {
                    const mesh_name = policy.meshName(
                        .benchmark_data,
                        mt,
                    );
                    const data_dir = try std.fmt.allocPrint(
                        allocator,
                        "data/bench/{s}_{s}",
                        .{ mesh_name, c.ds },
                    );
                    defer allocator.free(data_dir);

                    if (common.shouldRun(.{ .run = .all }, mt, st, sc, data_dir)) {
                        const case_name = if (st == .tex8_grey or st == .tex8_rgb)
                            try std.fmt.allocPrint(
                                allocator,
                                "{s}_{s}_{s}_{s}",
                                .{
                                    mesh_name,
                                    @tagName(st),
                                    @tagName(sc.sample),
                                    @tagName(sc.mode),
                                },
                            )
                        else
                            try std.fmt.allocPrint(
                                allocator,
                                "{s}_{s}",
                                .{ mesh_name, @tagName(st) },
                            );
                        defer allocator.free(case_name);

                        const gold_mesh_name = policy.meshName(
                            .sphere_gold_case,
                            mt,
                        );
                        const gold_case_name = if (st == .tex8_grey or st == .tex8_rgb)
                            try std.fmt.allocPrint(
                                allocator,
                                "{s}_{s}_{s}_{s}",
                                .{ gold_mesh_name, @tagName(st), @tagName(sc.sample), @tagName(sc.mode) },
                            )
                        else
                            try std.fmt.allocPrint(
                                allocator,
                                "{s}_{s}",
                                .{ gold_mesh_name, @tagName(st) },
                            );
                        defer allocator.free(gold_case_name);

                        // 1. Run benchmark
                        var r_config = tcfg.getRasterConfig(.bench);
                        r_config.save_strategy = if (c.out.len > 0) .both else .memory;

                        const test_dir_case = try std.fs.path.join(
                            allocator,
                            &[_][]const u8{ c.out, case_name },
                        );
                        defer allocator.free(test_dir_case);

                        var result = try common.runBenchmarkQuietWithImageOut(
                            u8,
                            allocator,
                            io,
                            mt,
                            st,
                            sc,
                            null,
                            data_dir,
                            render_defaults,
                            texture_grey,
                            texture_rgb,
                            r_config,
                            "",
                            test_dir_case,
                        );
                        result.deinit(allocator);

                        // 2. Map filenames
                        const is_rgb = (st == .nodal_rgb or st == .tex8_rgb);
                        const channels: usize = if (is_rgb) 3 else 1;

                        const test_path = try testcommon.findGoldPath(
                            allocator,
                            io,
                            test_dir_case,
                            0,
                            0,
                            0,
                            is_rgb,
                        );
                        defer allocator.free(test_path);

                        const gold_dir_case = try std.fs.path.join(allocator, &[_][]const u8{ c.gold, gold_case_name });
                        defer allocator.free(gold_dir_case);
                        const gold_path = try testcommon.findGoldPath(
                            allocator,
                            io,
                            gold_dir_case,
                            0,
                            0,
                            0,
                            is_rgb,
                        );
                        defer allocator.free(gold_path);

                        // 3. Load and Compare
                        const t_arr_res = common.loadNDArray(
                            allocator,
                            io,
                            test_path,
                            channels,
                            false,
                        );
                        if (t_arr_res) |t_arr| {
                            var t_mut = t_arr;
                            defer {
                                allocator.free(t_mut.slice);
                                t_mut.deinit(allocator);
                            }

                            const g_arr_res = common.loadNDArray(
                                allocator,
                                io,
                                gold_path,
                                channels,
                                false,
                            );
                            if (g_arr_res) |g_arr| {
                                var g_mut = g_arr;
                                defer {
                                    allocator.free(g_mut.slice);
                                    g_mut.deinit(allocator);
                                }

                                var diff_count: usize = 0;
                                for (t_mut.slice, 0..) |v_t, ii| {
                                    if (@abs(v_t - g_mut.slice[ii]) > tcfg.REL_TOL)
                                        diff_count += 1;
                                }

                                if (diff_count != 0) {
                                    total_fails += 1;

                                    const fail_dir_name = try std.fmt.allocPrint(
                                        allocator,
                                        "all_{s}_{s}{s}",
                                        .{ c.ds, case_name, impl_suffix },
                                    );
                                    defer allocator.free(fail_dir_name);
                                    try testcommon.saveComparisonArtifactsFromImages(
                                        allocator,
                                        io,
                                        fails_root,
                                        fail_dir_name,
                                        &t_mut,
                                        &g_mut,
                                    );
                                }
                            } else |err| {
                                std.debug.print(
                                    "GOLD LOAD ERROR: {s} ({s})\n",
                                    .{ gold_path, @errorName(err) },
                                );
                                total_fails += 1;
                            }
                        } else |err| {
                            std.debug.print(
                                "TEST LOAD ERROR: {s} ({s})\n",
                                .{ test_path, @errorName(err) },
                            );
                            total_fails += 1;
                        }
                    }
                }
            }
        }
    }

    if (total_fails == 0) {
        std.debug.print("\nALL SPHERE GOLD TESTS PASSED!\n", .{});
    } else {
        std.debug.print(
            "\n{d} TESTS FAILED! (Diagnostics in ./fails/)\n",
            .{total_fails},
        );
        try std.testing.expect(total_fails == 0);
    }
}
