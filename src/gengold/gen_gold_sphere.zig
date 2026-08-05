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
const tcfg = @import("../dev_support/testconfig.zig");
const mo = @import("../riley/zig/meshpipeline.zig");
const gk = @import("../riley/zig/geometrykernels.zig");
const iio = @import("../riley/zig/imageio.zig");
const texops = @import("../riley/zig/textureops.zig");
const buildconfig = @import("../riley/zig/buildconfig.zig");
const Rotation = @import("../riley/zig/rotation.zig").Rotation;

const cfg = buildconfig.config;
const F = buildconfig.F;

const simd_on = cfg.simd == .on;

pub fn main(init: std.process.Init) !void {
    @setEvalBranchQuota(buildconfig.comptime_eval_branch_quota);
    const io = init.io;
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const aa = arena.allocator();

    const texture_grey = try iio.loadImage(
        u8,
        1,
        aa,
        io,
        "texture/speckle.bmp",
        .bmp,
    );
    const texture_rgb = try iio.loadImage(
        u8,
        3,
        aa,
        io,
        "texture/speckle_rgb.bmp",
        .bmp,
    );

    const pixel_num = [_]u32{ 800, 500 };
    const render_defaults_base = common.BenchRenderDefaults{
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

    const cases = [_]struct {
        ds: []const u8,
        out: []const u8,
        fov_scale: F = 1.0,
    }{
        .{
            .ds = "sphere2000",
            .out = policy.goldRoot(.sphere2000),
        },
        .{
            .ds = "sphere2000",
            .out = policy.goldRoot(.sphere2000zoom),
            .fov_scale = 0.5,
        },
    };

    std.debug.print("Generating Sphere Gold data with .simd = .{s}...\n", .{
        if (simd_on) "on" else "off",
    });

    for (cases) |case| {
        inline for (mesh_types) |mt| {
            inline for (shader_types) |st| {
                inline for (samp_cfgs) |sc| {
                    const mesh_name = comptime policy.meshName(
                        .benchmark_data,
                        mt,
                    );
                    const data_dir = try std.fmt.allocPrint(
                        aa,
                        "data/bench/{s}_{s}",
                        .{ mesh_name, case.ds },
                    );

                    if (common.shouldRun(
                        .{ .run = .all, .skip_quad4ibi_sphere = true },
                        mt,
                        st,
                        sc,
                        data_dir,
                    )) {
                        const case_name = try common.calcCaseName(
                            aa,
                            policy.sphereGoldCaseMeshType(mt),
                            st,
                            sc,
                            null,
                            case.fov_scale,
                        );

                        std.debug.print(
                            "Rendering reference: {s}/{s}\n",
                            .{ case.out, case_name },
                        );

                        var r_config = tcfg.getRasterConfig(.bench);
                        r_config.save_strategy = .disk;
                        const is_rgb = (st == .nodal_rgb or st == .tex8_rgb);
                        r_config.image_save_opts = if (is_rgb)
                            &[_]iio.ImageSaveOpts{
                                .{ .format = .fimg, .bits = null, .scaling = .none, .channels = 3 },
                                .{ .format = .bmp, .bits = 8, .scaling = .auto, .channels = 3 },
                            }
                        else
                            &[_]iio.ImageSaveOpts{
                                .{ .format = .fimg, .bits = null, .scaling = .none },
                                .{ .format = .bmp, .bits = 8, .scaling = .auto },
                            };
                        const case_out_dir = try std.fs.path.join(
                            aa,
                            &[_][]const u8{ case.out, case_name },
                        );
                        _ = try common.runBenchmarkQuietWithImageOut(
                            u8,
                            aa,
                            io,
                            mt,
                            st,
                            sc,
                            null,
                            data_dir,
                            .{
                                .pixels_num = render_defaults_base.pixels_num,
                                .sub_sample = render_defaults_base.sub_sample,
                                .focal_leng = render_defaults_base.focal_leng,
                                .pixels_size = render_defaults_base.pixels_size,
                                .fov_scale = case.fov_scale,
                                .rot = render_defaults_base.rot,
                            },
                            texture_grey,
                            texture_rgb,
                            r_config,
                            "",
                            case_out_dir,
                        );
                    }
                }
            }
        }
    }

    std.debug.print("\nDone. Sphere gold references established.\n", .{});
}
