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

    const out_dir_base = policy.goldRoot(.fullscreen);
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

    std.debug.print(
        "Generating Unified Fullscreen Gold data to {s}/...\n",
        .{out_dir_base},
    );

    inline for (mesh_types) |mt| {
        inline for (shader_types) |st| {
            inline for (samp_cfgs) |sc| {
                const mesh_name = comptime policy.meshName(
                    .benchmark_data,
                    mt,
                );
                const data_dir = comptime "data/bench/" ++ mesh_name ++ "_fullraster";
                if (common.shouldRun(.{ .run = .all }, mt, st, sc, data_dir)) {
                    const case_name = try common.calcCaseName(
                        aa,
                        mt,
                        st,
                        sc,
                        null,
                        1.0,
                    );
                    std.debug.print("Rendering reference: {s}\n", .{case_name});

                    // We generate gold from the minimal 'fullraster' dataset
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
                        &[_][]const u8{ out_dir_base, case_name },
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
                        render_defaults,
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

    std.debug.print("\nDone. Unified gold references established.\n", .{});
}
