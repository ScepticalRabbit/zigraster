// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const std = @import("std");
const benchargs = @import("dev_support/benchargs.zig");
const benchstats = @import("dev_support/benchstats.zig");
const common = @import("dev_support/benchcommon.zig");
const tcfg = @import("dev_support/testconfig.zig");
const buildconfig = @import("riley/zig/buildconfig.zig");
const rastcfg = @import("riley/zig/rasterconfig.zig");
const riley = @import("riley/zig/riley.zig");
const gk = @import("riley/zig/geometrykernels.zig");
const iio = @import("riley/zig/imageio.zig");
const texops = @import("riley/zig/textureops.zig");
const Rotation = @import("riley/zig/rotation.zig").Rotation;
const F = buildconfig.F;

const DEFAULT_OUT_DIR = "out/bench_stats_sphere2000zoom";
const DEFAULT_IMAGE_OUT_DIR = "out/bench_images_sphere2000zoom";
const DEFAULT_DATA_DIR_SUFFIX = "sphere2000";
const DEFAULT_PIXELS_NUM = [2]u32{ 1600, 1000 };
const DEFAULT_SUB_SAMPLE: u32 = 1;
const DEFAULT_FOCAL_LENG: F = @floatCast(50.0e-3);
const DEFAULT_PIXELS_SIZE = [2]F{
    @floatCast(5.3e-6),
    @floatCast(5.3e-6),
};
const DEFAULT_FOV_SCALE: F = 0.5;
const DEFAULT_TEX_GREY_PATH = "texture/speckle.bmp";
const DEFAULT_TEX_RGB_PATH = "texture/speckle_rgb.bmp";
const DEFAULT_ROT = Rotation.init(0, 0, 0);

pub fn main(init: std.process.Init) !void {
    const outer_alloc = init.gpa;

    var base_raster_config = tcfg.getRasterConfig(.bench);
    base_raster_config.image_save_opts = &[_]iio.ImageSaveOpts{
        .{ .format = .bmp, .bits = 8, .scaling = .auto },
    };
    var default_bench_args = benchargs.defaultBenchArgs(
        DEFAULT_OUT_DIR,
        base_raster_config,
    );
    default_bench_args.image_out_dir = DEFAULT_IMAGE_OUT_DIR;
    default_bench_args.pixels_num = DEFAULT_PIXELS_NUM;
    default_bench_args.sub_sample = DEFAULT_SUB_SAMPLE;

    const bench_args = try benchargs.parseArgsWithDefaults(
        init.minimal.args.vector,
        default_bench_args,
    );
    var threaded_io = riley.getThreadedIo(
        outer_alloc,
        init.minimal,
        bench_args.total_threads,
    );
    defer threaded_io.deinit();
    const io = threaded_io.io();
    const render_defaults = common.BenchRenderDefaults{
        .pixels_num = bench_args.pixels_num,
        .sub_sample = bench_args.sub_sample,
        .focal_leng = DEFAULT_FOCAL_LENG,
        .pixels_size = DEFAULT_PIXELS_SIZE,
        .fov_scale = DEFAULT_FOV_SCALE,
        .rot = DEFAULT_ROT,
    };
    const texture_grey = try iio.loadImage(
        u8,
        1,
        outer_alloc,
        io,
        DEFAULT_TEX_GREY_PATH,
        .bmp,
    );
    defer texture_grey.deinit(outer_alloc);
    const texture_rgb = try iio.loadImage(
        u8,
        3,
        outer_alloc,
        io,
        DEFAULT_TEX_RGB_PATH,
        .bmp,
    );
    defer texture_rgb.deinit(outer_alloc);

    const mesh_types = comptime std.enums.values(gk.MeshType);
    const shader_types = [_]common.ShaderType{
        .nodal_grey,
        .nodal_rgb,
        .tex8_grey,
        .tex8_rgb,
    };
    const tex_func_shader_types = [_]common.ShaderType{
        .func,
        .func_rgb,
    };
    const samp_cfgs = [_]texops.TextureSampleConfig{
        .{ .sample = .linear, .mode = .direct },
        .{ .sample = .cubic_catmull_rom, .mode = .lut_lerp },
        .{ .sample = .cubic_mitchell_netravali, .mode = .lut_lerp },
        .{ .sample = .lanczos3, .mode = .lut_lerp },
        .{ .sample = .cubic_bspline, .mode = .lut_lerp },
        .{ .sample = .quintic_bspline, .mode = .lut_lerp },
    };
    const tex_func_cases = [_]common.TexFuncCase{
        .{ .builtin = .constant, .coord_mode = .param },
        .{ .builtin = .constant, .coord_mode = .uv },
        .{ .builtin = .sinusoidal, .coord_mode = .param },
        .{ .builtin = .sinusoidal, .coord_mode = .uv },
    };

    var stats = try benchstats.BenchStatsCollector.init(
        outer_alloc,
        bench_args.runs,
    );
    defer stats.deinit(outer_alloc);

    std.debug.print(
        "Starting Sphere 2000 Zoom Benchmark ({d}x{d}, {d} runs per case, scale={d:.1})...\n",
        .{
            bench_args.pixels_num[0],
            bench_args.pixels_num[1],
            bench_args.runs,
            DEFAULT_FOV_SCALE,
        },
    );

    const bench_raster_config = benchargs.applyRasterConfig(
        base_raster_config,
        bench_args,
    );
    const actual_tile_size = common.calcActualTileSize(
        bench_raster_config,
        bench_args.pixels_num,
        bench_args.sub_sample,
        0,
    );
    const render_group_workers = [_]u16{bench_args.total_threads};
    try common.writeBenchmarkConfig(
        outer_alloc,
        io,
        bench_args.out_dir,
        bench_args.image_out_dir,
        "bench_sphere2000zoom.zig",
        init.minimal.args.vector,
        bench_args.subpixel_center_map,
        bench_raster_config,
        render_group_workers[0..],
        bench_args.pixels_num,
        bench_args.sub_sample,
        bench_args.runs,
        DEFAULT_FOV_SCALE,
        actual_tile_size,
    );

    for (mesh_types) |mt| {
        for (shader_types) |st| {
            for (samp_cfgs) |sc| {
                const samp_cfg = if (st == .tex8_grey or st == .tex8_rgb) sc else null;
                const case_name = try common.calcCaseName(
                    outer_alloc,
                    mt,
                    st,
                    samp_cfg,
                    null,
                    DEFAULT_FOV_SCALE,
                );
                defer outer_alloc.free(case_name);

                std.debug.print("Case: {s}\n", .{case_name});

                var case_samples = try benchstats.CaseSamples.init(
                    outer_alloc,
                    bench_args.runs,
                );
                defer case_samples.deinit(outer_alloc);

                var data_dir_buf: [256]u8 = undefined;
                const data_dir = try std.fmt.bufPrint(
                    &data_dir_buf,
                    "data/bench/{s}_{s}",
                    .{ @tagName(mt), DEFAULT_DATA_DIR_SUFFIX },
                );
                for (0..bench_args.runs) |rr| {
                    const run_out_dir_base = if (bench_args.save_strategy == .disk or
                        bench_args.save_strategy == .both)
                        bench_args.out_dir
                    else
                        "";
                    const raster_config =
                        benchargs.applyRasterConfig(
                            base_raster_config,
                            bench_args,
                        );

                    var res = try common.runBenchmarkWithImageOut(
                        u8,
                        outer_alloc,
                        io,
                        mt,
                        st,
                        samp_cfg,
                        null,
                        data_dir,
                        render_defaults,
                        texture_grey,
                        texture_rgb,
                        raster_config,
                        run_out_dir_base,
                        bench_args.image_out_dir,
                    );
                    defer res.deinit(outer_alloc);

                    try stats.appendRunResult(
                        outer_alloc,
                        rr,
                        case_name,
                        mt,
                        st,
                        samp_cfg,
                        null,
                        res,
                    );
                    try stats.writeRunCSV(
                        outer_alloc,
                        io,
                        bench_args.out_dir,
                        rr,
                    );
                    case_samples.record(rr, res);
                }

                try stats.appendCaseStats(
                    outer_alloc,
                    case_name,
                    mt,
                    st,
                    samp_cfg,
                    null,
                    &case_samples,
                );
            }
        }

        for (tex_func_shader_types) |st| {
            for (tex_func_cases) |tex_func_case| {
                const case_name = try common.calcCaseName(
                    outer_alloc,
                    mt,
                    st,
                    null,
                    tex_func_case,
                    DEFAULT_FOV_SCALE,
                );
                defer outer_alloc.free(case_name);

                std.debug.print("Case: {s}\n", .{case_name});

                var case_samples = try benchstats.CaseSamples.init(
                    outer_alloc,
                    bench_args.runs,
                );
                defer case_samples.deinit(outer_alloc);

                var data_dir_buf: [256]u8 = undefined;
                const data_dir = try std.fmt.bufPrint(
                    &data_dir_buf,
                    "data/bench/{s}_{s}",
                    .{ @tagName(mt), DEFAULT_DATA_DIR_SUFFIX },
                );
                for (0..bench_args.runs) |rr| {
                    const run_out_dir_base = if (bench_args.save_strategy == .disk or
                        bench_args.save_strategy == .both)
                        bench_args.out_dir
                    else
                        "";
                    const raster_config =
                        benchargs.applyRasterConfig(
                            base_raster_config,
                            bench_args,
                        );

                    var res = try common.runBenchmarkWithImageOut(
                        u8,
                        outer_alloc,
                        io,
                        mt,
                        st,
                        null,
                        tex_func_case,
                        data_dir,
                        render_defaults,
                        texture_grey,
                        texture_rgb,
                        raster_config,
                        run_out_dir_base,
                        bench_args.image_out_dir,
                    );
                    defer res.deinit(outer_alloc);

                    try stats.appendRunResult(
                        outer_alloc,
                        rr,
                        case_name,
                        mt,
                        st,
                        null,
                        tex_func_case,
                        res,
                    );
                    try stats.writeRunCSV(
                        outer_alloc,
                        io,
                        bench_args.out_dir,
                        rr,
                    );
                    case_samples.record(rr, res);
                }

                try stats.appendCaseStats(
                    outer_alloc,
                    case_name,
                    mt,
                    st,
                    null,
                    tex_func_case,
                    &case_samples,
                );
            }
        }
    }

    try stats.writeRunCSVs(outer_alloc, io, bench_args.out_dir);
    try common.writeBenchmarkReport(
        outer_alloc,
        io,
        "Sphere 2000 Zoom Benchmark Results",
        bench_args.out_dir,
        bench_args.pixels_num,
        stats.stats_list.items,
        0,
    );
}
