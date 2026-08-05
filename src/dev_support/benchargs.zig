// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const std = @import("std");
const cam = @import("../riley/zig/camera.zig");
const rastcfg = @import("../riley/zig/rasterconfig.zig");
const texops = @import("../riley/zig/textureops.zig");

pub const BenchArgs = struct {
    out_dir: []const u8,
    image_out_dir: []const u8,
    render_mode: rastcfg.RenderMode,
    render_group_count: u16,
    total_threads: u16,
    frame_batch_size_per_group: u16,
    max_geom_jobs_in_flight_per_group: u16,
    max_geom_workers_per_job: u16,
    geom_scheduling_mode: rastcfg.GeometrySchedulingMode,
    max_raster_workers_per_job: u16,
    hull_mode: rastcfg.HullMode,
    subpixel_center_map: cam.SubPixelCenterMap,
    save_strategy: rastcfg.SaveStrategy,
    disk_save_overlap: bool,
    sample: ?texops.TexSample,
    sample_mode: ?texops.TexSampleMode,
    texture_storage: TextureStorage,
    shader_subset: ShaderSubset,
    mesh_subset: MeshSubset,
    pixels_num: [2]u32,
    sub_sample: u32,
    runs: usize,
    distortion: DistortionMode = .none,
};

pub const DistortionMode = enum {
    none,
    brown,
    brownext,
};

pub const TextureStorage = enum {
    u8,
    u16,
};

pub const ShaderSubset = enum {
    all,
    texture,
};

pub const MeshSubset = enum {
    all,
    tri3,
    tri3opt,
    tri3_compare,
};

pub fn defaultBenchArgs(
    default_out_dir: []const u8,
    raster_config: rastcfg.RasterConfig,
) BenchArgs {
    return .{
        .out_dir = default_out_dir,
        .image_out_dir = "",
        .render_mode = raster_config.render_mode,
        .render_group_count = 1,
        .total_threads = raster_config.total_threads,
        .frame_batch_size_per_group = raster_config.frame_batch_size_per_group,
        .max_geom_jobs_in_flight_per_group = raster_config.max_geom_jobs_in_flight_per_group,
        .max_geom_workers_per_job = raster_config.max_geom_workers_per_job,
        .geom_scheduling_mode = raster_config.geom_scheduling_mode,
        .max_raster_workers_per_job = raster_config.max_raster_workers_per_job,
        .hull_mode = raster_config.hull_mode,
        .subpixel_center_map = .per_tile,
        .save_strategy = .memory,
        .disk_save_overlap = raster_config.disk_save_overlap,
        .sample = null,
        .sample_mode = null,
        .texture_storage = .u8,
        .shader_subset = .all,
        .mesh_subset = .all,
        .pixels_num = .{ 800, 500 },
        .sub_sample = 2,
        .runs = 10,
        .distortion = .none,
    };
}

pub fn parseArgs(
    args: anytype,
    default_out_dir: []const u8,
    raster_config: rastcfg.RasterConfig,
) !BenchArgs {
    return parseArgsWithDefaults(
        args,
        defaultBenchArgs(
            default_out_dir,
            raster_config,
        ),
    );
}

pub fn parseArgsWithDefaults(
    args: anytype,
    defaults: BenchArgs,
) !BenchArgs {
    var bench_args = defaults;
    var total_threads_overridden = false;
    var frame_batch_overridden = false;
    var max_geom_jobs_overridden = false;
    var max_geom_workers_per_job_overridden = false;
    var max_raster_workers_per_job_overridden = false;
    var skip_next = false;
    var positional_arg_count: usize = 0;

    for (args[1..], 1..) |raw_arg, ii| {
        if (skip_next) {
            skip_next = false;
            continue;
        }

        const arg = argToSlice(raw_arg);
        if (std.mem.startsWith(u8, arg, "--")) {
            if (ii + 1 >= args.len) {
                return error.MissingArgumentValue;
            }

            const value = argToSlice(args[ii + 1]);
            skip_next = true;

            if (std.mem.eql(u8, arg, "--render-mode")) {
                bench_args.render_mode =
                    try parseEnum(rastcfg.RenderMode, value);
            } else if (std.mem.eql(u8, arg, "--render-group-count")) {
                bench_args.render_group_count = try parseInt(u16, value);
            } else if (std.mem.eql(u8, arg, "--total-threads")) {
                bench_args.total_threads = try parseInt(u16, value);
                total_threads_overridden = true;
            } else if (std.mem.eql(u8, arg, "--frame-batch-size-per-group")) {
                bench_args.frame_batch_size_per_group = try parseInt(u16, value);
                frame_batch_overridden = true;
            } else if (std.mem.eql(u8, arg, "--max-geom-jobs-in-flight-per-group")) {
                bench_args.max_geom_jobs_in_flight_per_group = try parseInt(u16, value);
                max_geom_jobs_overridden = true;
            } else if (std.mem.eql(u8, arg, "--max-geom-workers-per-job")) {
                bench_args.max_geom_workers_per_job = try parseInt(u16, value);
                max_geom_workers_per_job_overridden = true;
            } else if (std.mem.eql(u8, arg, "--geom-scheduling-mode")) {
                bench_args.geom_scheduling_mode =
                    try parseEnum(rastcfg.GeometrySchedulingMode, value);
            } else if (std.mem.eql(u8, arg, "--max-raster-workers-per-job")) {
                bench_args.max_raster_workers_per_job = try parseInt(u16, value);
                max_raster_workers_per_job_overridden = true;
            } else if (std.mem.eql(u8, arg, "--hull-mode")) {
                bench_args.hull_mode =
                    try parseEnum(rastcfg.HullMode, value);
            } else if (std.mem.eql(u8, arg, "--subpixel-center-map")) {
                bench_args.subpixel_center_map =
                    try parseEnum(cam.SubPixelCenterMap, value);
            } else if (std.mem.eql(u8, arg, "--save-strategy")) {
                bench_args.save_strategy =
                    try parseSaveStrategy(value);
            } else if (std.mem.eql(u8, arg, "--disk-save-overlap")) {
                bench_args.disk_save_overlap = try parseBool(value);
            } else if (std.mem.eql(u8, arg, "--image-out-dir")) {
                bench_args.image_out_dir = value;
            } else if (std.mem.eql(u8, arg, "--sample")) {
                bench_args.sample =
                    try parseEnum(texops.TexSample, value);
            } else if (std.mem.eql(u8, arg, "--sample-mode")) {
                bench_args.sample_mode =
                    try parseEnum(texops.TexSampleMode, value);
            } else if (std.mem.eql(u8, arg, "--texture-storage")) {
                bench_args.texture_storage =
                    try parseEnum(TextureStorage, value);
            } else if (std.mem.eql(u8, arg, "--shader-subset")) {
                bench_args.shader_subset =
                    try parseEnum(ShaderSubset, value);
            } else if (std.mem.eql(u8, arg, "--mesh-subset")) {
                bench_args.mesh_subset =
                    try parseEnum(MeshSubset, value);
            } else if (std.mem.eql(u8, arg, "--out-dir")) {
                bench_args.out_dir = value;
            } else if (std.mem.eql(u8, arg, "--pixels-x")) {
                bench_args.pixels_num[0] = try parseInt(u32, value);
            } else if (std.mem.eql(u8, arg, "--pixels-y")) {
                bench_args.pixels_num[1] = try parseInt(u32, value);
            } else if (std.mem.eql(u8, arg, "--sub-sample")) {
                bench_args.sub_sample = try parseInt(u32, value);
            } else if (std.mem.eql(u8, arg, "--runs")) {
                bench_args.runs = try parseInt(usize, value);
            } else if (std.mem.eql(u8, arg, "--distortion")) {
                bench_args.distortion =
                    try parseEnum(DistortionMode, value);
            } else {
                return error.UnknownArgument;
            }
        } else {
            if (positional_arg_count != 0) {
                return error.UnknownArgument;
            }
            bench_args.total_threads = try parseInt(u16, arg);
            total_threads_overridden = true;
            positional_arg_count += 1;
        }
    }

    if (bench_args.pixels_num[0] == 0 or bench_args.pixels_num[1] == 0) {
        return error.InvalidPixelsNum;
    }
    if (bench_args.render_group_count == 0) {
        return error.InvalidRenderGroupCount;
    }
    if (bench_args.sub_sample == 0) {
        return error.InvalidSubSample;
    }
    if (bench_args.runs == 0) {
        return error.InvalidRuns;
    }

    if (total_threads_overridden and !max_geom_workers_per_job_overridden) {
        bench_args.max_geom_workers_per_job = bench_args.total_threads;
    }
    if (total_threads_overridden and !max_raster_workers_per_job_overridden) {
        bench_args.max_raster_workers_per_job = bench_args.total_threads;
    }

    return bench_args;
}

pub fn applyRasterConfig(
    base_config: rastcfg.RasterConfig,
    bench_args: BenchArgs,
) rastcfg.RasterConfig {
    var raster_config = base_config;
    raster_config.render_mode = bench_args.render_mode;
    raster_config.total_threads = bench_args.total_threads;
    raster_config.frame_batch_size_per_group =
        bench_args.frame_batch_size_per_group;
    raster_config.max_geom_jobs_in_flight_per_group =
        bench_args.max_geom_jobs_in_flight_per_group;
    raster_config.max_geom_workers_per_job =
        bench_args.max_geom_workers_per_job;
    raster_config.geom_scheduling_mode =
        bench_args.geom_scheduling_mode;
    raster_config.max_raster_workers_per_job =
        bench_args.max_raster_workers_per_job;
    raster_config.hull_mode = bench_args.hull_mode;
    raster_config.save_strategy = bench_args.save_strategy;
    raster_config.disk_save_overlap = bench_args.disk_save_overlap;
    return raster_config;
}

fn parseInt(comptime T: type, value: []const u8) !T {
    return std.fmt.parseInt(T, value, 10);
}

fn parseEnum(comptime T: type, value: []const u8) !T {
    return std.meta.stringToEnum(T, value) orelse error.InvalidEnumValue;
}

fn parseBool(value: []const u8) !bool {
    if (std.mem.eql(u8, value, "true")) return true;
    if (std.mem.eql(u8, value, "false")) return false;
    return error.InvalidBoolValue;
}

fn parseSaveStrategy(value: []const u8) !rastcfg.SaveStrategy {
    if (std.mem.eql(u8, value, "memory_direct_write")) return .memory;
    if (std.mem.eql(u8, value, "memory_per_frame_copy")) return .memory;
    if (std.mem.eql(u8, value, "memory")) return .memory;
    if (std.mem.eql(u8, value, "both_direct_write")) return .both;
    if (std.mem.eql(u8, value, "both_per_frame_copy")) return .both;
    if (std.mem.eql(u8, value, "both")) return .both;
    return parseEnum(rastcfg.SaveStrategy, value);
}

fn argToSlice(arg: anytype) []const u8 {
    return switch (@typeInfo(@TypeOf(arg))) {
        .pointer => |pointer_info| switch (pointer_info.size) {
            .slice => arg,
            else => std.mem.span(arg),
        },
        .array => arg[0..],
        else => @compileError("Unsupported command line argument type."),
    };
}

test "parse bench args defaults" {
    const args = [_][]const u8{"bench_geom"};
    const raster_config = rastcfg.RasterConfig{
        .render_mode = .offline,
        .total_threads = 3,
        .frame_batch_size_per_group = 2,
        .max_geom_jobs_in_flight_per_group = 2,
        .max_geom_workers_per_job = 2,
        .max_raster_workers_per_job = 3,
        .hull_mode = .on_convex_fallback,
        .subpixel_center_map = .affine_jac,
    };
    const bench_args = try parseArgs(
        args[0..],
        "out/geom",
        raster_config,
    );

    try std.testing.expectEqual(
        defaultBenchArgs("out/geom", raster_config),
        bench_args,
    );
}

test "parse bench args named options" {
    const args = [_][]const u8{
        "bench_geom",
        "--render-mode",
        "offline",
        "--render-group-count",
        "2",
        "--total-threads",
        "8",
        "--frame-batch-size-per-group",
        "4",
        "--max-geom-jobs-in-flight-per-group",
        "6",
        "--max-geom-workers-per-job",
        "2",
        "--geom-scheduling-mode",
        "pack",
        "--max-raster-workers-per-job",
        "7",
        "--hull-mode",
        "off",
        "--subpixel-center-map",
        "affine_jac",
        "--save-strategy",
        "memory",
        "--sample",
        "linear",
        "--sample-mode",
        "direct",
        "--texture-storage",
        "u16",
        "--shader-subset",
        "texture",
        "--mesh-subset",
        "tri3opt",
        "--out-dir",
        "out/custom",
        "--pixels-x",
        "1024",
        "--pixels-y",
        "768",
        "--sub-sample",
        "4",
        "--runs",
        "7",
    };
    const bench_args = try parseArgs(
        args[0..],
        "out/geom",
        rastcfg.RasterConfig{},
    );

    try std.testing.expectEqual(rastcfg.RenderMode.offline, bench_args.render_mode);
    try std.testing.expectEqual(@as(u16, 2), bench_args.render_group_count);
    try std.testing.expectEqual(@as(u16, 8), bench_args.total_threads);
    try std.testing.expectEqual(
        @as(u16, 4),
        bench_args.frame_batch_size_per_group,
    );
    try std.testing.expectEqual(
        @as(u16, 6),
        bench_args.max_geom_jobs_in_flight_per_group,
    );
    try std.testing.expectEqual(
        @as(u16, 2),
        bench_args.max_geom_workers_per_job,
    );
    try std.testing.expectEqual(
        rastcfg.GeometrySchedulingMode.pack,
        bench_args.geom_scheduling_mode,
    );
    try std.testing.expectEqual(
        @as(u16, 7),
        bench_args.max_raster_workers_per_job,
    );
    try std.testing.expectEqual(rastcfg.HullMode.off, bench_args.hull_mode);
    try std.testing.expectEqual(
        cam.SubPixelCenterMap.affine_jac,
        bench_args.subpixel_center_map,
    );
    try std.testing.expectEqual(
        rastcfg.SaveStrategy.memory,
        bench_args.save_strategy,
    );
    try std.testing.expectEqual(
        texops.TexSample.linear,
        bench_args.sample.?,
    );
    try std.testing.expectEqual(
        texops.TexSampleMode.direct,
        bench_args.sample_mode.?,
    );
    try std.testing.expectEqual(
        TextureStorage.u16,
        bench_args.texture_storage,
    );
    try std.testing.expectEqual(
        ShaderSubset.texture,
        bench_args.shader_subset,
    );
    try std.testing.expectEqual(
        MeshSubset.tri3opt,
        bench_args.mesh_subset,
    );
    try std.testing.expectEqualStrings("out/custom", bench_args.out_dir);
    try std.testing.expectEqual([2]u32{ 1024, 768 }, bench_args.pixels_num);
    try std.testing.expectEqual(@as(u8, 4), bench_args.sub_sample);
    try std.testing.expectEqual(@as(usize, 7), bench_args.runs);
}

test "parse bench args legacy thread positional" {
    const args = [_][]const u8{ "bench_geom", "6" };
    const bench_args = try parseArgs(
        args[0..],
        "out/geom",
        rastcfg.RasterConfig{},
    );

    try std.testing.expectEqual(@as(u16, 6), bench_args.total_threads);
    try std.testing.expectEqual(
        @as(u16, 6),
        bench_args.max_geom_workers_per_job,
    );
    try std.testing.expectEqual(
        @as(u16, 6),
        bench_args.max_raster_workers_per_job,
    );
}
