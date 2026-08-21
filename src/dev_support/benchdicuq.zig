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
const F = buildconfig.F;

const benchargs = @import("benchargs.zig");
const common = @import("benchcommon.zig");
const riley = @import("../riley/zig/riley.zig");
const meshio = @import("../riley/zig/meshio.zig");
const uvio = @import("../riley/zig/uvio.zig");
const iio = @import("../riley/zig/imageio.zig");
const mo = @import("../riley/zig/meshpipeline.zig");
const gk = @import("../riley/zig/geometrykernels.zig");
const texops = @import("../riley/zig/textureops.zig");
const camera_mod = @import("../riley/zig/camera.zig");
const cameraops = @import("../riley/zig/cameraops.zig");
const Rotation = @import("../riley/zig/rotation.zig").Rotation;
const sceneops = @import("../riley/zig/sceneops.zig");
const report = @import("../riley/zig/report.zig");
const tcfg = @import("testconfig.zig");

const CameraInput = camera_mod.CameraInput;
const MeshInput = mo.MeshInput;

pub const PreparedDicuqBenchmark = struct {
    mesh_input: MeshInput,
    camera_inputs: [2]CameraInput,
    samp_cfg: texops.TextureSampleConfig,
    fov_scale: F,
};

pub const DicuqDefaults = struct {
    data_dir: []const u8,
    pixels_num: [2]u32,
    sub_sample: u32,
    focal_leng: F,
    pixels_size: [2]F,
    fov_scale: F,
    stereo_ang: F,
    tex_path: []const u8,
};

pub const DicuqFrameRow = struct {
    run_idx: usize,
    camera_idx: usize,
    frame_idx: usize,
    total_elems: usize,
    vis_elems: usize,
    total_px: u64,
    shaded_px: u64,
    geom_time_ms: F,
    cam_time_ms: F,
    elem_loop_time_ms: F,
    resolve_time_ms: F,
    raster_time_ms: F,
    save_time_ms: F,
    frame_time_ms: F,
    e2e_time_ms: ?F,
    geom_tpx_melem_s: F,
    raster_tpx_mpx_s: F,
    frame_tpx_mpx_s: F,
    e2e_tpx_mpx_s: ?F,
};

pub const DicuqE2ERow = struct {
    run_idx: usize,
    camera_idx: ?usize,
    total_elems: usize,
    vis_elems: usize,
    total_px: u64,
    shaded_px: u64,
    geom_time_ms: F,
    cam_time_ms: F,
    elem_loop_time_ms: F,
    resolve_time_ms: F,
    raster_time_ms: F,
    save_time_ms: F,
    frame_time_ms: F,
    e2e_time_ms: ?F,
    geom_tpx_melem_s: F,
    raster_tpx_mpx_s: F,
    frame_tpx_mpx_s: F,
    e2e_tpx_mpx_s: ?F,
};

pub const DicuqRunResult = struct {
    bench_result: common.BenchResult,
    frame_rows: []DicuqFrameRow,
    e2e_rows: []DicuqE2ERow,

    pub fn deinit(
        self: *DicuqRunResult,
        allocator: std.mem.Allocator,
    ) void {
        self.bench_result.deinit(allocator);
        allocator.free(self.frame_rows);
        allocator.free(self.e2e_rows);
    }
};

const DicuqStatsKind = enum {
    median,
    mad,
    min,
    max,
};

const DicuqFrameStatsRow = struct {
    run_idx: usize,
    camera_idx: ?usize,
    total_elems: F,
    vis_elems: F,
    total_px: F,
    shaded_px: F,
    geom_time_ms: F,
    cam_time_ms: F,
    elem_loop_time_ms: F,
    resolve_time_ms: F,
    raster_time_ms: F,
    save_time_ms: F,
    frame_time_ms: F,
    e2e_time_ms: ?F,
    geom_tpx_melem_s: F,
    raster_tpx_mpx_s: F,
    frame_tpx_mpx_s: F,
    e2e_tpx_mpx_s: ?F,
};

const DicuqE2EStatsRow = struct {
    camera_idx: ?usize,
    total_elems: F,
    vis_elems: F,
    total_px: F,
    shaded_px: F,
    geom_time_ms: F,
    cam_time_ms: F,
    elem_loop_time_ms: F,
    resolve_time_ms: F,
    raster_time_ms: F,
    save_time_ms: F,
    frame_time_ms: F,
    e2e_time_ms: ?F,
    geom_tpx_melem_s: F,
    raster_tpx_mpx_s: F,
    frame_tpx_mpx_s: F,
    e2e_tpx_mpx_s: ?F,
};

pub fn getBaseRasterConfig() riley.RasterConfig {
    var base_raster_config = tcfg.getRasterConfig(.bench);
    // Thread counts include the caller thread.
    base_raster_config.total_threads = 1;
    base_raster_config.frame_batch_size_per_group = 1;
    base_raster_config.max_geom_jobs_in_flight_per_group = 1;
    base_raster_config.max_geom_workers_per_job = 1;
    base_raster_config.max_raster_workers_per_job = 1;
    base_raster_config.geom_scheduling_mode = .auto;
    base_raster_config.save_strategy = .disk;
    base_raster_config.tile_size_min = 8;
    base_raster_config.tile_size_max = 128;
    base_raster_config.background_value = 128.0;
    base_raster_config.image_save_opts = &[_]iio.ImageSaveOpts{
        .{ .format = .bmp, .bits = 8, .scaling = .auto },
    };
    return base_raster_config;
}

pub fn makeSampleConfig(
    bench_args: benchargs.BenchArgs,
) !texops.TextureSampleConfig {
    const samp_cfg = texops.TextureSampleConfig{
        .sample = bench_args.sample orelse .cubic_catmull_rom,
        .mode = bench_args.sample_mode orelse .lut_lerp,
    };
    if (!samp_cfg.isValid()) {
        return error.InvalidTextureSampleConfig;
    }
    return samp_cfg;
}

pub fn prepareBenchmark(
    allocator: std.mem.Allocator,
    io: std.Io,
    defaults: DicuqDefaults,
    samp_cfg: texops.TextureSampleConfig,
) !PreparedDicuqBenchmark {
    const data_dir = defaults.data_dir;
    const coord_path = try std.fmt.allocPrint(
        allocator,
        "{s}coords.csv",
        .{data_dir},
    );
    const conn_path = try std.fmt.allocPrint(
        allocator,
        "{s}connect.csv",
        .{data_dir},
    );
    const field_disp_x_path = try std.fmt.allocPrint(
        allocator,
        "{s}field_disp_x.csv",
        .{data_dir},
    );
    const field_disp_y_path = try std.fmt.allocPrint(
        allocator,
        "{s}field_disp_y.csv",
        .{data_dir},
    );
    const field_disp_z_path = try std.fmt.allocPrint(
        allocator,
        "{s}field_disp_z.csv",
        .{data_dir},
    );
    const field_files = &[_][]const u8{
        field_disp_x_path,
        field_disp_y_path,
        field_disp_z_path,
    };
    const uv_path = try std.fmt.allocPrint(
        allocator,
        "{s}uvs.csv",
        .{data_dir},
    );

    const sim_data = try meshio.loadSimData(
        allocator,
        io,
        coord_path,
        conn_path,
        null,
        field_files,
    );
    const uvs = try uvio.loadUVMap(allocator, io, uv_path);
    const texture = try iio.loadImage(
        u8,
        1,
        allocator,
        io,
        defaults.tex_path,
        .bmp,
    );

    const mesh_input = MeshInput{
        .mesh_type = .quad8,
        .coords = sim_data.coords,
        .connect = sim_data.connect,
        .disp = sim_data.disp,
        .shader = .{ .tex_u8 = .{
            .uvs = uvs.array,
            .tex = texture,
            .samp_cfg = samp_cfg,
            .bits = 8,
            .scaling = .none,
        } },
    };

    const pixel_size = defaults.pixels_size;
    const focal_length = defaults.focal_leng;
    const fov_scale = defaults.fov_scale;
    const roi_pos = sceneops.boundsCenter(
        &sim_data.coords,
    );

    const cam0_rot = Rotation.init(
        std.math.degreesToRadians(0.0),
        std.math.degreesToRadians(0.0),
        std.math.degreesToRadians(0.0),
    );
    const cam0_pos = cameraops.posFillFrameFromRot(
        &sim_data.coords,
        defaults.pixels_num,
        pixel_size,
        focal_length,
        cam0_rot,
        fov_scale,
    );
    const cam1_rot = Rotation.init(
        std.math.degreesToRadians(0.0),
        std.math.degreesToRadians(defaults.stereo_ang),
        std.math.degreesToRadians(0.0),
    );
    const cam1_pos = cameraops.posFillFrameFromRot(
        &sim_data.coords,
        defaults.pixels_num,
        pixel_size,
        focal_length,
        cam1_rot,
        fov_scale,
    );

    return .{
        .mesh_input = mesh_input,
        .camera_inputs = .{
            .{
                .pixels_num = defaults.pixels_num,
                .pixels_size = pixel_size,
                .pos_world = cam0_pos,
                .rot_world = cam0_rot,
                .roi_cent_world = roi_pos,
                .focal_length = focal_length,
                .sub_sample = defaults.sub_sample,
            },
            .{
                .pixels_num = defaults.pixels_num,
                .pixels_size = pixel_size,
                .pos_world = cam1_pos,
                .rot_world = cam1_rot,
                .roi_cent_world = roi_pos,
                .focal_length = focal_length,
                .sub_sample = defaults.sub_sample,
            },
        },
        .samp_cfg = samp_cfg,
        .fov_scale = fov_scale,
    };
}

pub fn calcCaseName(
    allocator: std.mem.Allocator,
    samp_cfg: texops.TextureSampleConfig,
) ![]const u8 {
    return common.calcCaseName(
        allocator,
        .quad8,
        .tex8_grey,
        samp_cfg,
        null,
        1.0,
    );
}

pub fn runBenchmark(
    outer_alloc: std.mem.Allocator,
    io: std.Io,
    render_groups: []const riley.RenderGroupSpec,
    camera_inputs: []const CameraInput,
    mesh_input: MeshInput,
    config: riley.RasterConfig,
    out_dir_path: ?[]const u8,
) !DicuqRunResult {
    const frame_count = if (mesh_input.disp) |disp|
        disp.getTimeN() * camera_inputs.len
    else
        camera_inputs.len;
    const bench_capture = try outer_alloc.alloc(
        report.FrameBenchCapture,
        frame_count,
    );
    defer outer_alloc.free(bench_capture);

    const start = std.Io.Clock.Timestamp.now(io, .awake);
    const image_arr = try riley.rasterReport(
        outer_alloc,
        render_groups,
        camera_inputs,
        &[_]MeshInput{mesh_input},
        config,
        out_dir_path,
        bench_capture,
    );
    const end = std.Io.Clock.Timestamp.now(io, .awake);

    if (image_arr) |images| {
        outer_alloc.free(images.slice);
        var images_mut = images;
        images_mut.deinit(outer_alloc);
    }

    const frame_times = aggregateFrameTimes(bench_capture);
    const bench_log = aggregateBenchLog(bench_capture);
    const e2e_ms = @as(
        F,
        @floatFromInt(start.durationTo(end).raw.nanoseconds),
    ) / 1e6;
    const frame_count_f = @as(F, @floatFromInt(frame_count));
    const frame_rows = try buildFrameRows(
        outer_alloc,
        camera_inputs,
        bench_capture,
    );
    const e2e_rows = try buildE2ERows(
        outer_alloc,
        camera_inputs,
        frame_rows,
        e2e_ms,
    );

    return .{
        .bench_result = .{
            .e2e_ms = e2e_ms,
            .geom_ms = (frame_times.geometry_prep + frame_times.tile_overlap) /
                1e6,
            .raster_ms = frame_times.raster_loop / 1e6,
            .cam_ms = frame_times.cam_invert / 1e6,
            .resolve_ms = frame_times.scratch_resolve / 1e6,
            .fps = if (e2e_ms > 0)
                (frame_count_f * 1000.0 / e2e_ms)
            else
                0.0,
            .metrics = calcDicuqMetrics(
                mesh_input.mesh_type,
                camera_inputs[0].pixels_num,
                camera_inputs[0].sub_sample,
                frame_count,
                e2e_ms,
                frame_times,
                bench_log,
            ),
            .pipeline_times = frame_times,
            .image = null,
            .total_elems = bench_log.total_elems,
            .vis_elems = bench_log.vis_elems,
            .total_px = @as(u64, camera_inputs[0].pixels_num[0]) *
                @as(u64, camera_inputs[0].pixels_num[1]) *
                @as(u64, frame_count),
            .shaded_px = bench_log.total_shaded_px,
        },
        .frame_rows = frame_rows,
        .e2e_rows = e2e_rows,
    };
}

fn aggregateFrameTimes(
    bench_capture: []const report.FrameBenchCapture,
) report.FrameTimes {
    var frame_times = report.FrameTimes{};

    for (bench_capture) |capture| {
        frame_times.geometry_prep +=
            capture.bench_log.frame_times.geometry_prep;
        frame_times.tile_overlap +=
            capture.bench_log.frame_times.tile_overlap;
        frame_times.raster_loop +=
            capture.bench_log.frame_times.raster_loop;
        frame_times.cam_invert +=
            capture.bench_log.frame_times.cam_invert;
        frame_times.elem_loop +=
            capture.bench_log.frame_times.elem_loop;
        frame_times.scratch_resolve +=
            capture.bench_log.frame_times.scratch_resolve;
        frame_times.save_frame +=
            capture.bench_log.frame_times.save_frame;
        frame_times.active_time +=
            capture.bench_log.frame_times.active_time;
        frame_times.latency_time +=
            capture.bench_log.frame_times.latency_time;
    }

    return frame_times;
}

fn aggregateBenchLog(
    bench_capture: []const report.FrameBenchCapture,
) report.BenchLog {
    var bench_log = report.BenchLog{};

    for (bench_capture) |capture| {
        report.reduceBenchLog(&bench_log, &capture.bench_log);
        bench_log.total_elems += capture.bench_log.total_elems;
        bench_log.vis_elems += capture.bench_log.vis_elems;
    }

    bench_log.frame_times = aggregateFrameTimes(bench_capture);
    return bench_log;
}

fn calcDicuqMetrics(
    mesh_type: gk.MeshType,
    pixel_num: [2]u32,
    sub_sample: u32,
    frame_count: usize,
    e2e_ms: F,
    frame_times: report.FrameTimes,
    bench_log: report.BenchLog,
) common.CalculatedMetrics {
    const raster_sec = frame_times.raster_loop / 1e9;
    const geom_tiling_sec =
        (frame_times.geometry_prep + frame_times.tile_overlap) / 1e9;
    const active_sec = frame_times.active_time / 1e9;

    const nodes_per_elem = @as(F, @floatFromInt(mesh_type.getNodesNum()));
    const pixels_x = @as(F, @floatFromInt(pixel_num[0]));
    const pixels_y = @as(F, @floatFromInt(pixel_num[1]));
    const frame_count_f = @as(F, @floatFromInt(frame_count));
    const sub_samp_f = @as(F, @floatFromInt(sub_sample));

    const total_px = pixels_x * pixels_y * frame_count_f;
    const total_subpx = total_px * sub_samp_f * sub_samp_f;

    const shaded_subpx = @as(
        F,
        @floatFromInt(bench_log.total_shaded_px),
    );
    const est_shaded_px = shaded_subpx / (sub_samp_f * sub_samp_f);
    const total_elems = @as(
        F,
        @floatFromInt(bench_log.total_elems),
    );

    return .{
        .raster_tpx_mpx_s = if (raster_sec > 0)
            (total_px / (raster_sec * 1e6))
        else
            0,
        .frame_tpx_mpx_s = if (active_sec > 0)
            (total_px / (active_sec * 1e6))
        else
            0,
        .e2e_tpx_mpx_s = if (e2e_ms > 0.0)
            total_px / ((e2e_ms / 1e3) * 1e6)
        else
            0,
        .msubpx_sec = if (raster_sec > 0)
            (total_subpx / (raster_sec * 1e6))
        else
            0,
        .mshades_sec = if (raster_sec > 0)
            (est_shaded_px / (raster_sec * 1e6))
        else
            0,
        .msubshades_sec = if (raster_sec > 0)
            (shaded_subpx / (raster_sec * 1e6))
        else
            0,
        .melems_sec = if (geom_tiling_sec > 0)
            (total_elems / (geom_tiling_sec * 1e6))
        else
            0,
        .mnodes_sec = if (geom_tiling_sec > 0)
            (total_elems * nodes_per_elem / (geom_tiling_sec * 1e6))
        else
            0,
        .mops_sec = if (active_sec > 0)
            (nodes_per_elem * total_subpx / (active_sec * 1e6))
        else
            0,
    };
}

fn calcFrameMPxPerSec(
    camera_input: CameraInput,
    raster_time_ns: F,
) F {
    if (raster_time_ns <= 0.0) {
        return 0.0;
    }
    const pixels_x = @as(F, @floatFromInt(camera_input.pixels_num[0]));
    const pixels_y = @as(F, @floatFromInt(camera_input.pixels_num[1]));
    const sub_sample = @as(F, @floatFromInt(camera_input.sub_sample));
    const total_px = pixels_x * pixels_y;
    const total_subpx = total_px * sub_sample * sub_sample;
    _ = total_subpx;
    const raster_sec = raster_time_ns / 1e9;
    return total_px / (raster_sec * 1e6);
}

fn calcFrameActiveMPxPerSec(
    camera_input: CameraInput,
    frame_active_time_ns: F,
) F {
    if (frame_active_time_ns <= 0.0) {
        return 0.0;
    }
    const total_px = @as(F, @floatFromInt(camera_input.pixels_num[0])) *
        @as(F, @floatFromInt(camera_input.pixels_num[1]));
    const frame_active_sec = frame_active_time_ns / 1e9;
    return total_px / (frame_active_sec * 1e6);
}

fn calcFrameMElemPerSec(
    total_elems: usize,
    geom_time_ns: F,
) F {
    if (geom_time_ns <= 0.0) {
        return 0.0;
    }
    const geom_sec = geom_time_ns / 1e9;
    return @as(F, @floatFromInt(total_elems)) / (geom_sec * 1e6);
}

fn buildFrameRows(
    allocator: std.mem.Allocator,
    camera_inputs: []const CameraInput,
    bench_capture: []const report.FrameBenchCapture,
) ![]DicuqFrameRow {
    var frame_rows = try allocator.alloc(
        DicuqFrameRow,
        bench_capture.len,
    );

    for (bench_capture, 0..) |capture, ii| {
        const geom_time_ns =
            capture.bench_log.frame_times.geometry_prep +
            capture.bench_log.frame_times.tile_overlap;
        frame_rows[ii] = .{
            .run_idx = 0,
            .camera_idx = capture.camera_idx,
            .frame_idx = capture.frame_idx,
            .total_elems = capture.bench_log.total_elems,
            .vis_elems = capture.bench_log.vis_elems,
            .total_px = @as(
                u64,
                camera_inputs[capture.camera_idx].pixels_num[0],
            ) *
                @as(
                    u64,
                    camera_inputs[capture.camera_idx].pixels_num[1],
                ),
            .shaded_px = capture.bench_log.total_shaded_px,
            .geom_time_ms = geom_time_ns / 1e6,
            .cam_time_ms = capture.bench_log.frame_times.cam_invert / 1e6,
            .elem_loop_time_ms = capture.bench_log.frame_times.elem_loop / 1e6,
            .resolve_time_ms = capture.bench_log.frame_times.scratch_resolve / 1e6,
            .raster_time_ms = capture.bench_log.frame_times.raster_loop / 1e6,
            .save_time_ms = capture.bench_log.frame_times.save_frame / 1e6,
            .frame_time_ms = capture.bench_log.frame_times.active_time / 1e6,
            .e2e_time_ms = null,
            .geom_tpx_melem_s = calcFrameMElemPerSec(
                capture.bench_log.total_elems,
                geom_time_ns,
            ),
            .raster_tpx_mpx_s = calcFrameMPxPerSec(
                camera_inputs[capture.camera_idx],
                capture.bench_log.frame_times.raster_loop,
            ),
            .frame_tpx_mpx_s = calcFrameActiveMPxPerSec(
                camera_inputs[capture.camera_idx],
                capture.bench_log.frame_times.active_time,
            ),
            .e2e_tpx_mpx_s = null,
        };
    }

    return frame_rows;
}

fn buildE2ESummaryRow(
    frame_rows: []const DicuqFrameRow,
    camera_inputs: []const CameraInput,
    run_idx: usize,
    camera_idx: ?usize,
    e2e_ms: ?F,
) DicuqE2ERow {
    var geom_time_ms: F = 0.0;
    var elem_loop_time_ms: F = 0.0;
    var raster_time_ms: F = 0.0;
    var save_time_ms: F = 0.0;
    var frame_time_ms: F = 0.0;
    var total_elems: usize = 0;
    var vis_elems: usize = 0;
    var total_px: u64 = 0;
    var shaded_px: u64 = 0;
    var frame_count: usize = 0;

    for (frame_rows) |frame_row| {
        if (camera_idx) |cc| {
            if (frame_row.camera_idx != cc) {
                continue;
            }
        }
        geom_time_ms += frame_row.geom_time_ms;
        elem_loop_time_ms += frame_row.elem_loop_time_ms;
        raster_time_ms += frame_row.raster_time_ms;
        save_time_ms += frame_row.save_time_ms;
        frame_time_ms += frame_row.frame_time_ms;
        total_elems += frame_row.total_elems;
        vis_elems += frame_row.vis_elems;
        total_px += frame_row.total_px;
        shaded_px += frame_row.shaded_px;
        frame_count += 1;
    }

    const camera_ref = if (camera_idx) |cc|
        camera_inputs[cc]
    else
        camera_inputs[0];
    const total_px_f = @as(F, @floatFromInt(camera_ref.pixels_num[0])) *
        @as(F, @floatFromInt(camera_ref.pixels_num[1])) *
        @as(F, @floatFromInt(frame_count));
    const geom_tpx_melem_s = if (geom_time_ms > 0.0)
        @as(F, @floatFromInt(total_elems)) / ((geom_time_ms / 1e3) * 1e6)
    else
        0.0;
    const raster_tpx_mpx_s = if (raster_time_ms > 0.0)
        total_px_f / ((raster_time_ms / 1e3) * 1e6)
    else
        0.0;
    const frame_tpx_mpx_s = if (frame_time_ms > 0.0)
        total_px_f / ((frame_time_ms / 1e3) * 1e6)
    else
        0.0;
    const e2e_tpx_mpx_s = if (e2e_ms) |e2e_time_ms|
        if (e2e_time_ms > 0.0)
            total_px_f / ((e2e_time_ms / 1e3) * 1e6)
        else
            0.0
    else
        null;

    return .{
        .run_idx = run_idx,
        .camera_idx = camera_idx,
        .total_elems = total_elems,
        .vis_elems = vis_elems,
        .total_px = total_px,
        .shaded_px = shaded_px,
        .geom_time_ms = geom_time_ms,
        .cam_time_ms = blk: {
            var cam_sum: F = 0.0;
            for (frame_rows) |fr| {
                if (camera_idx) |cc| {
                    if (fr.camera_idx != cc) continue;
                }
                cam_sum += fr.cam_time_ms;
            }
            break :blk cam_sum;
        },
        .elem_loop_time_ms = elem_loop_time_ms,
        .resolve_time_ms = blk: {
            var res_sum: F = 0.0;
            for (frame_rows) |fr| {
                if (camera_idx) |cc| {
                    if (fr.camera_idx != cc) continue;
                }
                res_sum += fr.resolve_time_ms;
            }
            break :blk res_sum;
        },
        .raster_time_ms = raster_time_ms,
        .save_time_ms = save_time_ms,
        .frame_time_ms = frame_time_ms,
        .e2e_time_ms = e2e_ms,
        .geom_tpx_melem_s = geom_tpx_melem_s,
        .raster_tpx_mpx_s = raster_tpx_mpx_s,
        .frame_tpx_mpx_s = frame_tpx_mpx_s,
        .e2e_tpx_mpx_s = e2e_tpx_mpx_s,
    };
}

fn buildE2ERows(
    allocator: std.mem.Allocator,
    camera_inputs: []const CameraInput,
    frame_rows: []const DicuqFrameRow,
    e2e_ms: F,
) ![]DicuqE2ERow {
    const row_count = camera_inputs.len + 1;
    var e2e_rows = try allocator.alloc(DicuqE2ERow, row_count);

    for (camera_inputs, 0..) |_, cc| {
        e2e_rows[cc] = buildE2ESummaryRow(
            frame_rows,
            camera_inputs,
            0,
            cc,
            null,
        );
    }
    e2e_rows[camera_inputs.len] = buildE2ESummaryRow(
        frame_rows,
        camera_inputs,
        0,
        null,
        e2e_ms,
    );

    return e2e_rows;
}

fn cameraLabel(
    allocator: std.mem.Allocator,
    camera_idx: ?usize,
) ![]u8 {
    if (camera_idx) |cc| {
        return std.fmt.allocPrint(allocator, "{d}", .{cc});
    }
    return allocator.dupe(u8, "all");
}

fn writeDicuqFile(
    allocator: std.mem.Allocator,
    io: std.Io,
    out_dir_base: []const u8,
    file_name: []const u8,
    contents: []const u8,
) !void {
    const cwd = std.Io.Dir.cwd();
    cwd.createDir(
        io,
        out_dir_base,
        .default_dir,
    ) catch |err| if (err != error.PathAlreadyExists) return err;

    const csv_path = try std.fs.path.join(
        allocator,
        &[_][]const u8{ out_dir_base, file_name },
    );
    defer allocator.free(csv_path);

    var file = try cwd.createFile(io, csv_path, .{});
    defer file.close(io);

    var write_buf: [4096]u8 = undefined;
    var buffered_writer = file.writer(io, &write_buf);
    try buffered_writer.interface.writeAll(contents);
    try buffered_writer.interface.flush();
}

fn appendFrameRowsCSV(
    allocator: std.mem.Allocator,
    rows_buf: *std.ArrayList(u8),
    case_name: []const u8,
    frame_rows: []const DicuqFrameRow,
) !void {
    try rows_buf.appendSlice(
        allocator,
        "Run,Case,Camera,Frame,Total Elems,Vis Elems,Total Px,Shaded Px," ++
            "Geom Time [ms],Cam Inv Time [ms],Elem Loop Time [ms]," ++
            "Resolve Time [ms],Raster Time [ms],Save Time [ms]," ++
            "Frame Time [ms],E2E Time [ms],Geom TP [MElem/s]," ++
            "Raster TP [MPx/s],Frame TP [MPx/s],E2E TP [MPx/s]\n",
    );

    for (frame_rows) |frame_row| {
        const elem_loop_ms = frame_row.elem_loop_time_ms;
        const row = try std.fmt.allocPrint(
            allocator,
            "{d},{s},{d},{d},{d},{d},{d},{d}," ++
                "{d:.6},{d:.6},{d:.6},{d:.6},{d:.6}," ++
                "{d:.6},{d:.6},{s},{d:.6},{d:.6},{d:.6},{s}\n",
            .{
                frame_row.run_idx,
                case_name,
                frame_row.camera_idx,
                frame_row.frame_idx,
                frame_row.total_elems,
                frame_row.vis_elems,
                frame_row.total_px,
                frame_row.shaded_px,
                frame_row.geom_time_ms,
                frame_row.cam_time_ms,
                elem_loop_ms,
                frame_row.resolve_time_ms,
                frame_row.raster_time_ms,
                frame_row.save_time_ms,
                frame_row.frame_time_ms,
                "",
                frame_row.geom_tpx_melem_s,
                frame_row.raster_tpx_mpx_s,
                frame_row.frame_tpx_mpx_s,
                "",
            },
        );
        defer allocator.free(row);
        try rows_buf.appendSlice(allocator, row);
    }
}

fn appendE2ERowsCSV(
    allocator: std.mem.Allocator,
    rows_buf: *std.ArrayList(u8),
    case_name: []const u8,
    e2e_rows: []const DicuqE2ERow,
) !void {
    try rows_buf.appendSlice(
        allocator,
        "Run,Case,Camera,Total Elems,Vis Elems,Total Px,Shaded Px," ++
            "Geom Time [ms],Cam Inv Time [ms],Elem Loop Time [ms]," ++
            "Resolve Time [ms],Raster Time [ms],Save Time [ms]," ++
            "Frame Time [ms],E2E Time [ms],Geom TP [MElem/s]," ++
            "Raster TP [MPx/s],Frame TP [MPx/s],E2E TP [MPx/s]\n",
    );

    for (e2e_rows) |e2e_row| {
        const camera_text =
            try cameraLabel(allocator, e2e_row.camera_idx);
        defer allocator.free(camera_text);
        const e2e_time_text = if (e2e_row.e2e_time_ms) |val|
            try std.fmt.allocPrint(allocator, "{d:.6}", .{val})
        else
            try allocator.dupe(u8, "");
        defer allocator.free(e2e_time_text);
        const e2e_tp_text = if (e2e_row.e2e_tpx_mpx_s) |val|
            try std.fmt.allocPrint(allocator, "{d:.6}", .{val})
        else
            try allocator.dupe(u8, "");
        defer allocator.free(e2e_tp_text);
        const elem_loop_ms = e2e_row.elem_loop_time_ms;

        const row = try std.fmt.allocPrint(
            allocator,
            "{d},{s},{s},{d},{d},{d},{d}," ++
                "{d:.6},{d:.6},{d:.6},{d:.6},{d:.6}," ++
                "{d:.6},{d:.6},{s},{d:.6},{d:.6},{d:.6},{s}\n",
            .{
                e2e_row.run_idx,
                case_name,
                camera_text,
                e2e_row.total_elems,
                e2e_row.vis_elems,
                e2e_row.total_px,
                e2e_row.shaded_px,
                e2e_row.geom_time_ms,
                e2e_row.cam_time_ms,
                elem_loop_ms,
                e2e_row.resolve_time_ms,
                e2e_row.raster_time_ms,
                e2e_row.save_time_ms,
                e2e_row.frame_time_ms,
                e2e_time_text,
                e2e_row.geom_tpx_melem_s,
                e2e_row.raster_tpx_mpx_s,
                e2e_row.frame_tpx_mpx_s,
                e2e_tp_text,
            },
        );
        defer allocator.free(row);
        try rows_buf.appendSlice(allocator, row);
    }
}

fn calcFieldStats(
    allocator: std.mem.Allocator,
    values: []const F,
) !common.MedianMAD {
    const copy_vals = try allocator.dupe(F, values);
    defer allocator.free(copy_vals);
    return common.calcMedianMAD(allocator, copy_vals);
}

fn selectDicuqStat(
    stats: common.MedianMAD,
    kind: DicuqStatsKind,
) F {
    return switch (kind) {
        .median => stats.median,
        .mad => stats.mad,
        .min => stats.min,
        .max => stats.max,
    };
}

fn calcFrameStatsRow(
    allocator: std.mem.Allocator,
    run_idx: usize,
    camera_idx: ?usize,
    frame_rows: []const DicuqFrameRow,
    kind: DicuqStatsKind,
) !DicuqFrameStatsRow {
    var count: usize = 0;
    for (frame_rows) |frame_row| {
        if (camera_idx) |cc| {
            if (frame_row.camera_idx != cc) continue;
        }
        count += 1;
    }

    var elems_vals = try allocator.alloc(F, count);
    defer allocator.free(elems_vals);
    var total_px_vals = try allocator.alloc(F, count);
    defer allocator.free(total_px_vals);
    var geom_vals = try allocator.alloc(F, count);
    defer allocator.free(geom_vals);
    var cam_vals = try allocator.alloc(F, count);
    defer allocator.free(cam_vals);
    var elem_loop_vals = try allocator.alloc(F, count);
    defer allocator.free(elem_loop_vals);
    var resolve_vals = try allocator.alloc(F, count);
    defer allocator.free(resolve_vals);
    var raster_vals = try allocator.alloc(F, count);
    defer allocator.free(raster_vals);
    var save_vals = try allocator.alloc(F, count);
    defer allocator.free(save_vals);
    var frame_vals = try allocator.alloc(F, count);
    defer allocator.free(frame_vals);
    var e2e_vals = try allocator.alloc(F, count);
    defer allocator.free(e2e_vals);
    var vis_vals = try allocator.alloc(F, count);
    defer allocator.free(vis_vals);
    var shaded_vals = try allocator.alloc(F, count);
    defer allocator.free(shaded_vals);
    var geom_tpx_vals = try allocator.alloc(F, count);
    defer allocator.free(geom_tpx_vals);
    var raster_tpx_vals = try allocator.alloc(F, count);
    defer allocator.free(raster_tpx_vals);
    var frame_tpx_vals = try allocator.alloc(F, count);
    defer allocator.free(frame_tpx_vals);
    var e2e_tpx_vals = try allocator.alloc(F, count);
    defer allocator.free(e2e_tpx_vals);

    var ii: usize = 0;
    for (frame_rows) |frame_row| {
        if (camera_idx) |cc| {
            if (frame_row.camera_idx != cc) continue;
        }
        elems_vals[ii] = @floatFromInt(frame_row.total_elems);
        total_px_vals[ii] = @floatFromInt(frame_row.total_px);
        geom_vals[ii] = frame_row.geom_time_ms;
        cam_vals[ii] = frame_row.cam_time_ms;
        elem_loop_vals[ii] = frame_row.elem_loop_time_ms;
        resolve_vals[ii] = frame_row.resolve_time_ms;
        raster_vals[ii] = frame_row.raster_time_ms;
        save_vals[ii] = frame_row.save_time_ms;
        frame_vals[ii] = frame_row.frame_time_ms;
        e2e_vals[ii] = if (frame_row.e2e_time_ms) |val|
            val
        else
            0.0;
        elems_vals[ii] = @floatFromInt(frame_row.total_elems);
        vis_vals[ii] = @floatFromInt(frame_row.vis_elems);
        shaded_vals[ii] = @floatFromInt(frame_row.shaded_px);
        geom_tpx_vals[ii] = frame_row.geom_tpx_melem_s;
        raster_tpx_vals[ii] = frame_row.raster_tpx_mpx_s;
        frame_tpx_vals[ii] = frame_row.frame_tpx_mpx_s;
        e2e_tpx_vals[ii] = if (frame_row.e2e_tpx_mpx_s) |val|
            val
        else
            0.0;
        ii += 1;
    }

    return .{
        .run_idx = run_idx,
        .camera_idx = camera_idx,
        .total_elems = selectDicuqStat(
            try calcFieldStats(allocator, elems_vals),
            kind,
        ),
        .vis_elems = selectDicuqStat(
            try calcFieldStats(allocator, vis_vals),
            kind,
        ),
        .total_px = selectDicuqStat(
            try calcFieldStats(allocator, total_px_vals),
            kind,
        ),
        .shaded_px = selectDicuqStat(
            try calcFieldStats(allocator, shaded_vals),
            kind,
        ),
        .geom_time_ms = selectDicuqStat(
            try calcFieldStats(allocator, geom_vals),
            kind,
        ),
        .cam_time_ms = selectDicuqStat(
            try calcFieldStats(allocator, cam_vals),
            kind,
        ),
        .elem_loop_time_ms = selectDicuqStat(
            try calcFieldStats(allocator, elem_loop_vals),
            kind,
        ),
        .resolve_time_ms = selectDicuqStat(
            try calcFieldStats(allocator, resolve_vals),
            kind,
        ),
        .raster_time_ms = selectDicuqStat(
            try calcFieldStats(allocator, raster_vals),
            kind,
        ),
        .save_time_ms = selectDicuqStat(
            try calcFieldStats(allocator, save_vals),
            kind,
        ),
        .frame_time_ms = selectDicuqStat(
            try calcFieldStats(allocator, frame_vals),
            kind,
        ),
        .e2e_time_ms = null,
        .geom_tpx_melem_s = selectDicuqStat(
            try calcFieldStats(allocator, geom_tpx_vals),
            kind,
        ),
        .raster_tpx_mpx_s = selectDicuqStat(
            try calcFieldStats(allocator, raster_tpx_vals),
            kind,
        ),
        .frame_tpx_mpx_s = selectDicuqStat(
            try calcFieldStats(allocator, frame_tpx_vals),
            kind,
        ),
        .e2e_tpx_mpx_s = null,
    };
}

fn selectOptionalDicuqStat(
    allocator: std.mem.Allocator,
    values: []const F,
    kind: DicuqStatsKind,
) !?F {
    var any_nonzero = false;
    for (values) |val| {
        if (val != 0.0) {
            any_nonzero = true;
            break;
        }
    }
    if (!any_nonzero) {
        return null;
    }
    return selectDicuqStat(
        try calcFieldStats(allocator, values),
        kind,
    );
}

fn calcE2EStatsRow(
    allocator: std.mem.Allocator,
    camera_idx: ?usize,
    e2e_rows_by_run: []const []const DicuqE2ERow,
    kind: DicuqStatsKind,
) !DicuqE2EStatsRow {
    const count: usize = e2e_rows_by_run.len;

    var elems_vals = try allocator.alloc(F, count);
    defer allocator.free(elems_vals);
    var vis_vals = try allocator.alloc(F, count);
    defer allocator.free(vis_vals);
    var total_px_vals = try allocator.alloc(F, count);
    defer allocator.free(total_px_vals);
    var shaded_vals = try allocator.alloc(F, count);
    defer allocator.free(shaded_vals);
    var geom_vals = try allocator.alloc(F, count);
    defer allocator.free(geom_vals);
    var cam_vals = try allocator.alloc(F, count);
    defer allocator.free(cam_vals);
    var elem_loop_vals = try allocator.alloc(F, count);
    defer allocator.free(elem_loop_vals);
    var resolve_vals = try allocator.alloc(F, count);
    defer allocator.free(resolve_vals);
    var raster_vals = try allocator.alloc(F, count);
    defer allocator.free(raster_vals);
    var save_vals = try allocator.alloc(F, count);
    defer allocator.free(save_vals);
    var frame_vals = try allocator.alloc(F, count);
    defer allocator.free(frame_vals);
    var e2e_vals = try allocator.alloc(F, count);
    defer allocator.free(e2e_vals);
    var geom_tpx_vals = try allocator.alloc(F, count);
    defer allocator.free(geom_tpx_vals);
    var raster_tpx_vals = try allocator.alloc(F, count);
    defer allocator.free(raster_tpx_vals);
    var frame_tpx_vals = try allocator.alloc(F, count);
    defer allocator.free(frame_tpx_vals);
    var e2e_tpx_vals = try allocator.alloc(F, count);
    defer allocator.free(e2e_tpx_vals);

    var have_e2e = true;
    for (e2e_rows_by_run, 0..) |rows, rr| {
        var matched = false;
        for (rows) |row| {
            if (row.camera_idx != camera_idx) continue;
            matched = true;
            elems_vals[rr] = @floatFromInt(row.total_elems);
            vis_vals[rr] = @floatFromInt(row.vis_elems);
            total_px_vals[rr] = @floatFromInt(row.total_px);
            shaded_vals[rr] = @floatFromInt(row.shaded_px);
            geom_vals[rr] = row.geom_time_ms;
            cam_vals[rr] = row.cam_time_ms;
            elem_loop_vals[rr] = row.elem_loop_time_ms;
            resolve_vals[rr] = row.resolve_time_ms;
            raster_vals[rr] = row.raster_time_ms;
            save_vals[rr] = row.save_time_ms;
            frame_vals[rr] = row.frame_time_ms;
            if (row.e2e_time_ms) |val| {
                e2e_vals[rr] = val;
            } else {
                have_e2e = false;
                e2e_vals[rr] = 0.0;
            }
            geom_tpx_vals[rr] = row.geom_tpx_melem_s;
            raster_tpx_vals[rr] = row.raster_tpx_mpx_s;
            frame_tpx_vals[rr] = row.frame_tpx_mpx_s;
            e2e_tpx_vals[rr] = if (row.e2e_tpx_mpx_s) |val|
                val
            else
                0.0;
            break;
        }
        if (!matched) {
            return error.MissingE2ERowForCamera;
        }
    }

    return .{
        .camera_idx = camera_idx,
        .total_elems = selectDicuqStat(
            try calcFieldStats(allocator, elems_vals),
            kind,
        ),
        .vis_elems = selectDicuqStat(
            try calcFieldStats(allocator, vis_vals),
            kind,
        ),
        .total_px = selectDicuqStat(
            try calcFieldStats(allocator, total_px_vals),
            kind,
        ),
        .shaded_px = selectDicuqStat(
            try calcFieldStats(allocator, shaded_vals),
            kind,
        ),
        .geom_time_ms = selectDicuqStat(
            try calcFieldStats(allocator, geom_vals),
            kind,
        ),
        .cam_time_ms = selectDicuqStat(
            try calcFieldStats(allocator, cam_vals),
            kind,
        ),
        .elem_loop_time_ms = selectDicuqStat(
            try calcFieldStats(allocator, elem_loop_vals),
            kind,
        ),
        .resolve_time_ms = selectDicuqStat(
            try calcFieldStats(allocator, resolve_vals),
            kind,
        ),
        .raster_time_ms = selectDicuqStat(
            try calcFieldStats(allocator, raster_vals),
            kind,
        ),
        .save_time_ms = selectDicuqStat(
            try calcFieldStats(allocator, save_vals),
            kind,
        ),
        .frame_time_ms = selectDicuqStat(
            try calcFieldStats(allocator, frame_vals),
            kind,
        ),
        .e2e_time_ms = if (have_e2e)
            selectDicuqStat(
                try calcFieldStats(allocator, e2e_vals),
                kind,
            )
        else
            null,
        .geom_tpx_melem_s = selectDicuqStat(
            try calcFieldStats(allocator, geom_tpx_vals),
            kind,
        ),
        .raster_tpx_mpx_s = selectDicuqStat(
            try calcFieldStats(allocator, raster_tpx_vals),
            kind,
        ),
        .frame_tpx_mpx_s = selectDicuqStat(
            try calcFieldStats(allocator, frame_tpx_vals),
            kind,
        ),
        .e2e_tpx_mpx_s = if (have_e2e)
            try selectOptionalDicuqStat(
                allocator,
                e2e_tpx_vals,
                kind,
            )
        else
            null,
    };
}

fn appendFrameStatsRowsCSV(
    allocator: std.mem.Allocator,
    rows_buf: *std.ArrayList(u8),
    case_name: []const u8,
    run_idx: usize,
    camera_count: usize,
    frame_rows: []const DicuqFrameRow,
    kind: DicuqStatsKind,
) !void {
    try rows_buf.appendSlice(
        allocator,
        "Run,Case,Camera,Total Elems,Vis Elems,Total Px,Shaded Px," ++
            "Geom Time [ms],Cam Inv Time [ms],Elem Loop Time [ms]," ++
            "Resolve Time [ms],Raster Time [ms],Save Time [ms]," ++
            "Frame Time [ms],E2E Time [ms],Geom TP [MElem/s]," ++
            "Raster TP [MPx/s],Frame TP [MPx/s],E2E TP [MPx/s]\n",
    );

    for (0..camera_count + 1) |cc| {
        const camera_idx_opt: ?usize =
            if (cc < camera_count) cc else null;
        const stats_row = try calcFrameStatsRow(
            allocator,
            run_idx,
            camera_idx_opt,
            frame_rows,
            kind,
        );
        const camera_text =
            try cameraLabel(allocator, camera_idx_opt);
        defer allocator.free(camera_text);
        const e2e_time_text = if (stats_row.e2e_time_ms) |val|
            try std.fmt.allocPrint(
                allocator,
                "{d:.6}",
                .{val},
            )
        else
            try allocator.dupe(u8, "");
        defer allocator.free(e2e_time_text);
        const e2e_tp_text = if (stats_row.e2e_tpx_mpx_s) |val|
            try std.fmt.allocPrint(
                allocator,
                "{d:.6}",
                .{val},
            )
        else
            try allocator.dupe(u8, "");
        defer allocator.free(e2e_tp_text);
        const elem_loop_ms = stats_row.elem_loop_time_ms;
        const row = try std.fmt.allocPrint(
            allocator,
            "{d},{s},{s},{d:.6},{d:.6},{d:.6},{d:.6}," ++
                "{d:.6},{d:.6},{d:.6},{d:.6},{d:.6}," ++
                "{d:.6},{d:.6},{s},{d:.6},{d:.6},{d:.6},{s}\n",
            .{
                stats_row.run_idx,
                case_name,
                camera_text,
                stats_row.total_elems,
                stats_row.vis_elems,
                stats_row.total_px,
                stats_row.shaded_px,
                stats_row.geom_time_ms,
                stats_row.cam_time_ms,
                elem_loop_ms,
                stats_row.resolve_time_ms,
                stats_row.raster_time_ms,
                stats_row.save_time_ms,
                stats_row.frame_time_ms,
                e2e_time_text,
                stats_row.geom_tpx_melem_s,
                stats_row.raster_tpx_mpx_s,
                stats_row.frame_tpx_mpx_s,
                e2e_tp_text,
            },
        );
        defer allocator.free(row);
        try rows_buf.appendSlice(allocator, row);
    }
}

fn appendE2EStatsRowsCSV(
    allocator: std.mem.Allocator,
    rows_buf: *std.ArrayList(u8),
    case_name: []const u8,
    camera_count: usize,
    e2e_rows_by_run: []const []const DicuqE2ERow,
    kind: DicuqStatsKind,
) !void {
    try rows_buf.appendSlice(
        allocator,
        "Case,Camera,Total Elems,Vis Elems,Total Px,Shaded Px," ++
            "Geom Time [ms],Cam Inv Time [ms],Elem Loop Time [ms]," ++
            "Resolve Time [ms],Raster Time [ms],Save Time [ms]," ++
            "Frame Time [ms],E2E Time [ms],Geom TP [MElem/s]," ++
            "Raster TP [MPx/s],Frame TP [MPx/s],E2E TP [MPx/s]\n",
    );

    for (0..camera_count + 1) |cc| {
        const camera_idx_opt: ?usize =
            if (cc < camera_count) cc else null;
        const stats_row = try calcE2EStatsRow(
            allocator,
            camera_idx_opt,
            e2e_rows_by_run,
            kind,
        );
        const camera_text =
            try cameraLabel(allocator, camera_idx_opt);
        defer allocator.free(camera_text);
        const e2e_time_text = if (stats_row.e2e_time_ms) |val|
            try std.fmt.allocPrint(
                allocator,
                "{d:.6}",
                .{val},
            )
        else
            try allocator.dupe(u8, "");
        defer allocator.free(e2e_time_text);
        const e2e_tp_text = if (stats_row.e2e_tpx_mpx_s) |val|
            try std.fmt.allocPrint(
                allocator,
                "{d:.6}",
                .{val},
            )
        else
            try allocator.dupe(u8, "");
        defer allocator.free(e2e_tp_text);
        const elem_loop_ms = stats_row.elem_loop_time_ms;
        const row = try std.fmt.allocPrint(
            allocator,
            "{s},{s},{d:.6},{d:.6},{d:.6},{d:.6}," ++
                "{d:.6},{d:.6},{d:.6},{d:.6},{d:.6}," ++
                "{d:.6},{d:.6},{s},{d:.6},{d:.6},{d:.6},{s}\n",
            .{
                case_name,
                camera_text,
                stats_row.total_elems,
                stats_row.vis_elems,
                stats_row.total_px,
                stats_row.shaded_px,
                stats_row.geom_time_ms,
                stats_row.cam_time_ms,
                elem_loop_ms,
                stats_row.resolve_time_ms,
                stats_row.raster_time_ms,
                stats_row.save_time_ms,
                stats_row.frame_time_ms,
                e2e_time_text,
                stats_row.geom_tpx_melem_s,
                stats_row.raster_tpx_mpx_s,
                stats_row.frame_tpx_mpx_s,
                e2e_tp_text,
            },
        );
        defer allocator.free(row);
        try rows_buf.appendSlice(allocator, row);
    }
}

pub fn writeRunCSVs(
    allocator: std.mem.Allocator,
    io: std.Io,
    out_dir_base: []const u8,
    case_name: []const u8,
    run_idx: usize,
    camera_count: usize,
    run_result: DicuqRunResult,
) !void {
    var byframe_buf: std.ArrayList(u8) = .empty;
    defer byframe_buf.deinit(allocator);
    try appendFrameRowsCSV(
        allocator,
        &byframe_buf,
        case_name,
        run_result.frame_rows,
    );
    const byframe_name = try std.fmt.allocPrint(
        allocator,
        "bench_run{d}_byframe.csv",
        .{run_idx},
    );
    defer allocator.free(byframe_name);
    try writeDicuqFile(
        allocator,
        io,
        out_dir_base,
        byframe_name,
        byframe_buf.items,
    );

    var e2e_buf: std.ArrayList(u8) = .empty;
    defer e2e_buf.deinit(allocator);
    try appendE2ERowsCSV(
        allocator,
        &e2e_buf,
        case_name,
        run_result.e2e_rows,
    );
    const e2e_name = try std.fmt.allocPrint(
        allocator,
        "bench_run{d}_e2e.csv",
        .{run_idx},
    );
    defer allocator.free(e2e_name);
    try writeDicuqFile(
        allocator,
        io,
        out_dir_base,
        e2e_name,
        e2e_buf.items,
    );

    inline for (.{ .median, .mad, .min, .max }) |kind| {
        var stats_buf: std.ArrayList(u8) = .empty;
        defer stats_buf.deinit(allocator);
        try appendFrameStatsRowsCSV(
            allocator,
            &stats_buf,
            case_name,
            run_idx,
            camera_count,
            run_result.frame_rows,
            kind,
        );
        const file_name = try std.fmt.allocPrint(
            allocator,
            "bench_run{d}_overframes_{s}.csv",
            .{ run_idx, @tagName(kind) },
        );
        defer allocator.free(file_name);
        try writeDicuqFile(
            allocator,
            io,
            out_dir_base,
            file_name,
            stats_buf.items,
        );
    }
}

pub fn writeE2EOverRunsCSVs(
    allocator: std.mem.Allocator,
    io: std.Io,
    out_dir_base: []const u8,
    case_name: []const u8,
    camera_count: usize,
    e2e_rows_by_run: []const []const DicuqE2ERow,
) !void {
    inline for (.{ .median, .mad, .min, .max }) |kind| {
        var stats_buf: std.ArrayList(u8) = .empty;
        defer stats_buf.deinit(allocator);
        try appendE2EStatsRowsCSV(
            allocator,
            &stats_buf,
            case_name,
            camera_count,
            e2e_rows_by_run,
            kind,
        );
        const file_name = try std.fmt.allocPrint(
            allocator,
            "bench_e2e_overruns_{s}.csv",
            .{@tagName(kind)},
        );
        defer allocator.free(file_name);
        try writeDicuqFile(
            allocator,
            io,
            out_dir_base,
            file_name,
            stats_buf.items,
        );
    }
}
