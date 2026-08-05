// --------------------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const buildconfig = @import("buildconfig.zig");
const F = buildconfig.F;
const ndarray = @import("ndarray.zig");
const iio = @import("imageio.zig");
const matslice = @import("matslice.zig");
const cam = @import("camera.zig");
const mo = @import("meshpipeline.zig");
const newton = @import("newton.zig");
const rastcfg = @import("rasterconfig.zig");
const scalingpolicy = @import("scalingpolicy.zig");
pub const ReportMode = rastcfg.ReportMode;

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const OffLog = struct {};

pub const FrameTimes = struct {
    setup_frame_buff: F = 0,
    prepare_frame_context: F = 0,
    geometry_prep: F = 0,
    geom_coord_ops: F = 0,
    geom_cull_ops: F = 0,
    geom_prep_hulls_shaders: F = 0,
    geom_remap_inds: F = 0,
    tile_overlap: F = 0,
    raster_loop: F = 0,
    cam_invert: F = 0,
    scratch_resolve: F = 0,
    save_frame: F = 0,
    active_time: F = 0,
    latency_time: F = 0,
};

pub const EndToEndTimes = struct {
    setup_time: F = 0,
    setup_other_time: F = 0,
    setup_frame_buff_time: F = 0,
    dispatch_time: F = 0,
    total_time: F = 0,
};

pub const BenchLog = struct {
    frame_times: FrameTimes = .{},
    total_nodes: usize = 0,
    total_elems: usize = 0,
    vis_elems: usize = 0,
    solver_calls: u64 = 0,
    total_solver_iters: u64 = 0,
    solver_diverged: u64 = 0,
    tess_checks: u64 = 0,
    tess_passes: u64 = 0,
    total_shaded_px: u64 = 0,
    total_depth_test: u64 = 0,
    depth_tests_fail: u64 = 0,
    max_tile_elems: usize = 0,
    cam_time_ns: F = 0,
    resolve_time_ns: F = 0,
};

pub const FrameBenchCapture = struct {
    camera_idx: usize,
    frame_idx: usize,
    bench_log: BenchLog,
};

pub const FrameReportStorage = union(ReportMode) {
    off: OffLog,
    bench: BenchLog,
    full_stats: FullStatsLog,
};

// --------------------------------------------------------------------------------------
// Public Entry-Point Functions
// --------------------------------------------------------------------------------------

pub fn deinitFrameReportStorage(
    outer_alloc: std.mem.Allocator,
    config: rastcfg.RasterConfig,
    report_storage: *FrameReportStorage,
) void {
    if (config.report == .full_stats) {
        report_storage.full_stats.deinit(outer_alloc);
    }
    report_storage.* = .{ .off = .{} };
}

pub fn FrameReportPtr(comptime report_mode: ReportMode) type {
    return *LogType(report_mode);
}

pub fn calcBenchCaptureIdx(
    cameras_num: usize,
    camera_idx: usize,
    frame_idx: usize,
) usize {
    return frame_idx * cameras_num + camera_idx;
}

// --------------------------------------------------------------------------------------
// Publish Pipeline
// --------------------------------------------------------------------------------------

pub fn publishFrameResults(
    outer_alloc: std.mem.Allocator,
    io: std.Io,
    config: rastcfg.RasterConfig,
    actual_tile_size: u16,
    camera: *const cam.CameraPrepared,
    camera_idx: usize,
    frame_idx: usize,
    cameras_num: usize,
    out_dir: ?std.Io.Dir,
    bench_capture: ?[]FrameBenchCapture,
    report_storage: *FrameReportStorage,
    frame_times: FrameTimes,
    total_nodes_num: usize,
    total_elems_num: usize,
    total_elems_in_image: usize,
    prep_meshes: []const mo.MeshPrepared,
) !void {
    const nodes_per_elem = mo.calcNodesPerElem(prep_meshes);
    try publishFrameResultsWithNodesPerElem(
        outer_alloc,
        io,
        config,
        actual_tile_size,
        camera,
        camera_idx,
        frame_idx,
        cameras_num,
        out_dir,
        bench_capture,
        report_storage,
        frame_times,
        total_nodes_num,
        total_elems_num,
        total_elems_in_image,
        nodes_per_elem,
    );
}

pub fn publishFrameResultsWithNodesPerElem(
    outer_alloc: std.mem.Allocator,
    io: std.Io,
    config: rastcfg.RasterConfig,
    actual_tile_size: u16,
    camera: *const cam.CameraPrepared,
    camera_idx: usize,
    frame_idx: usize,
    cameras_num: usize,
    out_dir: ?std.Io.Dir,
    bench_capture: ?[]FrameBenchCapture,
    report_storage: *FrameReportStorage,
    frame_times: FrameTimes,
    total_nodes_num: usize,
    total_elems_num: usize,
    total_elems_in_image: usize,
    nodes_per_elem: F,
) !void {
    switch (config.report) {
        .off => {
            if (bench_capture) |capture| {
                const capture_idx = calcBenchCaptureIdx(
                    cameras_num,
                    camera_idx,
                    frame_idx,
                );
                capture[capture_idx] = .{
                    .camera_idx = camera_idx,
                    .frame_idx = frame_idx,
                    .bench_log = .{
                        .frame_times = frame_times,
                        .total_nodes = total_nodes_num,
                        .total_elems = total_elems_num,
                        .vis_elems = total_elems_in_image,
                    },
                };
            }
        },
        .bench => {
            report_storage.bench.frame_times = frame_times;
            report_storage.bench.total_nodes = total_nodes_num;
            report_storage.bench.total_elems = total_elems_num;
            report_storage.bench.vis_elems = total_elems_in_image;
            if (bench_capture) |capture| {
                const capture_idx = calcBenchCaptureIdx(
                    cameras_num,
                    camera_idx,
                    frame_idx,
                );
                capture[capture_idx] = .{
                    .camera_idx = camera_idx,
                    .frame_idx = frame_idx,
                    .bench_log = report_storage.bench,
                };
            }
            try standardReport(
                io,
                camera,
                actual_tile_size,
                frame_idx,
                camera_idx,
                frame_times,
                total_elems_num,
                total_elems_in_image,
                nodes_per_elem,
                &report_storage.bench,
            );
        },
        .full_stats => {
            report_storage.full_stats.bench.frame_times = frame_times;
            report_storage.full_stats.bench.total_nodes = total_nodes_num;
            report_storage.full_stats.bench.total_elems = total_elems_num;
            report_storage.full_stats.bench.vis_elems = total_elems_in_image;
            if (bench_capture) |capture| {
                const capture_idx = calcBenchCaptureIdx(
                    cameras_num,
                    camera_idx,
                    frame_idx,
                );
                capture[capture_idx] = .{
                    .camera_idx = camera_idx,
                    .frame_idx = frame_idx,
                    .bench_log = report_storage.full_stats.bench,
                };
            }
            try report_storage.full_stats.saveFrameReport(
                io,
                outer_alloc,
                out_dir,
                camera_idx,
                frame_idx,
                camera,
                actual_tile_size,
                config.full_stats_opts,
                nodes_per_elem,
            );
        },
    }
}

pub fn reduceBenchLog(dst: *BenchLog, src: *const BenchLog) void {
    dst.total_nodes += src.total_nodes;
    dst.solver_calls += src.solver_calls;
    dst.total_solver_iters += src.total_solver_iters;
    dst.solver_diverged += src.solver_diverged;
    dst.tess_checks += src.tess_checks;
    dst.tess_passes += src.tess_passes;
    dst.total_shaded_px += src.total_shaded_px;
    dst.total_depth_test += src.total_depth_test;
    dst.depth_tests_fail += src.depth_tests_fail;
    dst.max_tile_elems = @max(
        dst.max_tile_elems,
        src.max_tile_elems,
    );
    dst.cam_time_ns += src.cam_time_ns;
    dst.resolve_time_ns += src.resolve_time_ns;
}

pub const FullStatsLog = struct {
    bench: BenchLog = .{},
    iter_map: ?ndarray.NDArray(F) = null,
    xi_map: ?ndarray.NDArray(F) = null,
    eta_map: ?ndarray.NDArray(F) = null,
    conv_map: ?ndarray.NDArray(F) = null,
    solver_status_map: ?ndarray.NDArray(F) = null,
    pre_dom_conv_map: ?ndarray.NDArray(F) = null,
    hit_iter_lim_map: ?ndarray.NDArray(F) = null,
    jac_det_map: ?ndarray.NDArray(F) = null,
    resid_mag_map: ?ndarray.NDArray(F) = null,
    resid_x_map: ?ndarray.NDArray(F) = null,
    resid_y_map: ?ndarray.NDArray(F) = null,
    interp_w_map: ?ndarray.NDArray(F) = null,
    norm_resid_mag_map: ?ndarray.NDArray(F) = null,
    dom_violation_map: ?ndarray.NDArray(F) = null,
    pixel_occupancy_map: ?ndarray.NDArray(F) = null,
    depth_map: ?ndarray.NDArray(F) = null,
    normals_map: ?ndarray.NDArray(F) = null,
    earlyout_map: ?ndarray.NDArray(F) = null,
    tile_timing_map: ?ndarray.NDArray(F) = null,
    tile_density_map: ?ndarray.NDArray(F) = null,
    tile_occupancy_map: ?ndarray.NDArray(F) = null,

    pub fn deinit(self: *FullStatsLog, allocator: std.mem.Allocator) void {
        if (self.iter_map) |*imap| imap.deinit(allocator);
        if (self.xi_map) |*xmap| xmap.deinit(allocator);
        if (self.eta_map) |*emap| emap.deinit(allocator);
        if (self.conv_map) |*cmap| cmap.deinit(allocator);
        if (self.solver_status_map) |*smap| smap.deinit(allocator);
        if (self.pre_dom_conv_map) |*pmap| pmap.deinit(allocator);
        if (self.hit_iter_lim_map) |*hmap| hmap.deinit(allocator);
        if (self.jac_det_map) |*jmap| jmap.deinit(allocator);
        if (self.resid_mag_map) |*rmap| rmap.deinit(allocator);
        if (self.resid_x_map) |*rmap| rmap.deinit(allocator);
        if (self.resid_y_map) |*rmap| rmap.deinit(allocator);
        if (self.interp_w_map) |*wmap| wmap.deinit(allocator);
        if (self.norm_resid_mag_map) |*nmap| nmap.deinit(allocator);
        if (self.dom_violation_map) |*dmap| dmap.deinit(allocator);
        if (self.pixel_occupancy_map) |*pomap| pomap.deinit(allocator);
        if (self.depth_map) |*dmap| dmap.deinit(allocator);
        if (self.normals_map) |*nmap| nmap.deinit(allocator);
        if (self.earlyout_map) |*emap| emap.deinit(allocator);
        if (self.tile_timing_map) |*tmap| tmap.deinit(allocator);
        if (self.tile_density_map) |*dmap| dmap.deinit(allocator);
        if (self.tile_occupancy_map) |*omap| omap.deinit(allocator);
    }

    fn saveTileMapAsImage(
        io: std.Io,
        allocator: std.mem.Allocator,
        save_dir: std.Io.Dir,
        camera: *const cam.CameraPrepared,
        tile_size: u16,
        tile_data: []const F,
        name_prefix: []const u8,
        opts: rastcfg.FullStatsOpts,
    ) !void {
        const px_x = camera.pixels_num[0];
        const px_y = camera.pixels_num[1];
        const tiles_x = try std.math.divCeil(u32, px_x, tile_size);

        var expanded = try ndarray.NDArray(F).initFlat(
            allocator,
            &[_]usize{ px_y, px_x },
        );
        defer expanded.deinit(allocator);

        for (0..px_y) |yy| {
            const tile_y = yy / tile_size;
            for (0..px_x) |xx| {
                const tile_x = xx / tile_size;
                const tile_idx = tile_y * tiles_x + tile_x;
                expanded.slice[yy * px_x + xx] = tile_data[tile_idx];
            }
        }

        const mat = matslice.MatSlice(F).init(expanded.slice, px_y, px_x);
        for (opts.formats) |opt| {
            try iio.saveMatAsImage(io, save_dir, name_prefix, &mat, opt);
        }
    }

    fn saveSolverDiagnosticsCsv(
        self: *const FullStatsLog,
        io: std.Io,
        save_dir: std.Io.Dir,
        camera_idx: usize,
        frame_idx: usize,
        camera: *const cam.CameraPrepared,
    ) !void {
        const iter_map = self.iter_map orelse return;
        const conv_map = self.conv_map orelse return;
        const earlyout_map = self.earlyout_map orelse return;

        const sub_samp: usize = @intCast(camera.sub_sample);
        const rows_num = camera.pixels_num[1] * sub_samp;
        const cols_num = camera.pixels_num[0] * sub_samp;

        var name_buf: [1024]u8 = undefined;
        const file_name = try std.fmt.bufPrint(
            name_buf[0..],
            "diag_cam{d}_frame{d}_solver.csv",
            .{ camera_idx, frame_idx },
        );

        const csv_file = try save_dir.createFile(io, file_name, .{});
        defer csv_file.close(io);

        var write_buf: [4096]u8 = undefined;
        var file_writer = csv_file.writer(io, &write_buf);
        const writer = &file_writer.interface;

        try writer.writeAll("subpx_x,subpx_y,iters,conv,solver_status,");
        try writer.writeAll("pre_dom_conv,hit_iter_lim,resid_x,resid_y,");
        try writer.writeAll("interp_w,resid_mag,norm_resid_mag,");
        try writer.writeAll("jac_det,xi,eta,dom_violation,earlyout,inv_z\n");

        const iter_row_stride = iter_map.strides[0];

        for (0..rows_num) |yy| {
            for (0..cols_num) |xx| {
                const idx = yy * iter_row_stride + xx;
                const iters = iter_map.slice[idx];
                const conv = conv_map.slice[idx];
                const earlyout = earlyout_map.slice[idx];

                if (iters <= 0.0 and earlyout <= 0.0 and conv <= 0.0) {
                    continue;
                }

                const pre_dom_conv =
                    if (self.pre_dom_conv_map) |*m| m.slice[idx] else 0.0;
                const solver_status =
                    if (self.solver_status_map) |*m| m.slice[idx] else std.math.nan(F);
                const hit_iter_lim =
                    if (self.hit_iter_lim_map) |*m| m.slice[idx] else 0.0;
                const resid_mag =
                    if (self.resid_mag_map) |*m| m.slice[idx] else std.math.nan(F);
                const resid_x =
                    if (self.resid_x_map) |*m| m.slice[idx] else std.math.nan(F);
                const resid_y =
                    if (self.resid_y_map) |*m| m.slice[idx] else std.math.nan(F);
                const interp_w =
                    if (self.interp_w_map) |*m| m.slice[idx] else std.math.nan(F);
                const norm_resid_mag =
                    if (self.norm_resid_mag_map) |*m| m.slice[idx] else std.math.nan(F);
                const jac_det =
                    if (self.jac_det_map) |*m| m.slice[idx] else std.math.nan(F);
                const xi =
                    if (self.xi_map) |*m| m.slice[idx] else std.math.nan(F);
                const eta =
                    if (self.eta_map) |*m| m.slice[idx] else std.math.nan(F);
                const dom_violation =
                    if (self.dom_violation_map) |*m| m.slice[idx] else std.math.nan(F);
                const inv_z =
                    if (self.depth_map) |*m| m.slice[idx] else std.math.nan(F);

                const status_label =
                    if (std.math.isFinite(solver_status))
                        newton.statusLabel(
                            @enumFromInt(@as(
                                u8,
                                @intFromFloat(solver_status),
                            )),
                        )
                    else
                        "unknown";

                try writer.print(
                    "{d},{d},{d},{d},{s},{d},{d},{d},{d},{d},{d},{d},{d},{d},{d},{d},{d},{d}\n",
                    .{
                        xx,
                        yy,
                        iters,
                        conv,
                        status_label,
                        pre_dom_conv,
                        hit_iter_lim,
                        resid_x,
                        resid_y,
                        interp_w,
                        resid_mag,
                        norm_resid_mag,
                        jac_det,
                        xi,
                        eta,
                        dom_violation,
                        earlyout,
                        inv_z,
                    },
                );
            }
        }

        try file_writer.flush();
    }

    pub fn saveFrameReport(
        self: *const FullStatsLog,
        io: std.Io,
        allocator: std.mem.Allocator,
        out_dir: ?std.Io.Dir,
        camera_idx: usize,
        frame_idx: usize,
        camera: *const cam.CameraPrepared,
        tile_size: u16,
        opts: rastcfg.FullStatsOpts,
        nodes_per_elem: F,
    ) !void {
        const save_dir = out_dir orelse return;

        var name_buff: [1024]u8 = undefined;
        const stats_file_name = try std.fmt.bufPrint(
            name_buff[0..],
            "report_stats_cam{d}_frame{d}.txt",
            .{ camera_idx, frame_idx },
        );

        var stats_file = try save_dir.createFile(io, stats_file_name, .{});
        defer stats_file.close(io);

        var write_buf: [4096]u8 = undefined;
        var file_writer = stats_file.writer(io, &write_buf);
        try self.writeReport(&file_writer.interface, frame_idx, camera, nodes_per_elem);
        try self.fullReport(io, frame_idx, camera, nodes_per_elem);
        if (opts.save_solver_csv) {
            try self.saveSolverDiagnosticsCsv(
                io,
                save_dir,
                camera_idx,
                frame_idx,
                camera,
            );
        }

        if (self.iter_map) |*m| {
            const sub_samp: usize = @intCast(camera.sub_sample);
            const mat = matslice.MatSlice(F).init(
                m.slice,
                camera.pixels_num[1] * sub_samp,
                camera.pixels_num[0] * sub_samp,
            );
            const name = try std.fmt.bufPrint(
                name_buff[0..],
                "diag_cam{d}_frame{d}_iters",
                .{ camera_idx, frame_idx },
            );
            for (opts.formats) |opt| {
                try iio.saveMatAsImage(io, save_dir, name, &mat, opt);
            }
        }

        if (self.xi_map) |*m| {
            const sub_samp: usize = @intCast(camera.sub_sample);
            const mat = matslice.MatSlice(F).init(
                m.slice,
                camera.pixels_num[1] * sub_samp,
                camera.pixels_num[0] * sub_samp,
            );
            const name = try std.fmt.bufPrint(
                name_buff[0..],
                "diag_cam{d}_frame{d}_xi",
                .{ camera_idx, frame_idx },
            );
            for (opts.formats) |opt| {
                try iio.saveMatAsImage(io, save_dir, name, &mat, opt);
            }
        }

        if (self.eta_map) |*m| {
            const sub_samp: usize = @intCast(camera.sub_sample);
            const mat = matslice.MatSlice(F).init(
                m.slice,
                camera.pixels_num[1] * sub_samp,
                camera.pixels_num[0] * sub_samp,
            );
            const name = try std.fmt.bufPrint(
                name_buff[0..],
                "diag_cam{d}_frame{d}_eta",
                .{ camera_idx, frame_idx },
            );
            for (opts.formats) |opt| {
                try iio.saveMatAsImage(io, save_dir, name, &mat, opt);
            }
        }

        if (self.conv_map) |*m| {
            const sub_samp: usize = @intCast(camera.sub_sample);
            const mat = matslice.MatSlice(F).init(
                m.slice,
                camera.pixels_num[1] * sub_samp,
                camera.pixels_num[0] * sub_samp,
            );
            const name = try std.fmt.bufPrint(
                name_buff[0..],
                "diag_cam{d}_frame{d}_conv",
                .{ camera_idx, frame_idx },
            );
            for (opts.formats) |opt| {
                try iio.saveMatAsImage(io, save_dir, name, &mat, opt);
            }
        }

        if (self.pre_dom_conv_map) |*m| {
            const sub_samp: usize = @intCast(camera.sub_sample);
            const mat = matslice.MatSlice(F).init(
                m.slice,
                camera.pixels_num[1] * sub_samp,
                camera.pixels_num[0] * sub_samp,
            );
            const name = try std.fmt.bufPrint(
                name_buff[0..],
                "diag_cam{d}_frame{d}_pre_dom_conv",
                .{ camera_idx, frame_idx },
            );
            for (opts.formats) |opt| {
                try iio.saveMatAsImage(io, save_dir, name, &mat, opt);
            }
        }

        if (self.hit_iter_lim_map) |*m| {
            const sub_samp: usize = @intCast(camera.sub_sample);
            const mat = matslice.MatSlice(F).init(
                m.slice,
                camera.pixels_num[1] * sub_samp,
                camera.pixels_num[0] * sub_samp,
            );
            const name = try std.fmt.bufPrint(
                name_buff[0..],
                "diag_cam{d}_frame{d}_hit_iter_lim",
                .{ camera_idx, frame_idx },
            );
            for (opts.formats) |opt| {
                try iio.saveMatAsImage(io, save_dir, name, &mat, opt);
            }
        }

        if (self.jac_det_map) |*m| {
            const sub_samp: usize = @intCast(camera.sub_sample);
            const mat = matslice.MatSlice(F).init(
                m.slice,
                camera.pixels_num[1] * sub_samp,
                camera.pixels_num[0] * sub_samp,
            );
            const name = try std.fmt.bufPrint(
                name_buff[0..],
                "diag_cam{d}_frame{d}_Jdet",
                .{ camera_idx, frame_idx },
            );
            for (opts.formats) |opt| {
                try iio.saveMatAsImage(io, save_dir, name, &mat, opt);
            }
        }

        if (self.resid_mag_map) |*m| {
            const sub_samp: usize = @intCast(camera.sub_sample);
            const mat = matslice.MatSlice(F).init(
                m.slice,
                camera.pixels_num[1] * sub_samp,
                camera.pixels_num[0] * sub_samp,
            );
            const name = try std.fmt.bufPrint(
                name_buff[0..],
                "diag_cam{d}_frame{d}_resid_mag",
                .{ camera_idx, frame_idx },
            );
            for (opts.formats) |opt| {
                try iio.saveMatAsImage(io, save_dir, name, &mat, opt);
            }
        }

        if (self.interp_w_map) |*m| {
            const sub_samp: usize = @intCast(camera.sub_sample);
            const mat = matslice.MatSlice(F).init(
                m.slice,
                camera.pixels_num[1] * sub_samp,
                camera.pixels_num[0] * sub_samp,
            );
            const name = try std.fmt.bufPrint(
                name_buff[0..],
                "diag_cam{d}_frame{d}_W",
                .{ camera_idx, frame_idx },
            );
            for (opts.formats) |opt| {
                try iio.saveMatAsImage(io, save_dir, name, &mat, opt);
            }
        }

        if (self.norm_resid_mag_map) |*m| {
            const sub_samp: usize = @intCast(camera.sub_sample);
            const mat = matslice.MatSlice(F).init(
                m.slice,
                camera.pixels_num[1] * sub_samp,
                camera.pixels_num[0] * sub_samp,
            );
            const name = try std.fmt.bufPrint(
                name_buff[0..],
                "diag_cam{d}_frame{d}_norm_resid_mag",
                .{ camera_idx, frame_idx },
            );
            for (opts.formats) |opt| {
                try iio.saveMatAsImage(io, save_dir, name, &mat, opt);
            }
        }

        if (self.dom_violation_map) |*m| {
            const sub_samp: usize = @intCast(camera.sub_sample);
            const mat = matslice.MatSlice(F).init(
                m.slice,
                camera.pixels_num[1] * sub_samp,
                camera.pixels_num[0] * sub_samp,
            );
            const name = try std.fmt.bufPrint(
                name_buff[0..],
                "diag_cam{d}_frame{d}_dom_violation",
                .{ camera_idx, frame_idx },
            );
            for (opts.formats) |opt| {
                try iio.saveMatAsImage(io, save_dir, name, &mat, opt);
            }
        }

        if (self.pixel_occupancy_map) |*m| {
            const mat = matslice.MatSlice(F).init(
                m.slice,
                camera.pixels_num[1],
                camera.pixels_num[0],
            );
            const name = try std.fmt.bufPrint(
                name_buff[0..],
                "diag_cam{d}_frame{d}_occupancy",
                .{ camera_idx, frame_idx },
            );
            for (opts.formats) |opt| {
                try iio.saveMatAsImage(io, save_dir, name, &mat, opt);
            }
        }

        if (self.depth_map) |*m| {
            const sub_samp: usize = @intCast(camera.sub_sample);
            const mat = matslice.MatSlice(F).init(
                m.slice,
                camera.pixels_num[1] * sub_samp,
                camera.pixels_num[0] * sub_samp,
            );
            const name = try std.fmt.bufPrint(
                name_buff[0..],
                "diag_cam{d}_frame{d}_depth",
                .{ camera_idx, frame_idx },
            );
            for (opts.formats) |opt| {
                try iio.saveMatAsImage(io, save_dir, name, &mat, opt);
            }
        }

        if (self.normals_map) |*m| {
            const name = try std.fmt.bufPrint(
                name_buff[0..],
                "diag_cam{d}_frame{d}_normals",
                .{ camera_idx, frame_idx },
            );
            for (opts.formats) |opt| {
                var save_opt = opt;
                save_opt.channels = 3;
                try iio.saveImage(io, save_dir, name, m, 0, save_opt);
            }
        }

        if (self.earlyout_map) |*m| {
            const sub_samp: usize = @intCast(camera.sub_sample);
            const mat = matslice.MatSlice(F).init(
                m.slice,
                camera.pixels_num[1] * sub_samp,
                camera.pixels_num[0] * sub_samp,
            );
            const name = try std.fmt.bufPrint(
                name_buff[0..],
                "diag_cam{d}_frame{d}_earlyout",
                .{ camera_idx, frame_idx },
            );
            for (opts.formats) |opt| {
                try iio.saveMatAsImage(io, save_dir, name, &mat, opt);
            }
        }

        if (self.tile_timing_map) |*m| {
            const name = try std.fmt.bufPrint(
                name_buff[0..],
                "diag_cam{d}_frame{d}_tile_timing",
                .{ camera_idx, frame_idx },
            );
            try saveTileMapAsImage(
                io,
                allocator,
                save_dir,
                camera,
                tile_size,
                m.slice,
                name,
                opts,
            );
        }

        if (self.tile_density_map) |*m| {
            const name = try std.fmt.bufPrint(
                name_buff[0..],
                "diag_cam{d}_frame{d}_tile_density",
                .{ camera_idx, frame_idx },
            );
            try saveTileMapAsImage(
                io,
                allocator,
                save_dir,
                camera,
                tile_size,
                m.slice,
                name,
                opts,
            );
        }

        if (self.tile_occupancy_map) |*m| {
            const name = try std.fmt.bufPrint(
                name_buff[0..],
                "diag_cam{d}_frame{d}_tile_occupancy",
                .{ camera_idx, frame_idx },
            );
            try saveTileMapAsImage(
                io,
                allocator,
                save_dir,
                camera,
                tile_size,
                m.slice,
                name,
                opts,
            );
        }
    }

    pub fn fullReport(
        self: *const FullStatsLog,
        io: std.Io,
        frame_idx: usize,
        camera: *const cam.CameraPrepared,
        nodes_per_elem: F,
    ) !void {
        var buff: [4096]u8 = undefined;
        var stderr_writer = std.Io.File.stderr().writer(io, &buff);
        const writer = &stderr_writer.interface;
        try self.writeReport(writer, frame_idx, camera, nodes_per_elem);
    }

    pub fn writeReport(
        self: *const FullStatsLog,
        writer: anytype,
        frame_idx: usize,
        camera: *const cam.CameraPrepared,
        nodes_per_elem: F,
    ) !void {
        const active_ms = self.bench.frame_times.active_time / 1e6;
        const active_sec = self.bench.frame_times.active_time / 1e9;
        const latency_ms = self.bench.frame_times.latency_time / 1e6;
        const raster_sec = self.bench.frame_times.raster_loop / 1e9;
        const geom_tiling_sec =
            (self.bench.frame_times.geometry_prep + self.bench.frame_times.tile_overlap) /
            1e9;

        const border = [_]u8{'='} ** 80 ++ "\n";
        const line = [_]u8{'-'} ** 80 ++ "\n";

        try writer.print("{s}", .{border});
        try writer.print("SOFTWARE RASTER REPORT - FRAME {d}\n", .{frame_idx});
        try writer.print("{s}\n", .{border});

        try writer.print("--- GEOMETRY PIPELINE ---\n", .{});
        try writer.print("Total Elems in Mesh  = {d}\n", .{self.bench.total_elems});
        try writer.print(
            "Elems after Crop     = {d}\n",
            .{self.bench.vis_elems},
        );
        const cropped = self.bench.total_elems - self.bench.vis_elems;
        const crop_pct = if (self.bench.total_elems > 0)
            @as(F, @floatFromInt(cropped)) * 100.0 /
                @as(F, @floatFromInt(self.bench.total_elems))
        else
            0.0;
        try writer.print(
            "Elems Cropped        = {d} ({d:.2}%)\n\n",
            .{ cropped, crop_pct },
        );

        const solver_calls_f = @as(F, @floatFromInt(self.bench.solver_calls));
        const solver_diverged_f = @as(F, @floatFromInt(self.bench.solver_diverged));
        const solver_conv = self.bench.solver_calls - self.bench.solver_diverged;
        const solver_conv_f = @as(F, @floatFromInt(solver_conv));
        const solver_conv_pct = if (self.bench.solver_calls > 0)
            solver_conv_f * 100.0 / solver_calls_f
        else
            0.0;
        const solver_diverged_pct = if (self.bench.solver_calls > 0)
            solver_diverged_f * 100.0 / solver_calls_f
        else
            0.0;
        const avg_iters_per_call = if (self.bench.solver_calls > 0)
            @as(F, @floatFromInt(self.bench.total_solver_iters)) / solver_calls_f
        else
            0.0;

        const px_x = @as(F, @floatFromInt(camera.pixels_num[0]));
        const px_y = @as(F, @floatFromInt(camera.pixels_num[1]));
        const sub_samp_f = @as(F, @floatFromInt(camera.sub_sample));
        const total_px = px_x * px_y;
        const total_subpx = total_px * sub_samp_f * sub_samp_f;

        const tess_checks_f = @as(F, @floatFromInt(self.bench.tess_checks));
        const tess_passes_f = @as(F, @floatFromInt(self.bench.tess_passes));
        const solver_coverage_pct = if (total_subpx > 0)
            solver_calls_f * 100.0 / total_subpx
        else
            0.0;
        const tess_coverage_pct = if (total_subpx > 0)
            tess_checks_f * 100.0 / total_subpx
        else
            0.0;
        const tess_pass_pct = if (self.bench.tess_checks > 0)
            tess_passes_f * 100.0 / tess_checks_f
        else
            0.0;

        try writer.print("--- SOLVER & TESSELLATION STATS ---\n", .{});
        try writer.print("Total Solver Calls      = {d}\n", .{self.bench.solver_calls});
        try writer.print("Avg Iters / Solver Call = {d:.2}\n", .{avg_iters_per_call});
        try writer.print("Conv %             = {d:.2}%\n", .{solver_conv_pct});
        try writer.print("Diverged/Failed %       = {d:.2}%\n", .{solver_diverged_pct});
        try writer.print(
            "Solver Diverged/Failed  = {d}\n",
            .{self.bench.solver_diverged},
        );
        try writer.print("Solver Coverage %       = {d:.2}%\n", .{solver_coverage_pct});
        try writer.print("Tessellation Checks     = {d}\n", .{self.bench.tess_checks});
        try writer.print("Tessellation Coverage % = {d:.2}%\n", .{tess_coverage_pct});
        try writer.print("Tessellation Pass %     = {d:.2}%\n", .{tess_pass_pct});
        try writer.print("{s}", .{line});

        if (self.iter_map) |*imap| {
            var gpa: std.heap.DebugAllocator(.{}) = .init;
            defer _ = gpa.deinit();
            const allocator = gpa.allocator();

            const stats = try calcStats(allocator, imap.slice);
            try writer.print("Min Iterations          = {d}\n", .{stats.min});
            try writer.print("Max Iterations          = {d}\n", .{stats.max});
            try writer.print("Median Iterations       = {d:.2}\n", .{stats.median});
            try writer.print("Lower Quartile (Q1)     = {d:.2}\n", .{stats.q1});
            try writer.print("Upper Quartile (Q3)     = {d:.2}\n", .{stats.q3});
            try writer.print("Median Abs. Dev (MAD)   = {d:.2}\n\n", .{stats.mad});
        } else {
            try writer.print("No iter map data available.\n\n", .{});
        }

        try writer.print("--- TILING & RASTERIZATION ---\n", .{});
        try writer.print(
            "Total Shaded Pixels     = {d}\n",
            .{self.bench.total_shaded_px},
        );
        try writer.print(
            "Max Elems in a Tile  = {d}\n",
            .{self.bench.max_tile_elems},
        );
        try writer.print(
            "Total Depth Tests       = {d}\n",
            .{self.bench.total_depth_test},
        );
        const d_fail_pct = if (self.bench.total_depth_test > 0)
            @as(F, @floatFromInt(self.bench.depth_tests_fail)) * 100.0 /
                @as(F, @floatFromInt(self.bench.total_depth_test))
        else
            0.0;
        try writer.print("Depth Tests Failed      = {d} ({d:.2}%)\n", .{
            self.bench.depth_tests_fail,
            d_fail_pct,
        });
        try writer.print("{s}", .{line});

        const vis_elems_f = @as(F, @floatFromInt(self.bench.vis_elems));
        const total_elems_f = @as(F, @floatFromInt(self.bench.total_elems));
        const vis_pct = if (self.bench.total_elems > 0)
            (vis_elems_f * 100.0 / total_elems_f)
        else
            0;
        const shaded_subpx = @as(F, @floatFromInt(self.bench.total_shaded_px));
        const shaded_pct = if (total_subpx > 0)
            (shaded_subpx * 100.0 / total_subpx)
        else
            0;

        try writer.print(
            "Vis Elems               = {d}\n",
            .{self.bench.vis_elems},
        );
        try writer.print("Total Elems             = {d}\n", .{self.bench.total_elems});
        try writer.print("Vis %                   = {d:.2}%\n", .{vis_pct});
        try writer.print("Total SubPx             = {d:.0}\n", .{total_subpx});
        try writer.print("Shaded SubPx            = {d:.0}\n", .{shaded_subpx});
        try writer.print("Shaded %                = {d:.2}%\n\n", .{shaded_pct});

        try writer.print("--- PIPELINE TIMINGS (User Summary) ---\n", .{});
        const conv = 1.0 / 1e6;
        try writer.print("Setup Frame Buff      = {d:.6} ms\n", .{
            self.bench.frame_times.setup_frame_buff * conv,
        });
        try writer.print("Prep Frame              = {d:.6} ms\n", .{
            self.bench.frame_times.prepare_frame_context * conv,
        });
        try writer.print("Geometry Preparation    = {d:.6} ms\n", .{
            self.bench.frame_times.geometry_prep * conv,
        });
        try writer.print("  Coord Ops             = {d:.6} ms\n", .{
            self.bench.frame_times.geom_coord_ops * conv,
        });
        try writer.print("  Cull Ops              = {d:.6} ms\n", .{
            self.bench.frame_times.geom_cull_ops * conv,
        });
        try writer.print("  Prep Hulls Shaders    = {d:.6} ms\n", .{
            self.bench.frame_times.geom_prep_hulls_shaders * conv,
        });
        try writer.print("  Remap Inds            = {d:.6} ms\n", .{
            self.bench.frame_times.geom_remap_inds * conv,
        });
        try writer.print("Elem/Tile Overlap       = {d:.6} ms\n", .{
            self.bench.frame_times.tile_overlap * conv,
        });
        const cam_inv_ms =
            self.bench.frame_times.cam_invert * conv;
        const resolve_ms =
            self.bench.frame_times.scratch_resolve * conv;
        const elem_loop_ms =
            self.bench.frame_times.raster_loop * conv -
            cam_inv_ms - resolve_ms;
        try writer.print("Cam Invert Time         = {d:.6} ms\n", .{
            cam_inv_ms,
        });
        try writer.print("Elem Loop Time          = {d:.6} ms\n", .{
            elem_loop_ms,
        });
        try writer.print("Scratch Resolve Time    = {d:.6} ms\n", .{
            resolve_ms,
        });
        try writer.print("Raster loop time        = {d:.6} ms\n", .{
            self.bench.frame_times.raster_loop * conv,
        });
        try writer.print("Save Time               = {d:.6} ms\n", .{
            self.bench.frame_times.save_frame * conv,
        });
        try writer.print("{s}", .{line});
        try writer.print("ACTIVE FRAME TIME       = {d:.3} ms\n", .{active_ms});
        try writer.print("FRAME LATENCY           = {d:.3} ms\n", .{
            latency_ms,
        });
        try writer.print("{s}", .{line});

        const melems_sec = if (geom_tiling_sec > 0)
            (@as(F, @floatFromInt(self.bench.total_elems)) / (geom_tiling_sec * 1e6))
        else
            0;
        const mpx_sec = if (raster_sec > 0) (total_px / (raster_sec * 1e6)) else 0;
        const msubpx_sec = if (raster_sec > 0) (total_subpx / (raster_sec * 1e6)) else 0;
        const mops_sec = if (active_sec > 0)
            (nodes_per_elem * total_subpx / (active_sec * 1e6))
        else
            0;

        try writer.print("MElem/second            = {d:.2}\n", .{melems_sec});
        try writer.print("MPx/second              = {d:.2}\n", .{mpx_sec});
        try writer.print("MsubPx/second           = {d:.2}\n", .{msubpx_sec});
        try writer.print("MOps/second             = {d:.2}\n", .{mops_sec});
        try writer.print("{s}", .{border});
        try writer.flush();
    }
};

pub const ReportLog = union(ReportMode) {
    off: OffLog,
    bench: BenchLog,
    full_stats: FullStatsLog,
};

pub fn LogType(comptime mode: ReportMode) type {
    return switch (mode) {
        .off => OffLog,
        .bench => BenchLog,
        .full_stats => FullStatsLog,
    };
}

pub fn getBenchLog(
    comptime mode: ReportMode,
    log: *LogType(mode),
) ?*BenchLog {
    return switch (mode) {
        .off => null,
        .bench => log,
        .full_stats => &log.bench,
    };
}

pub fn initFullStatsLog(
    allocator: std.mem.Allocator,
    pixels_num: [2]u32,
    tile_size: u16,
    sub_sample: u32,
    opts: rastcfg.FullStatsOpts,
) !FullStatsLog {
    var self = FullStatsLog{};
    const sub_samp: usize = @intCast(sub_sample);
    const sub_pixels_num = [_]usize{ pixels_num[1] * sub_samp, pixels_num[0] * sub_samp };

    if (opts.save_iter_map) {
        self.iter_map = try ndarray.NDArray(F).initFlat(
            allocator,
            &sub_pixels_num,
        );
        @memset(self.iter_map.?.slice, 0);
    }

    if (opts.save_xi_map) {
        self.xi_map = try ndarray.NDArray(F).initFlat(allocator, &sub_pixels_num);
        @memset(self.xi_map.?.slice, std.math.nan(F));
    }

    if (opts.save_eta_map) {
        self.eta_map = try ndarray.NDArray(F).initFlat(allocator, &sub_pixels_num);
        @memset(self.eta_map.?.slice, std.math.nan(F));
    }

    if (opts.save_conv_map) {
        self.conv_map = try ndarray.NDArray(F).initFlat(
            allocator,
            &sub_pixels_num,
        );
        @memset(self.conv_map.?.slice, 0);
    }

    self.solver_status_map = try ndarray.NDArray(F).initFlat(
        allocator,
        &sub_pixels_num,
    );
    @memset(
        self.solver_status_map.?.slice,
        @floatFromInt(@intFromEnum(newton.NewtonStatus.fail_iter_lim)),
    );

    self.pre_dom_conv_map = try ndarray.NDArray(F).initFlat(
        allocator,
        &sub_pixels_num,
    );
    @memset(self.pre_dom_conv_map.?.slice, 0);

    self.hit_iter_lim_map = try ndarray.NDArray(F).initFlat(
        allocator,
        &sub_pixels_num,
    );
    @memset(self.hit_iter_lim_map.?.slice, 0);

    if (opts.save_jac_det_map) {
        self.jac_det_map = try ndarray.NDArray(F).initFlat(
            allocator,
            &sub_pixels_num,
        );
        @memset(self.jac_det_map.?.slice, std.math.nan(F));
    }

    self.resid_mag_map = try ndarray.NDArray(F).initFlat(
        allocator,
        &sub_pixels_num,
    );
    @memset(self.resid_mag_map.?.slice, std.math.nan(F));

    self.resid_x_map = try ndarray.NDArray(F).initFlat(
        allocator,
        &sub_pixels_num,
    );
    @memset(self.resid_x_map.?.slice, std.math.nan(F));

    self.resid_y_map = try ndarray.NDArray(F).initFlat(
        allocator,
        &sub_pixels_num,
    );
    @memset(self.resid_y_map.?.slice, std.math.nan(F));

    self.interp_w_map = try ndarray.NDArray(F).initFlat(
        allocator,
        &sub_pixels_num,
    );
    @memset(self.interp_w_map.?.slice, std.math.nan(F));

    self.norm_resid_mag_map = try ndarray.NDArray(F).initFlat(
        allocator,
        &sub_pixels_num,
    );
    @memset(self.norm_resid_mag_map.?.slice, std.math.nan(F));

    self.dom_violation_map = try ndarray.NDArray(F).initFlat(
        allocator,
        &sub_pixels_num,
    );
    @memset(self.dom_violation_map.?.slice, std.math.nan(F));

    if (opts.save_pixel_occupancy_map) {
        self.pixel_occupancy_map = try ndarray.NDArray(F).initFlat(
            allocator,
            &[_]usize{ pixels_num[1], pixels_num[0] },
        );
        @memset(self.pixel_occupancy_map.?.slice, 0);
    }

    if (opts.save_depth_map) {
        self.depth_map = try ndarray.NDArray(F).initFlat(allocator, &sub_pixels_num);
        @memset(self.depth_map.?.slice, 0);
    }

    if (opts.save_normals_map) {
        self.normals_map = try ndarray.NDArray(F).initFlat(
            allocator,
            &[_]usize{ 3, sub_pixels_num[0], sub_pixels_num[1] },
        );
        @memset(self.normals_map.?.slice, 0);
    }

    if (opts.save_earlyout_map) {
        self.earlyout_map = try ndarray.NDArray(F).initFlat(allocator, &sub_pixels_num);
        @memset(self.earlyout_map.?.slice, 0);
    }

    const tiles_num_x = try std.math.divCeil(usize, pixels_num[0], tile_size);
    const tiles_num_y = try std.math.divCeil(usize, pixels_num[1], tile_size);
    const tiles_num = tiles_num_x * tiles_num_y;

    if (opts.save_tile_timing_map) {
        self.tile_timing_map = try ndarray.NDArray(F).initFlat(
            allocator,
            &[_]usize{tiles_num},
        );
    }
    if (opts.save_tile_density_map) {
        self.tile_density_map = try ndarray.NDArray(F).initFlat(
            allocator,
            &[_]usize{tiles_num},
        );
    }
    if (opts.save_tile_occupancy_map) {
        self.tile_occupancy_map = try ndarray.NDArray(F).initFlat(
            allocator,
            &[_]usize{tiles_num},
        );
    }

    return self;
}

pub const Stats = struct {
    min: F,
    max: F,
    median: F,
    q1: F,
    q3: F,
    mad: F,
};

pub fn calcStats(allocator: std.mem.Allocator, data: []const F) !Stats {
    if (data.len == 0) {
        return .{ .min = 0, .max = 0, .median = 0, .q1 = 0, .q3 = 0, .mad = 0 };
    }

    var filtered: std.ArrayList(F) = .empty;
    defer filtered.deinit(allocator);
    for (data) |val| {
        if (val > 0) try filtered.append(allocator, val);
    }

    if (filtered.items.len == 0) {
        return .{ .min = 0, .max = 0, .median = 0, .q1 = 0, .q3 = 0, .mad = 0 };
    }

    const slice = filtered.items;
    std.mem.sort(F, slice, {}, std.sort.asc(F));

    const min = slice[0];
    const max = slice[slice.len - 1];
    const median = getMedian(slice);
    const q1 = getMedian(slice[0 .. slice.len / 2]);
    const q3 = getMedian(slice[slice.len / 2 ..]);

    var deviations = try allocator.alloc(F, slice.len);
    defer allocator.free(deviations);
    for (slice, 0..) |val, ii| {
        deviations[ii] = @abs(val - median);
    }
    std.mem.sort(F, deviations, {}, std.sort.asc(F));
    const mad = getMedian(deviations);

    return .{
        .min = min,
        .max = max,
        .median = median,
        .q1 = q1,
        .q3 = q3,
        .mad = mad,
    };
}

fn getMedian(sorted_data: []const F) F {
    if (sorted_data.len == 0) return 0;
    const mid = sorted_data.len / 2;
    if (sorted_data.len % 2 == 0) {
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2.0;
    } else {
        return sorted_data[mid];
    }
}

pub fn ReportContext(comptime mode: ReportMode) type {
    return struct {
        log: *LogType(mode),

        inline fn bench(self: @This()) ?*BenchLog {
            return getBenchLog(mode, self.log);
        }

        pub const mode_tag = mode;

        pub inline fn recordGeometry(self: @This(), total: usize, vis: usize) void {
            if (self.bench()) |bench_log| {
                bench_log.total_elems = total;
                bench_log.vis_elems = vis;
            }
        }

        pub inline fn recordTile(
            self: @This(),
            tile_idx: usize,
            time_ns: u64,
            shaded_px: u64,
            elem_count: usize,
        ) void {
            if (mode == .full_stats) {
                if (self.log.tile_timing_map) |*tmap| {
                    tmap.slice[tile_idx] = @floatFromInt(time_ns);
                }
                if (self.log.tile_occupancy_map) |*omap| {
                    omap.slice[tile_idx] = @floatFromInt(shaded_px);
                }
                if (self.log.tile_density_map) |*dmap| {
                    dmap.slice[tile_idx] = @floatFromInt(elem_count);
                }
            }

            if (self.bench()) |bench_log| {
                bench_log.total_shaded_px += shaded_px;
                if (elem_count > bench_log.max_tile_elems) {
                    bench_log.max_tile_elems = elem_count;
                }
            }
        }

        pub inline fn recordSolverCalls(self: @This(), solver_calls: u64) void {
            if (self.bench()) |bench_log| {
                bench_log.solver_calls += solver_calls;
            }
        }

        pub inline fn recordSolverIters(self: @This(), solver_iters: u64) void {
            if (self.bench()) |bench_log| {
                bench_log.total_solver_iters += solver_iters;
            }
        }

        pub inline fn recordSolverStats(
            self: @This(),
            solver_calls: u64,
            solver_iters: u64,
        ) void {
            if (self.bench()) |bench_log| {
                bench_log.solver_calls += solver_calls;
                bench_log.total_solver_iters += solver_iters;
            }
        }

        pub inline fn recordTessChecks(self: @This(), tess_checks: u64) void {
            if (self.bench()) |bench_log| {
                bench_log.tess_checks += tess_checks;
            }
        }

        pub inline fn recordTessPasses(self: @This(), tess_passes: u64) void {
            if (self.bench()) |bench_log| {
                bench_log.tess_passes += tess_passes;
            }
        }

        pub inline fn recordPixelIters(
            self: @This(),
            global_subx: usize,
            global_suby: usize,
            iters: u8,
        ) void {
            if (mode == .full_stats) {
                if (self.log.iter_map) |*imap| {
                    const row_stride = imap.strides[0];
                    imap.slice[global_suby * row_stride + global_subx] =
                        @floatFromInt(iters);
                }
            }
        }

        pub inline fn recordPixelXi(
            self: @This(),
            global_subx: usize,
            global_suby: usize,
            xi: F,
        ) void {
            if (mode == .full_stats) {
                if (self.log.xi_map) |*xmap| {
                    const row_stride = xmap.strides[0];
                    xmap.slice[global_suby * row_stride + global_subx] = xi;
                }
            }
        }

        pub inline fn recordPixelEta(
            self: @This(),
            global_subx: usize,
            global_suby: usize,
            eta: F,
        ) void {
            if (mode == .full_stats) {
                if (self.log.eta_map) |*emap| {
                    const row_stride = emap.strides[0];
                    emap.slice[global_suby * row_stride + global_subx] = eta;
                }
            }
        }

        pub inline fn recordPixelConv(
            self: @This(),
            global_subx: usize,
            global_suby: usize,
            conv: bool,
        ) void {
            if (mode == .full_stats) {
                if (self.log.conv_map) |*cmap| {
                    const row_stride = cmap.strides[0];
                    cmap.slice[global_suby * row_stride + global_subx] =
                        if (conv) 1.0 else 0.0;
                }
            }
        }

        pub inline fn recordPixelSolverStatus(
            self: @This(),
            global_subx: usize,
            global_suby: usize,
            status: newton.NewtonStatus,
        ) void {
            if (mode == .full_stats) {
                if (self.log.solver_status_map) |*smap| {
                    const row_stride = smap.strides[0];
                    smap.slice[global_suby * row_stride + global_subx] =
                        @floatFromInt(@intFromEnum(status));
                }
            }
        }

        pub inline fn recordPixelPreDomConv(
            self: @This(),
            global_subx: usize,
            global_suby: usize,
            conv: bool,
        ) void {
            if (mode == .full_stats) {
                if (self.log.pre_dom_conv_map) |*pmap| {
                    const row_stride = pmap.strides[0];
                    pmap.slice[global_suby * row_stride + global_subx] =
                        if (conv) 1.0 else 0.0;
                }
            }
        }

        pub inline fn recordPixelHitIterLimit(
            self: @This(),
            global_subx: usize,
            global_suby: usize,
            hit_limit: bool,
        ) void {
            if (mode == .full_stats) {
                if (self.log.hit_iter_lim_map) |*hmap| {
                    const row_stride = hmap.strides[0];
                    hmap.slice[global_suby * row_stride + global_subx] =
                        if (hit_limit) 1.0 else 0.0;
                }
            }
        }

        pub inline fn recordPixelJacDet(
            self: @This(),
            global_subx: usize,
            global_suby: usize,
            jac_det: F,
        ) void {
            if (mode == .full_stats) {
                if (self.log.jac_det_map) |*jmap| {
                    const row_stride = jmap.strides[0];
                    jmap.slice[global_suby * row_stride + global_subx] =
                        jac_det;
                }
            }
        }

        pub inline fn recordPixelResidualMag(
            self: @This(),
            global_subx: usize,
            global_suby: usize,
            resid_mag: F,
        ) void {
            if (mode == .full_stats) {
                if (self.log.resid_mag_map) |*rmap| {
                    const row_stride = rmap.strides[0];
                    rmap.slice[global_suby * row_stride + global_subx] =
                        resid_mag;
                }
            }
        }

        pub inline fn recordPixelResidualX(
            self: @This(),
            global_subx: usize,
            global_suby: usize,
            resid_x: F,
        ) void {
            if (mode == .full_stats) {
                if (self.log.resid_x_map) |*rmap| {
                    const row_stride = rmap.strides[0];
                    rmap.slice[global_suby * row_stride + global_subx] =
                        resid_x;
                }
            }
        }

        pub inline fn recordPixelResidualY(
            self: @This(),
            global_subx: usize,
            global_suby: usize,
            resid_y: F,
        ) void {
            if (mode == .full_stats) {
                if (self.log.resid_y_map) |*rmap| {
                    const row_stride = rmap.strides[0];
                    rmap.slice[global_suby * row_stride + global_subx] =
                        resid_y;
                }
            }
        }

        pub inline fn recordPixelInterpolatedW(
            self: @This(),
            global_subx: usize,
            global_suby: usize,
            interp_w: F,
        ) void {
            if (mode == .full_stats) {
                if (self.log.interp_w_map) |*wmap| {
                    const row_stride = wmap.strides[0];
                    wmap.slice[global_suby * row_stride + global_subx] =
                        interp_w;
                }
            }
        }

        pub inline fn recordPixelNormalizedResidualMag(
            self: @This(),
            global_subx: usize,
            global_suby: usize,
            norm_resid_mag: F,
        ) void {
            if (mode == .full_stats) {
                if (self.log.norm_resid_mag_map) |*nmap| {
                    const row_stride = nmap.strides[0];
                    nmap.slice[global_suby * row_stride + global_subx] =
                        norm_resid_mag;
                }
            }
        }

        pub inline fn recordPixelDomViolation(
            self: @This(),
            global_subx: usize,
            global_suby: usize,
            dom_violation: F,
        ) void {
            if (mode == .full_stats) {
                if (self.log.dom_violation_map) |*dmap| {
                    const row_stride = dmap.strides[0];
                    dmap.slice[global_suby * row_stride + global_subx] =
                        dom_violation;
                }
            }
        }

        pub inline fn recordPixelOccupancy(
            self: @This(),
            x: usize,
            y: usize,
        ) void {
            if (mode == .full_stats) {
                if (self.log.pixel_occupancy_map) |*pomap| {
                    const row_stride = pomap.strides[0];
                    pomap.slice[y * row_stride + x] += 1.0;
                }
            }
        }

        pub inline fn recordDepth(
            self: @This(),
            global_subx: usize,
            global_suby: usize,
            inv_z: F,
        ) void {
            if (mode == .full_stats) {
                if (self.log.depth_map) |*dmap| {
                    const row_stride = dmap.strides[0];
                    dmap.slice[global_suby * row_stride + global_subx] = inv_z;
                }
            }
        }

        pub inline fn recordNormal(
            self: @This(),
            global_subx: usize,
            global_suby: usize,
            nx: F,
            ny: F,
            nz: F,
        ) void {
            if (mode == .full_stats) {
                if (self.log.normals_map) |*nmap| {
                    const chan_stride = nmap.strides[0];
                    const row_stride = nmap.strides[1];
                    const col_idx = global_suby * row_stride + global_subx;

                    nmap.slice[0 * chan_stride + col_idx] = 0.5 * nx + 0.5;
                    nmap.slice[1 * chan_stride + col_idx] = 0.5 * ny + 0.5;
                    nmap.slice[2 * chan_stride + col_idx] = 0.5 * nz + 0.5;
                }
            }
        }

        pub inline fn recordEarlyOut(
            self: @This(),
            global_subx: usize,
            global_suby: usize,
            early: bool,
        ) void {
            if (mode == .full_stats) {
                if (self.log.earlyout_map) |*emap| {
                    const row_stride = emap.strides[0];
                    emap.slice[global_suby * row_stride + global_subx] =
                        if (early) 1.0 else 0.0;
                }
            }
        }

        pub inline fn recordDepthTest(self: @This(), fail: bool) void {
            if (self.bench()) |bench_log| {
                bench_log.total_depth_test += 1;
                if (fail) {
                    bench_log.depth_tests_fail += 1;
                }
            }
        }

        pub inline fn recordSolverDiverged(self: @This()) void {
            if (self.bench()) |bench_log| {
                bench_log.solver_diverged += 1;
            }
        }

        pub inline fn recordSolverDivergedCount(
            self: @This(),
            diverged_count: u64,
        ) void {
            if (self.bench()) |bench_log| {
                bench_log.solver_diverged += diverged_count;
            }
        }

        pub inline fn recordCamTime(
            self: @This(),
            cam_duration_ns: u64,
        ) void {
            if (self.bench()) |bench_log| {
                bench_log.cam_time_ns +=
                    @floatFromInt(cam_duration_ns);
            }
        }

        pub inline fn recordResolveTime(
            self: @This(),
            resolve_duration_ns: u64,
        ) void {
            if (self.bench()) |bench_log| {
                bench_log.resolve_time_ns +=
                    @floatFromInt(resolve_duration_ns);
            }
        }
    };
}

pub inline fn recordNormalSIMD(
    comptime nodes_num: usize,
    comptime lane_num: usize,
    ctx_report: anytype,
    ctx_shade: anytype,
    shader_buf: anytype,
    v_mask_active: @Vector(lane_num, bool),
    v_weights: [nodes_num]@Vector(lane_num, F),
) void {
    const lane_mask: [lane_num]bool = v_mask_active;
    inline for (0..lane_num) |ll| {
        if (lane_mask[ll]) {
            var normal = [3]F{ 0.0, 0.0, 0.0 };
            inline for (0..nodes_num) |nn| {
                normal[0] += v_weights[nn][ll] *
                    shader_buf.normals[0 * nodes_num + nn];
                normal[1] += v_weights[nn][ll] *
                    shader_buf.normals[1 * nodes_num + nn];
                normal[2] += v_weights[nn][ll] *
                    shader_buf.normals[2 * nodes_num + nn];
            }

            ctx_report.recordNormal(
                ctx_shade.global_subx + ll,
                ctx_shade.global_suby,
                normal[0],
                normal[1],
                normal[2],
            );
        }
    }
}

pub fn standardReport(
    io: std.Io,
    camera: *const cam.CameraPrepared,
    actual_tile_size: u16,
    frame_idx: usize,
    camera_idx: usize,
    frame_times: FrameTimes,
    total_elems: usize,
    vis_elems: usize,
    nodes_per_elem: F,
    bench_log: *const BenchLog,
) !void {
    var buff: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buff);
    const writer = &stdout_writer.interface;

    const px_x = @as(F, @floatFromInt(camera.pixels_num[0]));
    const px_y = @as(F, @floatFromInt(camera.pixels_num[1]));
    const sub_samp_f: F = @as(F, @floatFromInt(camera.sub_sample));

    const total_subpx = px_x * px_y * sub_samp_f * sub_samp_f;
    const total_px = px_x * px_y;

    const raster_sec = frame_times.raster_loop / 1e9;
    const active_sec = frame_times.active_time / 1e9;
    const geom_tiling_sec = (frame_times.geometry_prep + frame_times.tile_overlap) / 1e9;

    const melems_sec = if (geom_tiling_sec > 0)
        (@as(F, @floatFromInt(total_elems)) / (geom_tiling_sec * 1e6))
    else
        0;
    const mnodes_sec = if (geom_tiling_sec > 0)
        (@as(F, @floatFromInt(bench_log.total_nodes)) /
            (geom_tiling_sec * 1e6))
    else
        0;
    const mpx_sec = if (raster_sec > 0) (total_px / (raster_sec * 1e6)) else 0;
    const msubpx_sec = if (raster_sec > 0) (total_subpx / (raster_sec * 1e6)) else 0;
    _ = nodes_per_elem;
    const frame_mpx_sec = if (active_sec > 0)
        (total_px / (active_sec * 1e6))
    else
        0;

    const conv_units: F = 1.0 / 1.0e6;
    const print_break = [_]u8{'='} ** 80;
    const print_break_inner = [_]u8{'-'} ** 80;

    try writer.print("\n{s}\nRaster Frame Times: Frame {d}, Camera {d}\n{s}\n", .{
        print_break,
        frame_idx,
        camera_idx,
        print_break,
    });

    const shaded_subpx = @as(F, @floatFromInt(bench_log.total_shaded_px));
    const shaded_pct = if (total_subpx > 0)
        (shaded_subpx * 100.0 / total_subpx)
    else
        0;
    const vis_elems_f = @as(F, @floatFromInt(vis_elems));
    const total_elems_f = @as(F, @floatFromInt(total_elems));
    const vis_pct = if (total_elems > 0) (vis_elems_f * 100.0 / total_elems_f) else 0;

    try writer.print("Vis Elems = {d}\n", .{vis_elems});
    try writer.print("Total Elems   = {d}\n", .{total_elems});
    try writer.print("Vis %         = {d:.2}%\n", .{vis_pct});
    try writer.print("Total SubPx   = {d:.0}\n", .{total_subpx});
    try writer.print("Shaded SubPx  = {d:.0}\n", .{shaded_subpx});
    try writer.print("Shaded %      = {d:.2}%\n", .{shaded_pct});
    try writer.print("{s}\n", .{print_break_inner});

    try writer.print("Actual Tile Size        = {d}x{d}\n", .{
        actual_tile_size,
        actual_tile_size,
    });
    try writer.print("Setup Frame Buff      = {d:.6} ms\n", .{
        frame_times.setup_frame_buff * conv_units,
    });
    try writer.print("Prep Frame              = {d:.6} ms\n", .{
        frame_times.prepare_frame_context * conv_units,
    });
    try writer.print("Geometry Preparation    = {d:.6} ms\n", .{
        frame_times.geometry_prep * conv_units,
    });
    try writer.print("  Coord Ops             = {d:.6} ms\n", .{
        frame_times.geom_coord_ops * conv_units,
    });
    try writer.print("  Cull Ops              = {d:.6} ms\n", .{
        frame_times.geom_cull_ops * conv_units,
    });
    try writer.print("  Prep Hulls Shaders    = {d:.6} ms\n", .{
        frame_times.geom_prep_hulls_shaders * conv_units,
    });
    try writer.print("  Remap Inds            = {d:.6} ms\n", .{
        frame_times.geom_remap_inds * conv_units,
    });
    try writer.print("Elem/Tile Overlap       = {d:.6} ms\n", .{
        frame_times.tile_overlap * conv_units,
    });
    const cam_inv_print_ms =
        frame_times.cam_invert * conv_units;
    const resolve_print_ms =
        frame_times.scratch_resolve * conv_units;
    const elem_loop_print_ms =
        frame_times.raster_loop * conv_units -
        cam_inv_print_ms - resolve_print_ms;
    try writer.print("Cam Invert Time         = {d:.6} ms\n", .{
        cam_inv_print_ms,
    });
    try writer.print("Elem Loop Time          = {d:.6} ms\n", .{
        elem_loop_print_ms,
    });
    try writer.print("Scratch Resolve Time    = {d:.6} ms\n", .{
        resolve_print_ms,
    });
    try writer.print("Raster loop time        = {d:.6} ms\n", .{
        frame_times.raster_loop * conv_units,
    });
    try writer.print("Save Time               = {d:.6} ms\n", .{
        frame_times.save_frame * conv_units,
    });

    try writer.print("{s}\n", .{print_break_inner});
    try writer.print("ACTIVE FRAME TIME  = {d:.3} ms\n", .{
        frame_times.active_time * conv_units,
    });
    try writer.print("FRAME LATENCY      = {d:.3} ms\n", .{
        frame_times.latency_time * conv_units,
    });
    try writer.print("{s}\n", .{print_break_inner});

    try writer.print("Geom. Node Throughput  = {d:.2} MNodes/s\n", .{mnodes_sec});
    try writer.print("Geom. Elem. Throughput = {d:.2} MElem/s\n", .{melems_sec});
    try writer.print("Subpx Raster Throughput  = {d:.2} MSubPx/s\n", .{msubpx_sec});
    try writer.print("Raster Throughput        = {d:.2} MPx/s\n", .{mpx_sec});
    try writer.print("Active Frame Throughput  = {d:.2} MPx/s\n", .{frame_mpx_sec});

    try writer.print("{s}\n", .{print_break});
    try writer.flush();
}

pub fn printRenderSummary(
    io: std.Io,
    cameras: []const cam.CameraPrepared,
    config: rastcfg.RasterConfig,
    num_time: usize,
    report_mode: ReportMode,
    end_to_end_times: EndToEndTimes,
    bench_capture: ?[]const FrameBenchCapture,
) !void {
    if (report_mode == .off) {
        return;
    }

    var total_pixels: usize = 0;
    for (cameras) |camera| {
        total_pixels += camera.pixels_num[0] * camera.pixels_num[1];
    }
    total_pixels *= num_time;

    const actual_tile_size = scalingpolicy.tileSize(
        config.tile_size_override,
        config.tile_size_min,
        config.tile_size_max,
        cameras[0].pixels_num,
        cameras[0].sub_sample,
        cameras[0].prep_psf.halo_px,
    );

    const total_frames = cameras.len * num_time;
    const total_render_ms = end_to_end_times.total_time / 1e6;
    const total_render_sec = end_to_end_times.total_time / 1e9;
    const setup_ms = end_to_end_times.setup_time / 1e6;
    // const setup_other_ms = end_to_end_times.setup_other_time / 1e6;
    // const setup_frame_buff_ms =
    //     end_to_end_times.setup_frame_buff_time / 1e6;
    const dispatch_ms = end_to_end_times.dispatch_time / 1e6;
    const avg_frame_ms = total_render_ms / @as(
        F,
        @floatFromInt(total_frames),
    );
    const avg_frame_mpx_sec = if (total_render_sec > 0)
        @as(F, @floatFromInt(total_pixels)) / (total_render_sec * 1e6)
    else
        0.0;
    var total_raster_ns: F = 0.0;
    if (bench_capture) |capture| {
        for (capture) |frame_capture| {
            total_raster_ns += frame_capture.bench_log.frame_times.raster_loop;
        }
    }
    const total_raster_sec = total_raster_ns / 1e9;
    const avg_raster_mpx_sec = if (total_raster_sec > 0)
        @as(F, @floatFromInt(total_pixels)) / (total_raster_sec * 1e6)
    else
        0.0;

    var buff: [1024]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buff);
    const writer = &stdout_writer.interface;
    const print_break = [_]u8{'='} ** 80;

    try writer.print("\n{s}\nRiley Raster Render Summary\n{s}\n", .{
        print_break,
        print_break,
    });
    try writer.print("Actual Tile Size        = {d}x{d}\n", .{
        actual_tile_size,
        actual_tile_size,
    });
    try writer.print("Setup Time              = {d:.3} ms\n", .{setup_ms});
    // try writer.print("Setup other             = {d:.3} ms\n", .{
    //     setup_other_ms,
    // });
    // try writer.print("Setup frame buff      = {d:.3} ms\n", .{
    //     setup_frame_buff_ms,
    // });
    try writer.print("Dispatch Time           = {d:.3} ms\n", .{dispatch_ms});
    try writer.print("Total Render Time       = {d:.3} ms\n", .{total_render_ms});
    try writer.print("Avg. Frame Time         = {d:.3} ms\n", .{avg_frame_ms});
    if ((report_mode == .bench or report_mode == .full_stats) and
        bench_capture != null)
    {
        try writer.print(
            "Avg. Raster Throughput = {d:.3} MPx/s\n",
            .{avg_raster_mpx_sec},
        );
    }
    try writer.print(
        "Avg. Frame Throughput   = {d:.3} MPx/s\n",
        .{avg_frame_mpx_sec},
    );
    try writer.print("{s}\n", .{print_break});
    try writer.flush();
}
