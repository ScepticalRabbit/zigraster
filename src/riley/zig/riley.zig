// --------------------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const Timestamp = std.Io.Clock.Timestamp;
const matslice = @import("matslice.zig");
const ndarray = @import("ndarray.zig");
const buildconfig = @import("buildconfig.zig");

const sliceops = @import("sliceops.zig");

const cam = @import("camera.zig");
const camops = @import("cameraops.zig");
const rops = @import("rasterops.zig");
const mo = @import("meshpipeline.zig");
const shaderops = @import("shaderops.zig");

const iio = @import("imageio.zig");
const imageops = @import("imageops.zig");
const pce = @import("parachunkexec.zig");
const saveoverlap = @import("saveoverlap.zig");
const scalingpolicy = @import("scalingpolicy.zig");
const valinp = @import("validateinput.zig");

const geomkerns = @import("geometrykernels.zig");
const shadekerns = @import("shaderkernels.zig");
const rasterengine = @import("rasterengine.zig");

const rastcfg = @import("rasterconfig.zig");
pub const RasterConfig = rastcfg.RasterConfig;
pub const ImageSaveMode = rastcfg.ImageSaveMode;
pub const SaveStrategy = rastcfg.SaveStrategy;
pub const RenderMode = rastcfg.RenderMode;
pub const ReportMode = rastcfg.ReportMode;
pub const FullStatsOpts = rastcfg.FullStatsOpts;

const report = @import("report.zig");
const FrameReportStorage = report.FrameReportStorage;
const F = buildconfig.F;

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const RenderGroupSpec = struct {
    io: std.Io,
    save_frame_io: ?std.Io = null,
    workers: u16,
};

// --------------------------------------------------------------------------------------
// Public Entry-Point Func
// --------------------------------------------------------------------------------------

pub fn raster(
    outer_alloc: std.mem.Allocator,
    render_groups: []const RenderGroupSpec,
    cam_inps: []const cam.CameraInput,
    meshes: []const mo.MeshInput,
    config: RasterConfig,
    out_dir_path: ?[]const u8,
) !?ndarray.NDArray(F) {
    return rasterReport(
        outer_alloc,
        render_groups,
        cam_inps,
        meshes,
        config,
        out_dir_path,
        null,
    );
}

pub fn rasterInto(
    outer_alloc: std.mem.Allocator,
    render_groups: []const RenderGroupSpec,
    cam_inps: []const cam.CameraInput,
    meshes: []const mo.MeshInput,
    config: RasterConfig,
    out_dir_path: ?[]const u8,
    images_arr: ?*ndarray.NDArray(F),
) !void {
    try rasterReportInto(
        outer_alloc,
        render_groups,
        cam_inps,
        meshes,
        config,
        out_dir_path,
        images_arr,
        null,
    );
}

pub fn rasterReport(
    outer_alloc: std.mem.Allocator,
    render_groups: []const RenderGroupSpec,
    cam_inps: []const cam.CameraInput,
    meshes: []const mo.MeshInput,
    config: RasterConfig,
    out_dir_path: ?[]const u8,
    bench_capt: ?[]report.FrameBenchCapture,
) !?ndarray.NDArray(F) {
    const valid_summary = try valinp.checkRenderInpsErr(
        render_groups,
        cam_inps,
        meshes,
        config,
        null,
        false,
        bench_capt,
    );

    var images_arr_opt: ?ndarray.NDArray(F) = null;
    if (config.save_strategy == .memory or config.save_strategy == .both) {
        images_arr_opt = try ndarray.NDArray(F).initFlat(
            outer_alloc,
            valid_summary.img_dims[0..],
        );
    }
    errdefer if (images_arr_opt) |*images_arr| {
        outer_alloc.free(images_arr.slice);
        images_arr.deinit(outer_alloc);
    };

    try rasterReportInto(
        outer_alloc,
        render_groups,
        cam_inps,
        meshes,
        config,
        out_dir_path,
        if (images_arr_opt) |*images_arr| images_arr else null,
        bench_capt,
    );
    return images_arr_opt;
}

pub fn rasterReportInto(
    outer_alloc: std.mem.Allocator,
    render_groups: []const RenderGroupSpec,
    cam_inps: []const cam.CameraInput,
    meshes: []const mo.MeshInput,
    config: RasterConfig,
    out_dir_path: ?[]const u8,
    images_arr: ?*ndarray.NDArray(F),
    bench_capt: ?[]report.FrameBenchCapture,
) !void {
    const summary_io = render_groups[0].io;
    const time_start_render = Timestamp.now(summary_io, .awake);

    const valid_summary = try valinp.checkRenderInpsErr(
        render_groups,
        cam_inps,
        meshes,
        config,
        images_arr,
        true,
        bench_capt,
    );

    var out_dir: ?std.Io.Dir = null;
    if (out_dir_path) |path| {
        const cwd = std.Io.Dir.cwd();
        cwd.createDirPath(summary_io, path) catch |err| {
            if (err != error.PathAlreadyExists) {
                return err;
            }
        };
        out_dir = try cwd.openDir(summary_io, path, .{});
    }
    defer if (out_dir) |*od| od.close(summary_io);

    var static_arena = std.heap.ArenaAllocator.init(outer_alloc);
    defer static_arena.deinit();
    const static_alloc = static_arena.allocator();

    const cams = try camops.prepareCameraSlice(
        outer_alloc,
        cam_inps,
    );
    defer {
        for (cams) |cam_prep| cam_prep.deinit(outer_alloc);
        outer_alloc.free(cams);
    }

    const num_time = valid_summary.num_time;
    const num_fields = valid_summary.raw_num_fields;

    // Init. static data across all frames - here we reshape uv's once if we have them so
    // we don't need to do this every frames
    const mesh_static = try initMeshStaticSlice(static_alloc, meshes);
    const nodal_glob_scaling = try initNodalGlobalScaling(outer_alloc, meshes);
    defer outer_alloc.free(nodal_glob_scaling);

    // Timing hooks for frame buffer setup, render time and E2E times
    const time_start_frame_buff = Timestamp.now(summary_io, .awake);
    const time_end_setup = Timestamp.now(summary_io, .awake);
    var end_to_end_times = report.EndToEndTimes{
        .setup_time = @floatFromInt(
            time_start_render.durationTo(time_end_setup).raw.nanoseconds,
        ),
        .setup_other_time = @floatFromInt(
            time_start_render.durationTo(
                time_start_frame_buff,
            ).raw.nanoseconds,
        ),
        .setup_frame_buff_time = @floatFromInt(
            time_start_frame_buff.durationTo(
                time_end_setup,
            ).raw.nanoseconds,
        ),
    };
    const time_start_dispatch = Timestamp.now(summary_io, .awake);

    // Dispatch frame jobs to render groups to run the geomtry then raster pipelines
    if (config.render_mode == .in_order) {
        try dispatchFrameJobsInOrder(
            outer_alloc,
            render_groups,
            cams,
            config,
            out_dir,
            num_time,
            num_fields,
            mesh_static,
            nodal_glob_scaling,
            images_arr,
            bench_capt,
        );
    } else {
        try dispatchFrameJobsOffline(
            outer_alloc,
            render_groups,
            cams,
            config,
            out_dir,
            num_time,
            num_fields,
            mesh_static,
            nodal_glob_scaling,
            images_arr,
            bench_capt,
        );
    }

    const time_end_render = Timestamp.now(summary_io, .awake);
    end_to_end_times.dispatch_time = @floatFromInt(
        time_start_dispatch.durationTo(time_end_render).raw.nanoseconds,
    );
    end_to_end_times.total_time = @floatFromInt(
        time_start_render.durationTo(time_end_render).raw.nanoseconds,
    );

    try report.printRenderSummary(
        summary_io,
        cams,
        config,
        num_time,
        config.report,
        end_to_end_times,
        if (bench_capt) |capt| capt else null,
    );
}

pub fn calcAllFramesImageDims(
    cam_inps: []const cam.CameraInput,
    meshes: []const mo.MeshInput,
    config: RasterConfig,
) ![5]usize {
    std.debug.assert(cam_inps.len > 0);
    std.debug.assert(meshes.len > 0);

    const num_time = mo.countFrames(meshes);
    const raw_num_fields = mo.countOutputFields(meshes);
    const num_fields = try valinp.calcOutFieldsForImgSaveMode(
        config.image_save_mode,
        raw_num_fields,
    );

    var max_pix_num = cam_inps[0].pixels_num;
    for (cam_inps[1..]) |cam_inp| {
        max_pix_num[0] = @max(max_pix_num[0], cam_inp.pixels_num[0]);
        max_pix_num[1] = @max(max_pix_num[1], cam_inp.pixels_num[1]);
    }

    return .{
        cam_inps.len,
        num_time,
        @as(usize, num_fields),
        max_pix_num[1],
        max_pix_num[0],
    };
}

pub fn getThreadedIo(
    gpa: std.mem.Allocator,
    minimal: std.process.Init.Minimal,
    num_threads: u16,
) std.Io.Threaded {
    // User-facing thread counts in riley always include the caller thread.
    // Zig's std.Io.Threaded limits count only spawned worker threads, excluding
    // the caller. Translate here so:
    //   threads=1  -> caller only
    //   threads=N  -> caller + (N - 1) worker threads
    const limit: std.Io.Limit =
        if (num_threads <= 1) .nothing else .limited(num_threads - 1);

    return std.Io.Threaded.init(gpa, .{
        .argv0 = .init(minimal.args),
        .environ = minimal.environ,
        .async_limit = limit,
        .concurrent_limit = limit,
    });
}

// --------------------------------------------------------------------------------------
// Offline Dispatch Path
// --------------------------------------------------------------------------------------

const FrameJobErrorState = struct {
    mutex: std.atomic.Mutex = .unlocked,
    first_err: ?anyerror = null,

    fn setFirst(
        self: *FrameJobErrorState,
        err: anyerror,
    ) void {
        while (!self.mutex.tryLock()) {
            std.atomic.spinLoopHint();
        }
        defer self.mutex.unlock();
        if (self.first_err == null) {
            self.first_err = err;
        }
    }
};

const OfflineDispatchShared = struct {
    outer_alloc: std.mem.Allocator,
    cameras: []const cam.CameraPrepared,
    config: RasterConfig,
    out_dir: ?std.Io.Dir,
    num_time: usize,
    num_fields: u8,
    mesh_static: []const mo.MeshStatic,
    nodal_global_scaling: []const ?imageops.ScalingParams,
    images_arr: ?*ndarray.NDArray(F),
    bench_capture: ?[]report.FrameBenchCapture,
    total_scene_elems: usize,
    batch_size: usize,
    next_job: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
    err_state: *FrameJobErrorState,
};

fn dispatchFrameJobsOffline(
    outer_alloc: std.mem.Allocator,
    render_groups: []const RenderGroupSpec,
    cameras: []const cam.CameraPrepared,
    config: RasterConfig,
    out_dir: ?std.Io.Dir,
    num_time: usize,
    num_fields: u8,
    mesh_static: []const mo.MeshStatic,
    nodal_global_scaling: []const ?imageops.ScalingParams,
    images_arr: ?*ndarray.NDArray(F),
    bench_capture: ?[]report.FrameBenchCapture,
) !void {
    var err_state = FrameJobErrorState{};
    var shared = OfflineDispatchShared{
        .outer_alloc = outer_alloc,
        .cameras = cameras,
        .config = config,
        .out_dir = out_dir,
        .num_time = num_time,
        .num_fields = num_fields,
        .mesh_static = mesh_static,
        .nodal_global_scaling = nodal_global_scaling,
        .images_arr = images_arr,
        .bench_capture = bench_capture,
        .total_scene_elems = mo.countStaticMeshElems(mesh_static),
        .batch_size = @max(@as(usize, 1), config.frame_batch_size_per_group),
        .err_state = &err_state,
    };

    var threads = try outer_alloc.alloc(std.Thread, render_groups.len -| 1);
    defer outer_alloc.free(threads);

    for (render_groups[1..], 0..) |render_group, ii| {
        threads[ii] = try std.Thread.spawn(
            .{},
            processOfflineRenderGroupThread,
            .{ render_group, &shared },
        );
    }

    processOfflineRenderGroupLoop(render_groups[0], &shared) catch |err| {
        err_state.setFirst(err);
    };

    for (threads) |thread| {
        thread.join();
    }
    if (err_state.first_err) |err| return err;
}

fn processOfflineRenderGroupThread(
    render_group: RenderGroupSpec,
    shared: *OfflineDispatchShared,
) void {
    processOfflineRenderGroupLoop(render_group, shared) catch |err| {
        shared.err_state.setFirst(err);
    };
}

fn processOfflineRenderGroupLoop(
    render_group: RenderGroupSpec,
    shared: *OfflineDispatchShared,
) !void {
    var group_arena = std.heap.ArenaAllocator.init(shared.outer_alloc);
    defer group_arena.deinit();
    const group_alloc = group_arena.allocator();
    const jobs_num = shared.cameras.len * shared.num_time;

    var save_overlap = try saveoverlap.SaveOverlap.initMaybe(
        shared.outer_alloc,
        renderGroupSaveIo(render_group),
        shared.cameras,
        shared.num_fields,
        shared.config,
        saveOverlapEnabled(shared.config),
    );
    defer save_overlap.deinit();

    while (true) {
        const batch_start = shared.next_job.fetchAdd(
            shared.batch_size,
            .monotonic,
        );
        if (batch_start >= jobs_num) break;

        const batch_end = @min(jobs_num, batch_start + shared.batch_size);
        const batch_len = batch_end - batch_start;
        const job_indices = try group_alloc.alloc(usize, batch_len);
        for (0..batch_len) |ii| {
            job_indices[ii] = batch_start + ii;
        }

        const jobs = try prepareJobBatch(
            group_alloc,
            shared.cameras,
            shared.config,
            shared.out_dir,
            shared.num_fields,
            shared.mesh_static,
            shared.nodal_global_scaling,
            shared.images_arr,
            shared.bench_capture,
            job_indices,
        );

        try processGeometryBatch(
            group_alloc,
            render_group.io,
            render_group.workers,
            shared.config,
            shared.total_scene_elems,
            jobs,
        );

        try processRasterBatch(
            shared.outer_alloc,
            render_group.io,
            group_alloc,
            render_group.workers,
            shared.config,
            if (save_overlap.enabled()) &save_overlap else null,
            jobs,
        );
        _ = group_arena.reset(.retain_capacity);
    }

    try save_overlap.checkError();
}

// --------------------------------------------------------------------------------------
// In-Order Dispatch Path
// --------------------------------------------------------------------------------------

const InOrderDispatchShared = struct {
    outer_alloc: std.mem.Allocator,
    cameras: []const cam.CameraPrepared,
    config: RasterConfig,
    out_dir: ?std.Io.Dir,
    frame_idx: usize,
    num_fields: u8,
    mesh_static: []const mo.MeshStatic,
    nodal_global_scaling: []const ?imageops.ScalingParams,
    images_arr: ?*ndarray.NDArray(F),
    bench_capture: ?[]report.FrameBenchCapture,
    total_scene_elems: usize,
    batch_size: usize,
    next_camera: std.atomic.Value(usize) =
        std.atomic.Value(usize).init(0),
    err_state: *FrameJobErrorState,
};

fn dispatchFrameJobsInOrder(
    outer_alloc: std.mem.Allocator,
    render_groups: []const RenderGroupSpec,
    cameras: []const cam.CameraPrepared,
    config: RasterConfig,
    out_dir: ?std.Io.Dir,
    num_time: usize,
    num_fields: u8,
    mesh_static: []const mo.MeshStatic,
    nodal_global_scaling: []const ?imageops.ScalingParams,
    images_arr: ?*ndarray.NDArray(F),
    bench_capture: ?[]report.FrameBenchCapture,
) !void {
    const total_scene_elems = mo.countStaticMeshElems(mesh_static);
    const batch_size = @max(@as(usize, 1), config.frame_batch_size_per_group);

    for (0..num_time) |frame_idx| {
        var err_state = FrameJobErrorState{};
        var shared = InOrderDispatchShared{
            .outer_alloc = outer_alloc,
            .cameras = cameras,
            .config = config,
            .out_dir = out_dir,
            .frame_idx = frame_idx,
            .num_fields = num_fields,
            .mesh_static = mesh_static,
            .nodal_global_scaling = nodal_global_scaling,
            .images_arr = images_arr,
            .bench_capture = bench_capture,
            .total_scene_elems = total_scene_elems,
            .batch_size = batch_size,
            .err_state = &err_state,
        };

        var threads = try outer_alloc.alloc(
            std.Thread,
            render_groups.len -| 1,
        );
        defer outer_alloc.free(threads);

        for (render_groups[1..], 0..) |render_group, ii| {
            threads[ii] = try std.Thread.spawn(
                .{},
                processInOrderRenderGroupThread,
                .{ render_group, &shared },
            );
        }

        processInOrderRenderGroupLoop(render_groups[0], &shared) catch |err| {
            err_state.setFirst(err);
        };

        for (threads) |thread| {
            thread.join();
        }
        if (err_state.first_err) |err| return err;
    }
}

fn processInOrderRenderGroupThread(
    render_group: RenderGroupSpec,
    shared: *InOrderDispatchShared,
) void {
    processInOrderRenderGroupLoop(render_group, shared) catch |err| {
        shared.err_state.setFirst(err);
    };
}

fn processInOrderRenderGroupLoop(
    render_group: RenderGroupSpec,
    shared: *InOrderDispatchShared,
) !void {
    var group_arena = std.heap.ArenaAllocator.init(shared.outer_alloc);
    defer group_arena.deinit();
    const group_alloc = group_arena.allocator();

    var save_overlap = try saveoverlap.SaveOverlap.initMaybe(
        shared.outer_alloc,
        renderGroupSaveIo(render_group),
        shared.cameras,
        shared.num_fields,
        shared.config,
        saveOverlapEnabled(shared.config),
    );
    defer save_overlap.deinit();

    while (true) {
        const batch_start_camera = shared.next_camera.fetchAdd(
            shared.batch_size,
            .monotonic,
        );
        if (batch_start_camera >= shared.cameras.len) break;

        const batch_end_camera = @min(
            shared.cameras.len,
            batch_start_camera + shared.batch_size,
        );
        const batch_len = batch_end_camera - batch_start_camera;
        const job_indices = try group_alloc.alloc(usize, batch_len);
        for (0..batch_len) |ii| {
            job_indices[ii] =
                shared.frame_idx * shared.cameras.len + batch_start_camera + ii;
        }

        const jobs = try prepareJobBatch(
            group_alloc,
            shared.cameras,
            shared.config,
            shared.out_dir,
            shared.num_fields,
            shared.mesh_static,
            shared.nodal_global_scaling,
            shared.images_arr,
            shared.bench_capture,
            job_indices,
        );

        try processGeometryBatch(
            group_alloc,
            render_group.io,
            render_group.workers,
            shared.config,
            shared.total_scene_elems,
            jobs,
        );

        try processRasterBatch(
            shared.outer_alloc,
            render_group.io,
            group_alloc,
            render_group.workers,
            shared.config,
            if (save_overlap.enabled()) &save_overlap else null,
            jobs,
        );

        _ = group_arena.reset(.retain_capacity);
    }
    try save_overlap.checkError();
}

// --------------------------------------------------------------------------------------
// Batch Preparation and Wave Scheduling
// --------------------------------------------------------------------------------------

fn prepareJobBatch(
    group_alloc: std.mem.Allocator,
    cameras: []const cam.CameraPrepared,
    config: RasterConfig,
    out_dir: ?std.Io.Dir,
    num_fields: u8,
    mesh_static: []const mo.MeshStatic,
    nodal_global_scaling: []const ?imageops.ScalingParams,
    images_arr: ?*ndarray.NDArray(F),
    bench_capture: ?[]report.FrameBenchCapture,
    job_indices: []const usize,
) ![]PreparedFrameJob {
    const jobs = try group_alloc.alloc(PreparedFrameJob, job_indices.len);

    const can_write_result_direct = images_arr != null and
        cam.allCamerasSharePixels(cameras) and
        !needsOutputTransform(config.image_save_mode, num_fields);

    for (job_indices, 0..) |job_idx, ii| {
        const frame_idx = @divFloor(job_idx, cameras.len);
        const camera_idx = @mod(job_idx, cameras.len);
        jobs[ii] = PreparedFrameJob.init(
            group_alloc,
            .{
                .camera = &cameras[camera_idx],
                .camera_idx = camera_idx,
                .frame_idx = frame_idx,
                .num_fields = num_fields,
                .config = config,
                .out_dir = out_dir,
                .mesh_static = mesh_static,
                .nodal_global_scaling = nodal_global_scaling,
                .images_arr = images_arr,
                .bench_capture = bench_capture,
                .cameras_num = cameras.len,
                .can_write_result_direct = can_write_result_direct,
            },
        );
    }

    return jobs;
}

fn assignSpreadGeometryWorkers(
    allocator: std.mem.Allocator,
    group_workers: u16,
    jobs_in_wave: usize,
    max_geom_workers_per_job: u16,
) ![]u16 {
    const assigned = try allocator.alloc(u16, jobs_in_wave);
    @memset(assigned, 0);

    const max_jobs = @min(jobs_in_wave, @as(usize, @max(@as(u16, 1), group_workers)));
    for (0..max_jobs) |ii| {
        assigned[ii] = 1;
    }

    var remaining_workers = @as(usize, @max(@as(u16, 1), group_workers)) - max_jobs;
    while (remaining_workers > 0) {
        var added_any = false;
        for (assigned) |*workers| {
            if (remaining_workers == 0) break;
            if (workers.* < @max(@as(u16, 1), max_geom_workers_per_job)) {
                workers.* += 1;
                remaining_workers -= 1;
                added_any = true;
            }
        }
        if (!added_any) break;
    }

    for (assigned) |*workers| {
        if (workers.* == 0) workers.* = 1;
    }
    return assigned;
}

fn geometryJobsPerWave(
    config: RasterConfig,
    group_workers: u16,
    jobs_remaining: usize,
) usize {
    const requested_jobs = @max(@as(u16, 1), config.max_geom_jobs_in_flight_per_group);
    const worker_cap = @max(@as(u16, 1), group_workers);
    return @min(
        jobs_remaining,
        @as(usize, @intCast(@min(requested_jobs, worker_cap))),
    );
}

fn processGeometryWave(
    group_alloc: std.mem.Allocator,
    io: std.Io,
    jobs: []PreparedFrameJob,
    workers_per_job: []const u16,
) !void {
    const AsyncGeometryJob = struct {
        fn run(
            local_group_alloc: std.mem.Allocator,
            local_io: std.Io,
            job: *PreparedFrameJob,
            geom_workers: u16,
            err_state: *FrameJobErrorState,
        ) std.Io.Cancelable!void {
            runGeometryStage(
                local_group_alloc,
                local_io,
                job,
                geom_workers,
            ) catch |err| {
                err_state.setFirst(err);
            };
        }
    };

    var err_state = FrameJobErrorState{};
    var group: std.Io.Group = .init;
    errdefer group.cancel(io);

    const caller_idx = jobs.len - 1;
    for (jobs[0..caller_idx], workers_per_job[0..caller_idx]) |*job, geom_workers| {
        group.async(
            io,
            AsyncGeometryJob.run,
            .{ group_alloc, io, job, geom_workers, &err_state },
        );
    }

    try runGeometryStage(
        group_alloc,
        io,
        &jobs[caller_idx],
        workers_per_job[caller_idx],
    );
    try group.await(io);
    if (err_state.first_err) |err| return err;
}

fn processGeometryBatch(
    group_alloc: std.mem.Allocator,
    io: std.Io,
    group_workers: u16,
    config: RasterConfig,
    total_scene_elems: usize,
    jobs: []PreparedFrameJob,
) !void {
    const geom_mode = scalingpolicy.resolveGeometrySchedulingMode(
        config.geom_scheduling_mode,
        total_scene_elems,
    );
    var wave_start: usize = 0;

    while (wave_start < jobs.len) {
        const jobs_remaining = jobs.len - wave_start;

        const wave_jobs = switch (geom_mode) {
            .spread => geometryJobsPerWave(config, group_workers, jobs_remaining),
            .pack => @min(@as(usize, 1), jobs_remaining),
            .auto => unreachable,
        };

        const wave_end = wave_start + wave_jobs;
        const wave = jobs[wave_start..wave_end];

        const workers_per_job = switch (geom_mode) {
            .spread => try assignSpreadGeometryWorkers(
                group_alloc,
                group_workers,
                wave.len,
                config.max_geom_workers_per_job,
            ),
            .pack => blk: {
                const assigned = try group_alloc.alloc(u16, 1);
                assigned[0] = @min(
                    @max(@as(u16, 1), group_workers),
                    @max(@as(u16, 1), config.max_geom_workers_per_job),
                );
                break :blk assigned;
            },
            .auto => unreachable,
        };
        defer group_alloc.free(workers_per_job);

        try processGeometryWave(group_alloc, io, wave, workers_per_job);

        wave_start = wave_end;
    }
}

fn processRasterBatch(
    outer_alloc: std.mem.Allocator,
    group_io: std.Io,
    group_alloc: std.mem.Allocator,
    group_workers: u16,
    config: RasterConfig,
    save_overlap: ?*saveoverlap.SaveOverlap,
    jobs: []PreparedFrameJob,
) !void {
    const raster_workers = @min(
        @max(@as(u16, 1), group_workers),
        @max(@as(u16, 1), config.max_raster_workers_per_job),
    );
    for (jobs) |*job| {
        defer job.deinit(group_alloc);
        if (save_overlap) |so| {
            try so.runRasterStageAndQueue(
                outer_alloc,
                group_io,
                job,
                raster_workers,
                runRasterStage,
            );
            continue;
        }
        try runRasterAndSaveFrame(
            outer_alloc,
            group_io,
            job,
            raster_workers,
        );
    }
}

// --------------------------------------------------------------------------------------
// Stage Runners
// --------------------------------------------------------------------------------------

const PreparedFrameJob = struct {
    desc: FrameJobDesc,
    ctx: FrameContext,
    time_start_frame: ?Timestamp = null,

    fn init(
        group_alloc: std.mem.Allocator,
        desc: FrameJobDesc,
    ) PreparedFrameJob {
        return .{
            .desc = desc,
            .ctx = FrameContext.init(group_alloc),
            .time_start_frame = null,
        };
    }

    fn deinit(
        self: *PreparedFrameJob,
        group_alloc: std.mem.Allocator,
    ) void {
        self.ctx.deinit(group_alloc, self.desc.config);
    }
};

fn runGeometryStage(
    group_alloc: std.mem.Allocator,
    io: std.Io,
    job: *PreparedFrameJob,
    geom_workers: u16,
) !void {
    if (job.time_start_frame == null) {
        job.time_start_frame = Timestamp.now(io, .awake);
    }

    const time_start_geo = Timestamp.now(io, .awake);
    const time_start_pfc = time_start_geo;
    try prepareFrameContext(
        group_alloc,
        &job.ctx,
        &job.desc,
    );
    const time_end_pfc = Timestamp.now(io, .awake);
    job.ctx.frame_times.prepare_frame_context = @floatFromInt(
        time_start_pfc.durationTo(time_end_pfc).raw.nanoseconds,
    );

    var chunk_exec = pce.ParaChunkExecutor.init(io, geom_workers);
    const arena_alloc = job.ctx.arena.allocator();

    var timing = mo.GeomTimes{};
    const raster_halo_px = job.desc.config.raster_halo_px_override orelse
        job.desc.camera.prep_psf.halo_px;
    const geo_res = try mo.prepMeshFrames(
        arena_alloc,
        &chunk_exec,
        scalingpolicy.geometryWorkers(geom_workers),
        job.desc.camera,
        raster_halo_px,
        job.desc.config,
        job.desc.frame_idx,
        job.desc.mesh_static,
        job.desc.nodal_global_scaling,
        job.ctx.frame_meshes,
        &timing,
    );

    job.ctx.frame_times.geom_coord_ops = @floatFromInt(timing.coord_ops);
    job.ctx.frame_times.geom_cull_ops = @floatFromInt(timing.cull_ops);
    job.ctx.frame_times.geom_prep_hulls_shaders = @floatFromInt(
        timing.prep_hulls_shaders,
    );
    job.ctx.frame_times.geom_remap_inds = @floatFromInt(timing.remap_inds);

    for (job.ctx.frame_meshes, 0..) |*fm, ii| {
        job.ctx.prep_meshes[ii] = fm.mesh;
        job.ctx.elem_bboxes_by_mesh[ii] = fm.elem_bboxes;
        job.ctx.elems_in_image_by_mesh[ii] = fm.elems_in_image;
        job.ctx.raster_hulls[ii] = fm.raster_hull;
    }
    job.ctx.total_elems_num = geo_res.total_elems_num;
    job.ctx.total_elems_in_image = geo_res.total_elems_in_image;
    job.ctx.total_nodes_num = mo.countStaticMeshNodes(job.desc.mesh_static);

    const time_end_geo = Timestamp.now(io, .awake);
    job.ctx.frame_times.geometry_prep = @floatFromInt(
        time_start_geo.durationTo(time_end_geo).raw.nanoseconds,
    );

    try sceneTileOverlapBinning(
        io,
        &job.desc,
        &chunk_exec,
        geom_workers,
        &job.ctx,
    );
}

fn sceneTileOverlapBinning(
    io: std.Io,
    job: *const FrameJobDesc,
    chunk_exec: *pce.ParaChunkExecutor,
    geom_workers: u16,
    ctx: *FrameContext,
) !void {
    const arena_alloc = ctx.arena.allocator();

    const tiles_num_x: usize = try std.math.divCeil(
        usize,
        job.camera.pixels_num[0],
        ctx.actual_tile_size,
    );
    const tiles_num_y: usize = try std.math.divCeil(
        usize,
        job.camera.pixels_num[1],
        ctx.actual_tile_size,
    );

    const time_start_overlap = Timestamp.now(io, .awake);
    ctx.tiling = try rops.sceneTileElemOverlap(
        arena_alloc,
        chunk_exec,
        scalingpolicy.geometryWorkers(geom_workers),
        ctx.actual_tile_size,
        tiles_num_x,
        tiles_num_y,
        @intCast(job.camera.pixels_num[0]),
        @intCast(job.camera.pixels_num[1]),
        job.config.raster_halo_px_override orelse job.camera.prep_psf.halo_px,
        ctx.elems_in_image_by_mesh,
        ctx.elem_bboxes_by_mesh,
    );
    const time_end_overlap = Timestamp.now(io, .awake);
    ctx.frame_times.tile_overlap = @floatFromInt(
        time_start_overlap.durationTo(time_end_overlap).raw.nanoseconds,
    );
}

fn runRasterStage(
    outer_alloc: std.mem.Allocator,
    io: std.Io,
    job: *PreparedFrameJob,
    raster_workers: u16,
) !void {
    const time_start_fb = Timestamp.now(io, .awake);
    try prepareFrameBuff(&job.ctx, &job.desc);
    const time_end_fb = Timestamp.now(io, .awake);
    job.ctx.frame_times.setup_frame_buff = @floatFromInt(
        time_start_fb.durationTo(time_end_fb).raw.nanoseconds,
    );

    switch (job.desc.config.report) {
        .off => try rasterFrame(
            .off,
            outer_alloc,
            io,
            &job.desc,
            raster_workers,
            &job.ctx,
        ),
        .bench => try rasterFrame(
            .bench,
            outer_alloc,
            io,
            &job.desc,
            raster_workers,
            &job.ctx,
        ),
        .full_stats => try rasterFrame(
            .full_stats,
            outer_alloc,
            io,
            &job.desc,
            raster_workers,
            &job.ctx,
        ),
    }
}

fn runRasterAndSaveFrame(
    outer_alloc: std.mem.Allocator,
    io: std.Io,
    job: *PreparedFrameJob,
    raster_workers: u16,
) !void {
    try runRasterStage(
        outer_alloc,
        io,
        job,
        raster_workers,
    );

    const time_start_save = Timestamp.now(io, .awake);
    try saveFrame(io, &job.desc, &job.ctx);
    const time_end_save = Timestamp.now(io, .awake);

    job.ctx.frame_times.save_frame = @floatFromInt(
        time_start_save.durationTo(time_end_save).raw.nanoseconds,
    );
    job.ctx.frame_times.active_time =
        job.ctx.frame_times.setup_frame_buff +
        job.ctx.frame_times.geometry_prep +
        job.ctx.frame_times.tile_overlap +
        job.ctx.frame_times.raster_loop +
        job.ctx.frame_times.save_frame;

    const time_end_frame = Timestamp.now(io, .awake);

    job.ctx.frame_times.latency_time = @floatFromInt(
        job.time_start_frame.?.durationTo(time_end_frame).raw.nanoseconds,
    );

    try report.publishFrameResults(
        outer_alloc,
        io,
        job.desc.config,
        job.ctx.actual_tile_size,
        job.desc.camera,
        job.desc.camera_idx,
        job.desc.frame_idx,
        job.desc.cameras_num,
        job.desc.out_dir,
        job.desc.bench_capture,
        &job.ctx.report_storage,
        job.ctx.frame_times,
        job.ctx.total_nodes_num,
        job.ctx.total_elems_num,
        job.ctx.total_elems_in_image,
        job.ctx.prep_meshes,
    );
}

// --------------------------------------------------------------------------------------
// Private Helpers
// --------------------------------------------------------------------------------------

fn FrameReportPtr(comptime report_mode: ReportMode) type {
    return *report.LogType(report_mode);
}

fn getFrameReportPtr(
    comptime report_mode: ReportMode,
    ctx: *FrameContext,
) FrameReportPtr(report_mode) {
    return switch (report_mode) {
        .off => &ctx.report_storage.off,
        .bench => &ctx.report_storage.bench,
        .full_stats => &ctx.report_storage.full_stats,
    };
}

fn initNodalGlobalScaling(
    outer_alloc: std.mem.Allocator,
    meshes: []const mo.MeshInput,
) ![]?imageops.ScalingParams {
    var nodal_global_scaling = try outer_alloc.alloc(?imageops.ScalingParams, meshes.len);

    for (meshes, 0..) |mesh, ii| {
        nodal_global_scaling[ii] = null;
        switch (mesh.shader) {
            .nodal => |s| {
                if (s.scale_over == .over_frames) {
                    nodal_global_scaling[ii] = imageops.getScalingParamsNDArray(
                        &s.field.array,
                        null,
                        s.scaling,
                    );
                }
            },
            else => {},
        }
    }

    return nodal_global_scaling;
}

fn initMeshStaticSlice(
    allocator: std.mem.Allocator,
    meshes: []const mo.MeshInput,
) ![]mo.MeshStatic {
    const mesh_static = try allocator.alloc(mo.MeshStatic, meshes.len);

    for (meshes, 0..) |mesh, ii| {
        mesh_static[ii] = try mo.initMeshStatic(allocator, &mesh);
    }

    return mesh_static;
}

fn calcAllFramesDimsFromPixels(
    camera_pixels_num: []const [2]u32,
    num_time: usize,
    num_fields: u8,
) [5]usize {
    std.debug.assert(camera_pixels_num.len > 0);

    var max_pixels_num = camera_pixels_num[0];
    for (camera_pixels_num[1..]) |pixels_num| {
        max_pixels_num[0] = @max(max_pixels_num[0], pixels_num[0]);
        max_pixels_num[1] = @max(max_pixels_num[1], pixels_num[1]);
    }

    return .{
        camera_pixels_num.len,
        num_time,
        @as(usize, num_fields),
        max_pixels_num[1],
        max_pixels_num[0],
    };
}

fn needsOutputTransform(
    image_save_mode: ImageSaveMode,
    raw_num_fields: u8,
) bool {
    return switch (image_save_mode) {
        .multifield => false,
        .grey => raw_num_fields != 1,
        .rgb => raw_num_fields != 3,
    };
}

fn getFrameImageView(
    allocator: std.mem.Allocator,
    images_arr: *ndarray.NDArray(F),
    camera_idx: usize,
    frame_idx: usize,
) !ndarray.NDArray(F) {
    std.debug.assert(images_arr.dims.len == 5);
    return try images_arr.fixedPrefixView(
        allocator,
        &[_]usize{ camera_idx, frame_idx },
    );
}

fn initFrameReportStorage(
    outer_alloc: std.mem.Allocator,
    camera: *const cam.CameraPrepared,
    actual_tile_size: u16,
    config: RasterConfig,
) !report.FrameReportStorage {
    return switch (config.report) {
        .off => .{ .off = .{} },
        .bench => .{ .bench = .{} },
        .full_stats => .{ .full_stats = try report.initFullStatsLog(
            outer_alloc,
            camera.pixels_num,
            actual_tile_size,
            camera.sub_sample,
            config.full_stats_opts,
        ) },
    };
}

// -------------------------------------------------------------------------------------
// Frame Assembly and Output Helpers
// --------------------------------------------------------------------------------------
const FrameJobDesc = struct {
    camera: *const cam.CameraPrepared,
    camera_idx: usize,
    frame_idx: usize,
    num_fields: u8,
    config: RasterConfig,
    out_dir: ?std.Io.Dir,
    mesh_static: []const mo.MeshStatic,
    nodal_global_scaling: []const ?imageops.ScalingParams,
    images_arr: ?*ndarray.NDArray(F),
    bench_capture: ?[]report.FrameBenchCapture,
    cameras_num: usize,
    can_write_result_direct: bool,
    save_slot: ?*saveoverlap.SaveSlot = null,
};

const FrameContext = struct {
    arena: std.heap.ArenaAllocator,

    frame_meshes: []mo.MeshFrame = &.{},
    prep_meshes: []mo.MeshPrepared = &.{},
    elem_bboxes_by_mesh: [][]rops.ElemBBox = &.{},
    elems_in_image_by_mesh: []usize = &.{},
    raster_hulls: []?ndarray.NDArray(F) = &.{},
    tiling: ?rops.TilingOverlaps = null,
    total_nodes_num: usize = 0,
    total_elems_num: usize = 0,
    total_elems_in_image: usize = 0,
    actual_tile_size: u16 = 1,

    frame_arr: ndarray.NDArray(F) = undefined,

    report_storage: report.FrameReportStorage = .{ .off = .{} },
    frame_times: report.FrameTimes = .{},

    fn init(
        outer_alloc: std.mem.Allocator,
    ) FrameContext {
        return .{
            .arena = std.heap.ArenaAllocator.init(outer_alloc),
        };
    }

    fn deinit(
        self: *FrameContext,
        outer_alloc: std.mem.Allocator,
        config: RasterConfig,
    ) void {
        report.deinitFrameReportStorage(
            outer_alloc,
            config,
            &self.report_storage,
        );
        self.arena.deinit();
    }
};

fn prepareFrameContext(
    outer_alloc: std.mem.Allocator,
    ctx: *FrameContext,
    input: *const FrameJobDesc,
) !void {
    const arena_alloc = ctx.arena.allocator();
    ctx.actual_tile_size = scalingpolicy.tileSize(
        input.config.tile_size_override,
        input.config.tile_size_min,
        input.config.tile_size_max,
        input.camera.pixels_num,
        input.camera.sub_sample,
        input.config.raster_halo_px_override orelse input.camera.prep_psf.halo_px,
    );

    ctx.report_storage = try initFrameReportStorage(
        outer_alloc,
        input.camera,
        ctx.actual_tile_size,
        input.config,
    );

    const mesh_n = input.mesh_static.len;
    ctx.frame_meshes = try arena_alloc.alloc(mo.MeshFrame, mesh_n);
    ctx.prep_meshes = try arena_alloc.alloc(mo.MeshPrepared, mesh_n);
    ctx.elem_bboxes_by_mesh = try arena_alloc.alloc([]rops.ElemBBox, mesh_n);
    ctx.elems_in_image_by_mesh = try arena_alloc.alloc(usize, mesh_n);
    ctx.raster_hulls = try arena_alloc.alloc(?ndarray.NDArray(F), mesh_n);
}

fn prepareFrameBuff(
    ctx: *FrameContext,
    input: *const FrameJobDesc,
) !void {
    const arena_alloc = ctx.arena.allocator();
    const dims = [_]usize{
        @as(usize, input.num_fields),
        input.camera.pixels_num[1],
        input.camera.pixels_num[0],
    };

    if (input.save_slot) |save_slot| {
        ctx.frame_arr = save_slot.frame_arr;
        std.debug.assert(ctx.frame_arr.dims.len == dims.len);
        for (dims, 0..) |dim, ii| {
            std.debug.assert(ctx.frame_arr.dims[ii] == dim);
        }
        @memset(ctx.frame_arr.slice, input.config.background_value);
        return;
    }

    if (input.can_write_result_direct) {
        const images_arr = input.images_arr orelse return error.NoResult;
        ctx.frame_arr = try getFrameImageView(
            arena_alloc,
            images_arr,
            input.camera_idx,
            input.frame_idx,
        );
    } else {
        ctx.frame_arr = try ndarray.NDArray(F).initFlat(
            arena_alloc,
            dims[0..],
        );
    }

    std.debug.assert(ctx.frame_arr.dims.len == dims.len);
    for (dims, 0..) |dim, ii| {
        std.debug.assert(ctx.frame_arr.dims[ii] == dim);
    }
    @memset(ctx.frame_arr.slice, input.config.background_value);
}

fn copyFrameToImageBatch(
    background_val: F,
    images_arr: *ndarray.NDArray(F),
    camera_idx: usize,
    frame_idx: usize,
    frame_arr: *const ndarray.NDArray(F),
) void {
    std.debug.assert(images_arr.dims.len == 5);
    std.debug.assert(frame_arr.dims.len == 3);
    std.debug.assert(frame_arr.dims[0] == images_arr.dims[2]);
    std.debug.assert(frame_arr.dims[1] <= images_arr.dims[3]);
    std.debug.assert(frame_arr.dims[2] <= images_arr.dims[4]);

    const dst_base = camera_idx * images_arr.strides[0] + frame_idx * images_arr.strides[1];
    const dst_field_stride = images_arr.strides[2];
    const dst_row_stride = images_arr.strides[3];
    const src_field_stride = frame_arr.strides[0];
    const src_row_stride = frame_arr.strides[1];
    const dst_rows = images_arr.dims[3];
    const dst_cols = images_arr.dims[4];
    const src_rows = frame_arr.dims[1];
    const src_cols = frame_arr.dims[2];
    for (0..frame_arr.dims[0]) |ff| {
        const dst_field_base = dst_base + ff * dst_field_stride;
        const src_field_base = ff * src_field_stride;

        for (0..dst_rows) |rr| {
            const dst_row_base = dst_field_base + rr * dst_row_stride;
            @memset(
                images_arr.slice[dst_row_base .. dst_row_base + dst_cols],
                background_val,
            );
        }

        for (0..src_rows) |rr| {
            const dst_row_base = dst_field_base + rr * dst_row_stride;
            const src_row_base = src_field_base + rr * src_row_stride;
            @memcpy(
                images_arr.slice[dst_row_base .. dst_row_base + src_cols],
                frame_arr.slice[src_row_base .. src_row_base + src_cols],
            );
        }
    }
}

fn rasterFrame(
    comptime report_mode: ReportMode,
    outer_alloc: std.mem.Allocator,
    io: std.Io,
    input: *const FrameJobDesc,
    raster_workers: u16,
    ctx: *FrameContext,
) !void {
    const report_ptr = getFrameReportPtr(report_mode, ctx);
    const ctx_report = report.ReportContext(report_mode){ .log = report_ptr };
    const time_start_loop = Timestamp.now(io, .awake);

    const ctx_rast = rops.RasterContext{
        .camera = input.camera,
        .config = input.config,
        .frame_idx = input.frame_idx,
        .tile_size = ctx.actual_tile_size,
    };

    try rasterengine.rasterScene(
        report_mode,
        outer_alloc,
        io,
        ctx_rast,
        ctx_report,
        raster_workers,
        ctx.tiling.?,
        ctx.prep_meshes,
        ctx.raster_hulls,
        &ctx.frame_arr,
    );

    const time_end_loop = Timestamp.now(io, .awake);
    ctx.frame_times.raster_loop = @floatFromInt(
        time_start_loop.durationTo(time_end_loop).raw.nanoseconds,
    );
    if (report.getBenchLog(report_mode, report_ptr)) |bench_log| {
        ctx.frame_times.cam_invert = bench_log.cam_time_ns;
        ctx.frame_times.scratch_resolve = bench_log.resolve_time_ns;
    }
}

fn saveFrame(
    io: std.Io,
    input: *const FrameJobDesc,
    ctx: *FrameContext,
) !void {
    const arena_alloc = ctx.arena.allocator();
    const output_frame_arr = try saveoverlap.buildOutputFrameView(
        arena_alloc,
        input.config,
        &ctx.frame_arr,
    );
    if (input.config.save_strategy == .disk or input.config.save_strategy == .both) {
        std.debug.assert(output_frame_arr.dims[0] <= std.math.maxInt(u8));
        try iio.saveImages(
            io,
            input.out_dir,
            input.camera_idx,
            input.frame_idx,
            @intCast(output_frame_arr.dims[0]),
            input.camera.pixels_num,
            &output_frame_arr,
            saveoverlap.imageSaveChannelsOverride(input.config.image_save_mode),
            input.config.image_save_opts,
        );
    }
    if ((input.config.save_strategy == .memory or
        input.config.save_strategy == .both) and
        !input.can_write_result_direct)
    {
        const images_arr = input.images_arr orelse return error.NoResult;
        copyFrameToImageBatch(
            input.config.background_value,
            images_arr,
            input.camera_idx,
            input.frame_idx,
            &output_frame_arr,
        );
    }
}

fn renderGroupSaveIo(render_group: RenderGroupSpec) std.Io {
    return render_group.save_frame_io orelse render_group.io;
}

fn saveOverlapEnabled(config: RasterConfig) bool {
    return config.save_strategy == .disk and config.disk_save_overlap;
}
