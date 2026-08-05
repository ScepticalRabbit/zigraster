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

const Timestamp = std.Io.Clock.Timestamp;
const ndarray = @import("ndarray.zig");
const cam = @import("camera.zig");
const mo = @import("meshpipeline.zig");
const iio = @import("imageio.zig");
const rastcfg = @import("rasterconfig.zig");
const report = @import("report.zig");

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const RasterConfig = rastcfg.RasterConfig;
pub const ImageSaveMode = rastcfg.ImageSaveMode;
pub const FrameReportStorage = report.FrameReportStorage;

pub const SaveSlotState = enum {
    free,
    rendering,
    ready_to_save,
    saving,
};

pub const SaveSlot = struct {
    state: SaveSlotState = .free,
    frame_arr: ndarray.NDArray(F),
    camera: ?*const cam.CameraPrepared = null,
    camera_idx: usize = 0,
    frame_idx: usize = 0,
    cameras_num: usize = 0,
    num_fields: u8 = 0,
    pixels_num: [2]u32 = .{ 0, 0 },
    out_dir: ?std.Io.Dir = null,
    bench_capture: ?[]report.FrameBenchCapture = null,
    report_storage: FrameReportStorage = .{ .off = .{} },
    frame_times: report.FrameTimes = .{},
    total_nodes_num: usize = 0,
    total_elems_num: usize = 0,
    total_elems_in_image: usize = 0,
    nodes_per_elem: F = 0.0,
    actual_tile_size: u16 = 1,
    time_start_frame: ?Timestamp = null,

    pub fn resetReportStorage(
        self: *SaveSlot,
        outer_alloc: std.mem.Allocator,
        config: RasterConfig,
    ) void {
        report.deinitFrameReportStorage(
            outer_alloc,
            config,
            &self.report_storage,
        );
    }
};

pub const SaveCoordinator = struct {
    mutex: std.Io.Mutex = .init,
    ready_cond: std.Io.Condition = .init,
    free_cond: std.Io.Condition = .init,
    done_submitting: bool = false,
    first_err: ?anyerror = null,
    slots: []SaveSlot,

    pub fn setFirstError(
        self: *SaveCoordinator,
        io: std.Io,
        err: anyerror,
    ) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        if (self.first_err == null) {
            self.first_err = err;
        }
        self.done_submitting = true;
        self.ready_cond.broadcast(io);
        self.free_cond.broadcast(io);
    }
};

pub const SaveSlotBuff = struct {
    slots: []SaveSlot,
    frame_pool: ndarray.NDArray(F),
};

pub const SaveOverlap = struct {
    outer_alloc: std.mem.Allocator,
    save_io: std.Io,
    config: RasterConfig,
    enabled_flag: bool,
    arena: std.heap.ArenaAllocator,
    slot_buff: ?SaveSlotBuff = null,
    coordinator: ?*SaveCoordinator = null,
    thread: ?std.Thread = null,

    pub fn initMaybe(
        outer_alloc: std.mem.Allocator,
        save_io: std.Io,
        cameras: []const cam.CameraPrepared,
        num_fields: u8,
        config: RasterConfig,
        is_enabled: bool,
    ) !SaveOverlap {
        var session = SaveOverlap{
            .outer_alloc = outer_alloc,
            .save_io = save_io,
            .config = config,
            .enabled_flag = is_enabled,
            .arena = std.heap.ArenaAllocator.init(outer_alloc),
        };

        if (!is_enabled) return session;
        errdefer session.arena.deinit();

        session.slot_buff = try initSaveSlots(
            session.arena.allocator(),
            cameras,
            num_fields,
            @max(@as(usize, 1), config.save_frame_buff_count),
        );

        const coordinator = try outer_alloc.create(SaveCoordinator);
        errdefer outer_alloc.destroy(coordinator);
        coordinator.* = .{ .slots = session.slot_buff.?.slots };
        session.coordinator = coordinator;
        session.thread = try std.Thread.spawn(
            .{},
            saveWorkerLoop,
            .{
                outer_alloc,
                save_io,
                coordinator,
                config,
            },
        );
        return session;
    }

    pub fn deinit(self: *SaveOverlap) void {
        if (self.coordinator) |coordinator| {
            coordinator.mutex.lockUncancelable(self.save_io);
            coordinator.done_submitting = true;
            coordinator.ready_cond.broadcast(self.save_io);
            coordinator.free_cond.broadcast(self.save_io);
            coordinator.mutex.unlock(self.save_io);

            if (self.thread) |thread| {
                thread.join();
                self.thread = null;
            }
            for (coordinator.slots) |*slot| {
                slot.resetReportStorage(self.outer_alloc, self.config);
            }
            self.outer_alloc.destroy(coordinator);
            self.coordinator = null;
        }
        self.arena.deinit();
    }

    pub fn enabled(self: *const SaveOverlap) bool {
        return self.enabled_flag;
    }

    pub fn acquireSlot(
        self: *SaveOverlap,
        io: std.Io,
    ) !*SaveSlot {
        const coordinator = self.coordinator orelse unreachable;
        const slot_idx = try acquireSaveSlot(io, coordinator);
        return &coordinator.slots[slot_idx];
    }

    pub fn publishSlot(
        self: *SaveOverlap,
        io: std.Io,
        slot: *SaveSlot,
        meta: RenderedFrameMeta,
        report_storage: *FrameReportStorage,
        prep_meshes: []const mo.MeshPrepared,
    ) !void {
        const coordinator = self.coordinator orelse unreachable;
        try publishRenderedSlot(
            io,
            coordinator,
            slot,
            meta,
            report_storage,
            prep_meshes,
        );
    }

    fn releaseSlot(
        self: *SaveOverlap,
        io: std.Io,
        slot: *SaveSlot,
    ) void {
        const coordinator = self.coordinator orelse unreachable;
        coordinator.mutex.lockUncancelable(io);
        defer coordinator.mutex.unlock(io);
        slot.resetReportStorage(self.outer_alloc, self.config);
        slot.state = .free;
        coordinator.free_cond.signal(io);
    }

    pub fn checkError(self: *SaveOverlap) !void {
        if (!self.enabled_flag) return;
        const coordinator = self.coordinator orelse unreachable;
        coordinator.mutex.lockUncancelable(self.save_io);
        defer coordinator.mutex.unlock(self.save_io);
        if (coordinator.first_err) |err| return err;
    }

    pub fn runRasterStageAndQueue(
        self: *SaveOverlap,
        outer_alloc: std.mem.Allocator,
        io: std.Io,
        job: anytype,
        raster_workers: u16,
        comptime raster_stage_fn: anytype,
    ) !void {
        const slot = try self.acquireSlot(io);
        errdefer self.releaseSlot(io, slot);
        job.desc.save_slot = slot;
        try raster_stage_fn(
            outer_alloc,
            io,
            job,
            raster_workers,
        );
        try self.publishSlot(
            io,
            slot,
            .{
                .camera = job.desc.camera,
                .camera_idx = job.desc.camera_idx,
                .frame_idx = job.desc.frame_idx,
                .cameras_num = job.desc.cameras_num,
                .pixels_num = .{
                    job.desc.camera.pixels_num[0],
                    job.desc.camera.pixels_num[1],
                },
                .out_dir = job.desc.out_dir,
                .bench_capture = job.desc.bench_capture,
                .frame_times = job.ctx.frame_times,
                .total_nodes_num = job.ctx.total_nodes_num,
                .total_elems_num = job.ctx.total_elems_num,
                .total_elems_in_image = job.ctx.total_elems_in_image,
                .actual_tile_size = job.ctx.actual_tile_size,
                .time_start_frame = job.time_start_frame,
            },
            &job.ctx.report_storage,
            job.ctx.prep_meshes,
        );
    }
};

pub const RenderedFrameMeta = struct {
    camera: *const cam.CameraPrepared,
    camera_idx: usize,
    frame_idx: usize,
    cameras_num: usize,
    pixels_num: [2]u32,
    out_dir: ?std.Io.Dir,
    bench_capture: ?[]report.FrameBenchCapture,
    frame_times: report.FrameTimes,
    total_nodes_num: usize,
    total_elems_num: usize,
    total_elems_in_image: usize,
    actual_tile_size: u16,
    time_start_frame: ?Timestamp,
};

inline fn needsOutputTransform(
    image_save_mode: ImageSaveMode,
    raw_num_fields: u8,
) bool {
    return switch (image_save_mode) {
        .multifield => false,
        .grey => raw_num_fields != 1,
        .rgb => raw_num_fields != 3,
    };
}

inline fn outputFieldsForImageSaveMode(
    image_save_mode: ImageSaveMode,
    raw_num_fields: u8,
) !u8 {
    return switch (image_save_mode) {
        .multifield => raw_num_fields,
        .grey => switch (raw_num_fields) {
            1, 3 => 1,
            else => error.UnsuppedImageModeFieldCount,
        },
        .rgb => switch (raw_num_fields) {
            1, 3 => 3,
            else => error.UnsuppedImageModeFieldCount,
        },
    };
}

// --------------------------------------------------------------------------------------
// Public Entry-Point Func
// --------------------------------------------------------------------------------------

pub fn imageSaveChannelsOverride(image_save_mode: ImageSaveMode) ?usize {
    return switch (image_save_mode) {
        .grey => 1,
        .rgb => 3,
        .multifield => null,
    };
}

fn rgbFieldsToGrey(
    red_val: F,
    green_val: F,
    blue_val: F,
) F {
    return 0.299 * red_val + 0.587 * green_val + 0.114 * blue_val;
}

pub fn buildOutputFrameView(
    allocator: std.mem.Allocator,
    config: RasterConfig,
    raw_frame_arr: *const ndarray.NDArray(F),
) !ndarray.NDArray(F) {
    std.debug.assert(raw_frame_arr.dims.len == 3);
    const raw_num_fields: u8 = @intCast(raw_frame_arr.dims[0]);
    if (!needsOutputTransform(config.image_save_mode, raw_num_fields)) {
        return raw_frame_arr.*;
    }

    const out_num_fields = try outputFieldsForImageSaveMode(
        config.image_save_mode,
        raw_num_fields,
    );
    var output_frame_arr = try ndarray.NDArray(F).initFlat(
        allocator,
        &[_]usize{
            @as(usize, out_num_fields),
            raw_frame_arr.dims[1],
            raw_frame_arr.dims[2],
        },
    );

    switch (config.image_save_mode) {
        .multifield => unreachable,
        .grey => {
            std.debug.assert(raw_num_fields == 3);
            for (0..raw_frame_arr.dims[1]) |rr| {
                for (0..raw_frame_arr.dims[2]) |cc| {
                    const grey_val = rgbFieldsToGrey(
                        raw_frame_arr.get(&[_]usize{ 0, rr, cc }),
                        raw_frame_arr.get(&[_]usize{ 1, rr, cc }),
                        raw_frame_arr.get(&[_]usize{ 2, rr, cc }),
                    );
                    output_frame_arr.set(&[_]usize{ 0, rr, cc }, grey_val);
                }
            }
        },
        .rgb => {
            std.debug.assert(raw_num_fields == 1);
            for (0..raw_frame_arr.dims[1]) |rr| {
                for (0..raw_frame_arr.dims[2]) |cc| {
                    const grey_val = raw_frame_arr.get(&[_]usize{ 0, rr, cc });
                    for (0..3) |ff| {
                        output_frame_arr.set(&[_]usize{ ff, rr, cc }, grey_val);
                    }
                }
            }
        },
    }

    return output_frame_arr;
}

pub fn initSaveSlots(
    save_alloc: std.mem.Allocator,
    cameras: []const cam.CameraPrepared,
    num_fields: u8,
    slot_count: usize,
) !SaveSlotBuff {
    std.debug.assert(cameras.len > 0);
    var max_pixels_num = cameras[0].pixels_num;
    for (cameras[1..]) |camera| {
        max_pixels_num[0] = @max(max_pixels_num[0], camera.pixels_num[0]);
        max_pixels_num[1] = @max(max_pixels_num[1], camera.pixels_num[1]);
    }
    const frame_pool = try ndarray.NDArray(F).initFlat(
        save_alloc,
        &[_]usize{
            slot_count,
            @as(usize, num_fields),
            max_pixels_num[1],
            max_pixels_num[0],
        },
    );
    const slots = try save_alloc.alloc(SaveSlot, slot_count);
    for (0..slot_count) |ss| {
        slots[ss] = .{
            .frame_arr = try frame_pool.fixedPrefixView(
                save_alloc,
                &[_]usize{ss},
            ),
        };
    }
    return SaveSlotBuff{
        .slots = slots,
        .frame_pool = frame_pool,
    };
}

pub fn acquireSaveSlot(
    io: std.Io,
    coordinator: *SaveCoordinator,
) !usize {
    try coordinator.mutex.lock(io);
    defer coordinator.mutex.unlock(io);

    while (true) {
        if (coordinator.first_err) |err| return err;
        for (coordinator.slots, 0..) |*slot, ss| {
            if (slot.state == .free) {
                slot.state = .rendering;
                return ss;
            }
        }
        try coordinator.free_cond.wait(io, &coordinator.mutex);
    }
}

pub fn publishRenderedSlot(
    io: std.Io,
    coordinator: *SaveCoordinator,
    slot: *SaveSlot,
    meta: RenderedFrameMeta,
    report_storage: *FrameReportStorage,
    prep_meshes: []const mo.MeshPrepared,
) !void {
    slot.camera = meta.camera;
    slot.camera_idx = meta.camera_idx;
    slot.frame_idx = meta.frame_idx;
    slot.cameras_num = meta.cameras_num;
    slot.num_fields = @intCast(slot.frame_arr.dims[0]);
    slot.pixels_num = meta.pixels_num;
    slot.out_dir = meta.out_dir;
    slot.bench_capture = meta.bench_capture;
    slot.report_storage = report_storage.*;
    report_storage.* = .{ .off = .{} };
    slot.frame_times = meta.frame_times;
    slot.total_nodes_num = meta.total_nodes_num;
    slot.total_elems_num = meta.total_elems_num;
    slot.total_elems_in_image = meta.total_elems_in_image;
    slot.nodes_per_elem = mo.calcNodesPerElem(prep_meshes);
    slot.actual_tile_size = meta.actual_tile_size;
    slot.time_start_frame = meta.time_start_frame;

    try coordinator.mutex.lock(io);
    slot.state = .ready_to_save;
    coordinator.ready_cond.signal(io);
    coordinator.mutex.unlock(io);
}

pub fn completeSaveSlot(
    outer_alloc: std.mem.Allocator,
    save_io: std.Io,
    coordinator: *SaveCoordinator,
    slot_idx: usize,
    config: RasterConfig,
) !void {
    var slot = &coordinator.slots[slot_idx];
    const time_start_save = Timestamp.now(save_io, .awake);
    std.debug.assert(slot.camera != null);
    std.debug.assert(slot.pixels_num[0] > 0);
    std.debug.assert(slot.pixels_num[1] > 0);
    var save_arena = std.heap.ArenaAllocator.init(outer_alloc);
    defer save_arena.deinit();
    const output_frame_arr = try buildOutputFrameView(
        save_arena.allocator(),
        config,
        &slot.frame_arr,
    );
    try iio.saveImages(
        save_io,
        slot.out_dir,
        slot.camera_idx,
        slot.frame_idx,
        @intCast(output_frame_arr.dims[0]),
        slot.pixels_num,
        &output_frame_arr,
        imageSaveChannelsOverride(config.image_save_mode),
        config.image_save_opts,
    );
    const time_end_save = Timestamp.now(save_io, .awake);

    slot.frame_times.save_frame = @floatFromInt(
        time_start_save.durationTo(time_end_save).raw.nanoseconds,
    );
    slot.frame_times.active_time =
        slot.frame_times.geometry_prep +
        slot.frame_times.tile_overlap +
        slot.frame_times.raster_loop +
        slot.frame_times.save_frame;
    slot.frame_times.latency_time = @floatFromInt(
        slot.time_start_frame.?.durationTo(time_end_save).raw.nanoseconds,
    );

    try report.publishFrameResultsWithNodesPerElem(
        outer_alloc,
        save_io,
        config,
        slot.actual_tile_size,
        slot.camera.?,
        slot.camera_idx,
        slot.frame_idx,
        slot.cameras_num,
        slot.out_dir,
        slot.bench_capture,
        &slot.report_storage,
        slot.frame_times,
        slot.total_nodes_num,
        slot.total_elems_num,
        slot.total_elems_in_image,
        slot.nodes_per_elem,
    );

    try coordinator.mutex.lock(save_io);
    defer coordinator.mutex.unlock(save_io);
    slot.resetReportStorage(outer_alloc, config);
    slot.state = .free;
    coordinator.free_cond.signal(save_io);
}

pub fn saveWorkerLoop(
    outer_alloc: std.mem.Allocator,
    save_io: std.Io,
    coordinator: *SaveCoordinator,
    config: RasterConfig,
) void {
    while (true) {
        coordinator.mutex.lockUncancelable(save_io);
        while (true) {
            if (coordinator.first_err != null) {
                coordinator.mutex.unlock(save_io);
                return;
            }
            var ready_idx: ?usize = null;
            for (coordinator.slots, 0..) |slot, ss| {
                if (slot.state == .ready_to_save) {
                    ready_idx = ss;
                    break;
                }
            }
            if (ready_idx) |slot_idx| {
                coordinator.slots[slot_idx].state = .saving;
                coordinator.mutex.unlock(save_io);
                completeSaveSlot(
                    outer_alloc,
                    save_io,
                    coordinator,
                    slot_idx,
                    config,
                ) catch |err| {
                    coordinator.setFirstError(save_io, err);
                    return;
                };
                break;
            }
            if (coordinator.done_submitting) {
                coordinator.mutex.unlock(save_io);
                return;
            }
            coordinator.ready_cond.waitUncancelable(save_io, &coordinator.mutex);
        }
    }
}
