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
const camera = @import("../riley/zig/camera.zig");
const common = @import("../dev_support/tests.zig");
const common_full = @import("../gengold/gen_gold_full_common.zig");
const iio = @import("../riley/zig/imageio.zig");
const mo = @import("../riley/zig/meshpipeline.zig");
const ndarray = @import("../riley/zig/ndarray.zig");
const policy = @import("../dev_support/testpolicy.zig");
const rastcfg = @import("../riley/zig/rasterconfig.zig");
const riley = @import("../riley/zig/riley.zig");
const tcfg = @import("../dev_support/testconfig.zig");

const F = buildconfig.F;
const CameraInput = camera.CameraInput;
const MeshInput = mo.MeshInput;
const NDArray = ndarray.NDArray;

pub const FULL_SCENE_CAMERA_REL_TOL: F = if (F == f32) 1.0e-3 else 1.0e-5;
pub const FULL_SCENE_CAMERA_ABS_TOL: F = if (F == f32) 1.0e-3 else 1.0e-5;

pub const EQUIV_REL_TOL: F = if (F == f32) 1.0e-4 else 1.0e-6;
pub const EQUIV_ABS_TOL: F = if (F == f32) 1.0e-4 else 1.0e-6;

pub const ThreadingCase = struct {
    tag: []const u8,
    group_count: usize,
    workers_per_group: u16,
    total_threads: u16,
    max_geom_workers: u16,
    max_raster_workers: u16,
    geom_scheduling_mode: rastcfg.GeometrySchedulingMode,
};

pub const threading_cases = [_]ThreadingCase{
    .{
        .tag = "1g_1t",
        .group_count = 1,
        .workers_per_group = 1,
        .total_threads = 1,
        .max_geom_workers = 1,
        .max_raster_workers = 1,
        .geom_scheduling_mode = .auto,
    },
    .{
        .tag = "1g_4r",
        .group_count = 1,
        .workers_per_group = 4,
        .total_threads = 4,
        .max_geom_workers = 1,
        .max_raster_workers = 4,
        .geom_scheduling_mode = .auto,
    },
    .{
        .tag = "2g_2r_auto",
        .group_count = 2,
        .workers_per_group = 2,
        .total_threads = 4,
        .max_geom_workers = 2,
        .max_raster_workers = 2,
        .geom_scheduling_mode = .auto,
    },
    .{
        .tag = "2g_2r_spread",
        .group_count = 2,
        .workers_per_group = 2,
        .total_threads = 4,
        .max_geom_workers = 2,
        .max_raster_workers = 2,
        .geom_scheduling_mode = .spread,
    },
    .{
        .tag = "2g_2r_pack",
        .group_count = 2,
        .workers_per_group = 2,
        .total_threads = 4,
        .max_geom_workers = 2,
        .max_raster_workers = 2,
        .geom_scheduling_mode = .pack,
    },
    .{
        .tag = "4g_1t",
        .group_count = 4,
        .workers_per_group = 1,
        .total_threads = 4,
        .max_geom_workers = 1,
        .max_raster_workers = 1,
        .geom_scheduling_mode = .auto,
    },
};

pub const BufferModeCase = struct {
    tag: []const u8,
    mode: rastcfg.BufferMode,
};

pub const buffer_mode_cases = [_]BufferModeCase{
    .{ .tag = "tile_local", .mode = .tile_local },
    .{ .tag = "global_subpx", .mode = .global_subpx_full },
    .{ .tag = "global_stripe", .mode = .global_subpx_stripe },
};

pub const SchedulingCase = struct {
    tag: []const u8,
    mode: rastcfg.RenderMode,
};

pub const scheduling_cases = [_]SchedulingCase{
    .{ .tag = "in_order", .mode = .in_order },
    .{ .tag = "offline", .mode = .offline },
};

fn runRender(
    allocator: std.mem.Allocator,
    io: std.Io,
    cameras: []const CameraInput,
    meshes: []const MeshInput,
    thread_case: ThreadingCase,
    buf_case: BufferModeCase,
    sched_case: SchedulingCase,
    config: rastcfg.RasterConfig,
) !NDArray(F) {
    var run_config = config;
    run_config.save_strategy = .memory;
    run_config.render_mode = sched_case.mode;
    run_config.buffer_mode = buf_case.mode;
    run_config.total_threads = thread_case.total_threads;
    run_config.max_geom_workers_per_job = thread_case.max_geom_workers;
    run_config.max_raster_workers_per_job = thread_case.max_raster_workers;
    run_config.geom_scheduling_mode = thread_case.geom_scheduling_mode;

    var render_groups: [4]riley.RenderGroupSpec = undefined;
    for (0..thread_case.group_count) |gg| {
        render_groups[gg] = .{
            .io = io,
            .workers = thread_case.workers_per_group,
        };
    }

    const result = try riley.raster(
        allocator,
        render_groups[0..thread_case.group_count],
        cameras,
        meshes,
        run_config,
        null,
    );
    return result orelse error.MissingRenderOutput;
}

fn assertImagesEqual(
    actual: *const NDArray(F),
    expected: *const NDArray(F),
    rel_tol: F,
    abs_tol: F,
) !void {
    try std.testing.expectEqual(actual.dims.len, expected.dims.len);
    for (0..actual.dims.len) |dd| {
        try std.testing.expectEqual(actual.dims[dd], expected.dims[dd]);
    }
    for (actual.slice, 0..) |act_val, ii| {
        const exp_val = expected.slice[ii];
        if (!common.isApproxEqual(exp_val, act_val, rel_tol, abs_tol)) {
            const diff = @abs(exp_val - act_val);
            std.debug.print(
                "\nPixel mismatch: act={d} exp={d} diff={e}\n",
                .{ act_val, exp_val, diff },
            );
            return error.ImageMismatch;
        }
    }
}

fn assertImagesDiffer(
    img_a: *const NDArray(F),
    img_b: *const NDArray(F),
    diff_threshold: F,
) !void {
    var max_diff: F = 0.0;
    const len = @min(img_a.slice.len, img_b.slice.len);
    for (0..len) |ii| {
        const diff = @abs(img_a.slice[ii] - img_b.slice[ii]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    if (max_diff < diff_threshold) {
        std.debug.print(
            "\nExpected images to differ, but max diff was only {e}\n",
            .{max_diff},
        );
        return error.ImagesUnexpectedlyIdentical;
    }
}

pub fn run(allocator: std.mem.Allocator, io: std.Io) !void {
    var config = tcfg.getRasterConfig(.testing);
    config.save_strategy = .memory;
    config.image_save_mode = .grey;
    config.background_value = 32767.5;
    config.image_save_opts = &[_]iio.ImageSaveOpts{
        .{ .format = .fimg, .bits = null, .scaling = .none },
        .{ .format = .bmp, .bits = 8, .scaling = .auto },
    };

    const gold_dir_root = policy.goldRoot(.full_scene_camera_threads);

    var textures = try common_full.FullTextures.init(allocator, io);
    defer textures.deinit(allocator);

    var prep = try common_full.prepareScene3(allocator, io);
    defer prep.deinit(allocator);

    const meshes = common_full.buildScene3Meshes(&prep, &textures);
    const cameras = common_full.createScene3Cameras();

    // ----------------------------------------------------------------------
    // 1. Render Reference (1 thread, in_order, tile_local) for all 8 cameras
    // ----------------------------------------------------------------------
    const ref_thread = threading_cases[0];
    const ref_buf = buffer_mode_cases[0];
    const ref_sched = scheduling_cases[0];

    var ref_render = try runRender(
        allocator,
        io,
        &cameras,
        &meshes,
        ref_thread,
        ref_buf,
        ref_sched,
        config,
    );
    defer {
        allocator.free(ref_render.slice);
        ref_render.deinit(allocator);
    }

    // ----------------------------------------------------------------------
    // 2. Verify Reference Render Against Gold for All 8 Cameras & 4 Frames
    // ----------------------------------------------------------------------
    for (0..8) |cc| {
        for (0..4) |ff| {
            const cam_dir = try std.fmt.allocPrint(
                allocator,
                "{s}/cam{d}",
                .{ gold_dir_root, cc },
            );
            defer allocator.free(cam_dir);

            const gold_path = try common.findGoldPath(
                allocator,
                io,
                cam_dir,
                0,
                ff,
                0,
                false,
            );
            defer allocator.free(gold_path);

            const case_desc = try std.fmt.allocPrint(
                allocator,
                "cam{d}_frame{d}",
                .{ cc, ff },
            );
            defer allocator.free(case_desc);

            common.compareNDArrayToGold(
                allocator,
                io,
                &ref_render,
                cc,
                ff,
                0,
                1,
                gold_path,
                FULL_SCENE_CAMERA_REL_TOL,
                FULL_SCENE_CAMERA_ABS_TOL,
            ) catch |err| {
                const fail_dir_name = try std.fmt.allocPrint(
                    allocator,
                    "full_scene_camera_threads/{s}",
                    .{case_desc},
                );
                defer allocator.free(fail_dir_name);
                try common.saveComparisonArtifactsFromResult(
                    allocator,
                    io,
                    common.default_fails_root,
                    fail_dir_name,
                    &ref_render,
                    cc,
                    ff,
                    0,
                    gold_path,
                    1,
                );
                return err;
            };
        }
    }

    // ----------------------------------------------------------------------
    // 3. Multi-Camera Equivalence: Camera0 == Camera1 across all frames
    // ----------------------------------------------------------------------
    for (0..4) |ff| {
        var cam0_img = try common.extractFrameImage(allocator, &ref_render, 0, ff, 0, 1);
        defer {
            allocator.free(cam0_img.slice);
            cam0_img.deinit(allocator);
        }

        var cam1_img = try common.extractFrameImage(allocator, &ref_render, 1, ff, 0, 1);
        defer {
            allocator.free(cam1_img.slice);
            cam1_img.deinit(allocator);
        }

        try assertImagesEqual(&cam0_img, &cam1_img, EQUIV_REL_TOL, EQUIV_ABS_TOL);
    }

    // ----------------------------------------------------------------------
    // 4. Temporal Deformation Check: Frame 0 != Frame 1 != Frame 2 != Frame 3
    // ----------------------------------------------------------------------
    for (0..8) |cc| {
        var frame0_img = try common.extractFrameImage(allocator, &ref_render, cc, 0, 0, 1);
        defer {
            allocator.free(frame0_img.slice);
            frame0_img.deinit(allocator);
        }

        for (1..4) |ff| {
            var frame_img = try common.extractFrameImage(
                allocator,
                &ref_render,
                cc,
                ff,
                0,
                1,
            );
            defer {
                allocator.free(frame_img.slice);
                frame_img.deinit(allocator);
            }

            try assertImagesDiffer(&frame0_img, &frame_img, 0.05);
        }
    }

    // ----------------------------------------------------------------------
    // 5. Camera Distinctness Check: Cam0 != Cam2 != Cam3 != Cam4 != Cam5 != Cam6 != Cam7
    // ----------------------------------------------------------------------
    var cam0_f0 = try common.extractFrameImage(allocator, &ref_render, 0, 0, 0, 1);
    defer {
        allocator.free(cam0_f0.slice);
        cam0_f0.deinit(allocator);
    }

    const test_cams = [_]usize{ 2, 3, 4, 5, 6, 7 };
    for (test_cams) |cc| {
        var cam_img = try common.extractFrameImage(allocator, &ref_render, cc, 0, 0, 1);
        defer {
            allocator.free(cam_img.slice);
            cam_img.deinit(allocator);
        }

        try assertImagesDiffer(&cam0_f0, &cam_img, 0.05);
    }

    // ----------------------------------------------------------------------
    // 6. Factorial Threading, Scheduling & Buffer Equivalence Assertions
    // ----------------------------------------------------------------------
    const test_camera = [_]CameraInput{cameras[0]};

    var ref_cam0 = try common.extractFrameImage(allocator, &ref_render, 0, 0, 0, 1);
    defer {
        allocator.free(ref_cam0.slice);
        ref_cam0.deinit(allocator);
    }

    for (scheduling_cases) |sched_case| {
        for (threading_cases) |thread_case| {
            for (buffer_mode_cases) |buf_case| {
                if (std.mem.eql(u8, thread_case.tag, ref_thread.tag) and
                    std.mem.eql(u8, buf_case.tag, ref_buf.tag) and
                    std.mem.eql(u8, sched_case.tag, ref_sched.tag))
                {
                    continue;
                }

                var actual_render = try runRender(
                    allocator,
                    io,
                    &test_camera,
                    &meshes,
                    thread_case,
                    buf_case,
                    sched_case,
                    config,
                );
                defer {
                    allocator.free(actual_render.slice);
                    actual_render.deinit(allocator);
                }

                var actual_f0 = try common.extractFrameImage(
                    allocator,
                    &actual_render,
                    0,
                    0,
                    0,
                    1,
                );
                defer {
                    allocator.free(actual_f0.slice);
                    actual_f0.deinit(allocator);
                }

                try assertImagesEqual(
                    &actual_f0,
                    &ref_cam0,
                    EQUIV_REL_TOL,
                    EQUIV_ABS_TOL,
                );
            }
        }
    }

    try runAdditionalSceneCameraThreadTests(allocator, io, &textures, config);
}

fn runAdditionalSceneCameraThreadTests(
    allocator: std.mem.Allocator,
    io: std.Io,
    textures: *const common_full.FullTextures,
    config: rastcfg.RasterConfig,
) !void {
    var prep = try common_full.prepareScene3(allocator, io);
    defer prep.deinit(allocator);

    const meshes = common_full.buildScene3Meshes(&prep, textures);
    const cameras = common_full.createScene3Cameras();
    const cam_inp = cameras[0];

    // 1. Frame batch size per group sweeps (1, 2, 4) in offline and in_order modes
    const batch_sizes = [_]u16{ 1, 2, 4 };
    for (batch_sizes) |batch_size| {
        for ([_]rastcfg.RenderMode{ .in_order, .offline }) |render_mode| {
            var arena = std.heap.ArenaAllocator.init(allocator);
            defer arena.deinit();
            const aa = arena.allocator();

            var run_config = config;
            run_config.save_strategy = .memory;
            run_config.render_mode = render_mode;
            run_config.frame_batch_size_per_group = batch_size;

            const render_groups = [_]riley.RenderGroupSpec{
                .{ .io = io, .workers = 2 },
            };

            const result = try riley.raster(
                aa,
                &render_groups,
                &[_]CameraInput{cam_inp},
                &meshes,
                run_config,
                null,
            );
            const img = result orelse return error.NoResult;
            try std.testing.expect(img.slice.len > 0);
        }
    }

    // 2. Geometry scheduling mode & jobs in flight sweep
    const geom_sched_modes = [_]rastcfg.GeometrySchedulingMode{
        .auto,
        .pack,
        .spread,
    };
    const geom_jobs_in_flight = [_]u16{ 1, 2 };

    for (geom_sched_modes) |sched_mode| {
        for (geom_jobs_in_flight) |jobs_in_flight| {
            var arena = std.heap.ArenaAllocator.init(allocator);
            defer arena.deinit();
            const aa = arena.allocator();

            var run_config = config;
            run_config.save_strategy = .memory;
            run_config.geom_scheduling_mode = sched_mode;
            run_config.max_geom_jobs_in_flight_per_group = jobs_in_flight;

            const render_groups = [_]riley.RenderGroupSpec{
                .{ .io = io, .workers = 2 },
            };

            const result = try riley.raster(
                aa,
                &render_groups,
                &[_]CameraInput{cam_inp},
                &meshes,
                run_config,
                null,
            );
            const img = result orelse return error.NoResult;
            try std.testing.expect(img.slice.len > 0);
        }
    }

    // 3. Mixed-resolution multicamera call (camera 0 and camera 1)
    {
        var cam0 = cam_inp;
        cam0.pixels_num = .{ 160, 100 };
        var cam1 = cam_inp;
        cam1.pixels_num = .{ 128, 80 };

        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const aa = arena.allocator();

        var run_config = config;
        run_config.save_strategy = .memory;

        const render_groups = [_]riley.RenderGroupSpec{
            .{ .io = io, .workers = 2 },
        };

        const result = try riley.raster(
            aa,
            &render_groups,
            &[_]CameraInput{ cam0, cam1 },
            &meshes,
            run_config,
            null,
        );
        const img = result orelse return error.NoResult;
        try std.testing.expect(img.slice.len > 0);
    }

    // 4. OpenGL vs OpenCV coordinate systems parity
    for ([_]camera.CameraCoordSys{ .opengl, .opencv }) |coord_sys| {
        var cam_test = cam_inp;
        cam_test.coord_sys = coord_sys;

        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const aa = arena.allocator();

        var run_config = config;
        run_config.save_strategy = .memory;

        const render_groups = [_]riley.RenderGroupSpec{
            .{ .io = io, .workers = 1 },
        };

        const result = try riley.raster(
            aa,
            &render_groups,
            &[_]CameraInput{cam_test},
            &meshes,
            run_config,
            null,
        );
        const img = result orelse return error.NoResult;
        try std.testing.expect(img.slice.len > 0);
    }
}



