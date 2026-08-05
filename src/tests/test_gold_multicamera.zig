// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const std = @import("std");
const Timestamp = std.Io.Clock.Timestamp;

const benchcommon = @import("../dev_support/benchcommon.zig");
const policy = @import("../dev_support/testpolicy.zig");
const orch = @import("../dev_support/orchestration.zig");
const testcommon = @import("../dev_support/tests.zig");
const tcfg = @import("../dev_support/testconfig.zig");
const CameraInput = @import("../riley/zig/camera.zig").CameraInput;
const buildconfig = @import("../riley/zig/buildconfig.zig");
const F = buildconfig.F;
const cfg = buildconfig.config;
const camera_mod = @import("../riley/zig/camera.zig");
const CameraPrepared = camera_mod.CameraPrepared;
const gk = @import("../riley/zig/geometrykernels.zig");
const iio = @import("../riley/zig/imageio.zig");
const mo = @import("../riley/zig/meshpipeline.zig");
const meshio = @import("../riley/zig/meshio.zig");
const ndarray = @import("../riley/zig/ndarray.zig");
const NDArray = @import("../riley/zig/ndarray.zig").NDArray;
const texops = @import("../riley/zig/textureops.zig");
const uvio = @import("../riley/zig/uvio.zig");
const riley = @import("../riley/zig/riley.zig");
const GeometrySchedulingMode =
    @import("../riley/zig/rasterconfig.zig").GeometrySchedulingMode;

const simd_on = cfg.simd == .on;

const duplicate_rel_tol: F = 1.0e-7;
const duplicate_abs_tol: F = 1.0e-11;
const fails_root = "fails";

const RenderCase = struct {
    case_name: []const u8,
    data_dir: []const u8,
    mesh_type: gk.MeshType,
    channels: usize,
    shader: union(enum) {
        nodal_grey,
        tex8_rgb: texops.TextureSampleConfig,
    },
};

fn expectCamerasEqual(
    array: *const NDArray(F),
    camera_a: usize,
    camera_b: usize,
    channels: usize,
    rel_tol: F,
    abs_tol: F,
) !void {
    const rows_num = array.dims[3];
    const cols_num = array.dims[4];

    for (0..channels) |cc| {
        for (0..rows_num) |rr| {
            for (0..cols_num) |pp| {
                const value_a = array.get(
                    &[_]usize{ camera_a, 0, cc, rr, pp },
                );
                const value_b = array.get(
                    &[_]usize{ camera_b, 0, cc, rr, pp },
                );
                if (!testcommon.isApproxEqual(
                    value_a,
                    value_b,
                    rel_tol,
                    abs_tol,
                )) {
                    return error.CameraOutputsDiffer;
                }
            }
        }
    }
}

fn expectCamerasDifferent(
    array: *const NDArray(F),
    camera_a: usize,
    camera_b: usize,
    channels: usize,
    rel_tol: F,
    abs_tol: F,
) !void {
    const rows_num = array.dims[3];
    const cols_num = array.dims[4];

    for (0..channels) |cc| {
        for (0..rows_num) |rr| {
            for (0..cols_num) |pp| {
                const value_a = array.get(
                    &[_]usize{ camera_a, 0, cc, rr, pp },
                );
                const value_b = array.get(
                    &[_]usize{ camera_b, 0, cc, rr, pp },
                );
                if (!testcommon.isApproxEqual(
                    value_a,
                    value_b,
                    rel_tol,
                    abs_tol,
                )) {
                    return;
                }
            }
        }
    }

    return error.CameraOutputsIdentical;
}

fn expectCameraMatchesSingleResult(
    batch_result: *const NDArray(F),
    single_result: *const NDArray(F),
    camera_idx: usize,
    frame_idx: usize,
    channels: usize,
    rel_tol: F,
    abs_tol: F,
) !void {
    const rows_num = single_result.dims[3];
    const cols_num = single_result.dims[4];

    for (0..channels) |cc| {
        for (0..rows_num) |rr| {
            for (0..cols_num) |pp| {
                const batch_val = batch_result.get(
                    &[_]usize{ camera_idx, frame_idx, cc, rr, pp },
                );
                const single_val = single_result.get(
                    &[_]usize{ 0, frame_idx, cc, rr, pp },
                );
                try std.testing.expect(
                    testcommon.isApproxEqual(
                        batch_val,
                        single_val,
                        rel_tol,
                        abs_tol,
                    ),
                );
            }
        }
    }
}

fn expectCameraPaddingZero(
    batch_result: *const NDArray(F),
    camera_idx: usize,
    frame_idx: usize,
    channels: usize,
    valid_rows_num: usize,
    valid_cols_num: usize,
) !void {
    const rows_num = batch_result.dims[3];
    const cols_num = batch_result.dims[4];

    for (0..channels) |cc| {
        for (0..rows_num) |rr| {
            for (0..cols_num) |pp| {
                if (rr < valid_rows_num and pp < valid_cols_num) {
                    continue;
                }
                try std.testing.expectEqual(
                    @as(F, 0.0),
                    batch_result.get(&[_]usize{ camera_idx, frame_idx, cc, rr, pp }),
                );
            }
        }
    }
}

test "Multicamera duplicate sphere200 cameras match each other" {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const io = std.testing.io;
    const pixel_num = [_]u32{ 800, 500 };
    const render_case = RenderCase{
        .case_name = "tri3_nodal_grey",
        .data_dir = "data/min/tri3_sphere200",
        .mesh_type = .tri3,
        .channels = 1,
        .shader = .nodal_grey,
    };

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    const coord_path = try std.fs.path.join(
        aa,
        &[_][]const u8{ render_case.data_dir, "coords.csv" },
    );
    const connect_path = try std.fs.path.join(
        aa,
        &[_][]const u8{ render_case.data_dir, "connect.csv" },
    );
    const field_path = try std.fs.path.join(
        aa,
        &[_][]const u8{ render_case.data_dir, "field.csv" },
    );

    const sim_data = try meshio.loadSimData(
        aa,
        io,
        coord_path,
        connect_path,
        null,
        null,
    );
    const field_raw = try benchcommon.loadNDArrayFromCSV(
        aa,
        io,
        field_path,
        1,
        true,
    );
    const camera = try orch.initCameraForCoords(aa, &sim_data.coords, pixel_num, 1.0);
    defer camera.deinit(aa);
    const camera_input = CameraInput{
        .pixels_num = camera.pixels_num,
        .pixels_size = camera.pixels_size,
        .pos_world = camera.pos_world,
        .rot_world = camera.rot_world,
        .roi_cent_world = camera.roi_cent_world,
        .focal_length = camera.focal_length,
        .sub_sample = camera.sub_sample,
        .distortion = camera.distortion,
    };
    const cameras = [_]CameraInput{ camera_input, camera_input };

    const mesh_input = mo.MeshInput{
        .mesh_type = render_case.mesh_type,
        .coords = sim_data.coords,
        .connect = sim_data.connect,
        .disp = null,
        .shader = .{
            .nodal = .{
                .field = .{
                    .array = field_raw,
                    .array_mem = field_raw.slice,
                },
                .scaling = .auto,
            },
        },
    };

    var config = tcfg.getRasterConfig(.testing);
    config.save_strategy = .memory;
    config.image_save_opts = &[_]iio.ImageSaveOpts{
        .{ .format = .csv, .bits = null, .scaling = .none },
    };

    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = @max(@as(u16, 1), config.total_threads) },
    };
    const result = (try riley.raster(
        aa,
        &render_groups,
        &cameras,
        &[_]mo.MeshInput{mesh_input},
        config,
        null,
    )) orelse return error.NoResult;
    defer aa.free(result.slice);

    try std.testing.expectEqual(@as(usize, 2), result.dims[0]);
    try expectCamerasEqual(
        &result,
        0,
        1,
        render_case.channels,
        duplicate_rel_tol,
        duplicate_abs_tol,
    );
}

test "Multicamera grouped render groups match reference across scheduler modes" {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const io = std.testing.io;
    const pixel_num = [_]u32{ 320, 200 };
    const data_dir = "data/min/tri3_sphere200";

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    const coord_path = try std.fs.path.join(
        aa,
        &[_][]const u8{ data_dir, "coords.csv" },
    );
    const connect_path = try std.fs.path.join(
        aa,
        &[_][]const u8{ data_dir, "connect.csv" },
    );
    const field_path = try std.fs.path.join(
        aa,
        &[_][]const u8{ data_dir, "field.csv" },
    );

    const sim_data = try meshio.loadSimData(
        aa,
        io,
        coord_path,
        connect_path,
        null,
        null,
    );
    const field_raw = try benchcommon.loadNDArrayFromCSV(
        aa,
        io,
        field_path,
        1,
        true,
    );

    const camera_a = try orch.initCameraForCoords(
        aa,
        &sim_data.coords,
        pixel_num,
        1.0,
    );
    defer camera_a.deinit(aa);
    const camera_b = try orch.initCameraForCoords(
        aa,
        &sim_data.coords,
        pixel_num,
        1.0,
    );
    defer camera_b.deinit(aa);

    const mesh_input = mo.MeshInput{
        .mesh_type = .tri3,
        .coords = sim_data.coords,
        .connect = sim_data.connect,
        .disp = null,
        .shader = .{
            .nodal = .{
                .field = .{
                    .array = field_raw,
                    .array_mem = field_raw.slice,
                },
                .scaling = .auto,
            },
        },
    };

    const camera_inputs = [_]CameraInput{
        .{
            .pixels_num = camera_a.pixels_num,
            .pixels_size = camera_a.pixels_size,
            .pos_world = camera_a.pos_world,
            .rot_world = camera_a.rot_world,
            .roi_cent_world = camera_a.roi_cent_world,
            .focal_length = camera_a.focal_length,
            .sub_sample = camera_a.sub_sample,
            .distortion = camera_a.distortion,
        },
        .{
            .pixels_num = camera_b.pixels_num,
            .pixels_size = camera_b.pixels_size,
            .pos_world = camera_b.pos_world,
            .rot_world = camera_b.rot_world,
            .roi_cent_world = camera_b.roi_cent_world,
            .focal_length = camera_b.focal_length,
            .sub_sample = camera_b.sub_sample,
            .distortion = camera_b.distortion,
        },
    };

    const compareFrameStacks = struct {
        fn run(
            reference: anytype,
            grouped: anytype,
            duplicate_rel_tol_local: F,
            duplicate_abs_tol_local: F,
        ) !void {
            try std.testing.expectEqualSlices(usize, reference.dims, grouped.dims);
            for (0..reference.dims[0]) |camera_idx| {
                for (0..reference.dims[1]) |frame_idx| {
                    for (0..reference.dims[3]) |rr| {
                        for (0..reference.dims[4]) |cc| {
                            const ref_val = reference.get(&[_]usize{
                                camera_idx,
                                frame_idx,
                                0,
                                rr,
                                cc,
                            });
                            const grouped_val = grouped.get(&[_]usize{
                                camera_idx,
                                frame_idx,
                                0,
                                rr,
                                cc,
                            });
                            try std.testing.expect(
                                testcommon.isApproxEqual(
                                    ref_val,
                                    grouped_val,
                                    duplicate_rel_tol_local,
                                    duplicate_abs_tol_local,
                                ),
                            );
                        }
                    }
                }
            }
        }
    };

    var ref_config = tcfg.getRasterConfig(.testing);
    ref_config.save_strategy = .memory;
    ref_config.report = .off;
    ref_config.total_threads = 4;
    ref_config.frame_batch_size_per_group = 2;
    ref_config.max_geom_jobs_in_flight_per_group = 2;
    ref_config.max_geom_workers_per_job = 1;
    ref_config.max_raster_workers_per_job = 4;

    const ref_render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = @max(@as(u16, 1), ref_config.total_threads) },
    };
    const reference = (try riley.raster(
        aa,
        &ref_render_groups,
        &camera_inputs,
        &[_]mo.MeshInput{mesh_input},
        ref_config,
        null,
    )) orelse return error.NoResult;
    defer aa.free(reference.slice);

    const shared_io_grouped_cases = [_]struct {
        mode: GeometrySchedulingMode,
        render_groups: [2]riley.RenderGroupSpec,
    }{
        .{
            .mode = .spread,
            .render_groups = .{
                .{ .io = io, .workers = 2 },
                .{ .io = io, .workers = 2 },
            },
        },
        .{
            .mode = .pack,
            .render_groups = .{
                .{ .io = io, .workers = 2 },
                .{ .io = io, .workers = 2 },
            },
        },
    };

    for (shared_io_grouped_cases) |case| {
        var grouped_config = ref_config;
        grouped_config.geom_scheduling_mode = case.mode;

        const grouped = (try riley.raster(
            aa,
            case.render_groups[0..],
            &camera_inputs,
            &[_]mo.MeshInput{mesh_input},
            grouped_config,
            null,
        )) orelse return error.NoResult;
        defer aa.free(grouped.slice);

        try compareFrameStacks.run(
            reference,
            grouped,
            duplicate_rel_tol,
            duplicate_abs_tol,
        );
    }

    var managed_ios = [_]std.Io.Threaded{
        std.Io.Threaded.init(allocator, .{
            .argv0 = .empty,
            .environ = .empty,
            .async_limit = .limited(1),
            .concurrent_limit = .limited(1),
        }),
        std.Io.Threaded.init(allocator, .{
            .argv0 = .empty,
            .environ = .empty,
            .async_limit = .limited(1),
            .concurrent_limit = .limited(1),
        }),
        std.Io.Threaded.init(allocator, .{
            .argv0 = .empty,
            .environ = .empty,
            .async_limit = .limited(1),
            .concurrent_limit = .limited(1),
        }),
        std.Io.Threaded.init(allocator, .{
            .argv0 = .empty,
            .environ = .empty,
            .async_limit = .limited(1),
            .concurrent_limit = .limited(1),
        }),
    };
    defer for (&managed_ios) |*managed_io| managed_io.deinit();

    const independent_io_grouped_cases = [_]struct {
        mode: GeometrySchedulingMode,
    }{
        .{ .mode = .spread },
        .{ .mode = .pack },
    };

    for (independent_io_grouped_cases) |case| {
        var grouped_config = ref_config;
        grouped_config.geom_scheduling_mode = case.mode;

        const render_groups = [_]riley.RenderGroupSpec{
            .{ .io = managed_ios[0].io(), .workers = 2 },
            .{ .io = managed_ios[1].io(), .workers = 2 },
        };
        const grouped = (try riley.raster(
            aa,
            &render_groups,
            &camera_inputs,
            &[_]mo.MeshInput{mesh_input},
            grouped_config,
            null,
        )) orelse return error.NoResult;
        defer aa.free(grouped.slice);

        try compareFrameStacks.run(
            reference,
            grouped,
            duplicate_rel_tol,
            duplicate_abs_tol,
        );
    }

    const four_group_render_groups = [_]riley.RenderGroupSpec{
        .{ .io = managed_ios[0].io(), .workers = 1 },
        .{ .io = managed_ios[1].io(), .workers = 1 },
        .{ .io = managed_ios[2].io(), .workers = 1 },
        .{ .io = managed_ios[3].io(), .workers = 1 },
    };
    const four_group = (try riley.raster(
        aa,
        &four_group_render_groups,
        &camera_inputs,
        &[_]mo.MeshInput{mesh_input},
        ref_config,
        null,
    )) orelse return error.NoResult;
    defer aa.free(four_group.slice);

    try compareFrameStacks.run(
        reference,
        four_group,
        duplicate_rel_tol,
        duplicate_abs_tol,
    );
}

fn runMulticameraSaveCase(
    data_dir: []const u8,
    pixel_num: [2]u32,
    save_strategy: riley.SaveStrategy,
    overlap: bool,
) !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const io = std.testing.io;
    const out_dir: ?[]const u8 = if (save_strategy == .disk)
        if (overlap) "temp-tests/save-overlap-on" else "temp-tests/save-overlap-off"
    else
        null;
    const cwd = std.Io.Dir.cwd();
    if (out_dir) |path| cwd.deleteTree(io, path) catch {};
    defer if (out_dir) |path| cwd.deleteTree(io, path) catch {};

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    const coord_path = try std.fs.path.join(
        aa,
        &[_][]const u8{ data_dir, "coords.csv" },
    );
    const connect_path = try std.fs.path.join(
        aa,
        &[_][]const u8{ data_dir, "connect.csv" },
    );
    const field_path = try std.fs.path.join(
        aa,
        &[_][]const u8{ data_dir, "field.csv" },
    );
    const sim_data = try meshio.loadSimData(
        aa,
        io,
        coord_path,
        connect_path,
        null,
        null,
    );
    const field_raw = try benchcommon.loadNDArrayFromCSV(
        aa,
        io,
        field_path,
        1,
        true,
    );
    const camera_a = try orch.initCameraForCoords(
        aa,
        &sim_data.coords,
        pixel_num,
        1.0,
    );
    defer camera_a.deinit(aa);
    const camera_b = try orch.initCameraForCoords(
        aa,
        &sim_data.coords,
        pixel_num,
        1.0,
    );
    defer camera_b.deinit(aa);
    const camera_inputs = [_]CameraInput{
        .{
            .pixels_num = camera_a.pixels_num,
            .pixels_size = camera_a.pixels_size,
            .pos_world = camera_a.pos_world,
            .rot_world = camera_a.rot_world,
            .roi_cent_world = camera_a.roi_cent_world,
            .focal_length = camera_a.focal_length,
            .sub_sample = camera_a.sub_sample,
            .distortion = camera_a.distortion,
        },
        .{
            .pixels_num = camera_b.pixels_num,
            .pixels_size = camera_b.pixels_size,
            .pos_world = camera_b.pos_world,
            .rot_world = camera_b.rot_world,
            .roi_cent_world = camera_b.roi_cent_world,
            .focal_length = camera_b.focal_length,
            .sub_sample = camera_b.sub_sample,
            .distortion = camera_b.distortion,
        },
    };
    const mesh_input = mo.MeshInput{
        .mesh_type = .tri3,
        .coords = sim_data.coords,
        .connect = sim_data.connect,
        .disp = null,
        .shader = .{
            .nodal = .{
                .field = .{
                    .array = field_raw,
                    .array_mem = field_raw.slice,
                },
                .scaling = .auto,
            },
        },
    };

    var reference_config = tcfg.getRasterConfig(.testing);
    reference_config.save_strategy = .memory;
    reference_config.report = .off;
    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = @max(@as(u16, 1), reference_config.total_threads) },
    };
    const reference = (try riley.raster(
        aa,
        &render_groups,
        &camera_inputs,
        &[_]mo.MeshInput{mesh_input},
        reference_config,
        null,
    )) orelse return error.NoResult;
    defer aa.free(reference.slice);

    var target_config = reference_config;
    target_config.save_strategy = save_strategy;
    target_config.disk_save_overlap = overlap;
    if (save_strategy == .disk) {
        target_config.image_save_opts = &[_]iio.ImageSaveOpts{
            .{
                .format = .csv,
                .bits = null,
                .scaling = .none,
                .channels = 1,
            },
        };
    }
    const result = try riley.raster(
        aa,
        &render_groups,
        &camera_inputs,
        &[_]mo.MeshInput{mesh_input},
        target_config,
        out_dir,
    );
    defer if (result) |image| aa.free(image.slice);

    switch (save_strategy) {
        .both => {
            const both = result orelse return error.NoResult;
            try std.testing.expect(ndarray.matchArrayDims(F, &reference, &both));
            for (0..reference.slice.len) |ii| {
                try std.testing.expectApproxEqAbs(
                    reference.slice[ii],
                    both.slice[ii],
                    duplicate_abs_tol,
                );
            }
        },
        .disk => {
            try std.testing.expect(result == null);
            for (0..camera_inputs.len) |camera_idx| {
                const saved_path = try std.fmt.allocPrint(
                    aa,
                    "{s}/cam{d}_frame0_field0.csv",
                    .{ out_dir.?, camera_idx },
                );
                try testcommon.compareNDArrayToGold(
                    aa,
                    io,
                    &reference,
                    camera_idx,
                    0,
                    0,
                    1,
                    saved_path,
                    duplicate_rel_tol,
                    duplicate_abs_tol,
                );
            }
        },
        else => unreachable,
    }
}

test "Multicamera memory matches both" {
    try runMulticameraSaveCase(
        "data/bench/tri3_sphere200",
        .{ 320, 200 },
        .both,
        false,
    );
}

test "disk save without overlap" {
    try runMulticameraSaveCase(
        "data/min/tri3_sphere200",
        .{ 160, 100 },
        .disk,
        false,
    );
}

test "disk save with overlap" {
    try runMulticameraSaveCase(
        "data/min/tri3_sphere200",
        .{ 160, 100 },
        .disk,
        true,
    );
}

test "Sphere200 multicamera gold tests" {
    if (!simd_on) {
        std.debug.print(
            "Skipping scalar multicamera sphere gold tests.\n",
            .{},
        );
        return;
    }

    std.debug.print("Running Multicamera Gold Tests with .simd = .{s}...\n", .{
        if (simd_on) "on" else "off",
    });
    const suite_start = Timestamp.now(std.testing.io, .awake);

    var gpa: std.heap.DebugAllocator(.{}) = .init;
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const io = std.testing.io;
    const gold_root = policy.goldRoot(.sphere200multicam);
    const pixel_num = [_]u32{ 800, 500 };

    const texture_rgb = try iio.loadImage(
        u8,
        3,
        allocator,
        io,
        "texture/speckle_rgb.bmp",
        .bmp,
    );
    defer texture_rgb.deinit(allocator);

    const render_cases = [_]RenderCase{
        .{
            .case_name = "tri3_nodal_grey",
            .data_dir = "data/min/tri3_sphere200",
            .mesh_type = .tri3,
            .channels = 1,
            .shader = .nodal_grey,
        },
        .{
            .case_name = "tri6_tex8_rgb_cubic_catmull_rom_lut_lerp",
            .data_dir = "data/min/tri6_sphere200",
            .mesh_type = .tri6,
            .channels = 3,
            .shader = .{
                .tex8_rgb = .{
                    .sample = .cubic_catmull_rom,
                    .mode = .lut_lerp,
                },
            },
        },
    };

    for (render_cases) |render_case| {
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const aa = arena.allocator();

        const coord_path = try std.fs.path.join(
            aa,
            &[_][]const u8{ render_case.data_dir, "coords.csv" },
        );
        const connect_path = try std.fs.path.join(
            aa,
            &[_][]const u8{ render_case.data_dir, "connect.csv" },
        );
        const field_path = try std.fs.path.join(
            aa,
            &[_][]const u8{ render_case.data_dir, "field.csv" },
        );
        const sim_data = try meshio.loadSimData(
            aa,
            io,
            coord_path,
            connect_path,
            null,
            null,
        );
        const field_raw = try benchcommon.loadNDArrayFromCSV(
            aa,
            io,
            field_path,
            if (render_case.channels == 3) 3 else 1,
            true,
        );
        const cameras = try orch.initStereoCamerasForCoords(
            aa,
            &sim_data.coords,
            pixel_num,
            1.0,
            10.0,
        );
        defer for (cameras) |cam| cam.deinit(aa);
        const camera_inputs = [_]CameraInput{
            CameraInput{
                .pixels_num = cameras[0].pixels_num,
                .pixels_size = cameras[0].pixels_size,
                .pos_world = cameras[0].pos_world,
                .rot_world = cameras[0].rot_world,
                .roi_cent_world = cameras[0].roi_cent_world,
                .focal_length = cameras[0].focal_length,
                .sub_sample = cameras[0].sub_sample,
                .distortion = cameras[0].distortion,
            },
            CameraInput{
                .pixels_num = cameras[1].pixels_num,
                .pixels_size = cameras[1].pixels_size,
                .pos_world = cameras[1].pos_world,
                .rot_world = cameras[1].rot_world,
                .roi_cent_world = cameras[1].roi_cent_world,
                .focal_length = cameras[1].focal_length,
                .sub_sample = cameras[1].sub_sample,
                .distortion = cameras[1].distortion,
            },
        };

        const mesh_input = switch (render_case.shader) {
            .nodal_grey => mo.MeshInput{
                .mesh_type = render_case.mesh_type,
                .coords = sim_data.coords,
                .connect = sim_data.connect,
                .disp = null,
                .shader = .{
                    .nodal = .{
                        .field = .{
                            .array = field_raw,
                            .array_mem = field_raw.slice,
                        },
                        .scaling = .auto,
                    },
                },
            },
            .tex8_rgb => |samp_cfg| blk: {
                const uv_path = try std.fmt.allocPrint(
                    aa,
                    "{s}/uvs.csv",
                    .{render_case.data_dir},
                );
                const uv_map = try uvio.loadUVMap(
                    aa,
                    io,
                    uv_path,
                );
                break :blk mo.MeshInput{
                    .mesh_type = render_case.mesh_type,
                    .coords = sim_data.coords,
                    .connect = sim_data.connect,
                    .disp = null,
                    .shader = .{
                        .tex_rgb_u8 = .{
                            .uvs = uv_map.array,
                            .tex = texture_rgb,
                            .samp_cfg = samp_cfg,
                        },
                    },
                };
            },
        };

        var config = tcfg.getRasterConfig(.testing);
        config.save_strategy = .memory;
        config.image_save_opts = &[_]iio.ImageSaveOpts{
            .{
                .format = .csv,
                .bits = null,
                .scaling = .none,
                .channels = render_case.channels,
            },
        };

        if (tcfg.TEST_CASE_VERBOSE) {
            std.debug.print("Testing {s} ... ", .{render_case.case_name});
        }
        const time_start = Timestamp.now(io, .awake);
        const render_groups = [_]riley.RenderGroupSpec{
            .{ .io = io, .workers = @max(@as(u16, 1), config.total_threads) },
        };
        const result = (try riley.raster(
            aa,
            &render_groups,
            &camera_inputs,
            &[_]mo.MeshInput{mesh_input},
            config,
            null,
        )) orelse return error.NoResult;
        defer aa.free(result.slice);
        const time_end = Timestamp.now(io, .awake);
        const duration_ms = @as(F, @floatFromInt(time_start.durationTo(time_end).raw.nanoseconds)) / 1e6;

        try std.testing.expectEqual(@as(usize, 2), result.dims[0]);
        try expectCamerasDifferent(
            &result,
            0,
            1,
            render_case.channels,
            duplicate_rel_tol,
            duplicate_abs_tol,
        );

        const gold_dir = try std.fs.path.join(
            aa,
            &[_][]const u8{ gold_root, render_case.case_name },
        );

        for (0..2) |camera_idx| {
            const gold_path = try testcommon.findGoldPath(
                aa,
                io,
                gold_dir,
                camera_idx,
                0,
                0,
                render_case.channels == 3,
            );

            testcommon.compareNDArrayToGold(
                aa,
                io,
                &result,
                camera_idx,
                0,
                0,
                render_case.channels,
                gold_path,
                duplicate_rel_tol,
                duplicate_abs_tol,
            ) catch |err| {
                if (err == error.PixelMismatch) {
                    if (tcfg.TEST_CASE_VERBOSE) {
                        std.debug.print("MISMATCH! ({d:.2} ms)\n", .{duration_ms});
                    }
                } else {
                    if (tcfg.TEST_CASE_VERBOSE) {
                        std.debug.print("ERROR! ({d:.2} ms)\n", .{duration_ms});
                    }
                }
                const fail_dir_name = try std.fmt.allocPrint(
                    aa,
                    "multicam_{s}_cam{d}",
                    .{ render_case.case_name, camera_idx },
                );
                try testcommon.saveComparisonArtifactsFromResult(
                    aa,
                    io,
                    fails_root,
                    fail_dir_name,
                    &result,
                    camera_idx,
                    0,
                    0,
                    gold_path,
                    render_case.channels,
                );
                return err;
            };
        }
        if (tcfg.TEST_CASE_VERBOSE) {
            std.debug.print("MATCHED ({d:.2} ms)\n", .{duration_ms});
        }
    }

    const suite_end = Timestamp.now(io, .awake);
    const suite_ms = @as(
        F,
        @floatFromInt(suite_start.durationTo(suite_end).raw.nanoseconds),
    ) / 1e6;
    std.debug.print(
        "Multicamera Gold Test Suite took {d:.3} ms\n",
        .{suite_ms},
    );
}

test "Multicamera mixed sensor sizes return padded batch and save actual size" {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const io = std.testing.io;
    const temp_tests_root = "temp-tests";
    defer {
        const cwd = std.Io.Dir.cwd();
        cwd.deleteTree(io, temp_tests_root) catch {};
    }

    const pixel_num_small = [_]u32{ 320, 200 };
    const pixel_num_large = [_]u32{ 480, 300 };
    const data_dir = "data/min/tri3_sphere200";

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    const coord_path = try std.fs.path.join(
        aa,
        &[_][]const u8{ data_dir, "coords.csv" },
    );
    const connect_path = try std.fs.path.join(
        aa,
        &[_][]const u8{ data_dir, "connect.csv" },
    );
    const field_path = try std.fs.path.join(
        aa,
        &[_][]const u8{ data_dir, "field.csv" },
    );

    const sim_data = try meshio.loadSimData(
        aa,
        io,
        coord_path,
        connect_path,
        null,
        null,
    );
    const field_raw = try benchcommon.loadNDArrayFromCSV(
        aa,
        io,
        field_path,
        1,
        true,
    );

    const small_camera = try orch.initCameraForCoords(
        aa,
        &sim_data.coords,
        pixel_num_small,
        1.0,
    );
    defer small_camera.deinit(aa);
    const large_camera = try orch.initCameraForCoords(
        aa,
        &sim_data.coords,
        pixel_num_large,
        1.0,
    );
    defer large_camera.deinit(aa);

    const mesh_input = mo.MeshInput{
        .mesh_type = .tri3,
        .coords = sim_data.coords,
        .connect = sim_data.connect,
        .disp = null,
        .shader = .{
            .nodal = .{
                .field = .{
                    .array = field_raw,
                    .array_mem = field_raw.slice,
                },
                .scaling = .auto,
            },
        },
    };

    var memory_config = tcfg.getRasterConfig(.testing);
    memory_config.save_strategy = .memory;
    memory_config.image_save_opts = &[_]iio.ImageSaveOpts{
        .{ .format = .csv, .bits = null, .scaling = .none, .channels = 1 },
    };

    const small_render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = @max(@as(u16, 1), memory_config.total_threads) },
    };
    const small_single = (try riley.raster(
        aa,
        &small_render_groups,
        &[_]CameraInput{CameraInput{
            .pixels_num = small_camera.pixels_num,
            .pixels_size = small_camera.pixels_size,
            .pos_world = small_camera.pos_world,
            .rot_world = small_camera.rot_world,
            .roi_cent_world = small_camera.roi_cent_world,
            .focal_length = small_camera.focal_length,
            .sub_sample = small_camera.sub_sample,
            .distortion = small_camera.distortion,
        }},
        &[_]mo.MeshInput{mesh_input},
        memory_config,
        null,
    )) orelse return error.NoResult;
    defer aa.free(small_single.slice);

    const large_render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = @max(@as(u16, 1), memory_config.total_threads) },
    };
    const large_single = (try riley.raster(
        aa,
        &large_render_groups,
        &[_]CameraInput{CameraInput{
            .pixels_num = large_camera.pixels_num,
            .pixels_size = large_camera.pixels_size,
            .pos_world = large_camera.pos_world,
            .rot_world = large_camera.rot_world,
            .roi_cent_world = large_camera.roi_cent_world,
            .focal_length = large_camera.focal_length,
            .sub_sample = large_camera.sub_sample,
            .distortion = large_camera.distortion,
        }},
        &[_]mo.MeshInput{mesh_input},
        memory_config,
        null,
    )) orelse return error.NoResult;
    defer aa.free(large_single.slice);

    const out_dir = temp_tests_root ++ "/multicamera-mixed-sizes";
    var batch_config = tcfg.getRasterConfig(.testing);
    batch_config.save_strategy = .both;
    batch_config.image_save_opts = &[_]iio.ImageSaveOpts{
        .{ .format = .csv, .bits = null, .scaling = .none, .channels = 1 },
    };
    const batch_render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = @max(@as(u16, 1), batch_config.total_threads) },
    };
    const batch_result = (try riley.raster(
        aa,
        &batch_render_groups,
        &[_]CameraInput{
            CameraInput{
                .pixels_num = small_camera.pixels_num,
                .pixels_size = small_camera.pixels_size,
                .pos_world = small_camera.pos_world,
                .rot_world = small_camera.rot_world,
                .roi_cent_world = small_camera.roi_cent_world,
                .focal_length = small_camera.focal_length,
                .sub_sample = small_camera.sub_sample,
                .distortion = small_camera.distortion,
            },
            CameraInput{
                .pixels_num = large_camera.pixels_num,
                .pixels_size = large_camera.pixels_size,
                .pos_world = large_camera.pos_world,
                .rot_world = large_camera.rot_world,
                .roi_cent_world = large_camera.roi_cent_world,
                .focal_length = large_camera.focal_length,
                .sub_sample = large_camera.sub_sample,
                .distortion = large_camera.distortion,
            },
        },
        &[_]mo.MeshInput{mesh_input},
        batch_config,
        out_dir,
    )) orelse return error.NoResult;
    defer aa.free(batch_result.slice);

    try std.testing.expectEqual(@as(usize, 2), batch_result.dims[0]);
    try std.testing.expectEqual(@as(usize, 1), batch_result.dims[1]);
    try std.testing.expectEqual(@as(usize, 1), batch_result.dims[2]);
    try std.testing.expectEqual(
        @as(usize, pixel_num_large[1]),
        batch_result.dims[3],
    );
    try std.testing.expectEqual(
        @as(usize, pixel_num_large[0]),
        batch_result.dims[4],
    );

    try expectCameraMatchesSingleResult(
        &batch_result,
        &small_single,
        0,
        0,
        1,
        duplicate_rel_tol,
        duplicate_abs_tol,
    );
    try expectCameraMatchesSingleResult(
        &batch_result,
        &large_single,
        1,
        0,
        1,
        duplicate_rel_tol,
        duplicate_abs_tol,
    );
    try expectCameraPaddingZero(
        &batch_result,
        0,
        0,
        1,
        pixel_num_small[1],
        pixel_num_small[0],
    );

    const small_csv = try std.fmt.allocPrint(
        aa,
        "{s}/cam0_frame0_field0.csv",
        .{out_dir},
    );
    const large_csv = try std.fmt.allocPrint(
        aa,
        "{s}/cam1_frame0_field0.csv",
        .{out_dir},
    );
    const small_image = try iio.loadImage(
        F,
        1,
        aa,
        io,
        small_csv,
        .csv,
    );
    const large_image = try iio.loadImage(
        F,
        1,
        aa,
        io,
        large_csv,
        .csv,
    );
    defer small_image.deinit(aa);
    defer large_image.deinit(aa);

    try std.testing.expectEqual(
        @as(usize, pixel_num_small[1]),
        small_image.rows_num,
    );
    try std.testing.expectEqual(
        @as(usize, pixel_num_small[0]),
        small_image.cols_num,
    );
    try std.testing.expectEqual(
        @as(usize, pixel_num_large[1]),
        large_image.rows_num,
    );
    try std.testing.expectEqual(
        @as(usize, pixel_num_large[0]),
        large_image.cols_num,
    );
}
