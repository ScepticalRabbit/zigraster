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

const orch = @import("orchestration.zig");
const NDArray = @import("../riley/zig/ndarray.zig").NDArray;
const MatSlice = @import("../riley/zig/matslice.zig").MatSlice;

const meshio = @import("../riley/zig/meshio.zig");
const CameraPrepared = @import("../riley/zig/camera.zig").CameraPrepared;
const CameraInput = @import("../riley/zig/camera.zig").CameraInput;

const gk = @import("../riley/zig/geometrykernels.zig");
const mo = @import("../riley/zig/meshpipeline.zig");
const MeshType = gk.MeshType;
const MeshInput = mo.MeshInput;

const riley = @import("../riley/zig/riley.zig");
const iio = @import("../riley/zig/imageio.zig");
const texops = @import("../riley/zig/textureops.zig");
const buildconfig = @import("../riley/zig/buildconfig.zig");
const F = buildconfig.F;
const rastcfg = @import("../riley/zig/rasterconfig.zig");
const cfg = buildconfig.config;
const csvio = @import("../riley/zig/csvio.zig");
const policy = @import("testpolicy.zig");
const tcfg = @import("testconfig.zig");

pub const default_fails_root = "fails";
pub const impl_suffix = if (cfg.simd == .on) "_simd" else "_scalar";

// Default tolerances: for scientific accuracy and DIC
// F: rel= 1e-11, abs= 1e-11
// f32: rel= 1e-5, abs= 1e-4
pub fn isApproxEqual(v1: F, v2: F, rel_tol: F, abs_tol: F) bool {
    if (v1 == v2) return true;

    const diff = @abs(v1 - v2);

    if (diff <= abs_tol) return true;

    const abs_v1 = @abs(v1);
    const abs_v2 = @abs(v2);
    const largest = if (abs_v1 > abs_v2) abs_v1 else abs_v2;

    return (diff / largest) <= rel_tol;
}

fn getGoldValue(
    gold: *const NDArray(F),
    path_is_fimg: bool,
    channels: usize,
    ch: usize,
    row: usize,
    col: usize,
) F {
    if (path_is_fimg) {
        return gold.get(&[_]usize{ ch, row, col });
    }
    if (channels == 1) {
        return gold.get(&[_]usize{ row, col });
    }
    return gold.get(&[_]usize{ row, col, ch });
}

fn getActualValue(
    array: *const NDArray(F),
    camera_idx: usize,
    frame: usize,
    field_start: usize,
    ch: usize,
    row: usize,
    col: usize,
) F {
    return switch (array.dims.len) {
        5 => array.get(&[_]usize{
            camera_idx,
            frame,
            field_start + ch,
            row,
            col,
        }),
        4 => array.get(&[_]usize{ frame, field_start + ch, row, col }),
        else => array.get(&[_]usize{ field_start + ch, row, col }),
    };
}

pub fn compareNDArrayToGold(
    allocator: std.mem.Allocator,
    io: std.Io,
    array: *const NDArray(F),
    camera_idx: usize,
    frame: usize,
    field_start: usize,
    channels: usize,
    path: []const u8,
    rel_tol: F,
    abs_tol: F,
) !void {
    const path_is_fimg = std.mem.endsWith(u8, path, ".fimg");
    var gold = if (std.mem.endsWith(u8, path, ".fimg")) blk: {
        const gold_array = try iio.loadFIMG(allocator, io, path);
        break :blk gold_array;
    } else if (channels == 1)
        try csvio.loadScalarCsv2D(allocator, io, path)
    else
        try csvio.loadPackedCsv2D(allocator, io, path, channels);

    defer {
        allocator.free(gold.slice);
        gold.deinit(allocator);
    }

    // Handle different dimension layouts:
    // .fimg: [chans, rows, cols]
    // .csv (scalar): [rows, cols]
    // .csv (packed): [rows, cols, chans]
    const gold_rows = if (gold.dims.len == 3 and path_is_fimg)
        gold.dims[1]
    else
        gold.dims[0];
    const gold_cols = if (gold.dims.len == 3 and path_is_fimg)
        gold.dims[2]
    else
        gold.dims[1];

    const rows = switch (array.dims.len) {
        5 => array.dims[3],
        4 => array.dims[2],
        else => array.dims[1],
    };
    const cols = switch (array.dims.len) {
        5 => array.dims[4],
        4 => array.dims[3],
        else => array.dims[2],
    };

    if (gold_rows != rows) {
        std.debug.print(
            "Row count mismatch: Gold has {d}, array expects {d} (path: {s})\n",
            .{ gold_rows, rows, path },
        );
        return error.GoldRowsMismatch;
    }

    if (gold_cols != cols) return error.GoldColsMismatch;

    for (0..rows) |r| {
        for (0..cols) |c| {
            for (0..channels) |ch| {
                const gold_val = getGoldValue(
                    &gold,
                    path_is_fimg,
                    channels,
                    ch,
                    r,
                    c,
                );
                const actual_val = getActualValue(
                    array,
                    camera_idx,
                    frame,
                    field_start,
                    ch,
                    r,
                    c,
                );

                if (!isApproxEqual(gold_val, actual_val, rel_tol, abs_tol)) {
                    const abs_gold = @abs(gold_val);
                    const abs_act = @abs(actual_val);
                    const largest = if (abs_gold > abs_act) abs_gold else abs_act;

                    const diff = @abs(gold_val - actual_val);
                    const rel_diff = if (largest < abs_tol) diff else diff / largest;

                    std.debug.print(
                        "\n\nMismatch at:\n frame {d},\n field {d},\n " ++
                            "pixel ({d}, {d}): " ++
                            "\n gold={d},\n actual={d},\n rel_diff={e}\n (path: {s})\n\n",
                        .{
                            frame,
                            field_start + ch,
                            r,
                            c,
                            gold_val,
                            actual_val,
                            rel_diff,
                            path,
                        },
                    );
                    return error.PixelMismatch;
                }
            }
        }
    }
}

fn openFailsSubDir(
    io: std.Io,
    fails_root: []const u8,
    dir_name: []const u8,
) !std.Io.Dir {
    const cwd = std.Io.Dir.cwd();
    cwd.createDir(io, fails_root, .default_dir) catch |err| {
        if (err != error.PathAlreadyExists) return err;
    };
    var fails_dir = try cwd.openDir(io, fails_root, .{});
    defer fails_dir.close(io);
    fails_dir.createDir(io, dir_name, .default_dir) catch |err| {
        if (err != error.PathAlreadyExists) return err;
    };
    return try fails_dir.openDir(io, dir_name, .{});
}

fn saveResultToFails(
    allocator: std.mem.Allocator,
    io: std.Io,
    fails_root: []const u8,
    array: *const NDArray(F),
    dir_name: []const u8,
) !void {
    _ = allocator;
    var out_dir = try openFailsSubDir(io, fails_root, dir_name);
    defer out_dir.close(io);

    const cameras_num = if (array.dims.len == 5) array.dims[0] else 1;
    const frames_num = if (array.dims.len == 5) array.dims[1] else array.dims[0];
    const fields_num = if (array.dims.len == 5) array.dims[2] else array.dims[1];
    const rows = if (array.dims.len == 5) array.dims[3] else array.dims[2];
    const cols = if (array.dims.len == 5) array.dims[4] else array.dims[3];

    for (0..cameras_num) |camera_idx| {
        for (0..frames_num) |frame_idx| {
            for (0..fields_num) |field_idx| {
                const slice = if (array.dims.len == 5)
                    array.getSlice(&[_]usize{
                        camera_idx,
                        frame_idx,
                        field_idx,
                        0,
                        0,
                    }, 2)
                else
                    array.getSlice(&[_]usize{ frame_idx, field_idx, 0, 0 }, 1);
                const mat = MatSlice(F).init(slice, rows, cols);
                var name_buff: [128]u8 = undefined;
                const name = try iio.formatFrameFieldBaseName(
                    name_buff[0..],
                    camera_idx,
                    frame_idx,
                    field_idx,
                    1,
                );
                try iio.saveMatAsImage(
                    io,
                    out_dir,
                    name,
                    &mat,
                    .{ .format = .csv, .bits = null, .scaling = .none },
                );
                try iio.saveMatAsImage(
                    io,
                    out_dir,
                    name,
                    &mat,
                    .{ .format = .bmp, .bits = 8, .scaling = .auto },
                );
            }
        }
    }
}

fn extractFrameImage(
    allocator: std.mem.Allocator,
    array: *const NDArray(F),
    camera_idx: usize,
    frame: usize,
    field_start: usize,
    channels: usize,
) !NDArray(F) {
    const dims = array.dims;
    const rows = switch (dims.len) {
        5 => dims[3],
        4 => dims[2],
        else => dims[1],
    };
    const cols = switch (dims.len) {
        5 => dims[4],
        4 => dims[3],
        else => dims[2],
    };
    var image = try NDArray(F).initFlat(allocator, &[_]usize{ rows, cols, channels });

    for (0..rows) |rr| {
        for (0..cols) |cc| {
            for (0..channels) |ch| {
                const val = switch (dims.len) {
                    5 => array.get(&[_]usize{
                        camera_idx,
                        frame,
                        field_start + ch,
                        rr,
                        cc,
                    }),
                    4 => array.get(&[_]usize{ frame, field_start + ch, rr, cc }),
                    else => array.get(&[_]usize{ field_start + ch, rr, cc }),
                };
                image.set(&[_]usize{ rr, cc, ch }, val);
            }
        }
    }

    return image;
}

pub fn findGoldPath(
    allocator: std.mem.Allocator,
    io: std.Io,
    dir: []const u8,
    camera_idx: usize,
    frame: usize,
    field: usize,
    is_rgb: bool,
) ![]const u8 {
    const cwd = std.Io.Dir.cwd();

    const base_names = if (is_rgb)
        [_][]const u8{
            try std.fmt.allocPrint(
                allocator,
                "{s}/cam{d}_frame{d}_field{d}_rgb",
                .{ dir, camera_idx, frame, field },
            ),
            try std.fmt.allocPrint(
                allocator,
                "{s}/cam{d}_frame{d}_field{d}",
                .{ dir, camera_idx, frame, field },
            ),
            try std.fmt.allocPrint(
                allocator,
                "{s}/frame_{d}_field_{d}_rgb",
                .{ dir, frame, field },
            ),
            try std.fmt.allocPrint(
                allocator,
                "{s}/frame_{d}_field_{d}",
                .{ dir, frame, field },
            ),
        }
    else
        [_][]const u8{
            try std.fmt.allocPrint(
                allocator,
                "{s}/cam{d}_frame{d}_field{d}",
                .{ dir, camera_idx, frame, field },
            ),
            try std.fmt.allocPrint(
                allocator,
                "{s}/cam{d}_frame{d}_field{d}_rgb",
                .{ dir, camera_idx, frame, field },
            ),
            try std.fmt.allocPrint(
                allocator,
                "{s}/frame_{d}_field_{d}",
                .{ dir, frame, field },
            ),
            try std.fmt.allocPrint(
                allocator,
                "{s}/frame_{d}_field_{d}_rgb",
                .{ dir, frame, field },
            ),
        };
    defer for (base_names) |base_name| allocator.free(base_name);

    for (base_names) |base_name| {
        const fimg_path = try std.fmt.allocPrint(allocator, "{s}.fimg", .{base_name});
        errdefer allocator.free(fimg_path);
        if (cwd.access(io, fimg_path, .{})) |_| {
            return fimg_path;
        } else |_| {
            allocator.free(fimg_path);
        }

        const csv_path = try std.fmt.allocPrint(allocator, "{s}.csv", .{base_name});
        errdefer allocator.free(csv_path);
        if (cwd.access(io, csv_path, .{})) |_| {
            return csv_path;
        } else |_| {
            allocator.free(csv_path);
        }
    }

    return error.FileNotFound;
}

fn loadNDArrayFromGold(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    channels: usize,
) !NDArray(F) {
    if (std.mem.endsWith(u8, path, ".fimg")) {
        var array = try iio.loadFIMG(allocator, io, path);
        errdefer {
            allocator.free(array.slice);
            array.deinit(allocator);
        }

        if (array.dims[0] != channels) {
            return error.ChannelMismatch;
        }

        const rows = array.dims[1];
        const cols = array.dims[2];
        var image_packed = try NDArray(F).initFlat(
            allocator,
            &[_]usize{ rows, cols, channels },
        );

        for (0..rows) |rr| {
            for (0..cols) |cc| {
                for (0..channels) |ch| {
                    image_packed.set(
                        &[_]usize{ rr, cc, ch },
                        array.get(&[_]usize{ ch, rr, cc }),
                    );
                }
            }
        }

        allocator.free(array.slice);
        array.deinit(allocator);
        return image_packed;
    }
    return csvio.loadPackedCsv2D(allocator, io, path, channels);
}

pub fn calculateDiffImage(
    allocator: std.mem.Allocator,
    actual: *const NDArray(F),
    gold: *const NDArray(F),
) !NDArray(F) {
    var diff = try NDArray(F).initFlat(allocator, actual.dims);
    for (0..actual.slice.len) |ii| {
        diff.slice[ii] = @abs(actual.slice[ii] - gold.slice[ii]);
    }
    return diff;
}

fn saveImageCSV(
    io: std.Io,
    dir: std.Io.Dir,
    path: []const u8,
    image: *const NDArray(F),
) !void {
    const rows = image.dims[0];
    const cols = image.dims[1];
    const channels = image.dims[2];

    const SaveCtx = struct {
        fn getVal(
            ctx: *const NDArray(F),
            row: usize,
            col: usize,
            ch: usize,
        ) F {
            return ctx.get(&[_]usize{ row, col, ch });
        }
    };

    try csvio.savePackedGridCSV(
        io,
        dir,
        path,
        rows,
        cols,
        channels,
        image,
        SaveCtx.getVal,
    );
}

fn makeBMPImageArray(
    allocator: std.mem.Allocator,
    image: *const NDArray(F),
) !NDArray(F) {
    const rows = image.dims[0];
    const cols = image.dims[1];
    const channels = image.dims[2];
    var bmp_image = try NDArray(F).initFlat(
        allocator,
        &[_]usize{ channels, rows, cols },
    );

    for (0..rows) |rr| {
        for (0..cols) |cc| {
            for (0..channels) |ch| {
                bmp_image.set(
                    &[_]usize{ ch, rr, cc },
                    image.get(&[_]usize{ rr, cc, ch }),
                );
            }
        }
    }

    return bmp_image;
}

fn saveImageArtifacts(
    allocator: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    base_name: []const u8,
    image: *const NDArray(F),
) !void {
    const csv_name = try std.fmt.allocPrint(allocator, "{s}.csv", .{base_name});
    defer allocator.free(csv_name);
    try saveImageCSV(io, dir, csv_name, image);

    const bmp_name = try std.fmt.allocPrint(allocator, "{s}.bmp", .{base_name});
    defer allocator.free(bmp_name);
    var bmp_image = try makeBMPImageArray(allocator, image);
    defer {
        allocator.free(bmp_image.slice);
        bmp_image.deinit(allocator);
    }
    try iio.saveBMP(io, dir, bmp_name, &bmp_image, 0, .{
        .format = .bmp,
        .bits = 8,
        .scaling = .auto,
        .channels = bmp_image.dims[0],
    });
}

pub fn saveComparisonArtifactsFromResult(
    allocator: std.mem.Allocator,
    io: std.Io,
    fails_root: []const u8,
    dir_name: []const u8,
    result: *const NDArray(F),
    camera_idx: usize,
    frame: usize,
    field_start: usize,
    gold_csv_path: []const u8,
    channels: usize,
) !void {
    const prec_str = if (F == f32) "f32" else "f64";
    const simd_str = if (buildconfig.config.simd == .on) "simdon" else "simdoff";
    const prepended_dir = try std.fmt.allocPrint(
        allocator,
        "{s}_{s}_{s}",
        .{ prec_str, simd_str, dir_name },
    );
    defer allocator.free(prepended_dir);

    var out_dir = try openFailsSubDir(io, fails_root, prepended_dir);
    defer out_dir.close(io);

    var actual = try extractFrameImage(
        allocator,
        result,
        camera_idx,
        frame,
        field_start,
        channels,
    );
    defer {
        allocator.free(actual.slice);
        actual.deinit(allocator);
    }

    var gold = try loadNDArrayFromGold(allocator, io, gold_csv_path, channels);
    defer {
        allocator.free(gold.slice);
        gold.deinit(allocator);
    }

    var diff = try calculateDiffImage(allocator, &actual, &gold);
    defer {
        allocator.free(diff.slice);
        diff.deinit(allocator);
    }

    const base_name = try std.fmt.allocPrint(
        allocator,
        "cam{d}_frame{d}_field{d}",
        .{ camera_idx, frame, field_start },
    );
    defer allocator.free(base_name);

    try saveImageArtifacts(allocator, io, out_dir, base_name, &actual);

    const diff_name = try std.fmt.allocPrint(allocator, "{s}_diff", .{base_name});
    defer allocator.free(diff_name);
    try saveImageArtifacts(allocator, io, out_dir, diff_name, &diff);
}

pub fn saveComparisonArtifactsFromImages(
    allocator: std.mem.Allocator,
    io: std.Io,
    fails_root: []const u8,
    dir_name: []const u8,
    actual: *const NDArray(F),
    gold: *const NDArray(F),
) !void {
    const prec_str = if (F == f32) "f32" else "f64";
    const simd_str = if (buildconfig.config.simd == .on) "simdon" else "simdoff";
    const prepended_dir = try std.fmt.allocPrint(
        allocator,
        "{s}_{s}_{s}",
        .{ prec_str, simd_str, dir_name },
    );
    defer allocator.free(prepended_dir);

    var out_dir = try openFailsSubDir(io, fails_root, prepended_dir);
    defer out_dir.close(io);

    var diff = try calculateDiffImage(allocator, actual, gold);
    defer {
        allocator.free(diff.slice);
        diff.deinit(allocator);
    }

    try saveImageArtifacts(allocator, io, out_dir, "cam0_frame0_field0", actual);
    try saveImageArtifacts(allocator, io, out_dir, "cam0_frame0_field0_diff", &diff);
}

pub const ShaderFilter = enum { nodal, tex, both };

pub const RunMode = enum {
    generate,
    test_gold,
};

pub fn runSingleMeshSuiteDriver(
    mode: RunMode,
    outer_alloc: std.mem.Allocator,
    io: std.Io,
    test_type: []const u8,
    mesh_type: gk.MeshType,
    fov_scale: F,
    texture: texops.Tex(u8, 1),
    pixel_num: [2]u32,
    samp_cfgs: []const texops.TextureSampleConfig,
    gold_dir_root: []const u8,
    data_dir_root: []const u8,
    config: rastcfg.RasterConfig,
    rel_tol: F,
    abs_tol: F,
    shader_filter: ShaderFilter,
    report_perf: bool,
) !void {
    var arena = std.heap.ArenaAllocator.init(outer_alloc);
    defer arena.deinit();
    const aa = arena.allocator();

    const prepared = try orch.prepareSingleMeshCase(
        aa,
        io,
        test_type,
        mesh_type,
        pixel_num,
        fov_scale,
        data_dir_root,
    );

    const disps = [_]bool{ true, false };
    for (disps) |add_disp| {
        const d_str = if (add_disp) "dispon" else "dispoff";

        // --- Nodal Shader ---
        if (shader_filter == .nodal or shader_filter == .both) {
            const case_dir_name = try std.fmt.allocPrint(
                aa,
                "{s}_{s}_{s}_nodal",
                .{ test_type, @tagName(mesh_type), d_str },
            );
            const nodal_dir = try std.fmt.allocPrint(
                aa,
                "{s}/{s}",
                .{ gold_dir_root, case_dir_name },
            );

            const mesh_input = MeshInput{
                .mesh_type = mesh_type,
                .coords = prepared.sim_data.coords,
                .connect = prepared.sim_data.connect,
                .disp = if (add_disp) prepared.sim_data.field else null,
                .shader = .{
                    .nodal = .{
                        .field = prepared.sim_data.field.?,
                        .bits = 8,
                    },
                },
            };

            var run_config = config;
            if (mode == .test_gold) {
                run_config.save_strategy = .memory;
                run_config.image_save_opts = &[_]iio.ImageSaveOpts{
                    .{ .format = .csv, .bits = null, .scaling = .none },
                };
                run_config.report = if (report_perf) .full_stats else .off;
            } else {
                var out_dir = try orch.openDirEnsured(io, nodal_dir);
                out_dir.close(io);
            }

            const camera_input = CameraInput{
                .pixels_num = prepared.camera.pixels_num,
                .pixels_size = prepared.camera.pixels_size,
                .pos_world = prepared.camera.pos_world,
                .rot_world = prepared.camera.rot_world,
                .roi_cent_world = prepared.camera.roi_cent_world,
                .focal_length = prepared.camera.focal_length,
                .sub_sample = prepared.camera.sub_sample,
                .distortion = prepared.camera.distortion,
            };

            if (mode == .test_gold and tcfg.TEST_CASE_VERBOSE) {
                std.debug.print("Testing {s} ... ", .{case_dir_name});
            }
            const time_start = Timestamp.now(io, .awake);
            const render_groups = [_]riley.RenderGroupSpec{
                .{ .io = io, .workers = @max(@as(u16, 1), run_config.total_threads) },
            };
            const result = try riley.raster(
                aa,
                &render_groups,
                &[_]CameraInput{camera_input},
                &[_]MeshInput{mesh_input},
                run_config,
                if (mode == .generate) nodal_dir else null,
            );

            if (mode == .generate) {
                if (result) |images| {
                    aa.free(images.slice);
                    var images_mut = images;
                    images_mut.deinit(aa);
                }
            } else {
                var render_result = result orelse return error.NoResult;
                defer aa.free(render_result.slice);
                const time_end = Timestamp.now(io, .awake);
                const duration_ms = @as(F, @floatFromInt(time_start.durationTo(time_end).raw.nanoseconds)) / 1e6;

                const frames_num = if (render_result.dims.len == 5)
                    render_result.dims[1]
                else
                    render_result.dims[0];
                var first_err: ?anyerror = null;
                for (0..frames_num) |f| {
                    const fname = try findGoldPath(aa, io, nodal_dir, 0, f, 0, false);

                    compareNDArrayToGold(
                        aa,
                        io,
                        &render_result,
                        0,
                        f,
                        0,
                        1,
                        fname,
                        rel_tol,
                        abs_tol,
                    ) catch |err| {
                        if (first_err == null) {
                            if (err == error.PixelMismatch) {
                                if (tcfg.TEST_CASE_VERBOSE) {
                                    std.debug.print("MISMATCH! ({d:.2} ms)\n", .{duration_ms});
                                }
                            } else {
                                if (tcfg.TEST_CASE_VERBOSE) {
                                    std.debug.print("ERROR! ({d:.2} ms)\n", .{duration_ms});
                                }
                            }
                            first_err = err;
                        }
                        const fail_dir_name = try std.fmt.allocPrint(
                            aa,
                            "all_{s}{s}",
                            .{ case_dir_name, impl_suffix },
                        );
                        try saveComparisonArtifactsFromResult(
                            aa,
                            io,
                            default_fails_root,
                            fail_dir_name,
                            &render_result,
                            0,
                            f,
                            0,
                            fname,
                            1,
                        );
                    };
                }
                if (first_err) |err| return err;
                if (tcfg.TEST_CASE_VERBOSE) {
                    std.debug.print("MATCHED ({d:.2} ms)\n", .{duration_ms});
                }
            }
        }

        // --- Texture Shader ---
        if (shader_filter == .tex or shader_filter == .both) {
            for (samp_cfgs) |sc| {
                const case_dir_name = try std.fmt.allocPrint(
                    aa,
                    "{s}_{s}_{s}_tex_{s}_{s}",
                    .{ test_type, @tagName(mesh_type), d_str, @tagName(sc.sample), @tagName(sc.mode) },
                );
                const tex_dir = try std.fmt.allocPrint(
                    aa,
                    "{s}/{s}",
                    .{ gold_dir_root, case_dir_name },
                );

                const mesh_input = MeshInput{
                    .mesh_type = mesh_type,
                    .coords = prepared.sim_data.coords,
                    .connect = prepared.sim_data.connect,
                    .disp = if (add_disp) prepared.sim_data.field else null,
                    .shader = .{
                        .tex_u8 = .{
                            .uvs = prepared.uvs.array,
                            .tex = texture,
                            .samp_cfg = sc,
                        },
                    },
                };

                var run_config = config;
                if (mode == .test_gold) {
                    run_config.save_strategy = .memory;
                    run_config.image_save_opts = &[_]iio.ImageSaveOpts{
                        .{ .format = .csv, .bits = null, .scaling = .none },
                    };
                    run_config.report = if (report_perf) .full_stats else .off;
                } else {
                    var out_dir = try orch.openDirEnsured(io, tex_dir);
                    out_dir.close(io);
                }

                const camera_input = CameraInput{
                    .pixels_num = prepared.camera.pixels_num,
                    .pixels_size = prepared.camera.pixels_size,
                    .pos_world = prepared.camera.pos_world,
                    .rot_world = prepared.camera.rot_world,
                    .roi_cent_world = prepared.camera.roi_cent_world,
                    .focal_length = prepared.camera.focal_length,
                    .sub_sample = prepared.camera.sub_sample,
                    .distortion = prepared.camera.distortion,
                };

                if (mode == .test_gold and tcfg.TEST_CASE_VERBOSE) {
                    std.debug.print("Testing {s} ... ", .{case_dir_name});
                }
                const time_start = Timestamp.now(io, .awake);
                const render_groups = [_]riley.RenderGroupSpec{
                    .{ .io = io, .workers = @max(@as(u16, 1), run_config.total_threads) },
                };
                const result = try riley.raster(
                    aa,
                    &render_groups,
                    &[_]CameraInput{camera_input},
                    &[_]MeshInput{mesh_input},
                    run_config,
                    if (mode == .generate) tex_dir else null,
                );

                if (mode == .generate) {
                    if (result) |images| {
                        aa.free(images.slice);
                        var images_mut = images;
                        images_mut.deinit(aa);
                    }
                } else {
                    var render_result = result orelse return error.NoResult;
                    defer aa.free(render_result.slice);
                    const time_end = Timestamp.now(io, .awake);
                    const duration_ms = @as(F, @floatFromInt(time_start.durationTo(time_end).raw.nanoseconds)) / 1e6;

                    const frames_num = if (render_result.dims.len == 5)
                        render_result.dims[1]
                    else
                        render_result.dims[0];
                    var first_err: ?anyerror = null;
                    for (0..frames_num) |f| {
                        const fname = try findGoldPath(aa, io, tex_dir, 0, f, 0, false);

                        compareNDArrayToGold(
                            aa,
                            io,
                            &render_result,
                            0,
                            f,
                            0,
                            1,
                            fname,
                            rel_tol,
                            abs_tol,
                        ) catch |err| {
                            if (first_err == null) {
                                if (err == error.PixelMismatch) {
                                    if (tcfg.TEST_CASE_VERBOSE) {
                                        std.debug.print("MISMATCH! ({d:.2} ms)\n", .{duration_ms});
                                    }
                                } else {
                                    if (tcfg.TEST_CASE_VERBOSE) {
                                        std.debug.print("ERROR! ({d:.2} ms)\n", .{duration_ms});
                                    }
                                }
                                first_err = err;
                            }
                            const fail_dir_name = try std.fmt.allocPrint(
                                aa,
                                "all_{s}{s}",
                                .{ case_dir_name, impl_suffix },
                            );
                            try saveComparisonArtifactsFromResult(
                                aa,
                                io,
                                default_fails_root,
                                fail_dir_name,
                                &render_result,
                                0,
                                f,
                                0,
                                fname,
                                1,
                            );
                        };
                    }
                    if (first_err) |err| return err;
                    if (tcfg.TEST_CASE_VERBOSE) {
                        std.debug.print("MATCHED ({d:.2} ms)\n", .{duration_ms});
                    }
                }
            }
        }
    }
}

pub fn runTestInternal(
    outer_alloc: std.mem.Allocator,
    io: std.Io,
    test_type: []const u8,
    mesh_type: MeshType,
    fov_scale: F,
    texture: texops.Tex(u8, 1),
    pixel_num: [2]u32,
    samp_cfgs: []const texops.TextureSampleConfig,
    gold_dir_root: []const u8,
    data_dir_root: []const u8,
    rel_tol: F,
    abs_tol: F,
    shader_filter: ShaderFilter,
    report_perf: bool,
) !void {
    try runSingleMeshSuiteDriver(
        .test_gold,
        outer_alloc,
        io,
        test_type,
        mesh_type,
        fov_scale,
        texture,
        pixel_num,
        samp_cfgs,
        gold_dir_root,
        data_dir_root,
        tcfg.getRasterConfig(.testing),
        rel_tol,
        abs_tol,
        shader_filter,
        report_perf,
    );
}

pub fn runMeshTypesSuite(
    outer_alloc: std.mem.Allocator,
    io: std.Io,
    test_types: []const []const u8,
    mesh_types: []const gk.MeshType,
    fov_scale: F,
    texture: texops.Tex(u8, 1),
    pixel_num: [2]u32,
    samp_cfgs: []const texops.TextureSampleConfig,
    gold_dir_root: []const u8,
    data_dir_root: []const u8,
    rel_tol: F,
    abs_tol: F,
    shader_filter: ShaderFilter,
    report_perf: bool,
) !void {
    for (mesh_types) |mt| {
        for (test_types) |tt| {
            try runTestInternal(
                outer_alloc,
                io,
                tt,
                mt,
                fov_scale,
                texture,
                pixel_num,
                samp_cfgs,
                gold_dir_root,
                data_dir_root,
                rel_tol,
                abs_tol,
                shader_filter,
                report_perf,
            );
        }
    }
}

pub fn runMultimeshTest(
    outer_alloc: std.mem.Allocator,
    io: std.Io,
    rel_tol: F,
    abs_tol: F,
) !void {
    try runMultimeshTestExt(
        outer_alloc,
        io,
        policy.goldRoot(.multimesh),
        &orch.default_multimesh_dir_paths,
        .{ 1200, 800 },
        rel_tol,
        abs_tol,
    );
}

pub fn runMultimeshTestExt(
    outer_alloc: std.mem.Allocator,
    io: std.Io,
    gold_dir_root: []const u8,
    dir_paths: []const []const u8,
    pixel_num: [2]u32,
    rel_tol: F,
    abs_tol: F,
) !void {
    var arena = std.heap.ArenaAllocator.init(outer_alloc);
    defer arena.deinit();
    const aa = arena.allocator();

    const shader_modes = [_]enum { nodal, texture }{ .nodal, .texture };

    for (shader_modes) |mode| {
        _ = arena.reset(.free_all);
        const mesh_inputs = try orch.buildMultimeshInputs(
            aa,
            io,
            dir_paths,
            if (mode == .nodal) .nodal else .texture,
        );

        const fov_scale_factor: F = 1.1;
        const camera = try orch.initCameraForMeshes(
            aa,
            mesh_inputs,
            pixel_num,
            fov_scale_factor,
        );
        defer camera.deinit(aa);

        var config = tcfg.getRasterConfig(.testing);
        config.save_strategy = .memory;
        config.image_save_opts = &[_]iio.ImageSaveOpts{
            .{ .format = .csv, .bits = null, .scaling = .none },
        };

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
        const case_name = if (mode == .nodal)
            "multimesh_allelem_nodal"
        else
            "multimesh_allelem_tex";
        if (tcfg.TEST_CASE_VERBOSE) {
            std.debug.print("Testing {s} ... ", .{case_name});
        }

        const time_start = Timestamp.now(io, .awake);
        const render_groups = [_]riley.RenderGroupSpec{
            .{ .io = io, .workers = @max(@as(u16, 1), config.total_threads) },
        };
        const result = (try riley.raster(
            aa,
            &render_groups,
            &[_]CameraInput{camera_input},
            mesh_inputs,
            config,
            null,
        )) orelse return error.NoResult;
        const time_end = Timestamp.now(io, .awake);
        const duration_ms = @as(F, @floatFromInt(time_start.durationTo(time_end).raw.nanoseconds)) / 1e6;

        const gold_dir = if (mode == .nodal)
            try std.fmt.allocPrint(aa, "{s}/allelem_nodal", .{gold_dir_root})
        else
            try std.fmt.allocPrint(aa, "{s}/allelem_tex_cubic_lut_lerp", .{gold_dir_root});

        const frames_num = if (result.dims.len == 5) result.dims[1] else result.dims[0];
        for (0..frames_num) |f| {
            const fname = try findGoldPath(aa, io, gold_dir, 0, f, 0, false);
            compareNDArrayToGold(
                aa,
                io,
                &result,
                0,
                f,
                0,
                1,
                fname,
                rel_tol,
                abs_tol,
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
                    "all_{s}{s}",
                    .{ case_name, impl_suffix },
                );
                try saveComparisonArtifactsFromResult(
                    aa,
                    io,
                    default_fails_root,
                    fail_dir_name,
                    &result,
                    0,
                    f,
                    0,
                    fname,
                    1,
                );
                return err;
            };
        }
        if (tcfg.TEST_CASE_VERBOSE) {
            std.debug.print("MATCHED ({d:.2} ms)\n", .{duration_ms});
        }
    }
}

pub fn runMultimeshMixedTest(
    outer_alloc: std.mem.Allocator,
    io: std.Io,
    rel_tol: F,
    abs_tol: F,
) !void {
    const gold_dir = comptime policy.goldRoot(.multimesh) ++
        "/allelem_allshade";
    try runMultimeshMixedTestExt(
        outer_alloc,
        io,
        gold_dir,
        &orch.default_multimesh_dir_paths,
        .{ 1600, 800 },
        rel_tol,
        abs_tol,
    );
}

pub fn runMultimeshMixedTestExt(
    outer_alloc: std.mem.Allocator,
    io: std.Io,
    gold_dir: []const u8,
    dir_paths: []const []const u8,
    pixel_num: [2]u32,
    rel_tol: F,
    abs_tol: F,
) !void {
    var arena = std.heap.ArenaAllocator.init(outer_alloc);
    defer arena.deinit();
    const aa = arena.allocator();

    const texture = try iio.loadImage(
        u8,
        1,
        aa,
        io,
        "texture/speckle-simple.tiff",
        .tiff,
    );

    const mesh_inputs = try orch.buildMixedMeshInputs(
        aa,
        io,
        dir_paths,
        texture,
    );

    const fov_scale_factor: F = 1.2;
    const camera = try orch.initCameraForMeshes(
        aa,
        mesh_inputs,
        pixel_num,
        fov_scale_factor,
    );
    defer camera.deinit(aa);

    var config = tcfg.getRasterConfig(.testing);
    config.save_strategy = .memory;
    config.image_save_opts = &[_]iio.ImageSaveOpts{
        .{ .format = .csv, .bits = null, .scaling = .none },
    };
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

    const time_start = Timestamp.now(io, .awake);
    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = @max(@as(u16, 1), config.total_threads) },
    };
    const result = (try riley.raster(
        aa,
        &render_groups,
        &[_]CameraInput{camera_input},
        mesh_inputs,
        config,
        null,
    )) orelse return error.NoResult;
    const time_end = Timestamp.now(io, .awake);
    const duration_ms = @as(F, @floatFromInt(time_start.durationTo(time_end).raw.nanoseconds)) / 1e6;

    const frames_num = if (result.dims.len == 5) result.dims[1] else result.dims[0];
    for (0..frames_num) |f| {
        const fname = try findGoldPath(aa, io, gold_dir, 0, f, 0, false);
        compareNDArrayToGold(
            aa,
            io,
            &result,
            0,
            f,
            0,
            1,
            fname,
            rel_tol,
            abs_tol,
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
            try saveComparisonArtifactsFromResult(
                aa,
                io,
                default_fails_root,
                "all_multimesh_allelem_allshade" ++ impl_suffix,
                &result,
                0,
                f,
                0,
                fname,
                1,
            );
            return err;
        };
    }
}

pub fn runMultimeshMixedRGBTest(
    outer_alloc: std.mem.Allocator,
    io: std.Io,
    rel_tol: F,
    abs_tol: F,
) !void {
    const gold_dir = comptime policy.goldRoot(.multimesh) ++
        "/allelem_allshade_rgb";
    try runMultimeshMixedRGBTestExt(
        outer_alloc,
        io,
        gold_dir,
        &orch.default_multimesh_dir_paths,
        .{ 1200, 800 },
        rel_tol,
        abs_tol,
    );
}

pub fn runMultimeshMixedRGBTestExt(
    outer_alloc: std.mem.Allocator,
    io: std.Io,
    gold_dir: []const u8,
    dir_paths: []const []const u8,
    pixel_num: [2]u32,
    rel_tol: F,
    abs_tol: F,
) !void {
    var arena = std.heap.ArenaAllocator.init(outer_alloc);
    defer arena.deinit();
    const aa = arena.allocator();

    const texture = try iio.loadImage(
        u8,
        3,
        aa,
        io,
        "texture/speckle_rgb.bmp",
        .bmp,
    );

    const mesh_inputs = try orch.buildMixedRgbMeshInputs(
        aa,
        io,
        dir_paths,
        texture,
    );

    const fov_scale_factor: F = 1.1;
    const camera = try orch.initCameraForMeshes(
        aa,
        mesh_inputs,
        pixel_num,
        fov_scale_factor,
    );
    defer camera.deinit(aa);

    var config_rgb = tcfg.getRasterConfig(.testing);
    config_rgb.save_strategy = .memory;
    config_rgb.image_save_opts = &[_]iio.ImageSaveOpts{
        .{ .format = .csv, .bits = null, .scaling = .none, .channels = 3 },
    };

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

    const time_start = Timestamp.now(io, .awake);
    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = @max(@as(u16, 1), config_rgb.total_threads) },
    };
    const result = (try riley.raster(
        aa,
        &render_groups,
        &[_]CameraInput{camera_input},
        mesh_inputs,
        config_rgb,
        null,
    )) orelse return error.NoResult;
    const time_end = Timestamp.now(io, .awake);
    const duration_ms = @as(F, @floatFromInt(time_start.durationTo(time_end).raw.nanoseconds)) / 1e6;

    const frames_num = if (result.dims.len == 5) result.dims[1] else result.dims[0];
    for (0..frames_num) |f| {
        const fname = try findGoldPath(aa, io, gold_dir, 0, f, 0, true);
        compareNDArrayToGold(
            aa,
            io,
            &result,
            0,
            f,
            0,
            3,
            fname,
            rel_tol,
            abs_tol,
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
            try saveComparisonArtifactsFromResult(
                aa,
                io,
                default_fails_root,
                "all_multimesh_allelem_allshade_rgb" ++ impl_suffix,
                &result,
                0,
                f,
                0,
                fname,
                3,
            );
            return err;
        };
    }
    if (tcfg.TEST_CASE_VERBOSE) {
        std.debug.print("MATCHED ({d:.2} ms)\n", .{duration_ms});
    }
}

fn supportsMidsideDistortMesh(mesh_type: gk.MeshType) bool {
    return switch (mesh_type) {
        .tri6, .quad8, .quad9 => true,
        else => false,
    };
}

fn supportsAnyDistortMesh(mesh_type: gk.MeshType) bool {
    return switch (mesh_type) {
        .tri3, .tri6, .quad4ibi, .quad4newton, .quad8, .quad9 => true,
        .tri3opt => false,
    };
}

pub fn runEdgeTexFuncConstantSuiteDriver(
    mode: RunMode,
    outer_alloc: std.mem.Allocator,
    io: std.Io,
    test_type: []const u8,
    mesh_type: gk.MeshType,
    gold_dir_root: []const u8,
    data_dir_root: []const u8,
    pixel_num: [2]u32,
    hull_mode: ?rastcfg.HullMode,
    config: rastcfg.RasterConfig,
) !void {
    var arena = std.heap.ArenaAllocator.init(outer_alloc);
    defer arena.deinit();
    const aa = arena.allocator();
    const prepared = try orch.prepareSingleMeshCase(
        aa,
        io,
        test_type,
        mesh_type,
        pixel_num,
        1.1,
        data_dir_root,
    );

    const case_dir_name = if (hull_mode) |hm|
        try std.fmt.allocPrint(
            aa,
            "{s}_{s}_texfunc_constant_{s}",
            .{ test_type, @tagName(mesh_type), @tagName(hm) },
        )
    else
        try std.fmt.allocPrint(
            aa,
            "{s}_{s}_texfunc_constant",
            .{ test_type, @tagName(mesh_type) },
        );

    const gold_dir = try std.fmt.allocPrint(
        aa,
        "{s}/{s}",
        .{ gold_dir_root, case_dir_name },
    );

    const mesh_input = MeshInput{
        .mesh_type = mesh_type,
        .coords = prepared.sim_data.coords,
        .connect = prepared.sim_data.connect,
        .disp = prepared.sim_data.field,
        .shader = .{
            .func = .{
                .uvs = null,
                .coord_mode = .para,
                .builtin = .constant,
                .normal_type = .none,
            },
        },
    };

    var run_config = config;
    if (hull_mode) |hm| {
        run_config.hull_mode = hm;
    }
    if (mode == .test_gold) {
        run_config.save_strategy = .memory;
        run_config.image_save_opts = &[_]iio.ImageSaveOpts{
            .{ .format = .csv, .bits = null, .scaling = .none },
        };
    } else {
        var out_dir = try orch.openDirEnsured(io, gold_dir);
        out_dir.close(io);
    }

    const camera_input = CameraInput{
        .pixels_num = prepared.camera.pixels_num,
        .pixels_size = prepared.camera.pixels_size,
        .pos_world = prepared.camera.pos_world,
        .rot_world = prepared.camera.rot_world,
        .roi_cent_world = prepared.camera.roi_cent_world,
        .focal_length = prepared.camera.focal_length,
        .sub_sample = prepared.camera.sub_sample,
        .distortion = prepared.camera.distortion,
    };

    if (mode == .test_gold and tcfg.TEST_CASE_VERBOSE) {
        std.debug.print("Testing {s} ... ", .{case_dir_name});
    }
    const start_time = Timestamp.now(io, .awake);
    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = @max(@as(u16, 1), run_config.total_threads) },
    };
    const result = try riley.raster(
        aa,
        &render_groups,
        &[_]CameraInput{camera_input},
        &[_]MeshInput{mesh_input},
        run_config,
        if (mode == .generate) gold_dir else null,
    );

    if (mode == .generate) {
        if (result) |images| {
            aa.free(images.slice);
            var images_mut = images;
            images_mut.deinit(aa);
        }
    } else {
        var render_result = result orelse return error.NoResult;
        defer aa.free(render_result.slice);
        const end_time = Timestamp.now(io, .awake);
        const duration_ms = @as(
            F,
            @floatFromInt(start_time.durationTo(end_time).raw.nanoseconds),
        ) / 1e6;

        const frames_num = if (render_result.dims.len == 5) render_result.dims[1] else render_result.dims[0];
        var first_err: ?anyerror = null;
        for (0..frames_num) |frame_idx| {
            const gold_path = try findGoldPath(
                aa,
                io,
                gold_dir,
                0,
                frame_idx,
                0,
                false,
            );

            compareNDArrayToGold(
                aa,
                io,
                &render_result,
                0,
                frame_idx,
                0,
                1,
                gold_path,
                tcfg.REL_TOL,
                tcfg.ABS_TOL,
            ) catch |err| {
                if (first_err == null) {
                    if (err == error.PixelMismatch) {
                        if (tcfg.TEST_CASE_VERBOSE) {
                            std.debug.print("MISMATCH! ({d:.2} ms)\n", .{duration_ms});
                        }
                    } else {
                        if (tcfg.TEST_CASE_VERBOSE) {
                            std.debug.print("ERROR! ({d:.2} ms)\n", .{duration_ms});
                        }
                    }
                    first_err = err;
                }

                const fail_dir_name = try std.fmt.allocPrint(
                    aa,
                    "all_{s}{s}",
                    .{ case_dir_name, impl_suffix },
                );
                try saveComparisonArtifactsFromResult(
                    aa,
                    io,
                    default_fails_root,
                    fail_dir_name,
                    &render_result,
                    0,
                    frame_idx,
                    0,
                    gold_path,
                    1,
                );
            };
        }
        if (first_err) |err| return err;
        if (tcfg.TEST_CASE_VERBOSE) {
            std.debug.print("MATCHED ({d:.2} ms)\n", .{duration_ms});
        }
    }
}

pub fn runDistortEdgeTexFuncTest(
    allocator: std.mem.Allocator,
    io: std.Io,
    mesh_type: gk.MeshType,
    gold_dir_root: []const u8,
    data_dir_root: []const u8,
    pixel_num: [2]u32,
) !void {
    try runDistortEdgeTexFuncTestForHullModeExt(
        allocator,
        io,
        mesh_type,
        gold_dir_root,
        data_dir_root,
        pixel_num,
        null,
    );
}

pub fn runDistortEdgeTexFuncTestForHullMode(
    allocator: std.mem.Allocator,
    io: std.Io,
    mesh_type: gk.MeshType,
    gold_dir_root: []const u8,
    data_dir_root: []const u8,
    pixel_num: [2]u32,
    hull_mode: rastcfg.HullMode,
) !void {
    try runDistortEdgeTexFuncTestForHullModeExt(
        allocator,
        io,
        mesh_type,
        gold_dir_root,
        data_dir_root,
        pixel_num,
        hull_mode,
    );
}

fn runDistortEdgeTexFuncTestForHullModeExt(
    allocator: std.mem.Allocator,
    io: std.Io,
    mesh_type: gk.MeshType,
    gold_dir_root: []const u8,
    data_dir_root: []const u8,
    pixel_num: [2]u32,
    hull_mode: ?rastcfg.HullMode,
) !void {
    const distortion_cases = [_]struct {
        name: []const u8,
        supports: *const fn (gk.MeshType) bool,
    }{
        .{ .name = "distort_bulge", .supports = supportsMidsideDistortMesh },
        .{ .name = "distort_tan", .supports = supportsMidsideDistortMesh },
        .{ .name = "distort_stretch", .supports = supportsAnyDistortMesh },
        .{ .name = "distort_shear", .supports = supportsAnyDistortMesh },
        .{ .name = "distort_rot", .supports = supportsAnyDistortMesh },
    };

    for (distortion_cases) |distortion_case| {
        if (!distortion_case.supports(mesh_type)) continue;
        try runEdgeTexFuncConstantCaseForHullModeExt(
            allocator,
            io,
            distortion_case.name,
            mesh_type,
            gold_dir_root,
            data_dir_root,
            pixel_num,
            hull_mode,
        );
    }
}

pub fn runEdgeTexFuncConstantCaseForHullMode(
    allocator: std.mem.Allocator,
    io: std.Io,
    test_type: []const u8,
    mesh_type: gk.MeshType,
    gold_dir_root: []const u8,
    data_dir_root: []const u8,
    pixel_num: [2]u32,
    hull_mode: rastcfg.HullMode,
) !void {
    try runEdgeTexFuncConstantCaseForHullModeExt(
        allocator,
        io,
        test_type,
        mesh_type,
        gold_dir_root,
        data_dir_root,
        pixel_num,
        hull_mode,
    );
}

fn runEdgeTexFuncConstantCaseForHullModeExt(
    allocator: std.mem.Allocator,
    io: std.Io,
    test_type: []const u8,
    mesh_type: gk.MeshType,
    gold_dir_root: []const u8,
    data_dir_root: []const u8,
    pixel_num: [2]u32,
    hull_mode: ?rastcfg.HullMode,
) !void {
    try runEdgeTexFuncConstantSuiteDriver(
        .test_gold,
        allocator,
        io,
        test_type,
        mesh_type,
        gold_dir_root,
        data_dir_root,
        pixel_num,
        hull_mode,
        tcfg.getRasterConfig(.testing),
    );
}
