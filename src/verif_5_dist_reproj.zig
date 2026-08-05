// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const std = @import("std");
const buildconfig = @import("riley/zig/buildconfig.zig");
const F = buildconfig.F;
const cam = @import("riley/zig/camera.zig");
const orch = @import("dev_support/orchestration.zig");
const verif = @import("dev_support/verif.zig");
const vconst = @import("dev_support/verifconstants.zig");

const sample_grid_rows_num: usize = 250;
const sample_grid_cols_num: usize = 250;
const verif_subdir_name = "verif_5";

const DistortionRoundTripRecord = struct {
    ideal_x_true: F,
    ideal_y_true: F,
    ideal_x_rec: F,
    ideal_y_rec: F,
    observed_x: F,
    observed_y: F,
    observed_reproj_x: F,
    observed_reproj_y: F,
    err_x: F,
    err_y: F,
    err_dist: F,
    observed_reproj_err: F,
    iters: u8,
    converged: bool,
    in_bounds: bool,
    row_idx: usize,
    col_idx: usize,
};

const PixelSample = struct {
    ideal_x: F,
    ideal_y: F,
    row_idx: usize,
    col_idx: usize,
};

const DistortionInverseResult = struct {
    x: F,
    y: F,
    iters: u8,
};

fn frameFileStem(
    buf: []u8,
    field_idx: usize,
) ![]const u8 {
    return std.fmt.bufPrint(buf, "cam0_frame0_field{d}", .{field_idx});
}

fn statsFileName(buf: []u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "roundtrip_stats.csv", .{});
}

fn inverseDistortionWithIters(
    comptime DistortionType: type,
    distortion: DistortionType,
    x_dist: F,
    y_dist: F,
) !DistortionInverseResult {
    const tol = buildconfig.config.tolerance;
    const max_iters = buildconfig.config.distortion_newton_iter_max;

    var x_guess = x_dist;
    var y_guess = y_dist;

    for (0..max_iters) |ii| {
        const fwd = distortion.forwardWithJac(x_guess, y_guess);
        const resid_x = fwd.x_d - x_dist;
        const resid_y = fwd.y_d - y_dist;

        if (@max(@abs(resid_x), @abs(resid_y)) < tol.distortion.resid) {
            return .{
                .x = x_guess,
                .y = y_guess,
                .iters = @intCast(ii + 1),
            };
        }

        const jac00 = fwd.jac[0][0];
        const jac01 = fwd.jac[0][1];
        const jac10 = fwd.jac[1][0];
        const jac11 = fwd.jac[1][1];
        const det = jac00 * jac11 - jac01 * jac10;

        if (@abs(det) < tol.distortion.det) {
            return error.SingularJacobian;
        }

        const delta_x = (-resid_x * jac11 + jac01 * resid_y) / det;
        const delta_y = (jac10 * resid_x - jac00 * resid_y) / det;

        x_guess += delta_x;
        y_guess += delta_y;

        if (@max(@abs(delta_x), @abs(delta_y)) < tol.distortion.delta) {
            return .{
                .x = x_guess,
                .y = y_guess,
                .iters = @intCast(ii + 1),
            };
        }
    }

    return error.DistortionInverseFailed;
}

fn observedToIdealRasterWithIters(
    camera: *const cam.CameraPrepared,
    observed_xy: [2]F,
) !DistortionInverseResult {
    const focal_px = camera.calcFocalPx();
    const offsets = camera.calcRasterOffsets();
    const x_dist = (observed_xy[0] - offsets.x_off) / focal_px.fx;
    const y_dist = (observed_xy[1] - offsets.y_off) / focal_px.fy;

    return switch (camera.distortion) {
        .none => .{
            .x = observed_xy[0],
            .y = observed_xy[1],
            .iters = 0,
        },
        .brown_conrady => |distortion| blk: {
            const solved = try inverseDistortionWithIters(
                cam.BrownConrady,
                distortion,
                x_dist,
                y_dist,
            );
            break :blk .{
                .x = solved.x * focal_px.fx + offsets.x_off,
                .y = solved.y * focal_px.fy + offsets.y_off,
                .iters = solved.iters,
            };
        },
        .brown_conrady_ext => |distortion| blk: {
            const solved = try inverseDistortionWithIters(
                cam.BrownConradyExt,
                distortion,
                x_dist,
                y_dist,
            );
            break :blk .{
                .x = solved.x * focal_px.fx + offsets.x_off,
                .y = solved.y * focal_px.fy + offsets.y_off,
                .iters = solved.iters,
            };
        },
        .polynomial,
        .brown_conrady_polynomial,
        .brown_conrady_ext_polynomial,
        => blk: {
            const solved = try cam.inverseDistortionModelScalar(
                camera.distortion,
                x_dist,
                y_dist,
            );
            break :blk .{
                .x = solved.x * focal_px.fx + offsets.x_off,
                .y = solved.y * focal_px.fy + offsets.y_off,
                .iters = 0,
            };
        },
    };
}

fn buildPixelSampleGrid(
    allocator: std.mem.Allocator,
    camera_input: cam.CameraInput,
) !std.ArrayList(PixelSample) {
    var sample_list: std.ArrayList(PixelSample) = .empty;
    try sample_list.ensureTotalCapacity(
        allocator,
        sample_grid_rows_num * sample_grid_cols_num,
    );

    const width_px = @as(F, @floatFromInt(camera_input.pixels_num[0]));
    const height_px = @as(F, @floatFromInt(camera_input.pixels_num[1]));

    for (0..sample_grid_rows_num) |rr| {
        const row_blend = if (sample_grid_rows_num == 1)
            0.0
        else
            @as(F, @floatFromInt(rr)) /
                @as(F, @floatFromInt(sample_grid_rows_num - 1));
        const ideal_y = 0.5 + row_blend * (height_px - 1.0);

        for (0..sample_grid_cols_num) |cc| {
            const col_blend = if (sample_grid_cols_num == 1)
                0.0
            else
                @as(F, @floatFromInt(cc)) /
                    @as(F, @floatFromInt(sample_grid_cols_num - 1));
            const ideal_x = 0.5 + col_blend * (width_px - 1.0);
            try sample_list.append(allocator, .{
                .ideal_x = ideal_x,
                .ideal_y = ideal_y,
                .row_idx = rr,
                .col_idx = cc,
            });
        }
    }

    return sample_list;
}

fn isInSensorBounds(
    camera_input: cam.CameraInput,
    ideal_x: F,
    ideal_y: F,
) bool {
    const width_px = @as(F, @floatFromInt(camera_input.pixels_num[0]));
    const height_px = @as(F, @floatFromInt(camera_input.pixels_num[1]));
    return ideal_x >= 0.5 and ideal_x <= width_px - 0.5 and
        ideal_y >= 0.5 and ideal_y <= height_px - 0.5;
}

fn evalPixelSample(
    camera: *const cam.CameraPrepared,
    camera_input: cam.CameraInput,
    sample: PixelSample,
) DistortionRoundTripRecord {
    const nan = std.math.nan(F);
    const observed_xy = verif.idealToObservedRaster(
        camera,
        .{ sample.ideal_x, sample.ideal_y },
    ) catch {
        return .{
            .ideal_x_true = sample.ideal_x,
            .ideal_y_true = sample.ideal_y,
            .ideal_x_rec = nan,
            .ideal_y_rec = nan,
            .observed_x = nan,
            .observed_y = nan,
            .observed_reproj_x = nan,
            .observed_reproj_y = nan,
            .err_x = nan,
            .err_y = nan,
            .err_dist = nan,
            .observed_reproj_err = nan,
            .iters = 0,
            .converged = false,
            .in_bounds = false,
            .row_idx = sample.row_idx,
            .col_idx = sample.col_idx,
        };
    };

    const ideal_rec = observedToIdealRasterWithIters(
        camera,
        observed_xy,
    ) catch {
        return .{
            .ideal_x_true = sample.ideal_x,
            .ideal_y_true = sample.ideal_y,
            .ideal_x_rec = nan,
            .ideal_y_rec = nan,
            .observed_x = observed_xy[0],
            .observed_y = observed_xy[1],
            .observed_reproj_x = nan,
            .observed_reproj_y = nan,
            .err_x = nan,
            .err_y = nan,
            .err_dist = nan,
            .observed_reproj_err = nan,
            .iters = 0,
            .converged = false,
            .in_bounds = false,
            .row_idx = sample.row_idx,
            .col_idx = sample.col_idx,
        };
    };

    const observed_reproj = verif.idealToObservedRaster(
        camera,
        .{ ideal_rec.x, ideal_rec.y },
    ) catch {
        return .{
            .ideal_x_true = sample.ideal_x,
            .ideal_y_true = sample.ideal_y,
            .ideal_x_rec = ideal_rec.x,
            .ideal_y_rec = ideal_rec.y,
            .observed_x = observed_xy[0],
            .observed_y = observed_xy[1],
            .observed_reproj_x = nan,
            .observed_reproj_y = nan,
            .err_x = nan,
            .err_y = nan,
            .err_dist = nan,
            .observed_reproj_err = nan,
            .iters = ideal_rec.iters,
            .converged = false,
            .in_bounds = false,
            .row_idx = sample.row_idx,
            .col_idx = sample.col_idx,
        };
    };

    const err_x = ideal_rec.x - sample.ideal_x;
    const err_y = ideal_rec.y - sample.ideal_y;
    const observed_reproj_x = observed_reproj[0];
    const observed_reproj_y = observed_reproj[1];
    const observed_resid_x = observed_reproj_x - observed_xy[0];
    const observed_resid_y = observed_reproj_y - observed_xy[1];

    return .{
        .ideal_x_true = sample.ideal_x,
        .ideal_y_true = sample.ideal_y,
        .ideal_x_rec = ideal_rec.x,
        .ideal_y_rec = ideal_rec.y,
        .observed_x = observed_xy[0],
        .observed_y = observed_xy[1],
        .observed_reproj_x = observed_reproj_x,
        .observed_reproj_y = observed_reproj_y,
        .err_x = err_x,
        .err_y = err_y,
        .err_dist = @sqrt(err_x * err_x + err_y * err_y),
        .observed_reproj_err = @sqrt(
            observed_resid_x * observed_resid_x +
                observed_resid_y * observed_resid_y,
        ),
        .iters = ideal_rec.iters,
        .converged = true,
        .in_bounds = isInSensorBounds(
            camera_input,
            ideal_rec.x,
            ideal_rec.y,
        ),
        .row_idx = sample.row_idx,
        .col_idx = sample.col_idx,
    };
}

fn saveFieldMaps(
    allocator: std.mem.Allocator,
    io: std.Io,
    out_dir: std.Io.Dir,
    rows_num: usize,
    cols_num: usize,
    field0: []const F,
    field1: []const F,
) !void {
    var field0_stem_buf: [128]u8 = undefined;
    var field0_csv_buf: [128]u8 = undefined;
    var field1_stem_buf: [128]u8 = undefined;
    var field1_csv_buf: [128]u8 = undefined;

    const field0_stem = try frameFileStem(&field0_stem_buf, 0);
    const field0_csv = try std.fmt.bufPrint(&field0_csv_buf, "{s}.csv", .{
        field0_stem,
    });
    try verif.writeScalarMapCsv(
        io,
        out_dir,
        field0_csv,
        rows_num,
        cols_num,
        field0,
    );
    try verif.writeScalarMapBmp(
        allocator,
        io,
        out_dir,
        field0_stem,
        rows_num,
        cols_num,
        field0,
    );

    const field1_stem = try frameFileStem(&field1_stem_buf, 1);
    const field1_csv = try std.fmt.bufPrint(&field1_csv_buf, "{s}.csv", .{
        field1_stem,
    });
    try verif.writeScalarMapCsv(
        io,
        out_dir,
        field1_csv,
        rows_num,
        cols_num,
        field1,
    );
    try verif.writeScalarMapBmp(
        allocator,
        io,
        out_dir,
        field1_stem,
        rows_num,
        cols_num,
        field1,
    );
}

fn writeRoundTripStatsCsv(
    io: std.Io,
    out_dir: std.Io.Dir,
    records: []const DistortionRoundTripRecord,
) !void {
    var file_name_buf: [128]u8 = undefined;
    const file_name = try statsFileName(&file_name_buf);
    const csv_file = try out_dir.createFile(io, file_name, .{});
    defer csv_file.close(io);

    var write_buf: [4096]u8 = undefined;
    var file_writer = csv_file.writer(io, &write_buf);
    const writer = &file_writer.interface;

    try writer.writeAll(
        "ideal_x_true,ideal_y_true,ideal_x_rec,ideal_y_rec," ++
            "observed_x,observed_y,observed_reproj_x,observed_reproj_y," ++
            "err_x,err_y,err_dist,observed_reproj_err,iters,converged," ++
            "in_bounds,row_idx,col_idx\n",
    );

    for (records) |record| {
        try writer.print(
            "{d},{d},{d},{d},{d},{d},{d},{d},{d},{d},{d},{d},{d},{d}," ++
                "{d},{d},{d}\n",
            .{
                record.ideal_x_true,
                record.ideal_y_true,
                record.ideal_x_rec,
                record.ideal_y_rec,
                record.observed_x,
                record.observed_y,
                record.observed_reproj_x,
                record.observed_reproj_y,
                record.err_x,
                record.err_y,
                record.err_dist,
                record.observed_reproj_err,
                record.iters,
                @intFromBool(record.converged),
                @intFromBool(record.in_bounds),
                record.row_idx,
                record.col_idx,
            },
        );
    }

    try file_writer.flush();
}

fn runDistortionCase(
    case_spec: vconst.DistortCase,
    distortion_case: vconst.CameraDistortionCase,
    allocator: std.mem.Allocator,
    io: std.Io,
) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    const camera_input = vconst.cameraInputWithDistortion(
        case_spec.camera_input,
        distortion_case,
    );
    const camera = try cam.CameraPrepared.init(aa, camera_input);
    var sample_list = try buildPixelSampleGrid(aa, camera_input);
    defer sample_list.deinit(aa);

    const mesh_name = orch.meshDataName(case_spec.mesh_type);
    const out_dir_path = try std.fmt.allocPrint(
        aa,
        "{s}/{s}/verif_5_{s}_{s}_{s}",
        .{
            vconst.output_dir_name,
            verif_subdir_name,
            mesh_name,
            case_spec.case_name,
            distortion_case.case_name,
        },
    );
    var out_dir = try orch.openDirEnsured(io, out_dir_path);
    defer out_dir.close(io);

    const map_len = sample_grid_rows_num * sample_grid_cols_num;
    var ideal_x_rec_map = try allocator.alloc(F, map_len);
    defer allocator.free(ideal_x_rec_map);
    var ideal_y_rec_map = try allocator.alloc(F, map_len);
    defer allocator.free(ideal_y_rec_map);
    @memset(ideal_x_rec_map, std.math.nan(F));
    @memset(ideal_y_rec_map, std.math.nan(F));

    var records: std.ArrayList(DistortionRoundTripRecord) = .empty;
    defer records.deinit(allocator);
    try records.ensureTotalCapacity(allocator, sample_list.items.len);

    var err_vals: std.ArrayList(F) = .empty;
    defer err_vals.deinit(allocator);

    for (sample_list.items) |sample| {
        const record = evalPixelSample(&camera, camera_input, sample);
        try records.append(allocator, record);

        const map_idx = sample.row_idx * sample_grid_cols_num + sample.col_idx;
        ideal_x_rec_map[map_idx] = record.ideal_x_rec;
        ideal_y_rec_map[map_idx] = record.ideal_y_rec;

        if (record.converged and std.math.isFinite(record.err_dist)) {
            try err_vals.append(allocator, record.err_dist);
        }
    }

    try saveFieldMaps(
        allocator,
        io,
        out_dir,
        sample_grid_rows_num,
        sample_grid_cols_num,
        ideal_x_rec_map,
        ideal_y_rec_map,
    );
    try writeRoundTripStatsCsv(io, out_dir, records.items);

    if (err_vals.items.len > 0) {
        const err_stats = try verif.calcScalarStats(allocator, err_vals.items);
        std.debug.print(
            "verif_5_{s}_{s}_{s}: err max={e:.6}\n",
            .{
                mesh_name,
                case_spec.case_name,
                distortion_case.case_name,
                err_stats.max,
            },
        );
    } else {
        std.debug.print(
            "verif_5_{s}_{s}_{s}: no converged samples\n",
            .{
                mesh_name,
                case_spec.case_name,
                distortion_case.case_name,
            },
        );
    }
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const root_dir_path = try std.fmt.allocPrint(
        allocator,
        "{s}/{s}",
        .{ vconst.output_dir_name, verif_subdir_name },
    );
    var root_dir = try orch.openDirEnsured(io, root_dir_path);
    defer root_dir.close(io);

    for (vconst.distort_cases) |case_spec| {
        for (vconst.camera_distortion_cases) |distortion_case| {
            try runDistortionCase(
                case_spec,
                distortion_case,
                allocator,
                io,
            );
        }
    }
}
