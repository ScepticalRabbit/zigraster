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
const cameraops = @import("riley/zig/cameraops.zig");
const iio = @import("riley/zig/imageio.zig");
const matrix = @import("riley/zig/matstack.zig");
const meshio = @import("riley/zig/meshio.zig");
const mo = @import("riley/zig/meshpipeline.zig");
const orch = @import("dev_support/orchestration.zig");
const rastcfg = @import("riley/zig/rasterconfig.zig");
const tcfg = @import("dev_support/testconfig.zig");
const vconst = @import("dev_support/verifconstants.zig");
const verif = @import("dev_support/verif.zig");
const vector = @import("riley/zig/vecstack.zig");
const riley = @import("riley/zig/riley.zig");
const sceneops = @import("riley/zig/sceneops.zig");

const pixel_num = [_]u32{ 1024, 1024 };
const fov_scale: F = 1.05;
const verif_subdir_name = "verif_2";

const CentroidStats = struct {
    ideal_x: F,
    ideal_y: F,
    calc_x: F,
    calc_y: F,
    diff_x: F,
    diff_y: F,
    dist: F,
};

const ScalarMap = struct {
    rows_num: usize,
    cols_num: usize,
    vals: []F,
};

fn imageFormatExt(format: iio.ImageFormat) []const u8 {
    return switch (format) {
        .csv => ".csv",
        .fimg => ".fimg",
        .ppm => ".ppm",
        .bmp => ".bmp",
        .tiff => ".tiff",
    };
}

fn renameFullStatsOutputs(
    io: std.Io,
    out_dir: std.Io.Dir,
    frame_idx: usize,
    full_stats_opts: rastcfg.FullStatsOpts,
) !void {
    const diag_names = [_][]const u8{
        "iters",
        "xi",
        "eta",
        "converged",
        "Jdet",
        "occupancy",
        "depth",
        "earlyout",
        "tile_timing",
        "tile_density",
        "tile_occupancy",
        "normals",
    };

    var old_name_buf: [256]u8 = undefined;
    var new_name_buf: [256]u8 = undefined;

    const old_stats_name = try std.fmt.bufPrint(
        &old_name_buf,
        "report_stats_cam0_frame0.txt",
        .{},
    );
    const new_stats_name = try std.fmt.bufPrint(
        &new_name_buf,
        "report_stats_cam0_frame{d}.txt",
        .{frame_idx},
    );
    out_dir.rename(old_stats_name, out_dir, new_stats_name, io) catch |err| {
        if (err != error.FileNotFound) return err;
    };

    for (diag_names) |diag_name| {
        for (full_stats_opts.formats) |save_opt| {
            const ext = imageFormatExt(save_opt.format);
            const old_diag_name = try std.fmt.bufPrint(
                &old_name_buf,
                "diag_cam0_frame0_{s}{s}",
                .{ diag_name, ext },
            );
            const new_diag_name = try std.fmt.bufPrint(
                &new_name_buf,
                "diag_cam0_frame{d}_{s}{s}",
                .{ frame_idx, diag_name, ext },
            );
            out_dir.rename(old_diag_name, out_dir, new_diag_name, io) catch |err| {
                if (err != error.FileNotFound) return err;
            };
        }
    }
}

fn buildFrameCoords(
    allocator: std.mem.Allocator,
    sim_data: *const meshio.SimData,
    frame_idx: usize,
) !meshio.Coords {
    var coords = try meshio.Coords.initAlloc(allocator, sim_data.coords.mat.rows_num);

    for (0..sim_data.coords.mat.rows_num) |nn| {
        coords.mat.set(nn, 0, sim_data.coords.x(nn));
        coords.mat.set(nn, 1, sim_data.coords.y(nn));
        coords.mat.set(nn, 2, sim_data.coords.z(nn));

        if (sim_data.field) |field| {
            coords.mat.set(
                nn,
                0,
                coords.mat.get(nn, 0) + field.array.get(
                    &[_]usize{ frame_idx, nn, 0 },
                ),
            );
            coords.mat.set(
                nn,
                1,
                coords.mat.get(nn, 1) + field.array.get(
                    &[_]usize{ frame_idx, nn, 1 },
                ),
            );
            coords.mat.set(
                nn,
                2,
                coords.mat.get(nn, 2) + field.array.get(
                    &[_]usize{ frame_idx, nn, 2 },
                ),
            );
        }
    }

    return coords;
}

fn extractScalarMap(
    allocator: std.mem.Allocator,
    image_arr: *const @import("riley/zig/ndarray.zig").NDArray(F),
) !ScalarMap {
    const rows_num = if (image_arr.dims.len == 5)
        image_arr.dims[3]
    else
        image_arr.dims[2];
    const cols_num = if (image_arr.dims.len == 5)
        image_arr.dims[4]
    else
        image_arr.dims[3];

    const vals = try allocator.alloc(F, rows_num * cols_num);
    for (0..rows_num) |rr| {
        for (0..cols_num) |cc| {
            vals[rr * cols_num + cc] = if (image_arr.dims.len == 5)
                image_arr.get(&[_]usize{ 0, 0, 0, rr, cc })
            else
                image_arr.get(&[_]usize{ 0, 0, rr, cc });
        }
    }

    return .{
        .rows_num = rows_num,
        .cols_num = cols_num,
        .vals = vals,
    };
}

fn calcCentroidStats(
    camera_input: cam.CameraInput,
    rows_num: usize,
    cols_num: usize,
    vals: []const F,
) !CentroidStats {
    const ideal_x = 0.5 * @as(F, @floatFromInt(camera_input.pixels_num[0]));
    const ideal_y = 0.5 * @as(F, @floatFromInt(camera_input.pixels_num[1]));

    var sum_w: F = 0.0;
    var sum_x: F = 0.0;
    var sum_y: F = 0.0;

    for (0..rows_num) |rr| {
        for (0..cols_num) |cc| {
            const weight = vals[rr * cols_num + cc];
            if (!(weight > 0.0)) continue;

            const x = @as(F, @floatFromInt(cc)) + 0.5;
            const y = @as(F, @floatFromInt(rr)) + 0.5;
            sum_w += weight;
            sum_x += x * weight;
            sum_y += y * weight;
        }
    }

    if (sum_w == 0.0) return error.EmptySilhouette;

    const calc_x = sum_x / sum_w;
    const calc_y = sum_y / sum_w;
    const diff_x = calc_x - ideal_x;
    const diff_y = calc_y - ideal_y;

    return .{
        .ideal_x = ideal_x,
        .ideal_y = ideal_y,
        .calc_x = calc_x,
        .calc_y = calc_y,
        .diff_x = diff_x,
        .diff_y = diff_y,
        .dist = @sqrt(diff_x * diff_x + diff_y * diff_y),
    };
}

fn writeStatsCsv(
    io: std.Io,
    out_dir: std.Io.Dir,
    file_name: []const u8,
    stats: CentroidStats,
    centroid_world: vector.Vec3f,
    scaling: cam.FOVScaling,
    camera_prepared: *const cam.CameraPrepared,
    frame_coords: *const meshio.Coords,
) !void {
    const file = try out_dir.createFile(io, file_name, .{});
    defer file.close(io);

    var write_buf: [1024]u8 = undefined;
    var writer_buf = file.writer(io, &write_buf);
    const writer = &writer_buf.interface;

    try writer.writeAll("key,unit,value\n");
    try writer.print("cent_ideal_x,px,{d:.17}\n", .{stats.ideal_x});
    try writer.print("cent_ideal_y,px,{d:.17}\n", .{stats.ideal_y});
    try writer.print("cent_calc_x,px,{d:.17}\n", .{stats.calc_x});
    try writer.print("cent_calc_y,px,{d:.17}\n", .{stats.calc_y});
    try writer.print("diff_x,px,{d:.17}\n", .{stats.diff_x});
    try writer.print("diff_y,px,{d:.17}\n", .{stats.diff_y});
    try writer.print("dist,px,{d:.17}\n", .{stats.dist});
    try writer.print("sensor_pixels_x,px,{d}\n", .{
        camera_prepared.pixels_num[0],
    });
    try writer.print("sensor_pixels_y,px,{d}\n", .{
        camera_prepared.pixels_num[1],
    });
    try writer.print("centroid_x,length,{d:.17}\n", .{centroid_world.get(0)});
    try writer.print("centroid_y,length,{d:.17}\n", .{centroid_world.get(1)});
    try writer.print("centroid_z,length,{d:.17}\n", .{centroid_world.get(2)});
    try writer.print("plane_dist,length,{d:.17}\n", .{scaling.plane_dist});
    try writer.print("plane_size_x,length,{d:.17}\n", .{scaling.plane_size[0]});
    try writer.print("plane_size_y,length,{d:.17}\n", .{scaling.plane_size[1]});
    try writer.print(
        "leng_per_pixel_x,length/px,{d:.17}\n",
        .{scaling.leng_per_pixel[0]},
    );
    try writer.print(
        "leng_per_pixel_y,length/px,{d:.17}\n",
        .{scaling.leng_per_pixel[1]},
    );
    try writer.print(
        "pixel_per_leng_x,px/length,{d:.17}\n",
        .{scaling.pixel_per_leng[0]},
    );
    try writer.print(
        "pixel_per_leng_y,px/length,{d:.17}\n",
        .{scaling.pixel_per_leng[1]},
    );

    for (0..frame_coords.mat.rows_num) |nn| {
        const coord_raster = projectWorldNodeToRaster(
            camera_prepared,
            frame_coords.getVec3(nn),
        );
        try writer.print("N{d}_x,px,{d:.17}\n", .{ nn, coord_raster.get(0) });
        try writer.print("N{d}_y,px,{d:.17}\n", .{ nn, coord_raster.get(1) });
    }
    try writer.flush();
}

fn projectWorldNodeToRaster(
    camera_prepared: *const cam.CameraPrepared,
    coord_world: vector.Vec3f,
) vector.Vec3f {
    var coord_raster = matrix.Mat44Ops.mulVec3(
        F,
        camera_prepared.world_to_cam_mat,
        coord_world,
    );

    coord_raster.slice[0] = camera_prepared.image_dist * coord_raster.slice[0] /
        (-coord_raster.slice[2]);
    coord_raster.slice[1] = camera_prepared.image_dist * coord_raster.slice[1] /
        (-coord_raster.slice[2]);

    coord_raster.slice[0] = 2.0 * coord_raster.slice[0] /
        camera_prepared.image_dims[0];
    coord_raster.slice[1] = 2.0 * coord_raster.slice[1] /
        camera_prepared.image_dims[1];

    coord_raster.slice[0] = (coord_raster.slice[0] + 1.0) * 0.5 *
        @as(F, @floatFromInt(camera_prepared.pixels_num[0]));
    coord_raster.slice[1] = (1.0 - coord_raster.slice[1]) * 0.5 *
        @as(F, @floatFromInt(camera_prepared.pixels_num[1]));
    coord_raster.slice[2] = -coord_raster.slice[2];
    return coord_raster;
}

fn renderScalarMap(
    render_allocator: std.mem.Allocator,
    out_allocator: std.mem.Allocator,
    io: std.Io,
    case_spec: vconst.DistortCase,
    connect: meshio.Connect,
    frame_coords: meshio.Coords,
    camera_input: cam.CameraInput,
    config: @TypeOf(tcfg.getRasterConfig(.preview)),
    out_dir_path: []const u8,
) !ScalarMap {
    const mesh_input = mo.MeshInput{
        .mesh_type = case_spec.mesh_type,
        .coords = frame_coords,
        .connect = connect,
        .disp = null,
        .shader = .{
            .func = .{
                .uvs = null,
                .coord_mode = .para,
                .builtin = .constant,
                .normal_type = .none,
            },
        },
    };

    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = @max(@as(u16, 1), config.total_threads) },
    };
    const result = (try riley.raster(
        render_allocator,
        &render_groups,
        &[_]cam.CameraInput{camera_input},
        &[_]mo.MeshInput{mesh_input},
        config,
        out_dir_path,
    )) orelse return error.NoResult;

    return try extractScalarMap(out_allocator, &result);
}

fn buildCentroidCameraInput(ref_coords: *const meshio.Coords) cam.CameraInput {
    const initial_rot = orch.defaultRotation();
    const roi_cent_world = sceneops.meanCenter(ref_coords);
    const pos_world = cameraops.posFillFrameFromRotAndTarget(
        ref_coords,
        roi_cent_world,
        pixel_num,
        orch.default_pixel_size,
        orch.default_focal_length,
        initial_rot,
        fov_scale,
    );

    return .{
        .pixels_num = pixel_num,
        .pixels_size = orch.default_pixel_size,
        .pos_world = pos_world,
        .rot_world = initial_rot,
        .roi_cent_world = roi_cent_world,
        .focal_length = orch.default_focal_length,
        .sub_sample = 1,
        .distortion = .none,
    };
}

fn buildCentroidCameraInputOverFrames(
    allocator: std.mem.Allocator,
    sim_data: *const meshio.SimData,
    mesh_type: @TypeOf(vconst.distort_cases[0].mesh_type),
) !cam.CameraInput {
    const initial_rot = orch.defaultRotation();
    const roi_cent_world = sceneops.meanCenter(&sim_data.coords);
    const time_steps = if (sim_data.field) |field| field.getTimeN() else 1;

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    const frame_meshes = try aa.alloc(mo.MeshInput, time_steps);
    for (0..time_steps) |frame_idx| {
        const frame_coords = try buildFrameCoords(aa, sim_data, frame_idx);
        frame_meshes[frame_idx] = .{
            .mesh_type = mesh_type,
            .coords = frame_coords,
            .connect = sim_data.connect,
            .disp = null,
            .shader = .{
                .func = .{
                    .uvs = null,
                    .coord_mode = .para,
                    .builtin = .constant,
                    .normal_type = .none,
                },
            },
        };
    }

    const pos_world = cameraops.posFillFrameFromRotOverMeshesAndTarget(
        frame_meshes,
        roi_cent_world,
        pixel_num,
        orch.default_pixel_size,
        orch.default_focal_length,
        initial_rot,
        fov_scale,
    );

    return .{
        .pixels_num = pixel_num,
        .pixels_size = orch.default_pixel_size,
        .pos_world = pos_world,
        .rot_world = initial_rot,
        .roi_cent_world = roi_cent_world,
        .focal_length = orch.default_focal_length,
        .sub_sample = 1,
        .distortion = .none,
    };
}

fn runDistortCase(
    allocator: std.mem.Allocator,
    io: std.Io,
    case_spec: vconst.DistortCase,
) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    const sim_data = try orch.loadData(aa, io, case_spec.data_dir);
    const time_steps = if (sim_data.field) |field| field.getTimeN() else 1;
    const ref_coords = try buildFrameCoords(aa, &sim_data, 0);
    const base_camera_input = if (std.mem.eql(u8, case_spec.case_name, "rot"))
        try buildCentroidCameraInputOverFrames(allocator, &sim_data, case_spec.mesh_type)
    else
        buildCentroidCameraInput(&ref_coords);
    const base_camera_prepared = try cam.CameraPrepared.init(aa, base_camera_input);

    const out_dir_path = try std.fmt.allocPrint(
        aa,
        "{s}/{s}/b_{s}_{s}",
        .{
            vconst.output_dir_name,
            verif_subdir_name,
            orch.meshDataName(case_spec.mesh_type),
            case_spec.case_name,
        },
    );
    var out_dir = try orch.openDirEnsured(io, out_dir_path);
    defer out_dir.close(io);

    var config = tcfg.getRasterConfig(.preview);
    config.save_strategy = .memory;
    config.report = .off;

    for (0..time_steps) |frame_idx| {
        var frame_arena = std.heap.ArenaAllocator.init(allocator);
        defer frame_arena.deinit();
        const fa = frame_arena.allocator();

        const frame_coords = try buildFrameCoords(fa, &sim_data, frame_idx);
        const frame_cent_world = sceneops.meanCenter(&frame_coords);
        const fov_scaling = cameraops.calcFOVScaling(
            base_camera_input,
            frame_cent_world,
        );
        const scalar_map = try renderScalarMap(
            fa,
            allocator,
            io,
            case_spec,
            sim_data.connect,
            frame_coords,
            base_camera_input,
            config,
            out_dir_path,
        );
        defer allocator.free(scalar_map.vals);
        var base_name_buf: [128]u8 = undefined;
        const base_name = try iio.formatFrameFieldBaseName(
            &base_name_buf,
            0,
            frame_idx,
            0,
            1,
        );

        const csv_name = try std.fmt.allocPrint(aa, "{s}.csv", .{base_name});
        try verif.writeScalarMapCsv(
            io,
            out_dir,
            csv_name,
            scalar_map.rows_num,
            scalar_map.cols_num,
            scalar_map.vals,
        );
        try verif.writeScalarMapBmp(
            allocator,
            io,
            out_dir,
            base_name,
            scalar_map.rows_num,
            scalar_map.cols_num,
            scalar_map.vals,
        );
        const stats = calcCentroidStats(
            base_camera_input,
            scalar_map.rows_num,
            scalar_map.cols_num,
            scalar_map.vals,
        ) catch CentroidStats{
            .ideal_x = 0.5 * @as(F, @floatFromInt(base_camera_input.pixels_num[0])),
            .ideal_y = 0.5 * @as(F, @floatFromInt(base_camera_input.pixels_num[1])),
            .calc_x = std.math.nan(F),
            .calc_y = std.math.nan(F),
            .diff_x = std.math.nan(F),
            .diff_y = std.math.nan(F),
            .dist = std.math.nan(F),
        };
        const stats_name = try std.fmt.allocPrint(aa, "{s}_stats.csv", .{base_name});
        try writeStatsCsv(
            io,
            out_dir,
            stats_name,
            stats,
            frame_cent_world,
            fov_scaling,
            &base_camera_prepared,
            &frame_coords,
        );

        std.debug.print(
            "b_{s}_{s} frame {d}: centroid dist={e:.6}\n",
            .{
                orch.meshDataName(case_spec.mesh_type),
                case_spec.case_name,
                frame_idx,
                stats.dist,
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
        try runDistortCase(allocator, io, case_spec);
    }

    std.debug.print("Done.\n", .{});
}
