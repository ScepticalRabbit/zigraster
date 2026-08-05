// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const std = @import("std");
const riley = @import("riley/zig/riley.zig");
const meshio = @import("riley/zig/meshio.zig");
const iio = @import("riley/zig/imageio.zig");
const uvio = @import("riley/zig/uvio.zig");
const CameraPrepared = @import("riley/zig/camera.zig").CameraPrepared;
const CameraOps = @import("riley/zig/camera.zig").CameraOps;
const Rotation = @import("riley/zig/camera.zig").Rotation;
const sceneops = @import("riley/zig/sceneops.zig");
const MeshInput = @import("riley/zig/meshpipeline.zig").MeshInput;
const mo = @import("riley/zig/meshpipeline.zig");
const MatSlice = @import("riley/zig/matslice.zig").MatSlice;
const NDArray = @import("riley/zig/ndarray.zig").NDArray;
const csvio = @import("riley/zig/csvio.zig");

pub fn loadNDArrayFromCSVRGB(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    rows: usize,
    cols: usize,
) !NDArray(f64) {
    var image = try csvio.loadPackedCsv2D(allocator, io, path, 3);
    defer {
        allocator.free(image.slice);
        image.deinit(allocator);
    }

    if (image.dims[0] != rows) return error.CSVRowsMismatch;
    if (image.dims[1] != cols) return error.CSVColsMismatch;

    var array = try NDArray(f64).initFlat(allocator, &[_]usize{ 1, 3, rows, cols });
    for (0..rows) |rr| {
        for (0..cols) |cc| {
            for (0..3) |ch| {
                array.set(&[_]usize{ 0, ch, rr, cc }, image.get(&[_]usize{ rr, cc, ch }));
            }
        }
    }
    return array;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const out_dir_root = "out/diff";
    const cwd = std.Io.Dir.cwd();
    cwd.createDir(io, out_dir_root, .default_dir) catch |err| {
        if (err != error.PathAlreadyExists) return err;
    };
    var out_dir = try cwd.openDir(io, out_dir_root, .{});
    defer out_dir.close(io);

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    const dir_paths = [_][]const u8{
        "data/simple/tri3_twoelems/",
        "data/simple/tri6_twoelems/",
        "data/simple/quad4_twoelems/",
        "data/simple/quad8_twoelems/",
        "data/simple/quad9_twoelems/",
    };

    const mesh_types = [_]mo.MeshType{
        .tri3,
        .tri6,
        .quad4ibi,
        .quad8,
        .quad9,
    };

    const sim_datas = try meshio.loadMultiSimData(aa, io, &dir_paths, .{});
    const texture = try iio.loadImage(
        u8,
        3,
        aa,
        io,
        "texture/speckle_rgb.bmp",
        .bmp,
    );

    var mesh_inputs = try aa.alloc(MeshInput, 10);

    // Top Row (0-4): Texture RGB Shading
    for (0..5) |ii| {
        const uv_path = try std.fmt.allocPrint(aa, "{s}uvs.csv", .{dir_paths[ii]});
        const uvs = try uvio.loadUVMap(aa, io, uv_path);
        const coords_dup = try MatSlice(f64).initAlloc(
            aa,
            sim_datas[ii].coords.mat.rows_num,
            sim_datas[ii].coords.mat.cols_num,
        );
        @memcpy(coords_dup.slice, sim_datas[ii].coords.mat.slice);

        mesh_inputs[ii] = MeshInput{
            .mesh_type = mesh_types[ii],
            .coords = meshio.Coords.init(coords_dup.slice, coords_dup.rows_num),
            .connect = sim_datas[ii].connect,
            .disp = sim_datas[ii].field,
            .shader = .{ .tex_rgb = .{
                .uvs = uvs.array,
                .tex = texture,
                .samp_cfg = .{
                    .sample = .cubic_catmull_rom,
                    .mode = .lut_lerp,
                },
                .bits = 8,
                .scaling = .none,
            } },
        };
    }

    // Bottom Row (5-9): Flat RGB Shading with Gradient
    for (0..5) |ii| {
        const field = sim_datas[ii].field.?;
        const num_coords = sim_datas[ii].coords.mat.rows_num;
        var rgb_field_arr = try riley.NDArray(f64).initFlat(
            aa,
            &[_]usize{ field.array.dims[0], num_coords, 3 },
        );

        const coords = sim_datas[ii].coords;
        var min_x: f64 = std.math.inf(f64);
        var max_x: f64 = -std.math.inf(f64);
        for (0..num_coords) |nn| {
            const x_val = coords.x(nn);
            if (x_val < min_x) min_x = x_val;
            if (x_val > max_x) max_x = x_val;
        }
        const range_x = max_x - min_x;

        for (0..field.array.dims[0]) |tt| {
            for (0..num_coords) |nn| {
                const x_val = coords.x(nn);
                const t = if (range_x > 0) (x_val - min_x) / range_x else 0.5;
                var rr: f64 = 0;
                var gg: f64 = 0;
                var bb: f64 = 0;
                if (t < 0.5) {
                    const t_scaled = t * 2.0;
                    rr = 1.0 - t_scaled;
                    gg = t_scaled;
                    bb = 0.0;
                } else {
                    const t_scaled = (t - 0.5) * 2.0;
                    rr = 0.0;
                    gg = 1.0 - t_scaled;
                    bb = t_scaled;
                }
                rgb_field_arr.set(&[_]usize{ tt, nn, 0 }, rr);
                rgb_field_arr.set(&[_]usize{ tt, nn, 1 }, gg);
                rgb_field_arr.set(&[_]usize{ tt, nn, 2 }, bb);
            }
        }
        const rgb_field = meshio.Field{
            .array = rgb_field_arr,
            .array_mem = rgb_field_arr.slice,
        };
        const coords_dup = try MatSlice(f64).initAlloc(
            aa,
            sim_datas[ii].coords.mat.rows_num,
            sim_datas[ii].coords.mat.cols_num,
        );
        @memcpy(coords_dup.slice, sim_datas[ii].coords.mat.slice);

        mesh_inputs[ii + 5] = MeshInput{
            .mesh_type = mesh_types[ii],
            .coords = meshio.Coords.init(coords_dup.slice, coords_dup.rows_num),
            .connect = sim_datas[ii].connect,
            .disp = sim_datas[ii].field,
            .shader = .{ .nodal = .{
                .field = rgb_field,
                .bits = 8,
                .scaling = .auto,
                .scale_over = .within_frames,
            } },
        };
    }

    sceneops.arrangeMeshesGrid(mesh_inputs, .{
        .gap = .{ 0.15, 0.15, 0.0 },
        .max_divs = .{ 5, 2, 1 },
    });

    const pixel_num = [_]u32{ 1200, 800 };
    const pixel_size = [_]f64{ 5.3e-6, 5.3e-6 };
    const focal_leng: f64 = 50.0e-3;
    const rot = Rotation.init(0, 0, 0);
    const fov_scale_factor: f64 = 1.1;

    const roi_pos = sceneops.boundsCenterOverMeshes(mesh_inputs);
    const cam_pos = CameraOps.posFillFrameFromRotOverMeshes(
        mesh_inputs,
        pixel_num,
        pixel_size,
        focal_leng,
        rot,
        fov_scale_factor,
    );
    const camera = try CameraPrepared.init(
        aa,
        .{
            .pixels_num = pixel_num,
            .pixels_size = pixel_size,
            .pos_world = cam_pos,
            .rot_world = rot,
            .roi_cent_world = roi_pos,
            .focal_length = focal_leng,
            .sub_sample = 3,
        },
    );
    defer camera.deinit(aa);

    const config_rgb = riley.RasterConfig{
        .save_opt = .memory,
        .save_opts = &[_]iio.ImageSaveOpts{
            .{ .format = .csv, .bits = null, .scaling = .none, .channels = 3 },
            .{ .format = .bmp, .bits = 8, .scaling = .auto, .channels = 3 },
        },
    };

    std.debug.print("Rendering Mixed RGB Data for Difference analysis...\n", .{});
    const camera_input = camera.toInput();
    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = @max(@as(u16, 1), config_rgb.total_threads) },
    };
    const result = (try riley.raster(
        aa,
        &render_groups,
        &[_]@TypeOf(camera_input){camera_input},
        mesh_inputs,
        config_rgb,
        out_dir,
    )) orelse return error.NoResult;

    const gold_dir = "gold/multimesh/allelem_allshade_rgb";

    for (0..result.dims[0]) |f| {
        const fname = try std.fmt.allocPrint(
            aa,
            "{s}/frame_{d}_field_0_rgb.csv",
            .{ gold_dir, f },
        );
        std.debug.print("Comparing frame {d} with {s}\n", .{ f, fname });

        const gold_arr = try loadNDArrayFromCSVRGB(aa, io, fname, 800, 1200);

        var diff_arr = try NDArray(f64).initFlat(aa, &[_]usize{ 1, 3, 800, 1200 });
        for (0..3) |cc| {
            for (0..800) |rr| {
                for (0..1200) |col| {
                    const actual = result.get(&[_]usize{ f, cc, rr, col });
                    const gold = gold_arr.get(&[_]usize{ 0, cc, rr, col });
                    diff_arr.set(&[_]usize{ 0, cc, rr, col }, @abs(actual - gold));
                }
            }
        }

        const base_name = try std.fmt.allocPrint(aa, "frame_{d}_field_0_rgb", .{f});

        // Wrap frame in a 3D NDArray for saving
        var frame_dims = [_]usize{ result.dims[1], result.dims[2], result.dims[3] };
        var frame_strides = [_]usize{
            result.strides[1],
            result.strides[2],
            result.strides[3],
        };
        const frame_arr_3d = NDArray(f64){
            .slice = result.slice[f * result.strides[0] ..],
            .dims = &frame_dims,
            .strides = &frame_strides,
        };

        // Save Actual
        const act_csv_name = try std.fmt.allocPrint(aa, "{s}.csv", .{base_name});
        const act_bmp_name = try std.fmt.allocPrint(aa, "{s}.bmp", .{base_name});
        try iio.saveCSV(
            io,
            out_dir,
            act_csv_name,
            &frame_arr_3d,
            0,
            config_rgb.save_opts[0],
        );
        try iio.saveBMP(
            io,
            out_dir,
            act_bmp_name,
            &frame_arr_3d,
            0,
            config_rgb.save_opts[1],
        );

        // Save Diff
        const diff_base_name = try std.fmt.allocPrint(
            aa,
            "frame_{d}_field_0_rgb_diff",
            .{f},
        );
        const diff_csv_name = try std.fmt.allocPrint(aa, "{s}.csv", .{diff_base_name});
        const diff_bmp_name = try std.fmt.allocPrint(aa, "{s}.bmp", .{diff_base_name});

        var diff_dims = [_]usize{ diff_arr.dims[1], diff_arr.dims[2], diff_arr.dims[3] };
        var diff_strides = [_]usize{
            diff_arr.strides[1],
            diff_arr.strides[2],
            diff_arr.strides[3],
        };
        const diff_arr_3d = NDArray(f64){
            .slice = diff_arr.slice,
            .dims = &diff_dims,
            .strides = &diff_strides,
        };

        try iio.saveCSV(
            io,
            out_dir,
            diff_csv_name,
            &diff_arr_3d,
            0,
            config_rgb.save_opts[0],
        );
        try iio.saveBMP(
            io,
            out_dir,
            diff_bmp_name,
            &diff_arr_3d,
            0,
            config_rgb.save_opts[1],
        );
    }

    std.debug.print("Done. Results in {s}/\n", .{out_dir_root});
}
