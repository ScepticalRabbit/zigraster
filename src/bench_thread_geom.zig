// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const std = @import("std");
const benchargs = @import("dev_support/benchargs.zig");
const common = @import("dev_support/benchcommon.zig");
const tcfg = @import("dev_support/testconfig.zig");
const buildconfig = @import("riley/zig/buildconfig.zig");
const rastcfg = @import("riley/zig/rasterconfig.zig");
const riley = @import("riley/zig/riley.zig");
const gk = @import("riley/zig/geometrykernels.zig");
const iio = @import("riley/zig/imageio.zig");
const texops = @import("riley/zig/textureops.zig");
const Rotation = @import("riley/zig/rotation.zig").Rotation;
const meshio = @import("riley/zig/meshio.zig");
const mo = @import("riley/zig/meshpipeline.zig");
const cam = @import("riley/zig/camera.zig");
const cameraops = @import("riley/zig/cameraops.zig");
const sceneops = @import("riley/zig/sceneops.zig");
const report = @import("riley/zig/report.zig");
const NDArray = @import("riley/zig/ndarray.zig").NDArray;
const orch = @import("dev_support/orchestration.zig");
const shaderops = @import("riley/zig/shaderops.zig");
const Timestamp = std.Io.Clock.Timestamp;
const F = buildconfig.F;

const DEFAULT_OUT_DIR = "out/bench_stats_thread_geom";
const DEFAULT_IMAGE_OUT_DIR = "out/bench_images_thread_geom";
const DEFAULT_PIXELS_NUM = [2]u32{ 1600, 1000 };
const DEFAULT_SUB_SAMPLE: u32 = 1;
const DEFAULT_FOCAL_LENG: F = @floatCast(50.0e-3);
const DEFAULT_PIXELS_SIZE = [2]F{
    @floatCast(5.3e-6),
    @floatCast(5.3e-6),
};
const DEFAULT_FOV_SCALE: F = 1.0;
const DEFAULT_TEX_GREY_PATH = "texture/speckle.bmp";
const DEFAULT_TEX_RGB_PATH = "texture/speckle_rgb.bmp";
const DEFAULT_ROT = Rotation.init(0, 0, 0);

const EVAL_ELEMENTS = [_]gk.MeshType{
    .tri3,
    .tri6,
    .quad4ibi,
    .quad4newton,
    .quad8,
    .quad9,
};

const MESH_SIZES = [_][]const u8{
    "1e3",
    "1e4",
    "1e5",
    "1e6",
};

const MESH_COPIES_LIST = [_]usize{ 1, 2, 4 };

const RunTiming = struct {
    geom_prep_ms: F,
    active_frame_ms: F,
    total_time_ms: F,
};

fn duplicateMeshInput(
    allocator: std.mem.Allocator,
    base_mesh: mo.MeshInput,
    copies_num: usize,
    out_mesh: *mo.MeshInput,
) !void {
    const base_coords_num = base_mesh.coords.mat.rows_num;
    const new_coords_num = base_coords_num * copies_num;
    const new_coords_mem = try allocator.alloc(F, new_coords_num * 3);
    errdefer allocator.free(new_coords_mem);

    var new_coords = meshio.Coords.init(new_coords_mem, new_coords_num);

    var jj: usize = 0;
    while (jj < copies_num) : (jj += 1) {
        var ii: usize = 0;
        while (ii < base_coords_num) : (ii += 1) {
            const dest_idx = jj * base_coords_num + ii;
            const x_val = base_mesh.coords.x(ii);
            const y_val = base_mesh.coords.y(ii);
            const z_offset = @as(F, @floatFromInt(jj)) * 1.0;
            const z_val = base_mesh.coords.z(ii) - z_offset;

            new_coords.mat.set(dest_idx, 0, x_val);
            new_coords.mat.set(dest_idx, 1, y_val);
            new_coords.mat.set(dest_idx, 2, z_val);
        }
    }

    const base_elems_num = base_mesh.connect.getElemsNum();
    const nodes_per_elem = base_mesh.connect.getNodesPerElem();
    const new_elems_num = base_elems_num * copies_num;
    const new_connect_mem = try allocator.alloc(
        usize,
        new_elems_num * nodes_per_elem,
    );
    errdefer allocator.free(new_connect_mem);

    var new_connect = meshio.Connect.init(
        new_connect_mem,
        new_elems_num,
        nodes_per_elem,
    );

    jj = 0;
    while (jj < copies_num) : (jj += 1) {
        var ii: usize = 0;
        while (ii < base_elems_num) : (ii += 1) {
            const dest_elem_idx = jj * base_elems_num + ii;
            var kk: usize = 0;
            while (kk < nodes_per_elem) : (kk += 1) {
                const base_node_idx = base_mesh.connect.table.get(ii, kk);
                const new_node_idx = base_node_idx + jj * base_coords_num;
                new_connect.table.set(dest_elem_idx, kk, new_node_idx);
            }
        }
    }

    var new_shader: shaderops.ShaderInput = undefined;
    switch (base_mesh.shader) {
        .func => |base_func| {
            var new_uvs: ?NDArray(F) = null;
            if (base_func.uvs) |base_uvs| {
                const base_dims = base_uvs.dims;
                const new_dims = [_]usize{
                    base_dims[0] * copies_num,
                    base_dims[1],
                    base_dims[2],
                };
                new_uvs = try NDArray(F).initFlat(allocator, &new_dims);
                const base_elem_size = base_dims[1] * base_dims[2];
                jj = 0;
                while (jj < copies_num) : (jj += 1) {
                    const dest_start = jj * base_dims[0] * base_elem_size;
                    const uvs_slice = new_uvs.?.slice;
                    @memcpy(
                        uvs_slice[dest_start .. dest_start + base_uvs.slice.len],
                        base_uvs.slice,
                    );
                }
            }
            new_shader = .{
                .func = .{
                    .uvs = new_uvs,
                    .coord_mode = base_func.coord_mode,
                    .builtin = base_func.builtin,
                    .params = base_func.params,
                    .bits = base_func.bits,
                    .scaling = base_func.scaling,
                    .normal_type = base_func.normal_type,
                },
            };
        },
        else => {
            return error.UnsupportedShaderForGeomBenchmark;
        },
    }

    out_mesh.* = .{
        .mesh_type = base_mesh.mesh_type,
        .coords = new_coords,
        .connect = new_connect,
        .disp = null,
        .shader = new_shader,
    };
}

fn getSweepValues(
    total_threads: u16,
    out_vals: []u16,
    out_len: *usize,
) void {
    var val: u16 = 1;
    var idx: usize = 0;
    while (val <= total_threads) : (val *= 2) {
        out_vals[idx] = val;
        idx += 1;
    }
    if (idx > 0 and out_vals[idx - 1] != total_threads) {
        out_vals[idx] = total_threads;
        idx += 1;
    }
    out_len.* = idx;
}

fn runAndRecord(
    allocator: std.mem.Allocator,
    io: std.Io,
    etype: gk.MeshType,
    size_str: []const u8,
    copies_num: usize,
    config: rastcfg.RasterConfig,
    base_mesh: mo.MeshInput,
    base_camera: cam.CameraInput,
    out_timing: *RunTiming,
) !void {
    _ = etype;
    _ = size_str;

    var run_arena = std.heap.ArenaAllocator.init(allocator);
    defer run_arena.deinit();
    const ra = run_arena.allocator();

    var duplicated_mesh: mo.MeshInput = undefined;
    try duplicateMeshInput(ra, base_mesh, copies_num, &duplicated_mesh);

    const num_coords = duplicated_mesh.coords.mat.rows_num;
    const disp_field = try meshio.Field.initAlloc(ra, 64, num_coords, 3);
    duplicated_mesh.disp = disp_field;

    const camera_inputs = try ra.alloc(cam.CameraInput, 64);
    @memset(camera_inputs, base_camera);

    const bench_capture = try ra.alloc(report.FrameBenchCapture, 64);
    for (bench_capture, 0..) |*bc, ii| {
        bc.* = .{
            .camera_idx = 0,
            .frame_idx = ii,
            .bench_log = .{},
        };
    }

    const render_groups = [_]riley.RenderGroupSpec{
        .{
            .io = io,
            .workers = @max(@as(u16, 1), config.total_threads),
        },
    };

    const out_path: ?[]const u8 = if (config.save_strategy == .disk or
        config.save_strategy == .both)
        "out/debug_images"
    else
        null;

    const e2e_start = Timestamp.now(io, .awake);

    var image_arr = try riley.rasterReport(
        ra,
        &render_groups,
        camera_inputs[0..1],
        &[_]mo.MeshInput{duplicated_mesh},
        config,
        out_path,
        bench_capture,
    );

    if (image_arr) |*arr| {
        ra.free(arr.slice);
        arr.deinit(ra);
    }

    const e2e_end = Timestamp.now(io, .awake);
    const e2e_ms = @as(F, @floatFromInt(
        e2e_start.durationTo(e2e_end).raw.nanoseconds,
    )) / 1e6;

    var total_geom_prep_ns: F = 0;
    var total_active_time_ns: F = 0;
    for (bench_capture) |bc| {
        total_geom_prep_ns += bc.bench_log.frame_times.geometry_prep;
        total_active_time_ns += bc.bench_log.frame_times.active_time;
    }
    const avg_geom_prep_ms = total_geom_prep_ns / 1e6 / 64.0;
    const avg_active_frame_ms = total_active_time_ns / 1e6 / 64.0;

    out_timing.* = .{
        .geom_prep_ms = avg_geom_prep_ms,
        .active_frame_ms = avg_active_frame_ms,
        .total_time_ms = e2e_ms,
    };
}

pub fn main(init: std.process.Init) !void {
    const outer_alloc = init.gpa;

    var base_raster_config = tcfg.getRasterConfig(.bench);
    base_raster_config.image_save_opts = &[_]iio.ImageSaveOpts{
        .{ .format = .bmp, .bits = 8, .scaling = .auto },
    };
    var filtered_args: std.ArrayList([*:0]const u8) = .empty;
    defer filtered_args.deinit(outer_alloc);
    try filtered_args.append(outer_alloc, init.minimal.args.vector[0]);

    var element_filter: ?gk.MeshType = null;
    var size_filter: ?[]const u8 = null;
    var skip_arg = false;

    for (init.minimal.args.vector[1..], 1..) |raw_arg, ii| {
        if (skip_arg) {
            skip_arg = false;
            continue;
        }
        const arg = std.mem.span(raw_arg);
        if (std.mem.eql(u8, arg, "--element")) {
            if (ii + 1 >= init.minimal.args.vector.len) {
                std.debug.print("Missing value for --element\n", .{});
                return error.MissingArgumentValue;
            }
            const val = std.mem.span(init.minimal.args.vector[ii + 1]);
            skip_arg = true;
            if (std.mem.eql(u8, val, "tri3")) {
                element_filter = .tri3;
            } else if (std.mem.eql(u8, val, "tri6")) {
                element_filter = .tri6;
            } else if (std.mem.eql(u8, val, "quad4ibi")) {
                element_filter = .quad4ibi;
            } else if (std.mem.eql(u8, val, "quad4newton")) {
                element_filter = .quad4newton;
            } else if (std.mem.eql(u8, val, "quad8")) {
                element_filter = .quad8;
            } else if (std.mem.eql(u8, val, "quad9")) {
                element_filter = .quad9;
            } else {
                std.debug.print("Unknown element type: {s}\n", .{val});
                return error.UnknownElementFilter;
            }
        } else if (std.mem.eql(u8, arg, "--size")) {
            if (ii + 1 >= init.minimal.args.vector.len) {
                std.debug.print("Missing value for --size\n", .{});
                return error.MissingArgumentValue;
            }
            const val = std.mem.span(init.minimal.args.vector[ii + 1]);
            skip_arg = true;
            size_filter = val;
        } else {
            try filtered_args.append(outer_alloc, init.minimal.args.vector[ii]);
        }
    }

    var default_bench_args = benchargs.defaultBenchArgs(
        DEFAULT_OUT_DIR,
        base_raster_config,
    );
    default_bench_args.image_out_dir = DEFAULT_IMAGE_OUT_DIR;
    default_bench_args.pixels_num = DEFAULT_PIXELS_NUM;
    default_bench_args.sub_sample = DEFAULT_SUB_SAMPLE;

    const bench_args = try benchargs.parseArgsWithDefaults(
        filtered_args.items,
        default_bench_args,
    );

    var threaded_io = riley.getThreadedIo(
        outer_alloc,
        init.minimal,
        bench_args.total_threads,
    );
    defer threaded_io.deinit();
    const io = threaded_io.io();

    const render_defaults = common.BenchRenderDefaults{
        .pixels_num = bench_args.pixels_num,
        .sub_sample = bench_args.sub_sample,
        .focal_leng = DEFAULT_FOCAL_LENG,
        .pixels_size = DEFAULT_PIXELS_SIZE,
        .fov_scale = DEFAULT_FOV_SCALE,
        .rot = DEFAULT_ROT,
    };

    const texture_grey = try iio.loadImage(
        u8,
        1,
        outer_alloc,
        io,
        DEFAULT_TEX_GREY_PATH,
        .bmp,
    );
    defer texture_grey.deinit(outer_alloc);
    const texture_rgb = try iio.loadImage(
        u8,
        3,
        outer_alloc,
        io,
        DEFAULT_TEX_RGB_PATH,
        .bmp,
    );
    defer texture_rgb.deinit(outer_alloc);

    const total_threads = bench_args.total_threads;
    std.debug.print(
        "Starting Thread Geom Scaling Benchmark ({d}x{d}, {d} threads)...\n",
        .{
            bench_args.pixels_num[0],
            bench_args.pixels_num[1],
            total_threads,
        },
    );

    const bench_raster_config = benchargs.applyRasterConfig(
        base_raster_config,
        bench_args,
    );

    const cwd = std.Io.Dir.cwd();
    cwd.createDir(io, "out", .default_dir) catch |err| {
        if (err != error.PathAlreadyExists) return err;
    };

    const out_path = "out/bench_thread_geom.csv";
    var file = try cwd.createFile(io, out_path, .{});
    defer file.close(io);

    var write_buf: [4096]u8 = undefined;
    var buffered_writer = file.writer(io, &write_buf);
    const writer = &buffered_writer.interface;

    try writer.print(
        "experiment,element_type,mesh_size,mesh_copies,total_threads," ++
            "geom_scheduling_mode,max_geom_jobs_in_flight," ++
            "max_geom_workers_per_job,geom_prep_time_ms,active_frame_time_ms,total_time_ms\n",
        .{},
    );

    const tex_func_case = common.TexFuncCase{
        .builtin = .sinusoidal,
        .coord_mode = .uv,
    };

    for (EVAL_ELEMENTS) |etype| {
        for (MESH_SIZES) |size_str| {
            for (MESH_COPIES_LIST) |copies_num| {
                if (element_filter) |filter| {
                    if (filter != etype) continue;
                }
                if (size_filter) |filter| {
                    if (!std.mem.eql(u8, filter, size_str)) continue;
                }

                std.debug.print(
                    "Running mt={s}, size={s}, copies={d}\n",
                    .{ @tagName(etype), size_str, copies_num },
                );

                var arena = std.heap.ArenaAllocator.init(outer_alloc);
                defer arena.deinit();
                const aa = arena.allocator();

                var data_dir_buf: [256]u8 = undefined;
                const data_dir = try std.fmt.bufPrint(
                    &data_dir_buf,
                    "data/bench/{s}_geom_{s}",
                    .{ @tagName(etype), size_str },
                );

                var base_mesh = try common.loadBenchmarkMeshInput(
                    u8,
                    aa,
                    io,
                    etype,
                    .func,
                    null,
                    tex_func_case,
                    data_dir,
                    texture_grey,
                    texture_rgb,
                );

                const roi_pos = sceneops.boundsCenter(&base_mesh.coords);
                const cam_pos = cameraops.posFillFrameFromRot(
                    &base_mesh.coords,
                    render_defaults.pixels_num,
                    render_defaults.pixels_size,
                    render_defaults.focal_leng,
                    render_defaults.rot,
                    render_defaults.fov_scale,
                );
                const camera = try cam.CameraPrepared.init(
                    aa,
                    .{
                        .pixels_num = render_defaults.pixels_num,
                        .pixels_size = render_defaults.pixels_size,
                        .pos_world = cam_pos,
                        .rot_world = render_defaults.rot,
                        .roi_cent_world = roi_pos,
                        .focal_length = render_defaults.focal_leng,
                        .sub_sample = render_defaults.sub_sample,
                    },
                );
                const base_camera = cam.CameraInput{
                    .pixels_num = camera.pixels_num,
                    .pixels_size = camera.pixels_size,
                    .pos_world = camera.pos_world,
                    .rot_world = camera.rot_world,
                    .roi_cent_world = camera.roi_cent_world,
                    .focal_length = camera.focal_length,
                    .sub_sample = camera.sub_sample,
                    .distortion = camera.distortion,
                };

                // --- Experiment 1 ---
                {
                    var sweep_vals: [32]u16 = undefined;
                    var sweep_len: usize = 0;
                    getSweepValues(total_threads, &sweep_vals, &sweep_len);

                    var ii: usize = 0;
                    while (ii < sweep_len) : (ii += 1) {
                        const max_workers = sweep_vals[ii];
                        var run_config = bench_raster_config;
                        run_config.geom_scheduling_mode = .spread;
                        run_config.max_geom_jobs_in_flight_per_group = 1;
                        run_config.max_geom_workers_per_job = max_workers;
                        run_config.max_raster_workers_per_job = total_threads;

                        var timing = RunTiming{
                            .geom_prep_ms = 0.0,
                            .active_frame_ms = 0.0,
                            .total_time_ms = 0.0,
                        };
                        try runAndRecord(
                            outer_alloc,
                            io,
                            etype,
                            size_str,
                            copies_num,
                            run_config,
                            base_mesh,
                            base_camera,
                            &timing,
                        );

                        try writer.print(
                            "1,{s},{s},{d},{d},spread,1,{d},{d:.3},{d:.3},{d:.3}\n",
                            .{
                                @tagName(etype),
                                size_str,
                                copies_num,
                                total_threads,
                                max_workers,
                                timing.geom_prep_ms,
                                timing.active_frame_ms,
                                timing.total_time_ms,
                            },
                        );
                    }
                }

                // --- Experiment 2 ---
                {
                    var sweep_vals: [32]u16 = undefined;
                    var sweep_len: usize = 0;
                    getSweepValues(total_threads, &sweep_vals, &sweep_len);

                    const modes = [_]rastcfg.GeometrySchedulingMode{ .spread, .pack, .auto };
                    for (modes) |mode| {
                        var jj: usize = 0;
                        while (jj < sweep_len) : (jj += 1) {
                            const max_jobs = sweep_vals[jj];
                            var kk: usize = 0;
                            while (kk < sweep_len) : (kk += 1) {
                                const max_workers = sweep_vals[kk];

                                var run_config = bench_raster_config;
                                run_config.geom_scheduling_mode = mode;
                                run_config.max_geom_jobs_in_flight_per_group = max_jobs;
                                run_config.max_geom_workers_per_job = max_workers;
                                run_config.max_raster_workers_per_job = total_threads;

                                var timing = RunTiming{
                                    .geom_prep_ms = 0.0,
                                    .active_frame_ms = 0.0,
                                    .total_time_ms = 0.0,
                                };
                                try runAndRecord(
                                    outer_alloc,
                                    io,
                                    etype,
                                    size_str,
                                    copies_num,
                                    run_config,
                                    base_mesh,
                                    base_camera,
                                    &timing,
                                );

                                try writer.print(
                                    "2,{s},{s},{d},{d},{s},{d},{d},{d:.3},{d:.3},{d:.3}\n",
                                    .{
                                        @tagName(etype),
                                        size_str,
                                        copies_num,
                                        total_threads,
                                        @tagName(mode),
                                        max_jobs,
                                        max_workers,
                                        timing.geom_prep_ms,
                                        timing.active_frame_ms,
                                        timing.total_time_ms,
                                    },
                                );
                            }
                        }
                    }
                }

                // --- Experiment 3 ---
                {
                    var jj: u16 = 1;
                    while (jj <= total_threads) : (jj += 1) {
                        if (total_threads % jj == 0) {
                            const ww = total_threads / jj;

                            var run_config = bench_raster_config;
                            run_config.geom_scheduling_mode = .spread;
                            run_config.max_geom_jobs_in_flight_per_group = jj;
                            run_config.max_geom_workers_per_job = ww;
                            run_config.max_raster_workers_per_job = total_threads;

                            var timing = RunTiming{
                                .geom_prep_ms = 0.0,
                                .active_frame_ms = 0.0,
                                .total_time_ms = 0.0,
                            };
                            try runAndRecord(
                                outer_alloc,
                                io,
                                etype,
                                size_str,
                                copies_num,
                                run_config,
                                base_mesh,
                                base_camera,
                                &timing,
                            );

                            try writer.print(
                                "3,{s},{s},{d},{d},spread,{d},{d},{d:.3},{d:.3},{d:.3}\n",
                                .{
                                    @tagName(etype),
                                    size_str,
                                    copies_num,
                                    total_threads,
                                    jj,
                                    ww,
                                    timing.geom_prep_ms,
                                    timing.active_frame_ms,
                                    timing.total_time_ms,
                                },
                            );
                        }
                    }
                }
            }
        }
    }

    try buffered_writer.interface.flush();
    std.debug.print("Benchmark completed! Results written to {s}\n", .{out_path});
}
