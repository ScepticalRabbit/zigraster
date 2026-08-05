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
const benchstats = @import("dev_support/benchstats.zig");
const common = @import("dev_support/benchcommon.zig");
const tcfg = @import("dev_support/testconfig.zig");
const cam = @import("riley/zig/camera.zig");
const cameraops = @import("riley/zig/cameraops.zig");
const buildconfig = @import("riley/zig/buildconfig.zig");
const gk = @import("riley/zig/geometrykernels.zig");
const iio = @import("riley/zig/imageio.zig");
const meshio = @import("riley/zig/meshio.zig");
const mo = @import("riley/zig/meshpipeline.zig");
const rastcfg = @import("riley/zig/rasterconfig.zig");
const report = @import("riley/zig/report.zig");
const riley = @import("riley/zig/riley.zig");
const so = @import("riley/zig/shaderops.zig");
const orch = @import("dev_support/orchestration.zig");
const Rotation = @import("riley/zig/rotation.zig").Rotation;
const sceneops = @import("riley/zig/sceneops.zig");
const Timestamp = std.Io.Clock.Timestamp;
const F = buildconfig.F;

const DEFAULT_OUT_DIR = "out/bench_stats_cam";
const DEFAULT_IMAGE_OUT_DIR = "out/bench_images_cam";
const DEFAULT_DATA_DIR_SUFFIX = "sphere2000";
const DEFAULT_PIXELS_NUM = [2]u32{ 1600, 1000 };
const DEFAULT_SUB_SAMPLE: u32 = 1;
const DEFAULT_FOCAL_LENG: F = @floatCast(50.0e-3);
const DEFAULT_PIXELS_SIZE = [2]F{
    @floatCast(5.3e-6),
    @floatCast(5.3e-6),
};
const DEFAULT_FOV_SCALE: F = 1.0;
const DEFAULT_ROT = Rotation.init(0, 0, 0);
const DEFAULT_BACKGROUND_VALUE: F = 0.5;
const CHECKER_SQUARES_PER_AXIS: F = 36.0;

const DistortionCase = struct {
    tag: []const u8,
    model: cam.DistortionModel,
};

const PsfCase = struct {
    tag: []const u8,
    psf: cam.PointSpreadFunc,
};

const mesh_types = [_]gk.MeshType{
    .tri3,
    .tri6,
    // Add more element paths here later if broader camera-path coverage is
    // useful, e.g. .quad4ibi, .quad4newton, .quad8, .quad9.
};

const shader_types = [_]common.ShaderType{
    .func,
    // Re-enable RGB camera-path benchmarking later by adding .func_rgb here.
    // The checker input path below already supports it.
};

const distortion_cases = [_]DistortionCase{
    .{ .tag = "none", .model = .none },
    .{
        .tag = "brown_conrady",
        .model = .{ .brown_conrady = .{
            .k1 = -0.08,
            .k2 = 0.01,
            .k3 = -0.002,
            .p1 = 0.0004,
            .p2 = -0.0007,
        } },
    },
    .{
        .tag = "brown_conrady_ext",
        .model = .{ .brown_conrady_ext = .{
            .k1 = -0.09,
            .k2 = 0.012,
            .k3 = -0.0015,
            .k4 = 0.004,
            .k5 = -0.0008,
            .k6 = 0.00015,
            .p1 = 0.0005,
            .p2 = -0.0006,
        } },
    },
};

const psf_cases = [_]PsfCase{
    .{
        .tag = "pixel_box",
        .psf = .{ .pixel_box = .{} },
    },
    .{
        .tag = "gaussian_sep",
        .psf = .{ .gaussian = .{
            .sigma_px = 0.6,
            .supp_rad_px = 2.0,
            .separable = .yes,
        } },
    },
    .{
        .tag = "gaussian_nonsep",
        .psf = .{ .gaussian = .{
            .sigma_px = 0.6,
            .supp_rad_px = 2.0,
            .separable = .no,
        } },
    },
    .{
        .tag = "anisotropic_gaussian",
        .psf = .{ .anisotropic_gaussian = .{
            .sigma_x_px = 1.2,
            .sigma_y_px = 0.2,
            .theta_rad = std.math.pi / 6.0,
            .supp_rad_px = 3.0,
            .separable = .no,
        } },
    },
};

const tex_func_case = common.TexFuncCase{
    .builtin = .checker,
    .coord_mode = .uv,
};

const checker_params = so.FuncShaderParams{
    .coord_scale = .{
        CHECKER_SQUARES_PER_AXIS,
        CHECKER_SQUARES_PER_AXIS,
    },
    .coord_offset = .{ 0.0, 0.0 },
};

const CamCaseMeta = struct {
    case_name: []const u8,
    mesh_type: gk.MeshType,
    shader_type: common.ShaderType,
    coord_mode: common.TexFuncCoordMode,
    distortion_tag: []const u8,
    psf_tag: []const u8,
};

fn calcCaseName(
    allocator: std.mem.Allocator,
    mesh_type: gk.MeshType,
    shader_type: common.ShaderType,
    distortion_case: DistortionCase,
    psf_case: PsfCase,
) ![]u8 {
    return std.fmt.allocPrint(
        allocator,
        "{s}_{s}_{s}_{s}_{s}",
        .{
            @tagName(mesh_type),
            @tagName(shader_type),
            @tagName(tex_func_case.coord_mode),
            distortion_case.tag,
            psf_case.tag,
        },
    );
}

fn benchCamCSVHeader() []const u8 {
    return "Case,Element,Shader,CoordMode,DistortionModel,PSF,Interpolator," ++
        "Total Elems,Vis Elems,Total Px,Shaded Px," ++
        "Geom Time [ms],Raster Time [ms],Save Time [ms],Frame Time [ms]," ++
        "E2E Time [ms],Geom TP [MElem/s],Raster TP [MPx/s],Frame TP [MPx/s]," ++
        "E2E TP [MPx/s]," ++
        "Case_end,Element_end,Shader_end,CoordMode_end,DistortionModel_end,PSF_end\n";
}

fn formatBenchCamCSVRow(
    allocator: std.mem.Allocator,
    meta: CamCaseMeta,
    values: common.BenchmarkCSVValues,
) ![]u8 {
    return std.fmt.allocPrint(
        allocator,
        "{s},{s},{s},{s},{s},{s},{s}," ++
            "{d:.6},{d:.6},{d:.6},{d:.6}," ++
            "{d:.6},{d:.6},{d:.6},{d:.6},{d:.6}," ++
            "{d:.6},{d:.6},{d:.6},{d:.6}," ++
            "{s},{s},{s},{s},{s},{s}\n",
        .{
            meta.case_name,
            @tagName(meta.mesh_type),
            @tagName(meta.shader_type),
            @tagName(meta.coord_mode),
            meta.distortion_tag,
            meta.psf_tag,
            @tagName(tex_func_case.builtin),
            values.total_elems,
            values.vis_elems,
            values.total_px,
            values.shaded_px,
            values.geom,
            values.raster,
            values.save_frame,
            values.frame,
            values.e2e,
            values.geom_tpx,
            values.raster_tpx,
            values.frame_tpx,
            values.e2e_tpx,
            meta.case_name,
            @tagName(meta.mesh_type),
            @tagName(meta.shader_type),
            @tagName(meta.coord_mode),
            meta.distortion_tag,
            meta.psf_tag,
        },
    );
}

fn writeBenchCamStatsCSV(
    allocator: std.mem.Allocator,
    io: std.Io,
    out_dir_base: []const u8,
    stats_list: []const common.BenchStats,
    metas: []const CamCaseMeta,
    kind: common.BenchmarkCSVKind,
    file_name: []const u8,
) !void {
    std.debug.assert(stats_list.len == metas.len);

    const csv_path = try std.fs.path.join(
        allocator,
        &[_][]const u8{ out_dir_base, file_name },
    );
    defer allocator.free(csv_path);

    const cwd = std.Io.Dir.cwd();
    cwd.createDir(
        io,
        out_dir_base,
        .default_dir,
    ) catch |err| if (err != error.PathAlreadyExists) return err;

    var file = try cwd.createFile(io, csv_path, .{});
    defer file.close(io);

    var write_buf: [4096]u8 = undefined;
    var buffered_writer = file.writer(io, &write_buf);
    const writer = &buffered_writer.interface;

    try writer.writeAll(benchCamCSVHeader());

    for (stats_list, metas) |stats, meta| {
        const row = try formatBenchCamCSVRow(
            allocator,
            meta,
            common.calcBenchmarkCSVValuesFromStats(stats, kind),
        );
        defer allocator.free(row);
        try writer.writeAll(row);
    }

    try buffered_writer.flush();
}

fn writeBenchCamReport(
    allocator: std.mem.Allocator,
    io: std.Io,
    out_dir_base: []const u8,
    stats_list: []const common.BenchStats,
    metas: []const CamCaseMeta,
) !void {
    try writeBenchCamStatsCSV(
        allocator,
        io,
        out_dir_base,
        stats_list,
        metas,
        .median,
        "bench_stats_median.csv",
    );
    try writeBenchCamStatsCSV(
        allocator,
        io,
        out_dir_base,
        stats_list,
        metas,
        .min,
        "bench_stats_min.csv",
    );
    try writeBenchCamStatsCSV(
        allocator,
        io,
        out_dir_base,
        stats_list,
        metas,
        .max,
        "bench_stats_max.csv",
    );
    try writeBenchCamStatsCSV(
        allocator,
        io,
        out_dir_base,
        stats_list,
        metas,
        .mad,
        "bench_stats_mad.csv",
    );
    try writeBenchCamStatsCSV(
        allocator,
        io,
        out_dir_base,
        stats_list,
        metas,
        .cov,
        "bench_stats_cov.csv",
    );
}

fn calcTexFuncChannels(shader_type: common.ShaderType) u8 {
    return switch (shader_type) {
        .func_rgb => 3,
        else => 1,
    };
}

fn loadCheckerMeshInput(
    allocator: std.mem.Allocator,
    io: std.Io,
    mesh_type: gk.MeshType,
    shader_type: common.ShaderType,
    data_dir: []const u8,
) !mo.MeshInput {
    const coord_path = try std.fs.path.join(allocator, &[_][]const u8{
        data_dir,
        "coords.csv",
    });
    const conn_path = try std.fs.path.join(allocator, &[_][]const u8{
        data_dir,
        "connect.csv",
    });
    const uv_path = try std.fs.path.join(allocator, &[_][]const u8{
        data_dir,
        "uvs.csv",
    });

    const sim_data = try meshio.loadSimData(
        allocator,
        io,
        coord_path,
        conn_path,
        null,
        null,
    );
    const uvs_raw = try common.loadNDArrayFromCSV(
        allocator,
        io,
        uv_path,
        2,
        false,
    );

    const shader = switch (shader_type) {
        .func => so.ShaderInput{ .func = .{
            .uvs = uvs_raw,
            .coord_mode = .uv,
            .builtin = tex_func_case.builtin,
            .params = checker_params,
            .bits = 8,
            .scaling = .none,
            .normal_type = .none,
        } },
        .func_rgb => so.ShaderInput{ .func_rgb = .{
            .uvs = uvs_raw,
            .coord_mode = .uv,
            .builtin = tex_func_case.builtin,
            .params = checker_params,
            .bits = 8,
            .scaling = .none,
            .normal_type = .none,
        } },
        else => unreachable,
    };

    return .{
        .mesh_type = mesh_type,
        .coords = sim_data.coords,
        .connect = sim_data.connect,
        .disp = null,
        .shader = shader,
    };
}

fn runCameraBenchmarkWithImageOut(
    outer_alloc: std.mem.Allocator,
    io: std.Io,
    mesh_type: gk.MeshType,
    shader_type: common.ShaderType,
    distortion_case: DistortionCase,
    psf_case: PsfCase,
    data_dir: []const u8,
    render_defaults: common.BenchRenderDefaults,
    config: rastcfg.RasterConfig,
    stats_out_dir_base: []const u8,
    image_out_dir_base: []const u8,
) !common.BenchResult {
    var arena = std.heap.ArenaAllocator.init(outer_alloc);
    defer arena.deinit();
    const aa = arena.allocator();

    const mesh_input = try loadCheckerMeshInput(
        aa,
        io,
        mesh_type,
        shader_type,
        data_dir,
    );

    const roi_pos = sceneops.boundsCenter(&mesh_input.coords);
    const cam_pos = cameraops.posFillFrameFromRot(
        &mesh_input.coords,
        render_defaults.pixels_num,
        render_defaults.pixels_size,
        render_defaults.focal_leng,
        render_defaults.rot,
        render_defaults.fov_scale,
    );

    const camera_input = cam.CameraInput{
        .pixels_num = render_defaults.pixels_num,
        .pixels_size = render_defaults.pixels_size,
        .pos_world = cam_pos,
        .rot_world = render_defaults.rot,
        .roi_cent_world = roi_pos,
        .focal_length = render_defaults.focal_leng,
        .sub_sample = render_defaults.sub_sample,
        .distortion = distortion_case.model,
        .psf = psf_case.psf,
    };

    var config_run = config;
    config_run.report = .bench;
    config_run.image_save_mode = if (calcTexFuncChannels(shader_type) == 3)
        .rgb
    else
        .grey;

    if (stats_out_dir_base.len > 0) {
        var out_dir = try orch.openDirEnsured(io, stats_out_dir_base);
        out_dir.close(io);
    }

    const case_name = try calcCaseName(
        aa,
        mesh_type,
        shader_type,
        distortion_case,
        psf_case,
    );
    const out_path = if (image_out_dir_base.len > 0)
        image_out_dir_base
    else if (stats_out_dir_base.len > 0)
        try std.fs.path.join(
            aa,
            &[_][]const u8{ stats_out_dir_base, case_name },
        )
    else
        null;

    if (out_path) |case_out_path| {
        var case_out_dir = try orch.openDirEnsured(io, case_out_path);
        case_out_dir.close(io);
    }

    var bench_capture_storage: [1]report.FrameBenchCapture = undefined;
    const e2e_start = Timestamp.now(io, .awake);
    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = @max(@as(u16, 1), config_run.total_threads) },
    };
    var image_arr = try riley.rasterReport(
        outer_alloc,
        &render_groups,
        &[_]cam.CameraInput{camera_input},
        &[_]mo.MeshInput{mesh_input},
        config_run,
        out_path,
        bench_capture_storage[0..],
    );
    const e2e_end = Timestamp.now(io, .awake);

    if (image_arr) |images| {
        outer_alloc.free(images.slice);
        var images_mut = images;
        images_mut.deinit(outer_alloc);
        image_arr = null;
    }

    const e2e_ms = @as(F, @floatFromInt(
        e2e_start.durationTo(e2e_end).raw.nanoseconds,
    )) / 1e6;
    const geom_ms =
        (bench_capture_storage[0].bench_log.frame_times.geometry_prep +
            bench_capture_storage[0].bench_log.frame_times.tile_overlap) / 1e6;
    const raster_ms =
        bench_capture_storage[0].bench_log.frame_times.raster_loop / 1e6;
    const metrics = common.calcMetrics(
        mesh_type,
        camera_input.pixels_num,
        camera_input.sub_sample,
        e2e_ms,
        bench_capture_storage[0].bench_log.frame_times,
        bench_capture_storage[0].bench_log,
    );

    return .{
        .e2e_ms = e2e_ms,
        .geom_ms = geom_ms,
        .raster_ms = raster_ms,
        .cam_ms = bench_capture_storage[0]
            .bench_log.frame_times.cam_invert / 1e6,
        .resolve_ms = bench_capture_storage[0]
            .bench_log.frame_times.scratch_resolve / 1e6,
        .fps = if (e2e_ms > 0.0) 1000.0 / e2e_ms else 0.0,
        .total_elems = bench_capture_storage[0].bench_log.total_elems,
        .vis_elems = bench_capture_storage[0].bench_log.vis_elems,
        .total_px = @as(u64, camera_input.pixels_num[0]) *
            @as(u64, camera_input.pixels_num[1]),
        .shaded_px = bench_capture_storage[0].bench_log.total_shaded_px,
        .metrics = metrics,
        .pipeline_times = bench_capture_storage[0].bench_log.frame_times,
        .image = null,
    };
}

pub fn main(init: std.process.Init) !void {
    const outer_alloc = init.gpa;

    var base_raster_config = tcfg.getRasterConfig(.bench);
    base_raster_config.image_save_opts = &[_]iio.ImageSaveOpts{
        .{ .format = .bmp, .bits = 8, .scaling = .auto },
    };
    base_raster_config.background_value = DEFAULT_BACKGROUND_VALUE;
    var default_bench_args = benchargs.defaultBenchArgs(
        DEFAULT_OUT_DIR,
        base_raster_config,
    );
    default_bench_args.image_out_dir = DEFAULT_IMAGE_OUT_DIR;
    default_bench_args.pixels_num = DEFAULT_PIXELS_NUM;
    default_bench_args.sub_sample = DEFAULT_SUB_SAMPLE;

    const bench_args = try benchargs.parseArgsWithDefaults(
        init.minimal.args.vector,
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

    var stats = try benchstats.BenchStatsCollector.init(
        outer_alloc,
        bench_args.runs,
    );
    defer stats.deinit(outer_alloc);
    var case_metas = std.ArrayList(CamCaseMeta).empty;
    defer {
        for (case_metas.items) |meta| {
            outer_alloc.free(meta.case_name);
        }
        case_metas.deinit(outer_alloc);
    }

    std.debug.print(
        "Starting Camera Benchmark ({d}x{d}, {d} runs per case, {d} threads)...\n",
        .{
            bench_args.pixels_num[0],
            bench_args.pixels_num[1],
            bench_args.runs,
            bench_args.total_threads,
        },
    );

    const bench_raster_config = benchargs.applyRasterConfig(
        base_raster_config,
        bench_args,
    );
    const actual_tile_size = common.calcActualTileSize(
        bench_raster_config,
        bench_args.pixels_num,
        bench_args.sub_sample,
        0,
    );
    const render_group_workers = [_]u16{bench_args.total_threads};
    try common.writeBenchmarkConfig(
        outer_alloc,
        io,
        bench_args.out_dir,
        bench_args.image_out_dir,
        "benchcam.zig",
        init.minimal.args.vector,
        bench_args.subpixel_center_map,
        bench_raster_config,
        render_group_workers[0..],
        bench_args.pixels_num,
        bench_args.sub_sample,
        bench_args.runs,
        DEFAULT_FOV_SCALE,
        actual_tile_size,
    );

    for (mesh_types) |mesh_type| {
        for (shader_types) |shader_type| {
            for (distortion_cases) |distortion_case| {
                for (psf_cases) |psf_case| {
                    const case_name = try calcCaseName(
                        outer_alloc,
                        mesh_type,
                        shader_type,
                        distortion_case,
                        psf_case,
                    );
                    defer outer_alloc.free(case_name);
                    std.debug.print("Case: {s}\n", .{case_name});

                    var case_samples = try benchstats.CaseSamples.init(
                        outer_alloc,
                        bench_args.runs,
                    );
                    defer case_samples.deinit(outer_alloc);

                    var data_dir_buf: [256]u8 = undefined;
                    const data_dir = try std.fmt.bufPrint(
                        &data_dir_buf,
                        "data/bench/{s}_{s}",
                        .{ @tagName(mesh_type), DEFAULT_DATA_DIR_SUFFIX },
                    );

                    for (0..bench_args.runs) |rr| {
                        const run_out_dir_base = if (bench_args.save_strategy == .disk or
                            bench_args.save_strategy == .both)
                            bench_args.out_dir
                        else
                            "";
                        const raster_config = benchargs.applyRasterConfig(
                            base_raster_config,
                            bench_args,
                        );

                        var res = try runCameraBenchmarkWithImageOut(
                            outer_alloc,
                            io,
                            mesh_type,
                            shader_type,
                            distortion_case,
                            psf_case,
                            data_dir,
                            render_defaults,
                            raster_config,
                            run_out_dir_base,
                            bench_args.image_out_dir,
                        );
                        defer res.deinit(outer_alloc);

                        try stats.appendRunResult(
                            outer_alloc,
                            rr,
                            case_name,
                            mesh_type,
                            shader_type,
                            null,
                            tex_func_case,
                            res,
                        );
                        try stats.writeRunCSV(
                            outer_alloc,
                            io,
                            bench_args.out_dir,
                            rr,
                        );
                        case_samples.record(rr, res);
                    }

                    try stats.appendCaseStats(
                        outer_alloc,
                        case_name,
                        mesh_type,
                        shader_type,
                        null,
                        tex_func_case,
                        &case_samples,
                    );
                    try case_metas.append(
                        outer_alloc,
                        .{
                            .case_name = try outer_alloc.dupe(u8, case_name),
                            .mesh_type = mesh_type,
                            .shader_type = shader_type,
                            .coord_mode = tex_func_case.coord_mode,
                            .distortion_tag = distortion_case.tag,
                            .psf_tag = psf_case.tag,
                        },
                    );
                }
            }
        }
    }

    try stats.writeRunCSVs(outer_alloc, io, bench_args.out_dir);
    try writeBenchCamReport(
        outer_alloc,
        io,
        bench_args.out_dir,
        stats.stats_list.items,
        case_metas.items,
    );
}
