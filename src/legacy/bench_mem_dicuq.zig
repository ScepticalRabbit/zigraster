const std = @import("std");

const benchargs = @import("dev_support/benchargs.zig");
const benchdicuq = @import("dev_support/benchdicuq.zig");
const buildconfig = @import("riley/zig/buildconfig.zig");
const riley = @import("riley/zig/riley.zig");
const report = @import("riley/zig/report.zig");
const F = buildconfig.F;

const DEFAULT_OUT_DIR = "out/bench_stats_mem_dicuq";
const DEFAULT_DATA_DIR = "data/FE/platehole3d_6mr_63f/";
const DEFAULT_PIXELS_NUM = [2]u32{ 2464, 2056 };
const DEFAULT_SUB_SAMPLE: u32 = 2;
const DEFAULT_FOCAL_LENG: F = @floatCast(50.0e-3);
const DEFAULT_PIXELS_SIZE = [2]F{
    @floatCast(3.45e-6),
    @floatCast(3.45e-6),
};
const DEFAULT_FOV_SCALE: F = @floatCast(0.65);
const DEFAULT_STEREO_ANG: F = 20.0;
const DEFAULT_TEX_PATH = "texture/speckle.bmp";
const DEFAULT_TOTAL_THREADS: u16 = 1;
const DEFAULT_RUNS: usize = 1;

const TimeSums = struct {
    geom_ms: f64,
    raster_ms: f64,
    save_ms: f64,
    frame_ms: f64,
};

fn sumFrameTimes(
    capture: []const report.FrameBenchCapture,
) TimeSums {
    var sums = TimeSums{
        .geom_ms = 0.0,
        .raster_ms = 0.0,
        .save_ms = 0.0,
        .frame_ms = 0.0,
    };
    for (capture) |frame_capture| {
        sums.geom_ms += frame_capture.bench_log.frame_times.geometry_prep / 1e6;
        sums.raster_ms += frame_capture.bench_log.frame_times.raster_loop / 1e6;
        sums.save_ms += frame_capture.bench_log.frame_times.save_frame / 1e6;
        sums.frame_ms += frame_capture.bench_log.frame_times.active_time / 1e6;
    }
    return sums;
}

fn runCase(
    allocator: std.mem.Allocator,
    io: std.Io,
    render_groups: []const riley.RenderGroupSpec,
    camera_inputs: []const @import("riley/zig/camera.zig").CameraInput,
    mesh_input: @import("riley/zig/meshpipeline.zig").MeshInput,
    config: riley.RasterConfig,
    out_dir_path: ?[]const u8,
    case_name: []const u8,
) !void {
    const frame_count = if (mesh_input.disp) |disp|
        disp.getTimeN() * camera_inputs.len
    else
        camera_inputs.len;
    const bench_capture = try allocator.alloc(
        report.FrameBenchCapture,
        frame_count,
    );
    defer allocator.free(bench_capture);

    const start = std.Io.Clock.Timestamp.now(io, .awake);
    const images = try riley.rasterReport(
        allocator,
        render_groups,
        camera_inputs,
        &[_]@TypeOf(mesh_input){mesh_input},
        config,
        out_dir_path,
        bench_capture,
    );
    const end = std.Io.Clock.Timestamp.now(io, .awake);

    if (images) |result| {
        allocator.free(result.slice);
        var result_mut = result;
        result_mut.deinit(allocator);
    }

    const e2e_ms = @as(
        f64,
        @floatFromInt(start.durationTo(end).raw.nanoseconds),
    ) / 1e6;
    const sums = sumFrameTimes(bench_capture);
    std.debug.print(
        "case={s} e2e_ms={d:.3} geom_ms_sum={d:.3} " ++
            "raster_ms_sum={d:.3} save_ms_sum={d:.3} frame_ms_sum={d:.3}\n",
        .{
            case_name,
            e2e_ms,
            sums.geom_ms,
            sums.raster_ms,
            sums.save_ms,
            sums.frame_ms,
        },
    );
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const base_config = benchdicuq.getBaseRasterConfig();
    var defaults = benchargs.defaultBenchArgs(DEFAULT_OUT_DIR, base_config);
    defaults.total_threads = DEFAULT_TOTAL_THREADS;
    defaults.runs = DEFAULT_RUNS;
    defaults.pixels_num = DEFAULT_PIXELS_NUM;
    defaults.sub_sample = DEFAULT_SUB_SAMPLE;
    defaults.save_strategy = .memory;

    const bench_args = try benchargs.parseArgsWithDefaults(
        init.minimal.args.vector,
        defaults,
    );

    var threaded_io = riley.getThreadedIo(
        allocator,
        init.minimal,
        bench_args.total_threads,
    );
    defer threaded_io.deinit();
    const io = threaded_io.io();

    const samp_cfg = try benchdicuq.makeSampleConfig(bench_args);

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    const prepared = try benchdicuq.prepareBenchmark(
        aa,
        io,
        .{
            .data_dir = DEFAULT_DATA_DIR,
            .pixels_num = bench_args.pixels_num,
            .sub_sample = bench_args.sub_sample,
            .focal_leng = DEFAULT_FOCAL_LENG,
            .pixels_size = DEFAULT_PIXELS_SIZE,
            .fov_scale = DEFAULT_FOV_SCALE,
            .stereo_ang = DEFAULT_STEREO_ANG,
            .tex_path = DEFAULT_TEX_PATH,
        },
        samp_cfg,
    );

    const render_groups = [_]riley.RenderGroupSpec{
        .{
            .io = io,
            .workers = bench_args.total_threads,
        },
    };

    for (0..bench_args.runs) |run_idx| {
        std.debug.print("run={d}\n", .{run_idx});

        var disk_config = benchargs.applyRasterConfig(base_config, bench_args);
        disk_config.save_strategy = .disk;
        disk_config.report = .off;
        try runCase(
            allocator,
            io,
            render_groups[0..],
            &prepared.camera_inputs,
            prepared.mesh_input,
            disk_config,
            bench_args.out_dir,
            "disk_f64",
        );

        var memory_f64_config = disk_config;
        memory_f64_config.save_strategy = .memory;
        try runCase(
            allocator,
            io,
            render_groups[0..],
            &prepared.camera_inputs,
            prepared.mesh_input,
            memory_f64_config,
            null,
            "memory_f64",
        );
    }
}
