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
const F = buildconfig.F;
const common = @import("benchcommon.zig");
const gk = @import("../riley/zig/geometrykernels.zig");
const texops = @import("../riley/zig/textureops.zig");

pub const CaseSamples = struct {
    total_elems_vals: []F,
    vis_elems_vals: []F,
    total_px_vals: []F,
    shaded_px_vals: []F,
    e2e_times: []F,
    geom_times: []F,
    raster_times: []F,
    cam_times: []F,
    elem_loop_times: []F,
    resolve_times: []F,
    save_frame_times: []F,
    frame_times: []F,
    geom_tpx_vals: []F,
    raster_tpx_vals: []F,
    frame_tpx_vals: []F,
    e2e_tpx_vals: []F,
    setup_frame_buff_times: []F,
    prepare_frame_context_times: []F,
    geom_coord_ops_times: []F,
    geom_cull_ops_times: []F,
    geom_prep_hulls_shaders_times: []F,
    geom_remap_inds_times: []F,

    pub fn init(
        allocator: std.mem.Allocator,
        runs: usize,
    ) !CaseSamples {
        return .{
            .total_elems_vals = try allocator.alloc(F, runs),
            .vis_elems_vals = try allocator.alloc(F, runs),
            .total_px_vals = try allocator.alloc(F, runs),
            .shaded_px_vals = try allocator.alloc(F, runs),
            .e2e_times = try allocator.alloc(F, runs),
            .geom_times = try allocator.alloc(F, runs),
            .raster_times = try allocator.alloc(F, runs),
            .cam_times = try allocator.alloc(F, runs),
            .elem_loop_times = try allocator.alloc(F, runs),
            .resolve_times = try allocator.alloc(F, runs),
            .save_frame_times = try allocator.alloc(F, runs),
            .frame_times = try allocator.alloc(F, runs),
            .geom_tpx_vals = try allocator.alloc(F, runs),
            .raster_tpx_vals = try allocator.alloc(F, runs),
            .frame_tpx_vals = try allocator.alloc(F, runs),
            .e2e_tpx_vals = try allocator.alloc(F, runs),
            .setup_frame_buff_times = try allocator.alloc(F, runs),
            .prepare_frame_context_times = try allocator.alloc(F, runs),
            .geom_coord_ops_times = try allocator.alloc(F, runs),
            .geom_cull_ops_times = try allocator.alloc(F, runs),
            .geom_prep_hulls_shaders_times = try allocator.alloc(F, runs),
            .geom_remap_inds_times = try allocator.alloc(F, runs),
        };
    }

    pub fn deinit(
        self: *const CaseSamples,
        allocator: std.mem.Allocator,
    ) void {
        allocator.free(self.e2e_times);
        allocator.free(self.total_elems_vals);
        allocator.free(self.vis_elems_vals);
        allocator.free(self.total_px_vals);
        allocator.free(self.shaded_px_vals);
        allocator.free(self.geom_times);
        allocator.free(self.raster_times);
        allocator.free(self.cam_times);
        allocator.free(self.elem_loop_times);
        allocator.free(self.resolve_times);
        allocator.free(self.save_frame_times);
        allocator.free(self.frame_times);
        allocator.free(self.geom_tpx_vals);
        allocator.free(self.raster_tpx_vals);
        allocator.free(self.frame_tpx_vals);
        allocator.free(self.e2e_tpx_vals);
        allocator.free(self.setup_frame_buff_times);
        allocator.free(self.prepare_frame_context_times);
        allocator.free(self.geom_coord_ops_times);
        allocator.free(self.geom_cull_ops_times);
        allocator.free(self.geom_prep_hulls_shaders_times);
        allocator.free(self.geom_remap_inds_times);
    }

    pub fn record(
        self: *CaseSamples,
        rr: usize,
        result: common.BenchResult,
    ) void {
        self.total_elems_vals[rr] = @floatFromInt(result.total_elems);
        self.vis_elems_vals[rr] = @floatFromInt(result.vis_elems);
        self.total_px_vals[rr] = @floatFromInt(result.total_px);
        self.shaded_px_vals[rr] = @floatFromInt(result.shaded_px);
        self.e2e_times[rr] = result.e2e_ms;
        self.geom_times[rr] = result.geom_ms;
        self.raster_times[rr] = result.raster_ms;
        self.cam_times[rr] = result.cam_ms;
        self.elem_loop_times[rr] = result.pipeline_times.elem_loop / 1e6;
        self.resolve_times[rr] = result.resolve_ms;
        self.save_frame_times[rr] =
            result.pipeline_times.save_frame / 1e6;
        self.frame_times[rr] =
            result.pipeline_times.active_time / 1e6;
        self.geom_tpx_vals[rr] = result.metrics.melems_sec;
        self.raster_tpx_vals[rr] =
            result.metrics.raster_tpx_mpx_s;
        self.frame_tpx_vals[rr] =
            result.metrics.frame_tpx_mpx_s;
        self.e2e_tpx_vals[rr] =
            result.metrics.e2e_tpx_mpx_s;
        self.setup_frame_buff_times[rr] =
            result.pipeline_times.setup_frame_buff / 1e6;
        self.prepare_frame_context_times[rr] =
            result.pipeline_times.prepare_frame_context / 1e6;
        self.geom_coord_ops_times[rr] =
            result.pipeline_times.geom_coord_ops / 1e6;
        self.geom_cull_ops_times[rr] =
            result.pipeline_times.geom_cull_ops / 1e6;
        self.geom_prep_hulls_shaders_times[rr] =
            result.pipeline_times.geom_prep_hulls_shaders / 1e6;
        self.geom_remap_inds_times[rr] =
            result.pipeline_times.geom_remap_inds / 1e6;
    }

    pub fn toBenchStats(
        self: *const CaseSamples,
        allocator: std.mem.Allocator,
        case_name: []const u8,
        mesh_type: gk.MeshType,
        shader_type: common.ShaderType,
        samp_cfg: ?texops.TextureSampleConfig,
        tex_func_case: ?common.TexFuncCase,
    ) !common.BenchStats {
        return .{
            .name = try allocator.dupe(u8, case_name),
            .mesh_type = mesh_type,
            .shader_type = shader_type,
            .samp_cfg = samp_cfg,
            .tex_func_case = tex_func_case,
            .total_elems = try common.calcMedianMAD(
                allocator,
                self.total_elems_vals,
            ),
            .vis_elems = try common.calcMedianMAD(
                allocator,
                self.vis_elems_vals,
            ),
            .total_px = try common.calcMedianMAD(
                allocator,
                self.total_px_vals,
            ),
            .shaded_px = try common.calcMedianMAD(
                allocator,
                self.shaded_px_vals,
            ),
            .e2e = try common.calcMedianMAD(
                allocator,
                self.e2e_times,
            ),
            .geom = try common.calcMedianMAD(
                allocator,
                self.geom_times,
            ),
            .raster = try common.calcMedianMAD(
                allocator,
                self.raster_times,
            ),
            .cam_invert = try common.calcMedianMAD(
                allocator,
                self.cam_times,
            ),
            .elem_loop = try common.calcMedianMAD(
                allocator,
                self.elem_loop_times,
            ),
            .scratch_resolve = try common.calcMedianMAD(
                allocator,
                self.resolve_times,
            ),
            .save = try common.calcMedianMAD(
                allocator,
                self.save_frame_times,
            ),
            .frame = try common.calcMedianMAD(
                allocator,
                self.frame_times,
            ),
            .geom_tpx = try common.calcMedianMAD(
                allocator,
                self.geom_tpx_vals,
            ),
            .raster_tpx = try common.calcMedianMAD(
                allocator,
                self.raster_tpx_vals,
            ),
            .frame_tpx = try common.calcMedianMAD(
                allocator,
                self.frame_tpx_vals,
            ),
            .e2e_tpx = try common.calcMedianMAD(
                allocator,
                self.e2e_tpx_vals,
            ),
            .msubpx = undefined,
            .mshades = undefined,
            .msubshades = undefined,
            .melems = undefined,
            .mnodes = undefined,
            .mops = undefined,
            .geom_prep = undefined,
            .tile_overlap = undefined,
            .raster_loop = undefined,
            .save_frame = undefined,
            .setup_frame_buff = try common.calcMedianMAD(
                allocator,
                self.setup_frame_buff_times,
            ),
            .prepare_frame_context = try common.calcMedianMAD(
                allocator,
                self.prepare_frame_context_times,
            ),
            .geom_coord_ops = try common.calcMedianMAD(
                allocator,
                self.geom_coord_ops_times,
            ),
            .geom_cull_ops = try common.calcMedianMAD(
                allocator,
                self.geom_cull_ops_times,
            ),
            .geom_prep_hulls_shaders = try common.calcMedianMAD(
                allocator,
                self.geom_prep_hulls_shaders_times,
            ),
            .geom_remap_inds = try common.calcMedianMAD(
                allocator,
                self.geom_remap_inds_times,
            ),
        };
    }
};

pub const BenchStatsCollector = struct {
    stats_list: std.ArrayList(common.BenchStats) = .empty,
    run_csv_rows: []std.ArrayList(u8) = &.{},

    pub fn init(
        allocator: std.mem.Allocator,
        runs: usize,
    ) !BenchStatsCollector {
        const run_csv_rows = try allocator.alloc(std.ArrayList(u8), runs);
        for (0..runs) |rr| {
            run_csv_rows[rr] = .empty;
            try run_csv_rows[rr].appendSlice(
                allocator,
                common.benchmarkCSVHeader(),
            );
        }
        return .{
            .run_csv_rows = run_csv_rows,
        };
    }

    pub fn deinit(
        self: *BenchStatsCollector,
        allocator: std.mem.Allocator,
    ) void {
        for (self.stats_list.items) |stats| {
            allocator.free(stats.name);
        }
        self.stats_list.deinit(allocator);
        for (self.run_csv_rows) |*rows| {
            rows.deinit(allocator);
        }
        if (self.run_csv_rows.len > 0) {
            allocator.free(self.run_csv_rows);
        }
    }

    pub fn appendRunResult(
        self: *BenchStatsCollector,
        allocator: std.mem.Allocator,
        run_idx: usize,
        case_name: []const u8,
        mesh_type: gk.MeshType,
        shader_type: common.ShaderType,
        samp_cfg: ?texops.TextureSampleConfig,
        tex_func_case: ?common.TexFuncCase,
        result: common.BenchResult,
    ) !void {
        const row = try common.formatBenchmarkCSVRow(
            allocator,
            case_name,
            mesh_type,
            shader_type,
            samp_cfg,
            tex_func_case,
            common.calcBenchmarkCSVValuesFromResult(result),
        );
        defer allocator.free(row);
        try self.run_csv_rows[run_idx].appendSlice(allocator, row);
    }

    pub fn appendCaseStats(
        self: *BenchStatsCollector,
        allocator: std.mem.Allocator,
        case_name: []const u8,
        mesh_type: gk.MeshType,
        shader_type: common.ShaderType,
        samp_cfg: ?texops.TextureSampleConfig,
        tex_func_case: ?common.TexFuncCase,
        case_samples: *const CaseSamples,
    ) !void {
        try self.stats_list.append(
            allocator,
            try case_samples.toBenchStats(
                allocator,
                case_name,
                mesh_type,
                shader_type,
                samp_cfg,
                tex_func_case,
            ),
        );
    }

    pub fn writeRunCSVs(
        self: *const BenchStatsCollector,
        allocator: std.mem.Allocator,
        io: std.Io,
        out_dir_base: []const u8,
    ) !void {
        for (self.run_csv_rows, 0..) |_, rr| {
            try self.writeRunCSV(
                allocator,
                io,
                out_dir_base,
                rr,
            );
        }
    }

    pub fn writeRunCSV(
        self: *const BenchStatsCollector,
        allocator: std.mem.Allocator,
        io: std.Io,
        out_dir_base: []const u8,
        run_idx: usize,
    ) !void {
        const cwd = std.Io.Dir.cwd();
        cwd.createDir(
            io,
            out_dir_base,
            .default_dir,
        ) catch |err| if (err != error.PathAlreadyExists) return err;

        const file_name = try std.fmt.allocPrint(
            allocator,
            "bench_run{d}.csv",
            .{run_idx},
        );
        defer allocator.free(file_name);

        const csv_path = try std.fs.path.join(
            allocator,
            &[_][]const u8{ out_dir_base, file_name },
        );
        defer allocator.free(csv_path);

        var file = try cwd.createFile(io, csv_path, .{});
        defer file.close(io);

        var write_buf: [4096]u8 = undefined;
        var buffered_writer = file.writer(io, &write_buf);
        try buffered_writer.interface.writeAll(
            self.run_csv_rows[run_idx].items,
        );
        try buffered_writer.interface.flush();
    }
};
