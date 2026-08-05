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
const gk = @import("riley/zig/geometrykernels.zig");
const meshio = @import("riley/zig/meshio.zig");
const orch = @import("dev_support/orchestration.zig");
const verif = @import("dev_support/verif.zig");
const vconst = @import("dev_support/verifconstants.zig");

const structured_grid_quad_num: usize = 250;
const structured_grid_tri_num: usize = 250;
const verif_subdir_name = "verif_1";

const SampleGrid = struct {
    list: std.ArrayList(verif.SamplePoint),
    rows_num: usize,
    cols_num: usize,
};

fn frameFileStem(
    buf: []u8,
    frame_idx: usize,
    field_idx: usize,
) ![]const u8 {
    return std.fmt.bufPrint(buf, "cam0_frame{d}_field{d}", .{ frame_idx, field_idx });
}

fn frameStatsFileName(
    buf: []u8,
    frame_idx: usize,
) ![]const u8 {
    return std.fmt.bufPrint(buf, "solver_stats_frame{d}.csv", .{frame_idx});
}

fn isFinite(val: F) bool {
    return !std.math.isNan(val) and !std.math.isInf(val);
}

fn evalSample(
    comptime mesh_type: gk.MeshType,
    camera: *const cam.CameraPrepared,
    node_x: []const F,
    node_y: []const F,
    node_z: []const F,
    solver_nodes: *const verif.ElementNodes(mesh_type.getNodesNum()),
    sample: verif.SamplePoint,
) !verif.SampleRecord {
    const nan = std.math.nan(F);
    const world_true = verif.forwardMapWorldForMeshType(
        mesh_type,
        sample.xi_true,
        sample.eta_true,
        node_x,
        node_y,
        node_z,
    );
    const ideal_target = verif.worldToIdealRaster(camera, world_true);
    const observed_target = try verif.idealToObservedRaster(camera, ideal_target);
    const ideal_solve_target = try verif.observedToIdealRaster(camera, observed_target);
    const solve_result = verif.solveParentFromIdealRaster(
        mesh_type,
        camera,
        solver_nodes,
        ideal_solve_target[0],
        ideal_solve_target[1],
    );

    var xi_rec = nan;
    var eta_rec = nan;
    var xi_reproj = nan;
    var eta_reproj = nan;
    var observed_reproj = [2]F{ nan, nan };
    var reproj_err = nan;
    var err_xi = nan;
    var err_eta = nan;
    var err_param = nan;
    var in_domain = false;

    if (solve_result.converged) {
        xi_rec = solve_result.xi_rec;
        eta_rec = solve_result.eta_rec;
        err_xi = xi_rec - sample.xi_true;
        err_eta = eta_rec - sample.eta_true;
        err_param = @sqrt(err_xi * err_xi + err_eta * err_eta);
        in_domain = verif.isInParametricDomain(mesh_type, xi_rec, eta_rec);

        const world_rec = verif.forwardMapWorldForMeshType(
            mesh_type,
            xi_rec,
            eta_rec,
            node_x,
            node_y,
            node_z,
        );
        const ideal_reproj = verif.worldToIdealRaster(camera, world_rec);
        observed_reproj = try verif.idealToObservedRaster(camera, ideal_reproj);
        const dx_obs = observed_reproj[0] - observed_target[0];
        const dy_obs = observed_reproj[1] - observed_target[1];
        reproj_err = @sqrt(dx_obs * dx_obs + dy_obs * dy_obs);

        const reproj_ideal_target = try verif.observedToIdealRaster(
            camera,
            observed_reproj,
        );
        const reproj_result = verif.solveParentFromIdealRaster(
            mesh_type,
            camera,
            solver_nodes,
            reproj_ideal_target[0],
            reproj_ideal_target[1],
        );
        if (reproj_result.converged) {
            xi_reproj = reproj_result.xi_rec;
            eta_reproj = reproj_result.eta_rec;
        }
    }

    return .{
        .xi_true = sample.xi_true,
        .eta_true = sample.eta_true,
        .xi_rec = xi_rec,
        .eta_rec = eta_rec,
        .xi_reproj = xi_reproj,
        .eta_reproj = eta_reproj,
        .err_xi = err_xi,
        .err_eta = err_eta,
        .err_param = err_param,
        .ideal_target_x = ideal_target[0],
        .ideal_target_y = ideal_target[1],
        .observed_target_x = observed_target[0],
        .observed_target_y = observed_target[1],
        .observed_reproj_x = observed_reproj[0],
        .observed_reproj_y = observed_reproj[1],
        .reproj_err = reproj_err,
        .iters = solve_result.iters,
        .converged = solve_result.converged,
        .in_domain = in_domain,
        .row_idx = sample.row_idx,
        .col_idx = sample.col_idx,
    };
}

fn buildSampleList(
    comptime mesh_type: gk.MeshType,
    allocator: std.mem.Allocator,
) !SampleGrid {
    var sample_list: std.ArrayList(verif.SamplePoint) = .empty;
    const structured_grid_num = if (mesh_type == .tri3 or mesh_type == .tri6)
        structured_grid_tri_num
    else
        structured_grid_quad_num;
    const map_dims = try verif.appendStructuredSamples(
        mesh_type,
        allocator,
        &sample_list,
        structured_grid_num,
    );
    return .{
        .list = sample_list,
        .rows_num = map_dims.rows_num,
        .cols_num = map_dims.cols_num,
    };
}

fn saveFieldMaps(
    allocator: std.mem.Allocator,
    io: std.Io,
    out_dir: std.Io.Dir,
    frame_idx: usize,
    rows_num: usize,
    cols_num: usize,
    field0: []const F,
    field1: []const F,
) !void {
    var field0_stem_buf: [128]u8 = undefined;
    var field0_csv_buf: [128]u8 = undefined;
    var field1_stem_buf: [128]u8 = undefined;
    var field1_csv_buf: [128]u8 = undefined;

    const field0_stem = try frameFileStem(&field0_stem_buf, frame_idx, 0);
    const field0_csv = try std.fmt.bufPrint(
        &field0_csv_buf,
        "{s}.csv",
        .{field0_stem},
    );
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

    const field1_stem = try frameFileStem(&field1_stem_buf, frame_idx, 1);
    const field1_csv = try std.fmt.bufPrint(
        &field1_csv_buf,
        "{s}.csv",
        .{field1_stem},
    );
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

fn saveIdealMaps(
    comptime mesh_type: gk.MeshType,
    allocator: std.mem.Allocator,
    io: std.Io,
    sample_list: []const verif.SamplePoint,
    rows_num: usize,
    cols_num: usize,
) !void {
    const ideal_dir_path = try std.fmt.allocPrint(
        allocator,
        "{s}/{s}/a_ideal/{s}",
        .{
            vconst.output_dir_name,
            verif_subdir_name,
            @tagName(mesh_type),
        },
    );
    var ideal_dir = try orch.openDirEnsured(io, ideal_dir_path);
    defer ideal_dir.close(io);

    const map_len = rows_num * cols_num;
    var xi_map = try allocator.alloc(F, map_len);
    defer allocator.free(xi_map);
    var eta_map = try allocator.alloc(F, map_len);
    defer allocator.free(eta_map);
    @memset(xi_map, std.math.nan(F));
    @memset(eta_map, std.math.nan(F));

    for (sample_list) |sample| {
        const map_idx = sample.row_idx * cols_num + sample.col_idx;
        xi_map[map_idx] = sample.xi_true;
        eta_map[map_idx] = sample.eta_true;
    }

    try saveFieldMaps(
        allocator,
        io,
        ideal_dir,
        0,
        rows_num,
        cols_num,
        xi_map,
        eta_map,
    );
}

fn saveIdealMapsDynamic(
    mesh_type: gk.MeshType,
    allocator: std.mem.Allocator,
    io: std.Io,
    sample_list: []const verif.SamplePoint,
    rows_num: usize,
    cols_num: usize,
) !void {
    return switch (mesh_type) {
        .tri3 => try saveIdealMaps(
            .tri3,
            allocator,
            io,
            sample_list,
            rows_num,
            cols_num,
        ),
        .tri6 => try saveIdealMaps(
            .tri6,
            allocator,
            io,
            sample_list,
            rows_num,
            cols_num,
        ),
        .quad4ibi => try saveIdealMaps(
            .quad4ibi,
            allocator,
            io,
            sample_list,
            rows_num,
            cols_num,
        ),
        .quad4newton => try saveIdealMaps(
            .quad4newton,
            allocator,
            io,
            sample_list,
            rows_num,
            cols_num,
        ),
        .quad8 => try saveIdealMaps(
            .quad8,
            allocator,
            io,
            sample_list,
            rows_num,
            cols_num,
        ),
        .quad9 => try saveIdealMaps(
            .quad9,
            allocator,
            io,
            sample_list,
            rows_num,
            cols_num,
        ),
    };
}

fn frameNodes(
    comptime mesh_type: gk.MeshType,
    sim_data: *const meshio.SimData,
    frame_idx: usize,
) verif.ElementNodes(mesh_type.getNodesNum()) {
    const N = comptime mesh_type.getNodesNum();
    const elem = sim_data.connect.getElem(0);
    var nodes: verif.ElementNodes(N) = undefined;

    for (0..N) |nn| {
        const node_idx = elem[nn];
        nodes.x[nn] = sim_data.coords.x(node_idx);
        nodes.y[nn] = sim_data.coords.y(node_idx);
        nodes.z[nn] = sim_data.coords.z(node_idx);

        if (sim_data.field) |field| {
            nodes.x[nn] += field.array.get(&[_]usize{ frame_idx, node_idx, 0 });
            nodes.y[nn] += field.array.get(&[_]usize{ frame_idx, node_idx, 1 });
            nodes.z[nn] += field.array.get(&[_]usize{ frame_idx, node_idx, 2 });
        }
    }

    return nodes;
}

fn runDistortCase(
    comptime mesh_type: gk.MeshType,
    case_spec: vconst.DistortCase,
    allocator: std.mem.Allocator,
    io: std.Io,
    global_reproj_errs: *std.ArrayList(F),
) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    const camera = try cam.CameraPrepared.init(aa, case_spec.camera_input);
    const sim_data = try orch.loadData(aa, io, case_spec.data_dir);
    var sample_data = try buildSampleList(mesh_type, aa);
    defer sample_data.list.deinit(aa);

    try saveIdealMapsDynamic(
        mesh_type,
        allocator,
        io,
        sample_data.list.items,
        sample_data.rows_num,
        sample_data.cols_num,
    );

    const out_dir_path = try std.fmt.allocPrint(
        aa,
        "{s}/{s}/a_distort_{s}_{s}",
        .{
            vconst.output_dir_name,
            verif_subdir_name,
            case_spec.case_name,
            @tagName(mesh_type),
        },
    );
    var out_dir = try orch.openDirEnsured(io, out_dir_path);
    defer out_dir.close(io);

    const time_steps = if (sim_data.field) |field| field.getTimeN() else 1;
    const map_len = sample_data.rows_num * sample_data.cols_num;

    for (0..time_steps) |frame_idx| {
        const nodes = frameNodes(mesh_type, &sim_data, frame_idx);
        const solver_nodes = verif.worldNodesToSolverCoords(
            mesh_type,
            &camera,
            nodes.x[0..],
            nodes.y[0..],
            nodes.z[0..],
        );

        var xi_map = try allocator.alloc(F, map_len);
        defer allocator.free(xi_map);
        var eta_map = try allocator.alloc(F, map_len);
        defer allocator.free(eta_map);
        @memset(xi_map, std.math.nan(F));
        @memset(eta_map, std.math.nan(F));

        var records: std.ArrayList(verif.SampleRecord) = .empty;
        defer records.deinit(allocator);
        try records.ensureTotalCapacity(allocator, sample_data.list.items.len);

        var reproj_vals: std.ArrayList(F) = .empty;
        defer reproj_vals.deinit(allocator);
        var param_vals: std.ArrayList(F) = .empty;
        defer param_vals.deinit(allocator);

        for (sample_data.list.items) |sample| {
            const record = try evalSample(
                mesh_type,
                &camera,
                nodes.x[0..],
                nodes.y[0..],
                nodes.z[0..],
                &solver_nodes,
                sample,
            );

            try records.append(allocator, record);
            const map_idx = sample.row_idx * sample_data.cols_num + sample.col_idx;
            xi_map[map_idx] = record.xi_rec;
            eta_map[map_idx] = record.eta_rec;

            if (isFinite(record.reproj_err)) {
                try reproj_vals.append(allocator, record.reproj_err);
                try global_reproj_errs.append(allocator, record.reproj_err);
            }
            if (isFinite(record.err_param)) {
                try param_vals.append(allocator, record.err_param);
            }
        }

        try saveFieldMaps(
            allocator,
            io,
            out_dir,
            frame_idx,
            sample_data.rows_num,
            sample_data.cols_num,
            xi_map,
            eta_map,
        );

        var stats_name_buf: [128]u8 = undefined;
        const stats_file_name = try frameStatsFileName(&stats_name_buf, frame_idx);
        try verif.writeSolverStatsCsv(
            io,
            out_dir,
            stats_file_name,
            records.items,
        );

        if (reproj_vals.items.len > 0 and param_vals.items.len > 0) {
            const reproj_stats = try verif.calcScalarStats(allocator, reproj_vals.items);
            const param_stats = try verif.calcScalarStats(allocator, param_vals.items);
            std.debug.print(
                "a_distort_{s}_{s} frame {d}: reproj max={e:.6} param max={e:.6}\n",
                .{
                    case_spec.case_name,
                    @tagName(mesh_type),
                    frame_idx,
                    reproj_stats.max,
                    param_stats.max,
                },
            );
        } else {
            std.debug.print(
                "a_distort_{s}_{s} frame {d}: no converged samples\n",
                .{
                    case_spec.case_name,
                    @tagName(mesh_type),
                    frame_idx,
                },
            );
        }
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

    var global_reproj_errs: std.ArrayList(F) = .empty;
    defer global_reproj_errs.deinit(allocator);

    for (vconst.distort_cases) |case_spec| {
        switch (case_spec.mesh_type) {
            .tri3 => try runDistortCase(
                .tri3,
                case_spec,
                allocator,
                io,
                &global_reproj_errs,
            ),
            .tri6 => try runDistortCase(
                .tri6,
                case_spec,
                allocator,
                io,
                &global_reproj_errs,
            ),
            .quad4ibi => try runDistortCase(
                .quad4ibi,
                case_spec,
                allocator,
                io,
                &global_reproj_errs,
            ),
            .quad4newton => try runDistortCase(
                .quad4newton,
                case_spec,
                allocator,
                io,
                &global_reproj_errs,
            ),
            .quad8 => try runDistortCase(
                .quad8,
                case_spec,
                allocator,
                io,
                &global_reproj_errs,
            ),
            .quad9 => try runDistortCase(
                .quad9,
                case_spec,
                allocator,
                io,
                &global_reproj_errs,
            ),
        }
    }

    if (global_reproj_errs.items.len > 0) {
        const global_stats = try verif.calcScalarStats(
            allocator,
            global_reproj_errs.items,
        );
        const newton_tol = buildconfig.config.tolerance.newton.resid;

        std.debug.print(
            "\nGlobal reprojection error summary (px):\n" ++
                "  min={e:.6}\n  q1={e:.6}\n  median={e:.6}\n  q3={e:.6}\n" ++
                "  max={e:.6}\n  mean={e:.6}\n  rms={e:.6}\n",
            .{
                global_stats.min,
                global_stats.q1,
                global_stats.median,
                global_stats.q3,
                global_stats.max,
                global_stats.mean,
                global_stats.rms,
            },
        );
        std.debug.print(
            "Tolerance comparison:\n" ++
                "  newton resid tol = {e:.6}\n" ++
                "  max / newton tol = {e:.3}\n" ++
                "  median / newton tol = {e:.3}\n",
            .{
                newton_tol,
                global_stats.max / newton_tol,
                global_stats.median / newton_tol,
            },
        );
    }
}
