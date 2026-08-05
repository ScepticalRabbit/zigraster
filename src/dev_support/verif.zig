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
const cam = @import("../riley/zig/camera.zig");
const csvio = @import("../riley/zig/csvio.zig");
const gk = @import("../riley/zig/geometrykernels.zig");
const iio = @import("../riley/zig/imageio.zig");
const matrix = @import("../riley/zig/matstack.zig");
const ndarray = @import("../riley/zig/ndarray.zig");
const newton = @import("../riley/zig/newton.zig");
const rops = @import("../riley/zig/rasterops.zig");
const rastcfg = @import("../riley/zig/rasterconfig.zig");
const shapefun = @import("../riley/zig/shapefun.zig");
const vecstack = @import("../riley/zig/vecstack.zig");

pub const Vec3 = struct {
    x: F,
    y: F,
    z: F,
};

pub const invalid_grid_idx = std.math.maxInt(usize);

pub const SamplePoint = struct {
    xi_true: F,
    eta_true: F,
    row_idx: usize = invalid_grid_idx,
    col_idx: usize = invalid_grid_idx,
};

pub const SampleRecord = struct {
    xi_true: F,
    eta_true: F,
    xi_rec: F,
    eta_rec: F,
    xi_reproj: F,
    eta_reproj: F,
    err_xi: F,
    err_eta: F,
    err_param: F,
    ideal_target_x: F,
    ideal_target_y: F,
    observed_target_x: F,
    observed_target_y: F,
    observed_reproj_x: F,
    observed_reproj_y: F,
    reproj_err: F,
    iters: u8,
    converged: bool,
    in_domain: bool,
    row_idx: usize = invalid_grid_idx,
    col_idx: usize = invalid_grid_idx,
};

pub const ScalarStats = struct {
    min: F,
    q1: F,
    median: F,
    q3: F,
    max: F,
    mean: F,
    rms: F,
};

pub const CaseSummary = struct {
    mesh_type: gk.MeshType,
    geom_case_name: []const u8,
    camera_case_name: []const u8,
    samples_num: usize,
    nonconverged_num: usize,
    out_of_domain_num: usize,
    reproj_stats: ScalarStats,
    param_stats: ScalarStats,
    iter_stats: ScalarStats,
};

pub fn ElementNodes(comptime N: usize) type {
    return struct {
        x: [N]F,
        y: [N]F,
        z: [N]F,
    };
}

pub fn forwardMapWorld(
    comptime N: usize,
    xi: F,
    eta: F,
    node_values: *[N]F,
    deriv_xi: *[N]F,
    deriv_eta: *[N]F,
    node_x: []const F,
    node_y: []const F,
    node_z: []const F,
) Vec3 {
    shapefun.shapeFunctions(
        N,
        xi,
        eta,
        node_values,
        deriv_xi,
        deriv_eta,
    );

    var x_world: F = 0.0;
    var y_world: F = 0.0;
    var z_world: F = 0.0;

    for (0..N) |nn| {
        const node_weight = node_values[nn];
        x_world += node_weight * node_x[nn];
        y_world += node_weight * node_y[nn];
        z_world += node_weight * node_z[nn];
    }

    return .{
        .x = x_world,
        .y = y_world,
        .z = z_world,
    };
}

fn forwardMapQuad4Ibi(
    xi: F,
    eta: F,
    node_x: []const F,
    node_y: []const F,
    node_z: []const F,
) Vec3 {
    const weight_0 = (1.0 - xi) * (1.0 - eta);
    const weight_1 = xi * (1.0 - eta);
    const weight_2 = xi * eta;
    const weight_3 = (1.0 - xi) * eta;

    return .{
        .x = weight_0 * node_x[0] + weight_1 * node_x[1] +
            weight_2 * node_x[2] + weight_3 * node_x[3],
        .y = weight_0 * node_y[0] + weight_1 * node_y[1] +
            weight_2 * node_y[2] + weight_3 * node_y[3],
        .z = weight_0 * node_z[0] + weight_1 * node_z[1] +
            weight_2 * node_z[2] + weight_3 * node_z[3],
    };
}

pub fn forwardMapWorldForMeshType(
    comptime mesh_type: gk.MeshType,
    xi: F,
    eta: F,
    node_x: []const F,
    node_y: []const F,
    node_z: []const F,
) Vec3 {
    const N = comptime mesh_type.getNodesNum();

    if (mesh_type == .quad4ibi) {
        return forwardMapQuad4Ibi(
            xi,
            eta,
            node_x,
            node_y,
            node_z,
        );
    }

    var node_values: [N]F = undefined;
    var deriv_xi: [N]F = undefined;
    var deriv_eta: [N]F = undefined;
    return forwardMapWorld(
        N,
        xi,
        eta,
        &node_values,
        &deriv_xi,
        &deriv_eta,
        node_x,
        node_y,
        node_z,
    );
}

fn worldToCamera(
    camera: *const cam.CameraPrepared,
    world_point: Vec3,
) vecstack.Vec3f {
    return matrix.Mat44Ops.mulVec3(
        F,
        camera.world_to_cam_mat,
        .{ .slice = .{ world_point.x, world_point.y, world_point.z } },
    );
}

pub fn worldToIdealRaster(
    camera: *const cam.CameraPrepared,
    world_point: Vec3,
) [2]F {
    const coord_cam = worldToCamera(camera, world_point);
    const focal_px = camera.calcFocalPx();
    const offsets = camera.calcRasterOffsets();
    const inv_neg_z = 1.0 / (-coord_cam.slice[2]);

    return .{
        offsets.x_off + coord_cam.slice[0] * inv_neg_z * focal_px.fx,
        offsets.y_off - coord_cam.slice[1] * inv_neg_z * focal_px.fy,
    };
}

pub fn idealToObservedRaster(
    camera: *const cam.CameraPrepared,
    ideal_xy: [2]F,
) ![2]F {
    const focal_px = camera.calcFocalPx();
    const offsets = camera.calcRasterOffsets();
    const x_norm = (ideal_xy[0] - offsets.x_off) / focal_px.fx;
    const y_norm = (ideal_xy[1] - offsets.y_off) / focal_px.fy;

    const distorted = cam.forwardDistortionModelScal(
        camera.distortion,
        x_norm,
        y_norm,
    );
    return .{
        distorted[0] * focal_px.fx + offsets.x_off,
        distorted[1] * focal_px.fy + offsets.y_off,
    };
}

pub fn observedToIdealRaster(
    camera: *const cam.CameraPrepared,
    observed_xy: [2]F,
) ![2]F {
    const focal_px = camera.calcFocalPx();
    const offsets = camera.calcRasterOffsets();
    const x_dist = (observed_xy[0] - offsets.x_off) / focal_px.fx;
    const y_dist = (observed_xy[1] - offsets.y_off) / focal_px.fy;

    const solved = try cam.inverseDistortionModelScalar(
        camera.distortion,
        x_dist,
        y_dist,
    );
    return .{
        solved.x * focal_px.fx + offsets.x_off,
        solved.y * focal_px.fy + offsets.y_off,
    };
}

pub fn worldNodesToSolverCoords(
    comptime mesh_type: gk.MeshType,
    camera: *const cam.CameraPrepared,
    node_x: []const F,
    node_y: []const F,
    node_z: []const F,
) ElementNodes(mesh_type.getNodesNum()) {
    const N = comptime mesh_type.getNodesNum();
    const focal_px = camera.calcFocalPx();
    const offsets = camera.calcRasterOffsets();
    var solver_nodes: ElementNodes(N) = undefined;

    for (0..N) |nn| {
        const world_point = Vec3{
            .x = node_x[nn],
            .y = node_y[nn],
            .z = node_z[nn],
        };
        const coord_cam = worldToCamera(camera, world_point);
        const inv_neg_z = 1.0 / (-coord_cam.slice[2]);

        if (mesh_type == .tri3 or mesh_type == .tri3opt) {
            solver_nodes.x[nn] = offsets.x_off +
                coord_cam.slice[0] * inv_neg_z * focal_px.fx;
            solver_nodes.y[nn] = offsets.y_off -
                coord_cam.slice[1] * inv_neg_z * focal_px.fy;
            solver_nodes.z[nn] = -coord_cam.slice[2];
        } else {
            solver_nodes.x[nn] = coord_cam.slice[0] * focal_px.fx;
            solver_nodes.y[nn] = -coord_cam.slice[1] * focal_px.fy;
            solver_nodes.z[nn] = -coord_cam.slice[2];
        }
    }

    return solver_nodes;
}

pub fn toVec3Slices(
    comptime N: usize,
    nodes: *const ElementNodes(N),
) rops.Vec3Slices(F) {
    return .{
        .x = @constCast(nodes.x[0..]),
        .y = @constCast(nodes.y[0..]),
        .z = @constCast(nodes.z[0..]),
    };
}

pub const SolveResult = struct {
    converged: bool,
    xi_rec: F,
    eta_rec: F,
    iters: u8,
};

pub fn solveParentFromIdealRaster(
    comptime mesh_type: gk.MeshType,
    camera: *const cam.CameraPrepared,
    solver_nodes: *const ElementNodes(mesh_type.getNodesNum()),
    ideal_x: F,
    ideal_y: F,
) SolveResult {
    const offsets = camera.calcRasterOffsets();
    const nodes = toVec3Slices(mesh_type.getNodesNum(), solver_nodes);

    return switch (mesh_type) {
        .tri3, .tri3opt => blk: {
            const GK = if (mesh_type == .tri3) gk.Tri3Kernel() else gk.Tri3OptKernel();
            const inv_area = GK.getInvElemArea(nodes);
            const result = GK.solveWeightsHyperb(
                nodes,
                ideal_x,
                ideal_y,
                inv_area,
            );
            if (result.weights) |weights| {
                const inv_z = GK.calcInvZ(nodes, weights);
                const xi_rec = weights[1] * (1.0 / nodes.z[1]) / inv_z;
                const eta_rec = weights[2] * (1.0 / nodes.z[2]) / inv_z;
                break :blk .{
                    .converged = true,
                    .xi_rec = xi_rec,
                    .eta_rec = eta_rec,
                    .iters = result.iters,
                };
            }
            break :blk .{
                .converged = false,
                .xi_rec = 0.0,
                .eta_rec = 0.0,
                .iters = result.iters,
            };
        },
        .tri6 => blk: {
            const GK = gk.Tri6Kernel();
            const seed = GK.initSeed(.centroid, null);
            const result = GK.solveWeightsNewton(
                nodes,
                ideal_x,
                ideal_y,
                offsets.x_off,
                offsets.y_off,
                seed.xi,
                seed.eta,
            );
            break :blk .{
                .converged = result.weights != null,
                .xi_rec = result.xi_out,
                .eta_rec = result.eta_out,
                .iters = result.iters,
            };
        },
        .quad4ibi => blk: {
            const GK = gk.Quad4IBIKernel();
            const params = GK.getBilinearParams(nodes);
            const result = GK.solveWeightsInvBi(
                ideal_x,
                ideal_y,
                offsets.x_off,
                offsets.y_off,
                params,
            );
            break :blk .{
                .converged = result.weights != null,
                .xi_rec = result.xi_out,
                .eta_rec = result.eta_out,
                .iters = result.iters,
            };
        },
        .quad4newton => blk: {
            const GK = gk.Quad4NewtonKernel();
            const seed = GK.initSeed(.centroid, null);
            const result = GK.solveWeightsNewton(
                nodes,
                ideal_x,
                ideal_y,
                offsets.x_off,
                offsets.y_off,
                seed.xi,
                seed.eta,
            );
            break :blk .{
                .converged = result.weights != null,
                .xi_rec = result.xi_out,
                .eta_rec = result.eta_out,
                .iters = result.iters,
            };
        },
        .quad8 => blk: {
            const GK = gk.Quad89Kernel(8);
            const seed = GK.initSeed(.centroid, null);
            const result = GK.solveWeightsNewton(
                nodes,
                ideal_x,
                ideal_y,
                offsets.x_off,
                offsets.y_off,
                seed.xi,
                seed.eta,
            );
            break :blk .{
                .converged = result.weights != null,
                .xi_rec = result.xi_out,
                .eta_rec = result.eta_out,
                .iters = result.iters,
            };
        },
        .quad9 => blk: {
            const GK = gk.Quad89Kernel(9);
            const seed = GK.initSeed(.centroid, null);
            const result = GK.solveWeightsNewton(
                nodes,
                ideal_x,
                ideal_y,
                offsets.x_off,
                offsets.y_off,
                seed.xi,
                seed.eta,
            );
            break :blk .{
                .converged = result.weights != null,
                .xi_rec = result.xi_out,
                .eta_rec = result.eta_out,
                .iters = result.iters,
            };
        },
    };
}

pub fn isInParametricDomain(
    comptime mesh_type: gk.MeshType,
    xi: F,
    eta: F,
) bool {
    const eps = 1.0e-8;
    return switch (mesh_type) {
        .tri3, .tri3opt, .tri6 => xi >= -eps and eta >= -eps and xi + eta <= 1.0 + eps,
        .quad4ibi => xi >= -eps and xi <= 1.0 + eps and eta >= -eps and
            eta <= 1.0 + eps,
        .quad4newton, .quad8, .quad9 => @abs(xi) <= 1.0 + eps and
            @abs(eta) <= 1.0 + eps,
    };
}

pub fn structuredGridDims(comptime mesh_type: gk.MeshType) struct {
    rows_num: usize,
    cols_num: usize,
} {
    _ = mesh_type;
    return .{ .rows_num = 0, .cols_num = 0 };
}

pub fn appendStructuredSamples(
    comptime mesh_type: gk.MeshType,
    allocator: std.mem.Allocator,
    list: *std.ArrayList(SamplePoint),
    grid_num: usize,
) !struct { rows_num: usize, cols_num: usize } {
    if (mesh_type == .tri3 or mesh_type == .tri3opt or mesh_type == .tri6) {
        for (0..grid_num) |rr| {
            const eta = @as(F, @floatFromInt(rr)) /
                @as(F, @floatFromInt(grid_num - 1));
            for (0..grid_num) |cc| {
                const xi = @as(F, @floatFromInt(cc)) /
                    @as(F, @floatFromInt(grid_num - 1));
                if (xi + eta <= 1.0) {
                    try list.append(allocator, .{
                        .xi_true = xi,
                        .eta_true = eta,
                        .row_idx = rr,
                        .col_idx = cc,
                    });
                }
            }
        }
    } else {
        const xi_min = if (mesh_type == .quad4ibi) 0.0 else -1.0;
        const xi_max = 1.0;
        const eta_min = if (mesh_type == .quad4ibi) 0.0 else -1.0;
        const eta_max = 1.0;
        for (0..grid_num) |rr| {
            const eta = eta_min +
                (@as(F, @floatFromInt(rr)) /
                    @as(F, @floatFromInt(grid_num - 1))) * (eta_max - eta_min);
            for (0..grid_num) |cc| {
                const xi = xi_min +
                    (@as(F, @floatFromInt(cc)) /
                        @as(F, @floatFromInt(grid_num - 1))) * (xi_max - xi_min);
                try list.append(allocator, .{
                    .xi_true = xi,
                    .eta_true = eta,
                    .row_idx = rr,
                    .col_idx = cc,
                });
            }
        }
    }

    return .{ .rows_num = grid_num, .cols_num = grid_num };
}

fn quantileFromSorted(sorted_vals: []const F, q: F) F {
    if (sorted_vals.len == 0) {
        return std.math.nan(F);
    }
    if (sorted_vals.len == 1) {
        return sorted_vals[0];
    }

    const idx_f = q * @as(F, @floatFromInt(sorted_vals.len - 1));
    const idx_lo: usize = @intFromFloat(@floor(idx_f));
    const idx_hi: usize = @intFromFloat(@ceil(idx_f));
    const frac = idx_f - @as(F, @floatFromInt(idx_lo));

    return sorted_vals[idx_lo] * (1.0 - frac) + sorted_vals[idx_hi] * frac;
}

pub fn calcScalarStats(
    allocator: std.mem.Allocator,
    vals: []const F,
) !ScalarStats {
    const sorted_vals = try allocator.alloc(F, vals.len);
    defer allocator.free(sorted_vals);
    @memcpy(sorted_vals, vals);
    std.sort.pdq(
        F,
        sorted_vals,
        {},
        std.sort.asc(F),
    );

    var sum: F = 0.0;
    var sum_sq: F = 0.0;
    for (vals) |val| {
        sum += val;
        sum_sq += val * val;
    }

    const inv_len = 1.0 / @as(F, @floatFromInt(vals.len));
    return .{
        .min = sorted_vals[0],
        .q1 = quantileFromSorted(sorted_vals, 0.25),
        .median = quantileFromSorted(sorted_vals, 0.5),
        .q3 = quantileFromSorted(sorted_vals, 0.75),
        .max = sorted_vals[sorted_vals.len - 1],
        .mean = sum * inv_len,
        .rms = @sqrt(sum_sq * inv_len),
    };
}

pub fn openOutputDir(io: std.Io, dir_name: []const u8) !std.Io.Dir {
    const cwd = std.Io.Dir.cwd();
    cwd.createDir(io, dir_name, .default_dir) catch |err| {
        if (err != error.PathAlreadyExists) {
            return err;
        }
    };
    return try cwd.openDir(io, dir_name, .{});
}

pub fn writeSolverStatsCsv(
    io: std.Io,
    out_dir: std.Io.Dir,
    file_name: []const u8,
    records: []const SampleRecord,
) !void {
    const csv_file = try out_dir.createFile(io, file_name, .{});
    defer csv_file.close(io);

    var write_buf: [4096]u8 = undefined;
    var file_writer = csv_file.writer(io, &write_buf);
    const writer = &file_writer.interface;

    try writer.writeAll(
        "ideal_xi,ideal_eta,solved_xi,solved_eta,reproj_xi,reproj_eta," ++
            "err_xi,err_eta,err_param,ideal_target_x,ideal_target_y," ++
            "observed_target_x,observed_target_y,observed_reproj_x," ++
            "observed_reproj_y,reproj_err,iters,converged,in_domain," ++
            "row_idx,col_idx\n",
    );

    for (records) |record| {
        try writer.print(
            "{d},{d},{d},{d},{d},{d},{d},{d},{d},{d},{d},{d},{d},{d},{d}," ++
                "{d},{d},{d},{d},{d},{d}\n",
            .{
                record.xi_true,
                record.eta_true,
                record.xi_rec,
                record.eta_rec,
                record.xi_reproj,
                record.eta_reproj,
                record.err_xi,
                record.err_eta,
                record.err_param,
                record.ideal_target_x,
                record.ideal_target_y,
                record.observed_target_x,
                record.observed_target_y,
                record.observed_reproj_x,
                record.observed_reproj_y,
                record.reproj_err,
                record.iters,
                @intFromBool(record.converged),
                @intFromBool(record.in_domain),
                record.row_idx,
                record.col_idx,
            },
        );
    }

    try file_writer.flush();
}

pub fn writeSummaryCsv(
    io: std.Io,
    out_dir: std.Io.Dir,
    file_name: []const u8,
    summaries: []const CaseSummary,
) !void {
    const csv_file = try out_dir.createFile(io, file_name, .{});
    defer csv_file.close(io);

    var write_buf: [4096]u8 = undefined;
    var file_writer = csv_file.writer(io, &write_buf);
    const writer = &file_writer.interface;

    try writer.writeAll(
        "mesh_type,geom_case,camera_case,samples_num,nonconverged_num," ++
            "out_of_domain_num,reproj_min,reproj_q1,reproj_median,reproj_q3," ++
            "reproj_max,reproj_mean,reproj_rms,param_min,param_q1,param_median," ++
            "param_q3,param_max,param_mean,param_rms,iter_min,iter_q1," ++
            "iter_median,iter_q3,iter_max,iter_mean,iter_rms\n",
    );

    for (summaries) |summary| {
        try writer.print(
            "{s},{s},{s},{d},{d},{d},{d},{d},{d},{d},{d},{d},{d},{d},{d}," ++
                "{d},{d},{d},{d},{d},{d},{d},{d},{d},{d},{d},{d}\n",
            .{
                @tagName(summary.mesh_type),
                summary.geom_case_name,
                summary.camera_case_name,
                summary.samples_num,
                summary.nonconverged_num,
                summary.out_of_domain_num,
                summary.reproj_stats.min,
                summary.reproj_stats.q1,
                summary.reproj_stats.median,
                summary.reproj_stats.q3,
                summary.reproj_stats.max,
                summary.reproj_stats.mean,
                summary.reproj_stats.rms,
                summary.param_stats.min,
                summary.param_stats.q1,
                summary.param_stats.median,
                summary.param_stats.q3,
                summary.param_stats.max,
                summary.param_stats.mean,
                summary.param_stats.rms,
                summary.iter_stats.min,
                summary.iter_stats.q1,
                summary.iter_stats.median,
                summary.iter_stats.q3,
                summary.iter_stats.max,
                summary.iter_stats.mean,
                summary.iter_stats.rms,
            },
        );
    }

    try file_writer.flush();
}

pub fn writeScalarMapCsv(
    io: std.Io,
    out_dir: std.Io.Dir,
    file_name: []const u8,
    rows_num: usize,
    cols_num: usize,
    vals: []const F,
) !void {
    const Ctx = struct {
        vals: []const F,
        cols_num: usize,

        fn getVal(self: @This(), rr: usize, cc: usize) F {
            return self.vals[rr * self.cols_num + cc];
        }
    };

    try csvio.saveScalarGridCSV(
        io,
        out_dir,
        file_name,
        rows_num,
        cols_num,
        Ctx{
            .vals = vals,
            .cols_num = cols_num,
        },
        Ctx.getVal,
    );
}

pub fn writeScalarMapBmp(
    allocator: std.mem.Allocator,
    io: std.Io,
    out_dir: std.Io.Dir,
    file_name_no_ext: []const u8,
    rows_num: usize,
    cols_num: usize,
    vals: []const F,
) !void {
    var image = try ndarray.NDArray(F).initFlat(
        allocator,
        &[_]usize{ 1, rows_num, cols_num },
    );
    defer {
        allocator.free(image.slice);
        image.deinit(allocator);
    }

    for (0..rows_num) |rr| {
        for (0..cols_num) |cc| {
            image.set(&[_]usize{ 0, rr, cc }, vals[rr * cols_num + cc]);
        }
    }

    try iio.saveImage(
        io,
        out_dir,
        file_name_no_ext,
        &image,
        0,
        .{
            .format = .bmp,
            .bits = 8,
            .scaling = .auto,
            .channels = 1,
        },
    );
}

pub fn writeBinaryMaskBmp(
    allocator: std.mem.Allocator,
    io: std.Io,
    out_dir: std.Io.Dir,
    file_name_no_ext: []const u8,
    rows_num: usize,
    cols_num: usize,
    vals: []const F,
) !void {
    var image = try ndarray.NDArray(F).initFlat(
        allocator,
        &[_]usize{ 1, rows_num, cols_num },
    );
    defer {
        allocator.free(image.slice);
        image.deinit(allocator);
    }

    for (0..rows_num) |rr| {
        for (0..cols_num) |cc| {
            image.set(&[_]usize{ 0, rr, cc }, vals[rr * cols_num + cc]);
        }
    }

    try iio.saveImage(
        io,
        out_dir,
        file_name_no_ext,
        &image,
        0,
        .{
            .format = .bmp,
            .bits = 8,
            .scaling = .{ .fixed = .{ 0.0, 1.0 } },
            .channels = 1,
        },
    );
}

pub fn compareTol(stats: ScalarStats, tol_val: F) struct {
    median_ratio: F,
    q3_ratio: F,
    max_ratio: F,
} {
    return .{
        .median_ratio = stats.median / tol_val,
        .q3_ratio = stats.q3 / tol_val,
        .max_ratio = stats.max / tol_val,
    };
}
