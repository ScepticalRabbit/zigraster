// --------------------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const buildconfig = @import("buildconfig.zig");
const F = buildconfig.F;
const S = buildconfig.SimdWidth;
const newton = @import("newton.zig");
const Timestamp = std.Io.Clock.Timestamp;
const rastcfg = @import("rasterconfig.zig");
const rops = @import("rasterops.zig");

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const ReportMode = rastcfg.ReportMode;

pub const TileScope = struct {
    start: Timestamp,
};

// --------------------------------------------------------------------------------------
// Public Entry-Point Func
// --------------------------------------------------------------------------------------

pub inline fn beginTile(io: std.Io) TileScope {
    return .{
        .start = Timestamp.now(io, .awake),
    };
}

pub inline fn finishTile(
    ctx_report: anytype,
    ctx_rast: rops.RasterContext,
    tile: rops.ActiveTile,
    tile_duration_ns: u64,
    shaded_px: u64,
    elem_count: usize,
    cam_duration_ns: u64,
    resolve_duration_ns: u64,
) void {
    const screen_px_x = @as(
        u16,
        @intCast(ctx_rast.camera.pixels_num[0]),
    );

    const tiles_x =
        (screen_px_x + ctx_rast.tile_size - 1) /
        ctx_rast.tile_size;

    const spatial_idx =
        (tile.y_px_min / ctx_rast.tile_size) * tiles_x +
        (tile.x_px_min / ctx_rast.tile_size);

    ctx_report.recordTile(
        spatial_idx,
        tile_duration_ns,
        shaded_px,
        elem_count,
    );

    ctx_report.recordCamTime(cam_duration_ns);
    ctx_report.recordResolveTime(resolve_duration_ns);
}

pub inline fn recordEarlyOut(
    ctx_report: anytype,
    global_subx: usize,
    global_suby: usize,
    passed: bool,
) void {
    ctx_report.recordEarlyOut(global_subx, global_suby, passed);
}

pub inline fn recordPixelConvStats(
    ctx_report: anytype,
    global_subx: usize,
    global_suby: usize,
    conv: bool,
    xi: F,
    eta: F,
    jac_det: F,
) void {
    ctx_report.recordPixelConv(global_subx, global_suby, conv);
    ctx_report.recordPixelXi(global_subx, global_suby, xi);
    ctx_report.recordPixelEta(global_subx, global_suby, eta);
    ctx_report.recordPixelJacDet(global_subx, global_suby, jac_det);
}

pub inline fn recordTri3SIMDConvStats(
    ctx_report: anytype,
    tile: rops.ActiveTile,
    sub_samp: usize,
    scratch_x_u: usize,
    scratch_y_u: usize,
    v_x_mask: @Vector(S, bool),
    v_mask_active: @Vector(S, bool),
    v_weights: [3]@Vector(S, F),
    v_inv_z: @Vector(S, F),
    nodes_inv_z: [3]F,
    nodes_x: []const F,
    nodes_y: []const F,
) void {
    const lane_x_mask: [S]bool = v_x_mask;
    const lane_active_mask: [S]bool = v_mask_active;
    const lane_weights_0: [S]F = v_weights[0];
    const lane_weights_1: [S]F = v_weights[1];
    const lane_weights_2: [S]F = v_weights[2];
    const lane_inv_z: [S]F = v_inv_z;

    for (0..S) |ll| {
        if (!lane_x_mask[ll]) continue;

        const global_subx =
            @as(usize, @intCast(tile.scratch_x_px_min)) *
            sub_samp +
            scratch_x_u +
            ll;
        const global_suby =
            @as(usize, @intCast(tile.scratch_y_px_min)) *
            sub_samp +
            scratch_y_u;

        if (lane_active_mask[ll]) {
            const weights = [3]F{
                lane_weights_0[ll],
                lane_weights_1[ll],
                lane_weights_2[ll],
            };
            const inv_z = lane_inv_z[ll];
            const xi = weights[1] * nodes_inv_z[1] / inv_z;
            const eta = weights[2] * nodes_inv_z[2] / inv_z;
            recordPixelConvStats(
                ctx_report,
                global_subx,
                global_suby,
                true,
                xi,
                eta,
                newton.calcJacDet2D(
                    3,
                    xi,
                    eta,
                    nodes_x,
                    nodes_y,
                ),
            );
            continue;
        }

        const nan = std.math.nan(F);
        recordPixelConvStats(
            ctx_report,
            global_subx,
            global_suby,
            false,
            nan,
            nan,
            nan,
        );
    }
}

pub inline fn recordTri3SteppedSIMDConvStats(
    ctx_report: anytype,
    tile: rops.ActiveTile,
    sub_samp: usize,
    scratch_x_u: usize,
    global_suby: usize,
    v_x_mask: @Vector(S, bool),
    v_mask_active: @Vector(S, bool),
    v_weights: [3]@Vector(S, F),
    v_inv_z: @Vector(S, F),
    is_const_depth: bool,
    inv_z1: F,
    inv_z2: F,
    jac_det: F,
) void {
    const lane_x_mask: [S]bool = v_x_mask;
    const lane_active_mask: [S]bool = v_mask_active;
    const lane_weights_0: [S]F = v_weights[0];
    const lane_weights_1: [S]F = v_weights[1];
    const lane_weights_2: [S]F = v_weights[2];
    const lane_inv_z: [S]F = v_inv_z;

    for (0..S) |ll| {
        if (!lane_x_mask[ll]) continue;

        const global_subx =
            @as(usize, @intCast(tile.scratch_x_px_min)) *
            sub_samp +
            scratch_x_u +
            ll;

        if (lane_active_mask[ll]) {
            const weights = [3]F{
                lane_weights_0[ll],
                lane_weights_1[ll],
                lane_weights_2[ll],
            };
            const inv_z = lane_inv_z[ll];
            const xi = if (is_const_depth)
                weights[1]
            else
                @mulAdd(F, weights[1], inv_z1, 0.0) / inv_z;
            const eta = if (is_const_depth)
                weights[2]
            else
                @mulAdd(F, weights[2], inv_z2, 0.0) / inv_z;

            recordPixelConvStats(
                ctx_report,
                global_subx,
                global_suby,
                true,
                xi,
                eta,
                jac_det,
            );
            continue;
        }

        const nan = std.math.nan(F);
        recordPixelConvStats(
            ctx_report,
            global_subx,
            global_suby,
            false,
            nan,
            nan,
            nan,
        );
    }
}

pub inline fn recordNewtonSIMDChunkStats(
    comptime NodesNum: comptime_int,
    comptime DomViolation: anytype,
    ctx_report: anytype,
    tile: rops.ActiveTile,
    sub_samp: usize,
    subpx_simd_chunk: anytype,
    v_chunk_mask: @Vector(S, bool),
    v_conv_mask: @Vector(S, bool),
    v_iters: @Vector(S, u8),
    v_status: @Vector(S, u8),
    v_pre_dom_conv: @Vector(S, bool),
    v_xi_final: @Vector(S, F),
    v_eta_final: @Vector(S, F),
    subpx_x_off: F,
    subpx_y_off: F,
    nodes_x: []const F,
    nodes_y: []const F,
    nodes_z: []const F,
) void {
    const chunk_mask_arr: [S]bool = v_chunk_mask;
    const conv_mask_arr: [S]bool = v_conv_mask;
    const iters_arr: [S]u8 = v_iters;
    const status_arr_u8: [S]u8 = v_status;
    const pre_dom_arr: [S]bool = v_pre_dom_conv;
    const xi_final_arr: [S]F = v_xi_final;
    const eta_final_arr: [S]F = v_eta_final;

    for (0..S) |jj| {
        if (!chunk_mask_arr[jj]) continue;

        const global_subx =
            @as(usize, subpx_simd_chunk.scratch_x_u[jj]) +
            @as(usize, @intCast(tile.scratch_x_px_min)) * sub_samp;
        const global_suby =
            @as(usize, subpx_simd_chunk.scratch_y_u[jj]) +
            @as(usize, @intCast(tile.scratch_y_px_min)) * sub_samp;
        const solve_state = newton.evaluateSolveState(
            NodesNum,
            subpx_simd_chunk.px_f[jj] - subpx_x_off,
            subpx_simd_chunk.py_f[jj] - subpx_y_off,
            nodes_x,
            nodes_y,
            nodes_z,
            xi_final_arr[jj],
            eta_final_arr[jj],
        );
        const dom_violation = DomViolation(
            xi_final_arr[jj],
            eta_final_arr[jj],
        );
        const status: newton.NewtonStatus = @enumFromInt(status_arr_u8[jj]);
        const hit_iter_lim = newton.hitIterLimitStatus(status);
        const jac_det = newton.calcJacDet2D(
            NodesNum,
            xi_final_arr[jj],
            eta_final_arr[jj],
            nodes_x,
            nodes_y,
        );

        recordPixelIters(
            ctx_report,
            global_subx,
            global_suby,
            iters_arr[jj],
        );
        recordPixelConvStats(
            ctx_report,
            global_subx,
            global_suby,
            conv_mask_arr[jj],
            xi_final_arr[jj],
            eta_final_arr[jj],
            jac_det,
        );
        recordPixelSolverDiagnostics(
            ctx_report,
            global_subx,
            global_suby,
            status,
            pre_dom_arr[jj],
            hit_iter_lim,
            solve_state.resid_x,
            solve_state.resid_y,
            solve_state.interp_w,
            solve_state.resid_mag,
            solve_state.norm_resid_mag,
            dom_violation,
        );
    }
}

pub inline fn recordPixelSolverDiagnostics(
    ctx_report: anytype,
    global_subx: usize,
    global_suby: usize,
    status: newton.NewtonStatus,
    pre_dom_conv: bool,
    hit_iter_lim: bool,
    resid_x: F,
    resid_y: F,
    interp_w: F,
    resid_mag: F,
    norm_resid_mag: F,
    dom_violation: F,
) void {
    ctx_report.recordPixelSolverStatus(global_subx, global_suby, status);
    ctx_report.recordPixelPreDomConv(global_subx, global_suby, pre_dom_conv);
    ctx_report.recordPixelHitIterLimit(global_subx, global_suby, hit_iter_lim);
    ctx_report.recordPixelResidualX(global_subx, global_suby, resid_x);
    ctx_report.recordPixelResidualY(global_subx, global_suby, resid_y);
    ctx_report.recordPixelInterpolatedW(global_subx, global_suby, interp_w);
    ctx_report.recordPixelResidualMag(global_subx, global_suby, resid_mag);
    ctx_report.recordPixelNormalizedResidualMag(
        global_subx,
        global_suby,
        norm_resid_mag,
    );
    ctx_report.recordPixelDomViolation(global_subx, global_suby, dom_violation);
}

pub inline fn recordPixelIterAndOccupancy(
    ctx_report: anytype,
    global_subx: usize,
    global_suby: usize,
    iters: u8,
    occupancy_x: usize,
    occupancy_y: usize,
) void {
    ctx_report.recordPixelIters(global_subx, global_suby, iters);
    ctx_report.recordPixelOccupancy(occupancy_x, occupancy_y);
}

pub inline fn recordPixelIters(
    ctx_report: anytype,
    global_subx: usize,
    global_suby: usize,
    iters: u8,
) void {
    ctx_report.recordPixelIters(global_subx, global_suby, iters);
}
