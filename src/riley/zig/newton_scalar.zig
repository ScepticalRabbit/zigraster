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
const cfg = buildconfig.config;
const shapefun = @import("shapefun.zig");
const common = @import("newton_common.zig");

const iter_max = cfg.raster_newton_iter_max;
const tol = cfg.tol;

// --------------------------------------------------------------------------------------
// Public Entry-Point Func
// --------------------------------------------------------------------------------------

pub fn solveFastScal(
    comptime N: usize,
    targ_x: F,
    targ_y: F,
    elem_node_x: []const F,
    elem_node_y: []const F,
    elem_node_w: []const F,
    xi_seed: F,
    eta_seed: F,
    node_vals: *[N]F,
) common.NewtonResFastScal {
    const strict_resid_norm_tol = tol.newton.norm_resid;
    const relaxed_resid_norm_tol = tol.newton.stagnation_norm_resid;
    const abs_det_tol = tol.newton.rel_det;
    const eps = tol.newton.para_dom;
    const step_abs = tol.newton.para_step_abs;
    const step_rel = tol.newton.para_step_rel;
    const max_para_step = tol.newton.max_para_step;

    const use_compwise_resid = F == f64;
    const use_relaxed_resid = F == f32;
    const use_step_conv = F == f32;
    const lim_para_step = F == f32;

    var xi = xi_seed;
    var eta = eta_seed;

    var term_x: [N]F = undefined;
    var term_y: [N]F = undefined;
    inline for (0..N) |nn| {
        term_x[nn] = @mulAdd(F, targ_x, elem_node_w[nn], -elem_node_x[nn]);
        term_y[nn] = @mulAdd(F, targ_y, elem_node_w[nn], -elem_node_y[nn]);
    }

    var deriv_n_xi: [N]F = undefined;
    var deriv_n_eta: [N]F = undefined;
    var iters: u8 = 0;
    for (0..iter_max) |ii| {
        iters = @intCast(ii + 1);
        shapefun.shapeFunc(N, xi, eta, node_vals, &deriv_n_xi, &deriv_n_eta);

        var resid_x: F = 0.0;
        var resid_y: F = 0.0;
        var interp_w: F = 0.0;
        var jac_11: F = 0.0;
        var jac_12: F = 0.0;
        var jac_21: F = 0.0;
        var jac_22: F = 0.0;

        for (0..N) |nn| {
            resid_x = @mulAdd(F, node_vals[nn], term_x[nn], resid_x);
            resid_y = @mulAdd(F, node_vals[nn], term_y[nn], resid_y);
            interp_w = @mulAdd(F, node_vals[nn], elem_node_w[nn], interp_w);
            jac_11 = @mulAdd(F, deriv_n_xi[nn], term_x[nn], jac_11);
            jac_12 = @mulAdd(F, deriv_n_eta[nn], term_x[nn], jac_12);
            jac_21 = @mulAdd(F, deriv_n_xi[nn], term_y[nn], jac_21);
            jac_22 = @mulAdd(F, deriv_n_eta[nn], term_y[nn], jac_22);
        }

        const w_abs = @abs(interp_w);
        const strict_w_scaled_tol = w_abs * strict_resid_norm_tol;

        const strict_resid = blk: {
            if (comptime use_compwise_resid) {
                break :blk w_abs > 0.0 and
                    @abs(resid_x) <= strict_w_scaled_tol and
                    @abs(resid_y) <= strict_w_scaled_tol;
            }

            const resid_sq = resid_x * resid_x + resid_y * resid_y;
            break :blk w_abs > 0.0 and
                resid_sq <= strict_w_scaled_tol * strict_w_scaled_tol;
        };
        if (strict_resid) {
            const is_in = isInParaDom(N, xi, eta, eps);
            return .{
                .conv = is_in,
                .pre_dom_conv = true,
                .hit_iter_lim = false,
                .iters = iters,
                .xi = xi,
                .eta = eta,
            };
        }

        if (ii + 1 == iter_max) {
            break;
        }

        const det = @mulAdd(F, jac_11, jac_22, -(jac_12 * jac_21));
        if (@abs(det) <= abs_det_tol) {
            return .{
                .conv = false,
                .pre_dom_conv = false,
                .hit_iter_lim = false,
                .iters = iters,
                .xi = xi,
                .eta = eta,
            };
        }

        const inv_det = 1.0 / det;
        const delta_xi_num = @mulAdd(F, jac_22, resid_x, -(jac_12 * resid_y));
        const delta_eta_num = @mulAdd(F, jac_11, resid_y, -(jac_21 * resid_x));
        var step_xi = inv_det * delta_xi_num;
        var step_eta = inv_det * delta_eta_num;

        if (comptime use_relaxed_resid) {
            const relaxed_w_scaled_tol = w_abs * relaxed_resid_norm_tol;
            const resid_sq = resid_x * resid_x + resid_y * resid_y;
            const relaxed_resid = w_abs > 0.0 and
                resid_sq <= relaxed_w_scaled_tol * relaxed_w_scaled_tol;

            const step_tol_xi = step_abs + step_rel * @max(@abs(xi), @as(F, 1.0));
            const step_tol_eta = step_abs + step_rel * @max(@abs(eta), @as(F, 1.0));

            const met_step = if (comptime use_step_conv)
                @abs(step_xi) <= step_tol_xi and @abs(step_eta) <= step_tol_eta
            else
                false;

            if (met_step and relaxed_resid) {
                const is_in = isInParaDom(N, xi, eta, eps);
                return .{
                    .conv = is_in,
                    .pre_dom_conv = true,
                    .hit_iter_lim = false,
                    .iters = iters,
                    .xi = xi,
                    .eta = eta,
                };
            }
        }

        if (comptime lim_para_step) {
            const max_comp = @max(@abs(step_xi), @abs(step_eta));

            if (max_comp > max_para_step) {
                const step_scale = max_para_step / max_comp;
                step_xi *= step_scale;
                step_eta *= step_scale;
            }
        }

        xi -= step_xi;
        eta -= step_eta;
    }

    return .{
        .conv = false,
        .pre_dom_conv = false,
        .hit_iter_lim = true,
        .iters = iters,
        .xi = xi,
        .eta = eta,
    };
}

pub fn solveRobustScal(
    comptime N: usize,
    targ_x: F,
    targ_y: F,
    elem_node_x: []const F,
    elem_node_y: []const F,
    elem_node_w: []const F,
    xi_seed: F,
    eta_seed: F,
    node_vals: *[N]F,
) common.NewtonResRobustScal {
    const strict_resid_norm_tol = tol.newton.norm_resid;
    const relaxed_resid_norm_tol = tol.newton.stagnation_norm_resid;
    const rel_det_tol_sq = tol.newton.rel_det * tol.newton.rel_det;
    const eps = tol.newton.para_dom;
    const step_abs = tol.newton.para_step_abs;
    const step_rel = tol.newton.para_step_rel;
    const max_para_step = tol.newton.max_para_step;

    var xi = xi_seed;
    var eta = eta_seed;
    var xi_two_back: F = 0.0;
    var eta_two_back: F = 0.0;
    var has_two_back = false;

    var term_x: [N]F = undefined;
    var term_y: [N]F = undefined;
    inline for (0..N) |nn| {
        term_x[nn] = @mulAdd(F, targ_x, elem_node_w[nn], -elem_node_x[nn]);
        term_y[nn] = @mulAdd(F, targ_y, elem_node_w[nn], -elem_node_y[nn]);
    }

    var deriv_n_xi: [N]F = undefined;
    var deriv_n_eta: [N]F = undefined;
    var iters: u8 = 0;
    var resid_x: F = 0.0;
    var resid_y: F = 0.0;
    var status: common.NewtonStatus = .fail_iter_lim;
    var pre_dom_conv = false;

    for (0..iter_max) |ii| {
        iters = @intCast(ii + 1);
        shapefun.shapeFunc(N, xi, eta, node_vals, &deriv_n_xi, &deriv_n_eta);

        resid_x = 0.0;
        resid_y = 0.0;
        var interp_w: F = 0.0;
        var jac_11: F = 0.0;
        var jac_12: F = 0.0;
        var jac_21: F = 0.0;
        var jac_22: F = 0.0;

        for (0..N) |nn| {
            resid_x = @mulAdd(F, node_vals[nn], term_x[nn], resid_x);
            resid_y = @mulAdd(F, node_vals[nn], term_y[nn], resid_y);
            interp_w = @mulAdd(F, node_vals[nn], elem_node_w[nn], interp_w);
            jac_11 = @mulAdd(F, deriv_n_xi[nn], term_x[nn], jac_11);
            jac_12 = @mulAdd(F, deriv_n_eta[nn], term_x[nn], jac_12);
            jac_21 = @mulAdd(F, deriv_n_xi[nn], term_y[nn], jac_21);
            jac_22 = @mulAdd(F, deriv_n_eta[nn], term_y[nn], jac_22);
        }

        const invalid_resid_state =
            !std.math.isFinite(resid_x) or
            !std.math.isFinite(resid_y) or
            !std.math.isFinite(interp_w) or
            !std.math.isFinite(jac_11) or
            !std.math.isFinite(jac_12) or
            !std.math.isFinite(jac_21) or
            !std.math.isFinite(jac_22);
        if (invalid_resid_state) {
            status = .fail_invalid_state;
            break;
        }

        const w_abs = @abs(interp_w);
        const strict_w_scaled_tol = w_abs * strict_resid_norm_tol;
        const relaxed_w_scaled_tol = w_abs * relaxed_resid_norm_tol;
        const resid_sq = resid_x * resid_x + resid_y * resid_y;
        const strict_resid = w_abs > 0.0 and
            resid_sq <= strict_w_scaled_tol * strict_w_scaled_tol;
        const relaxed_resid = w_abs > 0.0 and
            resid_sq <= relaxed_w_scaled_tol * relaxed_w_scaled_tol;

        if (strict_resid) {
            status = .conv_resid;
            pre_dom_conv = true;
            break;
        }

        if (ii + 1 == iter_max) {
            status = .fail_iter_lim;
            break;
        }

        const det = @mulAdd(F, jac_11, jac_22, -(jac_12 * jac_21));
        const col_xi_norm_sq = @mulAdd(F, jac_11, jac_11, jac_21 * jac_21);
        const col_eta_norm_sq = @mulAdd(F, jac_12, jac_12, jac_22 * jac_22);
        const det_sq = det * det;
        const near_singular =
            det_sq <= rel_det_tol_sq * col_xi_norm_sq * col_eta_norm_sq;

        if (!std.math.isFinite(det)) {
            status = .fail_invalid_state;
            break;
        }
        if (near_singular) {
            status = .fail_near_singular;
            break;
        }

        const inv_det = 1.0 / det;
        if (!std.math.isFinite(inv_det)) {
            status = .fail_invalid_state;
            break;
        }

        const delta_xi_num = @mulAdd(F, jac_22, resid_x, -(jac_12 * resid_y));
        const delta_eta_num = @mulAdd(F, jac_11, resid_y, -(jac_21 * resid_x));
        const step_xi = inv_det * delta_xi_num;
        const step_eta = inv_det * delta_eta_num;
        const step_tol_xi = step_abs + step_rel * @max(@abs(xi), @as(F, 1.0));
        const step_tol_eta = step_abs + step_rel * @max(@abs(eta), @as(F, 1.0));
        const met_step = @abs(step_xi) <= step_tol_xi and @abs(step_eta) <= step_tol_eta;

        var lim_step_xi = step_xi;
        var lim_step_eta = step_eta;
        const max_comp = @max(@abs(step_xi), @abs(step_eta));
        if (max_comp > max_para_step) {
            const step_scale = max_para_step / max_comp;
            lim_step_xi *= step_scale;
            lim_step_eta *= step_scale;
        }

        const next_xi = xi - lim_step_xi;
        const next_eta = eta - lim_step_eta;
        const stagnated = next_xi == xi and next_eta == eta;
        const two_cycle = has_two_back and
            next_xi == xi_two_back and
            next_eta == eta_two_back;
        if ((met_step or stagnated or two_cycle) and relaxed_resid) {
            status = if (stagnated)
                .conv_stagnated
            else if (two_cycle)
                .conv_two_cycle
            else
                .conv_step;
            pre_dom_conv = true;
            break;
        }

        const invalid_step =
            !std.math.isFinite(step_xi) or
            !std.math.isFinite(step_eta) or
            !std.math.isFinite(next_xi) or
            !std.math.isFinite(next_eta);
        if (invalid_step) {
            status = .fail_invalid_step;
            break;
        }

        xi_two_back = xi;
        eta_two_back = eta;
        has_two_back = true;
        xi = next_xi;
        eta = next_eta;
    }

    if (common.isPreDomConvStatus(status)) {
        pre_dom_conv = true;
    }
    if (common.isConvStatus(status) and !isInParaDom(N, xi, eta, eps)) {
        status = .fail_dom;
        pre_dom_conv = true;
    }

    return .{
        .conv = common.isConvStatus(status) and status != .fail_dom,
        .pre_dom_conv = pre_dom_conv,
        .iters = iters,
        .status = status,
        .resid_x = resid_x,
        .resid_y = resid_y,
        .xi = xi,
        .eta = eta,
    };
}

pub fn reportSolve(
    comptime N: usize,
    writer: anytype,
    px_x: usize,
    px_y: usize,
    targ_x: F,
    targ_y: F,
    elem_node_x: []const F,
    elem_node_y: []const F,
    elem_node_w: []const F,
    xi_seed: F,
    eta_seed: F,
) !void {
    const strict_resid_norm_tol = tol.newton.norm_resid;
    const relaxed_resid_norm_tol = tol.newton.stagnation_norm_resid;
    const rel_det_tol_sq = tol.newton.rel_det * tol.newton.rel_det;
    const step_abs = tol.newton.para_step_abs;
    const step_rel = tol.newton.para_step_rel;
    const max_para_step = tol.newton.max_para_step;

    var xi = xi_seed;
    var eta = eta_seed;
    var xi_two_back: F = 0.0;
    var eta_two_back: F = 0.0;
    var has_two_back = false;

    var node_vals: [N]F = undefined;
    var deriv_n_xi: [N]F = undefined;
    var deriv_n_eta: [N]F = undefined;
    var term_x: [N]F = undefined;
    var term_y: [N]F = undefined;
    inline for (0..N) |nn| {
        term_x[nn] = @mulAdd(F, targ_x, elem_node_w[nn], -elem_node_x[nn]);
        term_y[nn] = @mulAdd(F, targ_y, elem_node_w[nn], -elem_node_y[nn]);
    }

    try writer.print(
        "Trace for pixel ({d}, {d}) seed=({d}, {d})\n",
        .{ px_x, px_y, xi_seed, eta_seed },
    );

    for (0..iter_max) |ii| {
        shapefun.shapeFunc(N, xi, eta, &node_vals, &deriv_n_xi, &deriv_n_eta);

        var resid_x: F = 0.0;
        var resid_y: F = 0.0;
        var interp_w: F = 0.0;
        var jac_11: F = 0.0;
        var jac_12: F = 0.0;
        var jac_21: F = 0.0;
        var jac_22: F = 0.0;

        for (0..N) |nn| {
            resid_x = @mulAdd(F, node_vals[nn], term_x[nn], resid_x);
            resid_y = @mulAdd(F, node_vals[nn], term_y[nn], resid_y);
            interp_w = @mulAdd(F, node_vals[nn], elem_node_w[nn], interp_w);
            jac_11 = @mulAdd(F, deriv_n_xi[nn], term_x[nn], jac_11);
            jac_12 = @mulAdd(F, deriv_n_eta[nn], term_x[nn], jac_12);
            jac_21 = @mulAdd(F, deriv_n_xi[nn], term_y[nn], jac_21);
            jac_22 = @mulAdd(F, deriv_n_eta[nn], term_y[nn], jac_22);
        }

        const w_abs = @abs(interp_w);
        const strict_w_scaled_tol = w_abs * strict_resid_norm_tol;
        const relaxed_w_scaled_tol = w_abs * relaxed_resid_norm_tol;
        const resid_sq = resid_x * resid_x + resid_y * resid_y;
        const strict_resid = w_abs > 0.0 and
            resid_sq <= strict_w_scaled_tol * strict_w_scaled_tol;
        const relaxed_resid = w_abs > 0.0 and
            resid_sq <= relaxed_w_scaled_tol * relaxed_w_scaled_tol;
        const det = @mulAdd(F, jac_11, jac_22, -(jac_12 * jac_21));
        const col_xi_norm_sq = @mulAdd(F, jac_11, jac_11, jac_21 * jac_21);
        const col_eta_norm_sq = @mulAdd(F, jac_12, jac_12, jac_22 * jac_22);
        const det_sq = det * det;
        const near_singular =
            det_sq <= rel_det_tol_sq * col_xi_norm_sq * col_eta_norm_sq;

        try writer.print(
            "iter={d} xi={d} eta={d} rx={d} ry={d} w={d} nres={d} det={d} strict={any} relaxed={any} near_singular={any}\n",
            .{
                ii + 1,
                xi,
                eta,
                resid_x,
                resid_y,
                interp_w,
                if (w_abs > 0.0) @sqrt(resid_sq) / w_abs else std.math.nan(F),
                det,
                strict_resid,
                relaxed_resid,
                near_singular,
            },
        );

        if (strict_resid) {
            try writer.writeAll("  -> conv_resid\n\n");
            return;
        }
        if (ii + 1 == iter_max) {
            try writer.writeAll("  -> fail_iter_lim\n\n");
            return;
        }
        if (near_singular) {
            try writer.writeAll("  -> fail_near_singular\n\n");
            return;
        }

        const inv_det = 1.0 / det;
        const delta_xi_num = @mulAdd(F, jac_22, resid_x, -(jac_12 * resid_y));
        const delta_eta_num = @mulAdd(F, jac_11, resid_y, -(jac_21 * resid_x));
        const step_xi = inv_det * delta_xi_num;
        const step_eta = inv_det * delta_eta_num;
        const step_tol_xi =
            step_abs + step_rel * @max(@abs(xi), @as(F, 1.0));
        const step_tol_eta =
            step_abs + step_rel * @max(@abs(eta), @as(F, 1.0));
        const met_step =
            @abs(step_xi) <= step_tol_xi and
            @abs(step_eta) <= step_tol_eta;

        var lim_step_xi = step_xi;
        var lim_step_eta = step_eta;
        const max_comp = @max(@abs(step_xi), @abs(step_eta));
        if (max_comp > max_para_step) {
            const step_scale = max_para_step / max_comp;
            lim_step_xi *= step_scale;
            lim_step_eta *= step_scale;
        }

        const next_xi = xi - lim_step_xi;
        const next_eta = eta - lim_step_eta;
        const stagnated = next_xi == xi and next_eta == eta;
        const two_cycle = has_two_back and
            next_xi == xi_two_back and
            next_eta == eta_two_back;

        try writer.print(
            "  step_xi={d} step_eta={d} lim_step_xi={d} lim_step_eta={d} met_step={any} stagnated={any} two_cycle={any} next_xi={d} next_eta={d}\n",
            .{
                step_xi,
                step_eta,
                lim_step_xi,
                lim_step_eta,
                met_step,
                stagnated,
                two_cycle,
                next_xi,
                next_eta,
            },
        );

        if ((met_step or stagnated or two_cycle) and relaxed_resid) {
            const label = if (stagnated)
                "conv_stagnated"
            else if (two_cycle)
                "conv_two_cycle"
            else
                "conv_step";
            try writer.print("  -> {s}\n\n", .{label});
            return;
        }

        xi_two_back = xi;
        eta_two_back = eta;
        has_two_back = true;
        xi = next_xi;
        eta = next_eta;
    }
}

// --------------------------------------------------------------------------------------
// Generic Low-Level Helpers
// --------------------------------------------------------------------------------------

inline fn isInParaDom(
    comptime N: usize,
    xi: F,
    eta: F,
    eps: F,
) bool {
    if (comptime N == 6) {
        return xi >= -eps and eta >= -eps and (xi + eta) <= 1.0 + eps;
    }

    return xi >= -1.0 - eps and xi <= 1.0 + eps and
        eta >= -1.0 - eps and eta <= 1.0 + eps;
}
