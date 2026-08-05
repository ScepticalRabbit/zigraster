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

const VecSB = buildconfig.VecSB;
const VecSF = buildconfig.VecSF;
const VecSU8 = buildconfig.VecSU8;
const iter_max = cfg.raster_newton_iter_max;
const tol = cfg.tol;

// --------------------------------------------------------------------------------------
// Public Entry-Point Func
// --------------------------------------------------------------------------------------

pub fn solveFastSIMD(
    comptime N: usize,
    v_targ_x: VecSF,
    v_targ_y: VecSF,
    elem_node_x: []const F,
    elem_node_y: []const F,
    elem_node_w: []const F,
    v_xi_seed: VecSF,
    v_eta_seed: VecSF,
    v_node_vals: *[N]VecSF,
) common.NewtonResFastSIMD {
    const v_strict_resid_norm_tol: VecSF = @splat(tol.newton.norm_resid);
    const v_relaxed_resid_norm_tol: VecSF = @splat(tol.newton.stagnation_norm_resid);
    const v_abs_det_tol: VecSF = @splat(tol.newton.rel_det);
    const v_step_abs: VecSF = @splat(tol.newton.para_step_abs);
    const v_step_rel: VecSF = @splat(tol.newton.para_step_rel);
    const v_max_para_step: VecSF = @splat(tol.newton.max_para_step);
    const v_one: VecSF = @splat(1.0);
    const v_zero: VecSF = @splat(0.0);

    const use_compwise_resid = F == f64;
    const use_relaxed_resid = F == f32;
    const use_step_conv = F == f32;
    const lim_para_step = F == f32;

    var v_xi = v_xi_seed;
    var v_eta = v_eta_seed;
    var v_conv: VecSB = @splat(false);
    var v_fail_near_singular: VecSB = @splat(false);
    var v_iters: VecSU8 = @splat(0);
    var v_active: VecSB = @splat(true);
    var v_resid_x_final: VecSF = @splat(0.0);
    var v_resid_y_final: VecSF = @splat(0.0);

    var v_term_x: [N]VecSF = undefined;
    var v_term_y: [N]VecSF = undefined;
    inline for (0..N) |nn| {
        const v_node_x: VecSF = @splat(elem_node_x[nn]);
        const v_node_y: VecSF = @splat(elem_node_y[nn]);
        const v_node_w: VecSF = @splat(elem_node_w[nn]);
        v_term_x[nn] = @mulAdd(VecSF, v_targ_x, v_node_w, -v_node_x);
        v_term_y[nn] = @mulAdd(VecSF, v_targ_y, v_node_w, -v_node_y);
    }

    var v_deriv_n_xi: [N]VecSF = undefined;
    var v_deriv_n_eta: [N]VecSF = undefined;

    for (0..iter_max) |ii| {
        if (!@reduce(.Or, v_active)) break;

        const v_idx: VecSU8 = @splat(@intCast(ii + 1));
        v_iters = @select(u8, v_active, v_idx, v_iters);

        shapefun.shapeFuncSIMD(N, v_xi, v_eta, v_node_vals, &v_deriv_n_xi, &v_deriv_n_eta);

        var v_resid_x: VecSF = @splat(0.0);
        var v_resid_y: VecSF = @splat(0.0);
        var v_interp_w: VecSF = @splat(0.0);
        var v_jac_11: VecSF = @splat(0.0);
        var v_jac_12: VecSF = @splat(0.0);
        var v_jac_21: VecSF = @splat(0.0);
        var v_jac_22: VecSF = @splat(0.0);

        inline for (0..N) |nn| {
            v_resid_x = @mulAdd(VecSF, v_node_vals[nn], v_term_x[nn], v_resid_x);
            v_resid_y = @mulAdd(VecSF, v_node_vals[nn], v_term_y[nn], v_resid_y);

            const v_node_w: VecSF = @splat(elem_node_w[nn]);
            v_interp_w = @mulAdd(VecSF, v_node_vals[nn], v_node_w, v_interp_w);
            v_jac_11 = @mulAdd(VecSF, v_deriv_n_xi[nn], v_term_x[nn], v_jac_11);
            v_jac_12 = @mulAdd(VecSF, v_deriv_n_eta[nn], v_term_x[nn], v_jac_12);
            v_jac_21 = @mulAdd(VecSF, v_deriv_n_xi[nn], v_term_y[nn], v_jac_21);
            v_jac_22 = @mulAdd(VecSF, v_deriv_n_eta[nn], v_term_y[nn], v_jac_22);
        }

        v_resid_x_final = v_resid_x;
        v_resid_y_final = v_resid_y;

        const v_w_abs = @abs(v_interp_w);
        const v_strict_w_scaled_tol = v_w_abs * v_strict_resid_norm_tol;
        const v_strict_resid = blk: {
            if (comptime use_compwise_resid) {
                break :blk v_active & (v_w_abs > v_zero) &
                    (@abs(v_resid_x) <= v_strict_w_scaled_tol) &
                    (@abs(v_resid_y) <= v_strict_w_scaled_tol);
            }

            const v_resid_sq = v_resid_x * v_resid_x + v_resid_y * v_resid_y;
            break :blk v_active & (v_w_abs > v_zero) &
                (v_resid_sq <= v_strict_w_scaled_tol * v_strict_w_scaled_tol);
        };

        v_conv |= v_strict_resid;
        v_active &= !v_strict_resid;
        if (!@reduce(.Or, v_active)) break;

        const v_det = @mulAdd(VecSF, v_jac_11, v_jac_22, -(v_jac_12 * v_jac_21));
        const v_near_singular = v_active & (@abs(v_det) <= v_abs_det_tol);
        v_fail_near_singular |= v_near_singular;
        v_active &= !v_near_singular;
        if (!@reduce(.Or, v_active)) break;

        const v_safe_det = @select(F, v_active, v_det, v_one);
        const v_inv_det = v_one / v_safe_det;
        var v_step_xi = v_inv_det * @mulAdd(
            VecSF,
            v_jac_22,
            v_resid_x,
            -(v_jac_12 * v_resid_y),
        );
        var v_step_eta = v_inv_det * @mulAdd(
            VecSF,
            v_jac_11,
            v_resid_y,
            -(v_jac_21 * v_resid_x),
        );

        if (comptime use_relaxed_resid) {
            const v_relaxed_w_scaled_tol = v_w_abs * v_relaxed_resid_norm_tol;
            const v_resid_sq = v_resid_x * v_resid_x + v_resid_y * v_resid_y;
            const v_relaxed_resid = v_active & (v_w_abs > v_zero) &
                (v_resid_sq <= v_relaxed_w_scaled_tol * v_relaxed_w_scaled_tol);
            const v_step_tol_xi =
                v_step_abs + v_step_rel * @max(@abs(v_xi), v_one);
            const v_step_tol_eta =
                v_step_abs + v_step_rel * @max(@abs(v_eta), v_one);
            const v_met_step = if (comptime use_step_conv)
                v_active & (@abs(v_step_xi) <= v_step_tol_xi) &
                    (@abs(v_step_eta) <= v_step_tol_eta)
            else
                @as(VecSB, @splat(false));

            const v_conv_step = v_met_step & v_relaxed_resid;
            v_conv |= v_conv_step;
            v_active &= !v_conv_step;
            if (!@reduce(.Or, v_active)) break;
        }

        if (comptime lim_para_step) {
            const v_max_comp = @max(@abs(v_step_xi), @abs(v_step_eta));
            const v_safe_max_comp = @select(
                F,
                v_max_comp > v_zero,
                v_max_comp,
                v_one,
            );
            const v_step_scale = @min(v_one, v_max_para_step / v_safe_max_comp);
            v_step_xi *= v_step_scale;
            v_step_eta *= v_step_scale;
        }

        v_xi = @select(F, v_active, v_xi - v_step_xi, v_xi);
        v_eta = @select(F, v_active, v_eta - v_step_eta, v_eta);
    }

    const v_eps: VecSF = @splat(tol.newton.para_dom);
    const v_splat_one: VecSF = @splat(1.0);
    const v_splat_neg_one: VecSF = @splat(-1.0);
    const v_is_in = if (comptime N == 6)
        (v_xi >= -v_eps) & (v_eta >= -v_eps) &
            ((v_xi + v_eta) <= v_splat_one + v_eps)
    else
        (v_xi >= v_splat_neg_one - v_eps) & (v_xi <= v_splat_one + v_eps) &
            (v_eta >= v_splat_neg_one - v_eps) & (v_eta <= v_splat_one + v_eps);

    const v_final_conv = v_conv & v_is_in;
    const v_pre_dom_conv = v_conv;
    const v_hit_iter_lim = !v_pre_dom_conv & !v_fail_near_singular;

    return .{
        .v_conv = v_final_conv,
        .v_pre_dom_conv = v_pre_dom_conv,
        .v_hit_iter_lim = v_hit_iter_lim,
        .v_iters = v_iters,
        .v_resid_x = v_resid_x_final,
        .v_resid_y = v_resid_y_final,
        .v_xi = v_xi,
        .v_eta = v_eta,
    };
}

pub fn solveRobustSIMD(
    comptime N: usize,
    v_targ_x: VecSF,
    v_targ_y: VecSF,
    elem_node_x: []const F,
    elem_node_y: []const F,
    elem_node_w: []const F,
    v_xi_seed: VecSF,
    v_eta_seed: VecSF,
    v_node_vals: *[N]VecSF,
) common.NewtonResRobustSIMD {
    const v_strict_resid_norm_tol: VecSF = @splat(tol.newton.norm_resid);
    const v_relaxed_resid_norm_tol: VecSF =
        @splat(tol.newton.stagnation_norm_resid);
    const rel_det_tol_sq = tol.newton.rel_det * tol.newton.rel_det;
    const v_rel_det_tol_sq: VecSF = @splat(rel_det_tol_sq);
    const v_eps: VecSF = @splat(tol.newton.para_dom);
    const v_step_abs: VecSF = @splat(tol.newton.para_step_abs);
    const v_step_rel: VecSF = @splat(tol.newton.para_step_rel);
    const v_max_para_step: VecSF = @splat(tol.newton.max_para_step);
    const v_one: VecSF = @splat(1.0);
    const v_zero: VecSF = @splat(0.0);

    var v_xi = v_xi_seed;
    var v_eta = v_eta_seed;
    var v_xi_two_back: VecSF = @splat(0.0);
    var v_eta_two_back: VecSF = @splat(0.0);
    var v_has_two_back: VecSB = @splat(false);

    var v_conv: VecSB = @splat(false);
    var v_iters: VecSU8 = @splat(0);
    var v_active: VecSB = @splat(true);
    var v_fail_near_singular: VecSB = @splat(false);
    var v_resid_x_final: VecSF = @splat(0.0);
    var v_resid_y_final: VecSF = @splat(0.0);

    var v_term_x: [N]VecSF = undefined;
    var v_term_y: [N]VecSF = undefined;
    inline for (0..N) |nn| {
        const v_node_x: VecSF = @splat(elem_node_x[nn]);
        const v_node_y: VecSF = @splat(elem_node_y[nn]);
        const v_node_w: VecSF = @splat(elem_node_w[nn]);
        v_term_x[nn] = @mulAdd(VecSF, v_targ_x, v_node_w, -v_node_x);
        v_term_y[nn] = @mulAdd(VecSF, v_targ_y, v_node_w, -v_node_y);
    }

    var v_deriv_n_xi: [N]VecSF = undefined;
    var v_deriv_n_eta: [N]VecSF = undefined;

    for (0..iter_max) |ii| {
        if (!@reduce(.Or, v_active)) break;

        const v_idx: VecSU8 = @splat(@intCast(ii + 1));
        v_iters = @select(u8, v_active, v_idx, v_iters);

        shapefun.shapeFuncSIMD(
            N,
            v_xi,
            v_eta,
            v_node_vals,
            &v_deriv_n_xi,
            &v_deriv_n_eta,
        );

        var v_resid_x: VecSF = @splat(0.0);
        var v_resid_y: VecSF = @splat(0.0);
        var v_interp_w: VecSF = @splat(0.0);
        var v_jac_11: VecSF = @splat(0.0);
        var v_jac_12: VecSF = @splat(0.0);
        var v_jac_21: VecSF = @splat(0.0);
        var v_jac_22: VecSF = @splat(0.0);

        inline for (0..N) |nn| {
            v_resid_x = @mulAdd(
                VecSF,
                v_node_vals[nn],
                v_term_x[nn],
                v_resid_x,
            );
            v_resid_y = @mulAdd(
                VecSF,
                v_node_vals[nn],
                v_term_y[nn],
                v_resid_y,
            );
            const v_node_w: VecSF = @splat(elem_node_w[nn]);
            v_interp_w = @mulAdd(
                VecSF,
                v_node_vals[nn],
                v_node_w,
                v_interp_w,
            );
            v_jac_11 = @mulAdd(
                VecSF,
                v_deriv_n_xi[nn],
                v_term_x[nn],
                v_jac_11,
            );
            v_jac_12 = @mulAdd(
                VecSF,
                v_deriv_n_eta[nn],
                v_term_x[nn],
                v_jac_12,
            );
            v_jac_21 = @mulAdd(
                VecSF,
                v_deriv_n_xi[nn],
                v_term_y[nn],
                v_jac_21,
            );
            v_jac_22 = @mulAdd(
                VecSF,
                v_deriv_n_eta[nn],
                v_term_y[nn],
                v_jac_22,
            );
        }

        v_resid_x_final = v_resid_x;
        v_resid_y_final = v_resid_y;

        const v_invalid_state = v_active & !finiteMask(v_resid_x) |
            v_active & !finiteMask(v_resid_y) |
            v_active & !finiteMask(v_interp_w) |
            v_active & !finiteMask(v_jac_11) |
            v_active & !finiteMask(v_jac_12) |
            v_active & !finiteMask(v_jac_21) |
            v_active & !finiteMask(v_jac_22);
        v_active &= !v_invalid_state;

        const v_w_abs = @abs(v_interp_w);
        const v_strict_w_scaled_tol = v_w_abs * v_strict_resid_norm_tol;
        const v_relaxed_w_scaled_tol = v_w_abs * v_relaxed_resid_norm_tol;
        const v_resid_sq = v_resid_x * v_resid_x + v_resid_y * v_resid_y;
        const v_strict_resid = v_active & (v_w_abs > v_zero) &
            (v_resid_sq <= v_strict_w_scaled_tol * v_strict_w_scaled_tol);
        const v_relaxed_resid = v_active & (v_w_abs > v_zero) &
            (v_resid_sq <= v_relaxed_w_scaled_tol * v_relaxed_w_scaled_tol);

        v_conv |= v_strict_resid;
        v_active &= !v_strict_resid;
        if (!@reduce(.Or, v_active)) break;

        const v_det = @mulAdd(
            VecSF,
            v_jac_11,
            v_jac_22,
            -(v_jac_12 * v_jac_21),
        );
        const v_invalid_det_state = v_active & !finiteMask(v_det);
        v_active &= !v_invalid_det_state;

        const v_col_xi_norm_sq = @mulAdd(
            VecSF,
            v_jac_11,
            v_jac_11,
            v_jac_21 * v_jac_21,
        );
        const v_col_eta_norm_sq = @mulAdd(
            VecSF,
            v_jac_12,
            v_jac_12,
            v_jac_22 * v_jac_22,
        );
        const v_det_sq = v_det * v_det;
        const v_near_singular = v_active &
            (v_det_sq <= v_rel_det_tol_sq * v_col_xi_norm_sq * v_col_eta_norm_sq);
        v_fail_near_singular |= v_near_singular;
        v_active &= !v_near_singular;
        if (!@reduce(.Or, v_active)) break;

        const v_safe_det = @select(F, v_active, v_det, v_one);
        const v_inv_det = v_one / v_safe_det;
        const v_invalid_inv_det = v_active & !finiteMask(v_inv_det);
        v_active &= !v_invalid_inv_det;
        if (!@reduce(.Or, v_active)) break;

        const v_delta_xi_num = @mulAdd(
            VecSF,
            v_jac_22,
            v_resid_x,
            -(v_jac_12 * v_resid_y),
        );
        const v_delta_eta_num = @mulAdd(
            VecSF,
            v_jac_11,
            v_resid_y,
            -(v_jac_21 * v_resid_x),
        );
        const v_step_xi = v_inv_det * v_delta_xi_num;
        const v_step_eta = v_inv_det * v_delta_eta_num;
        const v_step_tol_xi = v_step_abs + v_step_rel * @max(@abs(v_xi), v_one);
        const v_step_tol_eta = v_step_abs + v_step_rel * @max(@abs(v_eta), v_one);
        const v_met_step = v_active &
            (@abs(v_step_xi) <= v_step_tol_xi) &
            (@abs(v_step_eta) <= v_step_tol_eta);

        var v_lim_step_xi = v_step_xi;
        var v_lim_step_eta = v_step_eta;
        const v_max_comp = @max(@abs(v_step_xi), @abs(v_step_eta));
        const v_safe_max_comp = @select(F, v_max_comp > v_zero, v_max_comp, v_one);
        const v_step_scale = @min(v_one, v_max_para_step / v_safe_max_comp);
        v_lim_step_xi *= v_step_scale;
        v_lim_step_eta *= v_step_scale;

        const v_next_xi = v_xi - v_lim_step_xi;
        const v_next_eta = v_eta - v_lim_step_eta;
        const v_stagnated = v_active & (v_next_xi == v_xi) & (v_next_eta == v_eta);
        const v_two_cycle = v_active & v_has_two_back &
            (v_next_xi == v_xi_two_back) &
            (v_next_eta == v_eta_two_back);
        const v_machine_lim = v_met_step | v_stagnated | v_two_cycle;
        const v_conv_step = v_machine_lim & v_relaxed_resid;

        v_conv |= v_conv_step;
        v_active &= !v_conv_step;

        const v_invalid_step = v_active &
            (!finiteMask(v_step_xi) |
                !finiteMask(v_step_eta) |
                !finiteMask(v_next_xi) |
                !finiteMask(v_next_eta));
        v_active &= !v_invalid_step;
        if (!@reduce(.Or, v_active)) break;

        v_xi_two_back = @select(F, v_active, v_xi, v_xi_two_back);
        v_eta_two_back = @select(F, v_active, v_eta, v_eta_two_back);
        v_has_two_back |= v_active;
        v_xi = @select(F, v_active, v_next_xi, v_xi);
        v_eta = @select(F, v_active, v_next_eta, v_eta);
    }

    const v_splat_one: VecSF = @splat(1.0);
    const v_splat_neg_one: VecSF = @splat(-1.0);
    const v_is_in = if (comptime N == 6)
        (v_xi >= -v_eps) & (v_eta >= -v_eps) &
            ((v_xi + v_eta) <= v_splat_one + v_eps)
    else
        (v_xi >= v_splat_neg_one - v_eps) & (v_xi <= v_splat_one + v_eps) &
            (v_eta >= v_splat_neg_one - v_eps) & (v_eta <= v_splat_one + v_eps);

    const v_fail_dom = v_conv & !v_is_in;
    const v_final_conv = v_conv & v_is_in;
    const v_pre_dom_conv = v_conv;
    const v_status_fail_iter_lim: VecSU8 =
        @splat(@intFromEnum(common.NewtonStatus.fail_iter_lim));
    const v_status_fail_near_singular: VecSU8 =
        @splat(@intFromEnum(common.NewtonStatus.fail_near_singular));
    const v_status_conv_resid: VecSU8 =
        @splat(@intFromEnum(common.NewtonStatus.conv_resid));
    const v_status_fail_dom: VecSU8 =
        @splat(@intFromEnum(common.NewtonStatus.fail_dom));
    var v_status: VecSU8 =
        v_status_fail_iter_lim;
    v_status = @select(
        u8,
        v_fail_near_singular,
        v_status_fail_near_singular,
        v_status,
    );
    v_status = @select(
        u8,
        v_pre_dom_conv,
        v_status_conv_resid,
        v_status,
    );
    v_status = @select(
        u8,
        v_fail_dom,
        v_status_fail_dom,
        v_status,
    );

    return .{
        .v_conv = v_final_conv,
        .v_pre_dom_conv = v_pre_dom_conv | v_fail_dom,
        .v_iters = v_iters,
        .v_status = v_status,
        .v_resid_x = v_resid_x_final,
        .v_resid_y = v_resid_y_final,
        .v_xi = v_xi,
        .v_eta = v_eta,
    };
}

// --------------------------------------------------------------------------------------
// Generic Low-Level Helpers
// --------------------------------------------------------------------------------------

inline fn finiteMask(vals: VecSF) VecSB {
    const v_inf: VecSF = @splat(std.math.inf(F));
    return (vals == vals) & (vals != v_inf) & (vals != -v_inf);
}
