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
const rastcfg = @import("rasterconfig.zig");

const S = buildconfig.SimdWidth;
const VecSB = buildconfig.VecSB;
const VecSF = buildconfig.VecSF;
const VecSU8 = buildconfig.VecSU8;
const tol = cfg.tol;

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const NewtonSeed = struct {
    xi: F,
    eta: F,
};

pub const NewtonSeedSIMD = struct {
    v_xi: VecSF,
    v_eta: VecSF,
};

pub const NewtonSeedState = struct {
    is_valid: bool = false,
    xi: F = 0.0,
    eta: F = 0.0,
};

pub const NewtonSeedQuality = struct {
    is_usable: bool,
    dom_violation: F,
    resid_sq: F,
    det_abs: F,
};

pub const NewtonEvalState = struct {
    resid_x: F,
    resid_y: F,
    interp_w: F,
    resid_mag: F,
    norm_resid_x: F,
    norm_resid_y: F,
    norm_resid_mag: F,
};

pub const NewtonStatus = enum(u8) {
    conv_resid,
    conv_step,
    conv_stagnated,
    conv_two_cycle,
    fail_dom,
    fail_iter_lim,
    fail_near_singular,
    fail_invalid_state,
    fail_invalid_step,
};

pub const NewtonResFastScal = struct {
    conv: bool,
    pre_dom_conv: bool,
    hit_iter_lim: bool,
    iters: u8,
    xi: F,
    eta: F,
};

pub const NewtonResRobustScal = struct {
    conv: bool,
    pre_dom_conv: bool,
    iters: u8,
    status: NewtonStatus,
    resid_x: F,
    resid_y: F,
    xi: F,
    eta: F,
};

pub const NewtonResFastSIMD = struct {
    v_conv: VecSB,
    v_pre_dom_conv: VecSB,
    v_hit_iter_lim: VecSB,
    v_iters: VecSU8,
    v_resid_x: VecSF,
    v_resid_y: VecSF,
    v_xi: VecSF,
    v_eta: VecSF,
};

pub const NewtonResRobustSIMD = struct {
    v_conv: VecSB,
    v_pre_dom_conv: VecSB,
    v_iters: VecSU8,
    v_status: VecSU8,
    v_resid_x: VecSF,
    v_resid_y: VecSF,
    v_xi: VecSF,
    v_eta: VecSF,
};

// --------------------------------------------------------------------------------------
// Public Entry-Point Functions
// --------------------------------------------------------------------------------------

pub inline fn selectSeed(
    seed_reuse: rastcfg.NewtonSeedReuse,
    base_seed: NewtonSeed,
    seed_state: NewtonSeedState,
) NewtonSeed {
    if (seed_reuse == .last_conv and seed_state.is_valid) {
        return .{
            .xi = seed_state.xi,
            .eta = seed_state.eta,
        };
    }
    return base_seed;
}

pub inline fn isConvStatus(status: NewtonStatus) bool {
    return switch (status) {
        .conv_resid,
        .conv_step,
        .conv_stagnated,
        .conv_two_cycle,
        => true,
        else => false,
    };
}

pub inline fn isPreDomConvStatus(status: NewtonStatus) bool {
    return switch (status) {
        .conv_resid,
        .conv_step,
        .conv_stagnated,
        .conv_two_cycle,
        .fail_dom,
        => true,
        else => false,
    };
}

pub inline fn hitIterLimitStatus(status: NewtonStatus) bool {
    return status == .fail_iter_lim;
}

pub inline fn statusLabel(status: NewtonStatus) []const u8 {
    return switch (status) {
        .conv_resid => "conv_resid",
        .conv_step => "conv_step",
        .conv_stagnated => "conv_stagnated",
        .conv_two_cycle => "conv_two_cycle",
        .fail_dom => "fail_dom",
        .fail_iter_lim => "fail_iter_lim",
        .fail_near_singular => "fail_near_singular",
        .fail_invalid_state => "fail_invalid_state",
        .fail_invalid_step => "fail_invalid_step",
    };
}

pub inline fn isSeedFinite(seed: NewtonSeed) bool {
    return std.math.isFinite(seed.xi) and std.math.isFinite(seed.eta);
}

pub inline fn updateSeedState(
    seed_state: *NewtonSeedState,
    xi: F,
    eta: F,
) void {
    seed_state.* = .{
        .is_valid = true,
        .xi = xi,
        .eta = eta,
    };
}

pub inline fn applySeedReuseInPlace(
    lane_count: usize,
    seed_state: NewtonSeedState,
    seed_xi: []F,
    seed_eta: []F,
) void {
    if (seed_state.is_valid) {
        for (0..lane_count) |jj| {
            seed_xi[jj] = seed_state.xi;
            seed_eta[jj] = seed_state.eta;
        }
    }
}

pub inline fn updateSeedStateFromSIMDResult(
    seed_state: *NewtonSeedState,
    v_chunk_mask: VecSB,
    v_conv_mask: VecSB,
    v_xi: VecSF,
    v_eta: VecSF,
    v_resid_x: VecSF,
    v_resid_y: VecSF,
) void {
    const v_mask_valid = v_chunk_mask & v_conv_mask;
    if (!@reduce(.Or, v_mask_valid)) return;

    const lane_mask_valid: [S]bool = v_mask_valid;
    const lane_xi: [S]F = v_xi;
    const lane_eta: [S]F = v_eta;
    const lane_resid_x: [S]F = v_resid_x;
    const lane_resid_y: [S]F = v_resid_y;

    var best_lane_idx: ?usize = null;
    var best_resid_sq = std.math.inf(F);

    for (0..S) |jj| {
        if (lane_mask_valid[jj]) {
            const resid_sq =
                lane_resid_x[jj] * lane_resid_x[jj] +
                lane_resid_y[jj] * lane_resid_y[jj];
            if (best_lane_idx == null or resid_sq < best_resid_sq) {
                best_lane_idx = jj;
                best_resid_sq = resid_sq;
            }
        }
    }

    if (best_lane_idx) |jj| {
        updateSeedState(seed_state, lane_xi[jj], lane_eta[jj]);
    }
}

pub fn evaluateSeedQuality(
    comptime N: usize,
    comptime domViolationFn: anytype,
    targ_screen_x: F,
    targ_screen_y: F,
    elem_node_x: []const F,
    elem_node_y: []const F,
    elem_node_w: []const F,
    seed: NewtonSeed,
) NewtonSeedQuality {
    if (!isSeedFinite(seed)) {
        return .{
            .is_usable = false,
            .dom_violation = std.math.inf(F),
            .resid_sq = std.math.inf(F),
            .det_abs = 0.0,
        };
    }

    var node_vals: [N]F = undefined;
    var deriv_n_xi: [N]F = undefined;
    var deriv_n_eta: [N]F = undefined;
    shapefun.shapeFunc(
        N,
        seed.xi,
        seed.eta,
        &node_vals,
        &deriv_n_xi,
        &deriv_n_eta,
    );

    var resid_x: F = 0.0;
    var resid_y: F = 0.0;
    var jac_11: F = 0.0;
    var jac_12: F = 0.0;
    var jac_21: F = 0.0;
    var jac_22: F = 0.0;

    for (0..N) |nn| {
        const term_x = @mulAdd(
            F,
            targ_screen_x,
            elem_node_w[nn],
            -elem_node_x[nn],
        );
        const term_y = @mulAdd(
            F,
            targ_screen_y,
            elem_node_w[nn],
            -elem_node_y[nn],
        );

        resid_x = @mulAdd(F, node_vals[nn], term_x, resid_x);
        resid_y = @mulAdd(F, node_vals[nn], term_y, resid_y);

        jac_11 = @mulAdd(F, deriv_n_xi[nn], term_x, jac_11);
        jac_12 = @mulAdd(F, deriv_n_eta[nn], term_x, jac_12);
        jac_21 = @mulAdd(F, deriv_n_xi[nn], term_y, jac_21);
        jac_22 = @mulAdd(F, deriv_n_eta[nn], term_y, jac_22);
    }

    const determinant = @mulAdd(
        F,
        jac_11,
        jac_22,
        -(jac_12 * jac_21),
    );
    const det_abs = @abs(determinant);
    const resid_sq = resid_x * resid_x + resid_y * resid_y;
    const dom_violation = domViolationFn(seed.xi, seed.eta);
    const seed_tol = tol.newton_seed;

    const is_usable = dom_violation <= seed_tol.para_dom and
        det_abs >= seed_tol.det and
        resid_sq <= seed_tol.resid_sq and
        std.math.isFinite(resid_sq);

    return .{
        .is_usable = is_usable,
        .dom_violation = dom_violation,
        .resid_sq = resid_sq,
        .det_abs = det_abs,
    };
}

pub fn calcJacDet2D(
    comptime N: usize,
    xi: F,
    eta: F,
    node_x: []const F,
    node_y: []const F,
) F {
    var node_vals: [N]F = undefined;
    var deriv_n_xi: [N]F = undefined;
    var deriv_n_eta: [N]F = undefined;
    shapefun.shapeFunc(
        N,
        xi,
        eta,
        &node_vals,
        &deriv_n_xi,
        &deriv_n_eta,
    );

    var dx_dxi: F = 0.0;
    var dx_deta: F = 0.0;
    var dy_dxi: F = 0.0;
    var dy_deta: F = 0.0;

    for (0..N) |nn| {
        dx_dxi = @mulAdd(F, deriv_n_xi[nn], node_x[nn], dx_dxi);
        dx_deta = @mulAdd(F, deriv_n_eta[nn], node_x[nn], dx_deta);
        dy_dxi = @mulAdd(F, deriv_n_xi[nn], node_y[nn], dy_dxi);
        dy_deta = @mulAdd(F, deriv_n_eta[nn], node_y[nn], dy_deta);
    }

    return @mulAdd(F, dx_dxi, dy_deta, -(dx_deta * dy_dxi));
}

pub fn evaluateSolveState(
    comptime N: usize,
    targ_screen_x: F,
    targ_screen_y: F,
    elem_node_x: []const F,
    elem_node_y: []const F,
    elem_node_w: []const F,
    xi: F,
    eta: F,
) NewtonEvalState {
    var node_vals: [N]F = undefined;
    var deriv_n_xi: [N]F = undefined;
    var deriv_n_eta: [N]F = undefined;
    shapefun.shapeFunc(
        N,
        xi,
        eta,
        &node_vals,
        &deriv_n_xi,
        &deriv_n_eta,
    );

    var resid_x: F = 0.0;
    var resid_y: F = 0.0;
    var interp_w: F = 0.0;

    for (0..N) |nn| {
        const term_x = @mulAdd(
            F,
            targ_screen_x,
            elem_node_w[nn],
            -elem_node_x[nn],
        );
        const term_y = @mulAdd(
            F,
            targ_screen_y,
            elem_node_w[nn],
            -elem_node_y[nn],
        );

        resid_x = @mulAdd(F, node_vals[nn], term_x, resid_x);
        resid_y = @mulAdd(F, node_vals[nn], term_y, resid_y);
        interp_w = @mulAdd(
            F,
            node_vals[nn],
            elem_node_w[nn],
            interp_w,
        );
    }

    const resid_mag = @sqrt(
        resid_x * resid_x +
            resid_y * resid_y,
    );
    const w_abs = @abs(interp_w);
    const norm_resid_x = if (w_abs > 0.0)
        resid_x / interp_w
    else
        std.math.nan(F);
    const norm_resid_y = if (w_abs > 0.0)
        resid_y / interp_w
    else
        std.math.nan(F);
    const norm_resid_mag = if (w_abs > 0.0)
        @sqrt(
            norm_resid_x * norm_resid_x +
                norm_resid_y * norm_resid_y,
        )
    else
        std.math.nan(F);

    return .{
        .resid_x = resid_x,
        .resid_y = resid_y,
        .interp_w = interp_w,
        .resid_mag = resid_mag,
        .norm_resid_x = norm_resid_x,
        .norm_resid_y = norm_resid_y,
        .norm_resid_mag = norm_resid_mag,
    };
}

// --------------------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------------------

test "calcJacDet2D regular elems" {
    const testing = std.testing;
    const det_tol: F = if (comptime F == f32) 1e-4 else 1e-9;
    const quad_det_tol: F = if (comptime F == f32) 1e-4 else 1e-12;

    const tri_x = [_]F{ 0.0, 10.0, 5.0 };
    const tri_y = [_]F{ 0.0, 0.0, 8.660254037844386 };
    const tri_det = calcJacDet2D(3, 0.2, 0.3, &tri_x, &tri_y);
    try testing.expectApproxEqAbs(86.60254037844386, tri_det, det_tol);

    const quad_x = [_]F{ 0.0, 10.0, 10.0, 0.0 };
    const quad_y = [_]F{ 0.0, 0.0, 10.0, 10.0 };
    const quad_det = calcJacDet2D(4, 0.0, 0.0, &quad_x, &quad_y);
    try testing.expectApproxEqAbs(25.0, quad_det, quad_det_tol);
}
