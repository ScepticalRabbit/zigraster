// --------------------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const buildconfig = @import("buildconfig.zig");
const F = buildconfig.F;

const S = buildconfig.SimdWidth;
const VecSF = buildconfig.VecSF;

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const NodalDerivs = struct {
    dNu: [9][9]F,
    dNv: [9][9]F,
};

// --------------------------------------------------------------------------------------
// Public Entry-Point Func
// --------------------------------------------------------------------------------------

pub fn getNodalDerivs(comptime N: usize) NodalDerivs {
    var nodal_derivs = NodalDerivs{
        .dNu = [_][9]F{[_]F{0} ** 9} ** 9,
        .dNv = [_][9]F{[_]F{0} ** 9} ** 9,
    };

    const node_coords = switch (N) {
        3 => [3][2]F{
            .{ 0, 0 }, .{ 1, 0 }, .{ 0, 1 },
        },
        4 => [4][2]F{
            .{ -1, -1 }, .{ 1, -1 }, .{ 1, 1 }, .{ -1, 1 },
        },
        6 => [6][2]F{
            .{ 0, 0 }, .{ 1, 0 }, .{ 0, 1 }, .{ 0.5, 0 }, .{ 0.5, 0.5 }, .{ 0, 0.5 },
        },
        8 => [8][2]F{
            .{ -1, -1 }, .{ 1, -1 }, .{ 1, 1 }, .{ -1, 1 },
            .{ 0, -1 },  .{ 1, 0 },  .{ 0, 1 }, .{ -1, 0 },
        },
        9 => [9][2]F{
            .{ -1, -1 }, .{ 1, -1 }, .{ 1, 1 }, .{ -1, 1 },
            .{ 0, -1 },  .{ 1, 0 },  .{ 0, 1 }, .{ -1, 0 },
            .{ 0, 0 },
        },
        else => return nodal_derivs,
    };

    for (0..N) |ii| {
        var n_v: [N]F = undefined;
        var dNu: [N]F = undefined;
        var dNv: [N]F = undefined;

        shapeFunc(N, node_coords[ii][0], node_coords[ii][1], &n_v, &dNu, &dNv);

        for (0..N) |jj| {
            nodal_derivs.dNu[ii][jj] = dNu[jj];
            nodal_derivs.dNv[ii][jj] = dNv[jj];
        }
    }

    return nodal_derivs;
}

pub fn shapeFunc(
    comptime N: usize,
    xi: F,
    eta: F,
    n_v: *[N]F,
    dNu: *[N]F,
    dNv: *[N]F,
) void {
    switch (N) {
        3 => shapeFunc3(xi, eta, n_v, dNu, dNv),
        4 => shapeFunc4(xi, eta, n_v, dNu, dNv),
        6 => shapeFunc6(xi, eta, n_v, dNu, dNv),
        8 => shapeFunc8(xi, eta, n_v, dNu, dNv),
        9 => shapeFunc9(xi, eta, n_v, dNu, dNv),
        else => @compileError("Unsupped number of nodes"),
    }
}

pub fn shapeFuncSIMD(
    comptime N: usize,
    v_xi: VecSF,
    v_eta: VecSF,
    v_shape: *[N]VecSF,
    v_dN_dxi: *[N]VecSF,
    v_dN_deta: *[N]VecSF,
) void {
    switch (N) {
        3 => shapeFunc3SIMD(v_xi, v_eta, v_shape, v_dN_dxi, v_dN_deta),
        4 => shapeFunc4SIMD(v_xi, v_eta, v_shape, v_dN_dxi, v_dN_deta),
        6 => shapeFunc6SIMD(v_xi, v_eta, v_shape, v_dN_dxi, v_dN_deta),
        8 => shapeFunc8SIMD(v_xi, v_eta, v_shape, v_dN_dxi, v_dN_deta),
        9 => shapeFunc9SIMD(v_xi, v_eta, v_shape, v_dN_dxi, v_dN_deta),
        else => @compileError("Unsupped number of nodes"),
    }
}

fn shapeFunc3(xi: F, eta: F, n_v: *[3]F, dNu: *[3]F, dNv: *[3]F) void {
    const L1 = 1.0 - xi - eta;
    const L2 = xi;
    const L3 = eta;

    n_v[0] = L1;
    n_v[1] = L2;
    n_v[2] = L3;

    dNu[0] = -1.0;
    dNu[1] = 1.0;
    dNu[2] = 0.0;

    dNv[0] = -1.0;
    dNv[1] = 0.0;
    dNv[2] = 1.0;
}

fn shapeFunc3SIMD(
    v_xi: VecSF,
    v_eta: VecSF,
    v_shape: *[3]VecSF,
    v_dN_dxi: *[3]VecSF,
    v_dN_deta: *[3]VecSF,
) void {
    const v_splat_one: VecSF = @splat(1.0);
    const v_splat_zero: VecSF = @splat(0.0);
    const v_splat_neg_one: VecSF = @splat(-1.0);

    const v_L1 = v_splat_one - v_xi - v_eta;
    const v_L2 = v_xi;
    const v_L3 = v_eta;

    v_shape[0] = v_L1;
    v_shape[1] = v_L2;
    v_shape[2] = v_L3;

    v_dN_dxi[0] = v_splat_neg_one;
    v_dN_dxi[1] = v_splat_one;
    v_dN_dxi[2] = v_splat_zero;

    v_dN_deta[0] = v_splat_neg_one;
    v_dN_deta[1] = v_splat_zero;
    v_dN_deta[2] = v_splat_one;
}

fn shapeFunc4(xi: F, eta: F, n_v: *[4]F, dNu: *[4]F, dNv: *[4]F) void {
    n_v[0] = 0.25 * (1.0 - xi) * (1.0 - eta);
    n_v[1] = 0.25 * (1.0 + xi) * (1.0 - eta);
    n_v[2] = 0.25 * (1.0 + xi) * (1.0 + eta);
    n_v[3] = 0.25 * (1.0 - xi) * (1.0 + eta);

    dNu[0] = -0.25 * (1.0 - eta);
    dNu[1] = 0.25 * (1.0 - eta);
    dNu[2] = 0.25 * (1.0 + eta);
    dNu[3] = -0.25 * (1.0 + eta);

    dNv[0] = -0.25 * (1.0 - xi);
    dNv[1] = -0.25 * (1.0 + xi);
    dNv[2] = 0.25 * (1.0 + xi);
    dNv[3] = 0.25 * (1.0 - xi);
}

fn shapeFunc4SIMD(
    v_xi: VecSF,
    v_eta: VecSF,
    v_shape: *[4]VecSF,
    v_dN_dxi: *[4]VecSF,
    v_dN_deta: *[4]VecSF,
) void {
    const v_splat_quarter: VecSF = @splat(0.25);
    const v_splat_neg_quarter: VecSF = @splat(-0.25);
    const v_splat_one: VecSF = @splat(1.0);

    const v_one_minus_xi = v_splat_one - v_xi;
    const v_one_plus_xi = v_splat_one + v_xi;
    const v_one_minus_eta = v_splat_one - v_eta;
    const v_one_plus_eta = v_splat_one + v_eta;

    v_shape[0] = v_splat_quarter * v_one_minus_xi * v_one_minus_eta;
    v_shape[1] = v_splat_quarter * v_one_plus_xi * v_one_minus_eta;
    v_shape[2] = v_splat_quarter * v_one_plus_xi * v_one_plus_eta;
    v_shape[3] = v_splat_quarter * v_one_minus_xi * v_one_plus_eta;

    v_dN_dxi[0] = v_splat_neg_quarter * v_one_minus_eta;
    v_dN_dxi[1] = v_splat_quarter * v_one_minus_eta;
    v_dN_dxi[2] = v_splat_quarter * v_one_plus_eta;
    v_dN_dxi[3] = v_splat_neg_quarter * v_one_plus_eta;

    v_dN_deta[0] = v_splat_neg_quarter * v_one_minus_xi;
    v_dN_deta[1] = v_splat_neg_quarter * v_one_plus_xi;
    v_dN_deta[2] = v_splat_quarter * v_one_plus_xi;
    v_dN_deta[3] = v_splat_quarter * v_one_minus_xi;
}

fn shapeFunc6(
    xi: F,
    eta: F,
    n_vals: *[6]F,
    dN_dxi: *[6]F,
    dN_deta: *[6]F,
) void {
    const L1 = 1.0 - xi - eta;
    const L2 = xi;
    const L3 = eta;

    n_vals[0] = L1 * (2.0 * L1 - 1.0);
    dN_dxi[0] = -(4.0 * L1 - 1.0);
    dN_deta[0] = -(4.0 * L1 - 1.0);

    n_vals[1] = L2 * (2.0 * L2 - 1.0);
    dN_dxi[1] = 4.0 * L2 - 1.0;
    dN_deta[1] = 0.0;

    n_vals[2] = L3 * (2.0 * L3 - 1.0);
    dN_dxi[2] = 0.0;
    dN_deta[2] = 4.0 * L3 - 1.0;

    n_vals[3] = 4.0 * L1 * L2;
    dN_dxi[3] = 4.0 * (L1 - L2);
    dN_deta[3] = -4.0 * L2;

    n_vals[4] = 4.0 * L2 * L3;
    dN_dxi[4] = 4.0 * L3;
    dN_deta[4] = 4.0 * L2;

    n_vals[5] = 4.0 * L3 * L1;
    dN_dxi[5] = -4.0 * L3;
    dN_deta[5] = 4.0 * (L1 - L3);
}

fn shapeFunc6SIMD(
    v_xi: VecSF,
    v_eta: VecSF,
    v_shape: *[6]VecSF,
    v_dN_dxi: *[6]VecSF,
    v_dN_deta: *[6]VecSF,
) void {
    const v_splat_one: VecSF = @splat(1.0);
    const v_splat_two: VecSF = @splat(2.0);
    const v_splat_four: VecSF = @splat(4.0);
    const v_splat_zero: VecSF = @splat(0.0);

    const v_L1 = v_splat_one - v_xi - v_eta;
    const v_L2 = v_xi;
    const v_L3 = v_eta;

    const v_four_L1_minus_one = v_splat_four * v_L1 - v_splat_one;
    const v_four_L2_minus_one = v_splat_four * v_L2 - v_splat_one;
    const v_four_L3_minus_one = v_splat_four * v_L3 - v_splat_one;

    v_shape[0] = v_L1 * (v_splat_two * v_L1 - v_splat_one);
    v_dN_dxi[0] = -v_four_L1_minus_one;
    v_dN_deta[0] = -v_four_L1_minus_one;

    v_shape[1] = v_L2 * (v_splat_two * v_L2 - v_splat_one);
    v_dN_dxi[1] = v_four_L2_minus_one;
    v_dN_deta[1] = v_splat_zero;

    v_shape[2] = v_L3 * (v_splat_two * v_L3 - v_splat_one);
    v_dN_dxi[2] = v_splat_zero;
    v_dN_deta[2] = v_four_L3_minus_one;

    v_shape[3] = v_splat_four * v_L1 * v_L2;
    v_dN_dxi[3] = v_splat_four * (v_L1 - v_L2);
    v_dN_deta[3] = -v_splat_four * v_L2;

    v_shape[4] = v_splat_four * v_L2 * v_L3;
    v_dN_dxi[4] = v_splat_four * v_L3;
    v_dN_deta[4] = v_splat_four * v_L2;

    v_shape[5] = v_splat_four * v_L3 * v_L1;
    v_dN_dxi[5] = -v_splat_four * v_L3;
    v_dN_deta[5] = v_splat_four * (v_L1 - v_L3);
}

fn shapeFunc8(xi: F, eta: F, n_v: *[8]F, dNu: *[8]F, dNv: *[8]F) void {
    const x = xi;
    const y = eta;
    n_v[0] = -0.25 * (1.0 - x) * (1.0 - y) * (1.0 + x + y);
    n_v[1] = -0.25 * (1.0 + x) * (1.0 - y) * (1.0 - x + y);
    n_v[2] = -0.25 * (1.0 + x) * (1.0 + y) * (1.0 - x - y);
    n_v[3] = -0.25 * (1.0 - x) * (1.0 + y) * (1.0 + x - y);
    n_v[4] = 0.5 * (1.0 - x * x) * (1.0 - y);
    n_v[5] = 0.5 * (1.0 + x) * (1.0 - y * y);
    n_v[6] = 0.5 * (1.0 - x * x) * (1.0 + y);
    n_v[7] = 0.5 * (1.0 - x) * (1.0 - y * y);

    dNu[0] = 0.25 * (1.0 - y) * (2.0 * x + y);
    dNu[1] = 0.25 * (1.0 - y) * (2.0 * x - y);
    dNu[2] = 0.25 * (1.0 + y) * (2.0 * x + y);
    dNu[3] = 0.25 * (1.0 + y) * (2.0 * x - y);
    dNu[4] = -x * (1.0 - y);
    dNu[5] = 0.5 * (1.0 - y * y);
    dNu[6] = -x * (1.0 + y);
    dNu[7] = -0.5 * (1.0 - y * y);

    dNv[0] = 0.25 * (1.0 - x) * (x + 2.0 * y);
    dNv[1] = 0.25 * (1.0 + x) * (2.0 * y - x);
    dNv[2] = 0.25 * (1.0 + x) * (x + 2.0 * y);
    dNv[3] = 0.25 * (1.0 - x) * (2.0 * y - x);
    dNv[4] = -0.5 * (1.0 - x * x);
    dNv[5] = -y * (1.0 + x);
    dNv[6] = 0.5 * (1.0 - x * x);
    dNv[7] = -y * (1.0 - x);
}

fn shapeFunc8SIMD(
    v_xi: VecSF,
    v_eta: VecSF,
    v_shape: *[8]VecSF,
    v_dN_dxi: *[8]VecSF,
    v_dN_deta: *[8]VecSF,
) void {
    const v_x = v_xi;
    const v_y = v_eta;
    const v_splat_one: VecSF = @splat(1.0);
    const v_splat_two: VecSF = @splat(2.0);
    const v_splat_half: VecSF = @splat(0.5);
    const v_splat_neg_half: VecSF = @splat(-0.5);
    const v_splat_quarter: VecSF = @splat(0.25);
    const v_splat_neg_quarter: VecSF = @splat(-0.25);

    const v_x_sq = v_x * v_x;
    const v_y_sq = v_y * v_y;
    const v_one_minus_x = v_splat_one - v_x;
    const v_one_plus_x = v_splat_one + v_x;
    const v_one_minus_y = v_splat_one - v_y;
    const v_one_plus_y = v_splat_one + v_y;

    v_shape[0] =
        v_splat_neg_quarter * v_one_minus_x * v_one_minus_y *
        (v_splat_one + v_x + v_y);
    v_shape[1] =
        v_splat_neg_quarter * v_one_plus_x * v_one_minus_y *
        (v_splat_one - v_x + v_y);
    v_shape[2] =
        v_splat_neg_quarter * v_one_plus_x * v_one_plus_y *
        (v_splat_one - v_x - v_y);
    v_shape[3] =
        v_splat_neg_quarter * v_one_minus_x * v_one_plus_y *
        (v_splat_one + v_x - v_y);
        
    v_shape[4] = v_splat_half * (v_splat_one - v_x_sq) * v_one_minus_y;
    v_shape[5] = v_splat_half * v_one_plus_x * (v_splat_one - v_y_sq);
    v_shape[6] = v_splat_half * (v_splat_one - v_x_sq) * v_one_plus_y;
    v_shape[7] = v_splat_half * v_one_minus_x * (v_splat_one - v_y_sq);

    const v_two_x = v_splat_two * v_x;
    const v_two_y = v_splat_two * v_y;
    const v_one_minus_y_sq = v_splat_one - v_y_sq;
    const v_one_minus_x_sq = v_splat_one - v_x_sq;

    v_dN_dxi[0] = v_splat_quarter * v_one_minus_y * (v_two_x + v_y);
    v_dN_dxi[1] = v_splat_quarter * v_one_minus_y * (v_two_x - v_y);
    v_dN_dxi[2] = v_splat_quarter * v_one_plus_y * (v_two_x + v_y);
    v_dN_dxi[3] = v_splat_quarter * v_one_plus_y * (v_two_x - v_y);
    v_dN_dxi[4] = -v_x * v_one_minus_y;
    v_dN_dxi[5] = v_splat_half * v_one_minus_y_sq;
    v_dN_dxi[6] = -v_x * v_one_plus_y;
    v_dN_dxi[7] = v_splat_neg_half * v_one_minus_y_sq;

    v_dN_deta[0] = v_splat_quarter * v_one_minus_x * (v_x + v_two_y);
    v_dN_deta[1] = v_splat_quarter * v_one_plus_x * (v_two_y - v_x);
    v_dN_deta[2] = v_splat_quarter * v_one_plus_x * (v_x + v_two_y);
    v_dN_deta[3] = v_splat_quarter * v_one_minus_x * (v_two_y - v_x);
    v_dN_deta[4] = v_splat_neg_half * v_one_minus_x_sq;
    v_dN_deta[5] = -v_y * v_one_plus_x;
    v_dN_deta[6] = v_splat_half * v_one_minus_x_sq;
    v_dN_deta[7] = -v_y * v_one_minus_x;
}

fn shapeFunc9(xi: F, eta: F, n_v: *[9]F, dNu: *[9]F, dNv: *[9]F) void {
    const x = xi;
    const y = eta;
    const phi = [3]F{ 0.5 * x * (x - 1.0), 1.0 - x * x, 0.5 * x * (x + 1.0) };
    const psi = [3]F{ 0.5 * y * (y - 1.0), 1.0 - y * y, 0.5 * y * (y + 1.0) };
    const dphi = [3]F{ x - 0.5, -2.0 * x, x + 0.5 };
    const dpsi = [3]F{ y - 0.5, -2.0 * y, y + 0.5 };

    n_v[0] = phi[0] * psi[0];
    n_v[1] = phi[2] * psi[0];
    n_v[2] = phi[2] * psi[2];
    n_v[3] = phi[0] * psi[2];
    n_v[4] = phi[1] * psi[0];
    n_v[5] = phi[2] * psi[1];
    n_v[6] = phi[1] * psi[2];
    n_v[7] = phi[0] * psi[1];
    n_v[8] = phi[1] * psi[1];

    dNu[0] = dphi[0] * psi[0];
    dNu[1] = dphi[2] * psi[0];
    dNu[2] = dphi[2] * psi[2];
    dNu[3] = dphi[0] * psi[2];
    dNu[4] = dphi[1] * psi[0];
    dNu[5] = dphi[2] * psi[1];
    dNu[6] = dphi[1] * psi[2];
    dNu[7] = dphi[0] * psi[1];
    dNu[8] = dphi[1] * psi[1];

    dNv[0] = phi[0] * dpsi[0];
    dNv[1] = phi[2] * dpsi[0];
    dNv[2] = phi[2] * dpsi[2];
    dNv[3] = phi[0] * dpsi[2];
    dNv[4] = phi[1] * dpsi[0];
    dNv[5] = phi[2] * dpsi[1];
    dNv[6] = phi[1] * dpsi[2];
    dNv[7] = phi[0] * dpsi[1];
    dNv[8] = phi[1] * dpsi[1];
}

fn shapeFunc9SIMD(
    v_xi: VecSF,
    v_eta: VecSF,
    v_shape: *[9]VecSF,
    v_dN_dxi: *[9]VecSF,
    v_dN_deta: *[9]VecSF,
) void {
    const v_x = v_xi;
    const v_y = v_eta;
    const v_splat_one: VecSF = @splat(1.0);
    const v_splat_half: VecSF = @splat(0.5);
    const v_splat_neg_two: VecSF = @splat(-2.0);

    const v_x_minus_one = v_x - v_splat_one;
    const v_x_plus_one = v_x + v_splat_one;
    const v_y_minus_one = v_y - v_splat_one;
    const v_y_plus_one = v_y + v_splat_one;

    const v_phi = [3]VecSF{
        v_splat_half * v_x * v_x_minus_one,
        v_splat_one - v_x * v_x,
        v_splat_half * v_x * v_x_plus_one,
    };
    const v_psi = [3]VecSF{
        v_splat_half * v_y * v_y_minus_one,
        v_splat_one - v_y * v_y,
        v_splat_half * v_y * v_y_plus_one,
    };
    const v_dphi = [3]VecSF{
        v_x - v_splat_half,
        v_splat_neg_two * v_x,
        v_x + v_splat_half,
    };
    const v_dpsi = [3]VecSF{
        v_y - v_splat_half,
        v_splat_neg_two * v_y,
        v_y + v_splat_half,
    };

    v_shape[0] = v_phi[0] * v_psi[0];
    v_shape[1] = v_phi[2] * v_psi[0];
    v_shape[2] = v_phi[2] * v_psi[2];
    v_shape[3] = v_phi[0] * v_psi[2];
    v_shape[4] = v_phi[1] * v_psi[0];
    v_shape[5] = v_phi[2] * v_psi[1];
    v_shape[6] = v_phi[1] * v_psi[2];
    v_shape[7] = v_phi[0] * v_psi[1];
    v_shape[8] = v_phi[1] * v_psi[1];

    v_dN_dxi[0] = v_dphi[0] * v_psi[0];
    v_dN_dxi[1] = v_dphi[2] * v_psi[0];
    v_dN_dxi[2] = v_dphi[2] * v_psi[2];
    v_dN_dxi[3] = v_dphi[0] * v_psi[2];
    v_dN_dxi[4] = v_dphi[1] * v_psi[0];
    v_dN_dxi[5] = v_dphi[2] * v_psi[1];
    v_dN_dxi[6] = v_dphi[1] * v_psi[2];
    v_dN_dxi[7] = v_dphi[0] * v_psi[1];
    v_dN_dxi[8] = v_dphi[1] * v_psi[1];

    v_dN_deta[0] = v_phi[0] * v_dpsi[0];
    v_dN_deta[1] = v_phi[2] * v_dpsi[0];
    v_dN_deta[2] = v_phi[2] * v_dpsi[2];
    v_dN_deta[3] = v_phi[0] * v_dpsi[2];
    v_dN_deta[4] = v_phi[1] * v_dpsi[0];
    v_dN_deta[5] = v_phi[2] * v_dpsi[1];
    v_dN_deta[6] = v_phi[1] * v_dpsi[2];
    v_dN_deta[7] = v_phi[0] * v_dpsi[1];
    v_dN_deta[8] = v_phi[1] * v_dpsi[1];
}
