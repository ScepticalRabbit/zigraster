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

const common = @import("cameramodels_common.zig");
const cfg = buildconfig.config;
const S = buildconfig.SimdWidth;
const VecSB = buildconfig.VecSB;
const VecSF = buildconfig.VecSF;
const tol = cfg.tol;

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const DistortionForwardJacSIMDResult = struct {
    x_d: VecSF,
    y_d: VecSF,
    j11: VecSF,
    j12: VecSF,
    j21: VecSF,
    j22: VecSF,
};

pub const DistortionInvSIMDResult = struct {
    x: VecSF,
    y: VecSF,
};

// --------------------------------------------------------------------------------------
// Brown Conrady
// --------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------
// Public Entry-Point Func
// --------------------------------------------------------------------------------------

pub fn forwardDistortionSIMD(
    comptime DistortionType: type,
    distortion: DistortionType,
    x: VecSF,
    y: VecSF,
) struct { x_d: VecSF, y_d: VecSF } {
    const fwd = forwardDistortionWithJacSIMD(
        DistortionType,
        distortion,
        x,
        y,
    );
    return .{
        .x_d = fwd.x_d,
        .y_d = fwd.y_d,
    };
}

pub fn forwardDistortionWithJacSIMD(
    comptime DistortionType: type,
    distortion: DistortionType,
    x: VecSF,
    y: VecSF,
) DistortionForwardJacSIMDResult {
    const r2 = x * x + y * y;
    const r4 = r2 * r2;
    const r6 = r4 * r2;

    const radial_and_deriv = if (@hasField(DistortionType, "k4")) blk: {
        const numerator = @as(VecSF, @splat(1.0)) +
            @as(VecSF, @splat(distortion.k1)) * r2 +
            @as(VecSF, @splat(distortion.k2)) * r4 +
            @as(VecSF, @splat(distortion.k3)) * r6;

        const denominator = @as(VecSF, @splat(1.0)) +
            @as(VecSF, @splat(distortion.k4)) * r2 +
            @as(VecSF, @splat(distortion.k5)) * r4 +
            @as(VecSF, @splat(distortion.k6)) * r6;

        const dnum_dr2 = @as(VecSF, @splat(distortion.k1)) +
            @as(VecSF, @splat(2.0 * distortion.k2)) * r2 +
            @as(VecSF, @splat(3.0 * distortion.k3)) * r4;

        const dden_dr2 = @as(VecSF, @splat(distortion.k4)) +
            @as(VecSF, @splat(2.0 * distortion.k5)) * r2 +
            @as(VecSF, @splat(3.0 * distortion.k6)) * r4;

        const radial_scale = numerator / denominator;
        const dradial_dr2 =
            (dnum_dr2 * denominator - numerator * dden_dr2) /
            (denominator * denominator);

        break :blk .{
            .radial_scale = radial_scale,
            .dradial_dr2 = dradial_dr2,
        };
    } else blk: {
        const radial_scale = @as(VecSF, @splat(1.0)) +
            @as(VecSF, @splat(distortion.k1)) * r2 +
            @as(VecSF, @splat(distortion.k2)) * r4 +
            @as(VecSF, @splat(distortion.k3)) * r6;

        const dradial_dr2 = @as(VecSF, @splat(distortion.k1)) +
            @as(VecSF, @splat(2.0 * distortion.k2)) * r2 +
            @as(VecSF, @splat(3.0 * distortion.k3)) * r4;

        break :blk .{
            .radial_scale = radial_scale,
            .dradial_dr2 = dradial_dr2,
        };
    };

    const radial_scale = radial_and_deriv.radial_scale;
    const dradial_dr2 = radial_and_deriv.dradial_dr2;
    const dradial_dx = dradial_dr2 * @as(VecSF, @splat(2.0)) * x;
    const dradial_dy = dradial_dr2 * @as(VecSF, @splat(2.0)) * y;
    const p1: VecSF = @splat(distortion.p1);
    const p2: VecSF = @splat(distortion.p2);

    const x_d = x * radial_scale + @as(VecSF, @splat(2.0)) * p1 * x * y +
        p2 * (r2 + @as(VecSF, @splat(2.0)) * x * x);
    const y_d = y * radial_scale + p1 * (r2 + @as(VecSF, @splat(2.0)) * y * y) +
        @as(VecSF, @splat(2.0)) * p2 * x * y;

    const j11 = radial_scale + x * dradial_dx +
        @as(VecSF, @splat(2.0)) * p1 * y +
        @as(VecSF, @splat(6.0)) * p2 * x;
    const j12 = x * dradial_dy +
        @as(VecSF, @splat(2.0)) * p1 * x +
        @as(VecSF, @splat(2.0)) * p2 * y;
    const j21 = y * dradial_dx +
        @as(VecSF, @splat(2.0)) * p1 * x +
        @as(VecSF, @splat(2.0)) * p2 * y;
    const j22 = radial_scale + y * dradial_dy +
        @as(VecSF, @splat(6.0)) * p1 * y +
        @as(VecSF, @splat(2.0)) * p2 * x;

    return .{
        .x_d = x_d,
        .y_d = y_d,
        .j11 = j11,
        .j12 = j12,
        .j21 = j21,
        .j22 = j22,
    };
}

pub fn invDistortionSIMD(
    comptime DistortionType: type,
    distortion: DistortionType,
    v_x_d: VecSF,
    v_y_d: VecSF,
    v_lane_active_init: VecSB,
) !DistortionInvSIMDResult {
    const v_resid_tol: VecSF = @splat(tol.distortion.resid);
    const v_delta_tol: VecSF = @splat(tol.distortion.delta);
    const v_det_tol: VecSF = @splat(tol.distortion.det);

    var v_x = v_x_d;
    var v_y = v_y_d;
    var v_active = v_lane_active_init;

    for (0..cfg.distortion_newton_iter_max) |_| {
        if (!@reduce(.Or, v_active)) {
            return .{ .x = v_x, .y = v_y };
        }

        const fwd = forwardDistortionWithJacSIMD(
            DistortionType,
            distortion,
            v_x,
            v_y,
        );
        const f0 = fwd.x_d - v_x_d;
        const f1 = fwd.y_d - v_y_d;

        const v_met_resid = (@abs(f0) < v_resid_tol) & (@abs(f1) < v_resid_tol);
        v_active = v_active & !v_met_resid;
        if (!@reduce(.Or, v_active)) {
            return .{ .x = v_x, .y = v_y };
        }

        const v_det = fwd.j11 * fwd.j22 - fwd.j12 * fwd.j21;
        const v_bad_det = @abs(v_det) < v_det_tol;
        if (@reduce(.Or, v_active & v_bad_det)) {
            return error.SingularJac;
        }

        const v_safe_det = @select(
            F,
            v_active,
            v_det,
            @as(VecSF, @splat(1.0)),
        );
        const v_delta_x = (-f0 * fwd.j22 + fwd.j12 * f1) / v_safe_det;
        const v_delta_y = (fwd.j21 * f0 - fwd.j11 * f1) / v_safe_det;

        v_x += @select(F, v_active, v_delta_x, @as(VecSF, @splat(0.0)));
        v_y += @select(F, v_active, v_delta_y, @as(VecSF, @splat(0.0)));

        const v_met_delta =
            (@abs(v_delta_x) < v_delta_tol) & (@abs(v_delta_y) < v_delta_tol);
        v_active = v_active & !v_met_delta;
    }

    if (@reduce(.Or, v_active)) {
        return error.DistortionInvFailed;
    }
    return .{ .x = v_x, .y = v_y };
}

// --------------------------------------------------------------------------------------
// Polynomial Distortion
// --------------------------------------------------------------------------------------

fn invPolynomialSIMD(
    polynomial: common.BidirectionalPolynomial,
    v_x_d: VecSF,
    v_y_d: VecSF,
    v_lane_active: VecSB,
) !DistortionInvSIMDResult {
    if (polynomial.inv_map) |inv_map| {
        const eval = evaluatePolynomialMapSIMD(inv_map, v_x_d, v_y_d);
        return .{ .x = eval.x_d, .y = eval.y_d };
    }
    if (polynomial.forward_map) |forward_map| {
        return try invertPolynomialMapSIMD(
            forward_map,
            v_x_d,
            v_y_d,
            v_lane_active,
        );
    }
    return error.MissingPolynomialMap;
}

fn evaluatePolynomialMapSIMD(
    polynomial: common.PolynomialMap,
    x: VecSF,
    y: VecSF,
) struct { x_d: VecSF, y_d: VecSF } {
    const poly = evaluatePolynomialMapWithJacSIMD(polynomial, x, y);
    return .{ .x_d = poly.x_d, .y_d = poly.y_d };
}

fn evaluatePolynomialMapWithJacSIMD(
    polynomial: common.PolynomialMap,
    x: VecSF,
    y: VecSF,
) DistortionForwardJacSIMDResult {
    var du: VecSF = @splat(0.0);
    var dv: VecSF = @splat(0.0);
    var ddu_dx: VecSF = @splat(0.0);
    var ddu_dy: VecSF = @splat(0.0);
    var ddv_dx: VecSF = @splat(0.0);
    var ddv_dy: VecSF = @splat(0.0);
    const term_count = polynomial.order.termCount();

    for (0..term_count) |ii| {
        const pu = common.poly_powers_u[ii];
        const pv = common.poly_powers_v[ii];
        const basis = powSmallSIMD(x, pu) * powSmallSIMD(y, pv);
        du += @as(VecSF, @splat(polynomial.coeffs_u[ii])) * basis;
        dv += @as(VecSF, @splat(polynomial.coeffs_v[ii])) * basis;

        if (pu > 0) {
            const basis_dx = @as(VecSF, @splat(@as(F, @floatFromInt(pu)))) *
                powSmallSIMD(x, pu - 1) *
                powSmallSIMD(y, pv);
            ddu_dx += @as(VecSF, @splat(polynomial.coeffs_u[ii])) * basis_dx;
            ddv_dx += @as(VecSF, @splat(polynomial.coeffs_v[ii])) * basis_dx;
        }
        if (pv > 0) {
            const basis_dy = @as(VecSF, @splat(@as(F, @floatFromInt(pv)))) *
                powSmallSIMD(x, pu) *
                powSmallSIMD(y, pv - 1);
            ddu_dy += @as(VecSF, @splat(polynomial.coeffs_u[ii])) * basis_dy;
            ddv_dy += @as(VecSF, @splat(polynomial.coeffs_v[ii])) * basis_dy;
        }
    }

    return .{
        .x_d = x + du,
        .y_d = y + dv,
        .j11 = @as(VecSF, @splat(1.0)) + ddu_dx,
        .j12 = ddu_dy,
        .j21 = ddv_dx,
        .j22 = @as(VecSF, @splat(1.0)) + ddv_dy,
    };
}

fn invertPolynomialMapSIMD(
    polynomial: common.PolynomialMap,
    v_x_d: VecSF,
    v_y_d: VecSF,
    v_lane_active_init: VecSB,
) !DistortionInvSIMDResult {
    const v_resid_tol: VecSF = @splat(tol.distortion.resid);
    const v_delta_tol: VecSF = @splat(tol.distortion.delta);
    const v_det_tol: VecSF = @splat(tol.distortion.det);

    var v_x = v_x_d;
    var v_y = v_y_d;
    var v_active = v_lane_active_init;

    for (0..cfg.distortion_newton_iter_max) |_| {
        if (!@reduce(.Or, v_active)) {
            return .{ .x = v_x, .y = v_y };
        }

        const fwd = evaluatePolynomialMapWithJacSIMD(polynomial, v_x, v_y);
        const f0 = fwd.x_d - v_x_d;
        const f1 = fwd.y_d - v_y_d;

        const v_met_resid = (@abs(f0) < v_resid_tol) & (@abs(f1) < v_resid_tol);
        v_active = v_active & !v_met_resid;
        if (!@reduce(.Or, v_active)) {
            return .{ .x = v_x, .y = v_y };
        }

        const v_det = fwd.j11 * fwd.j22 - fwd.j12 * fwd.j21;
        const v_bad_det = @abs(v_det) < v_det_tol;
        if (@reduce(.Or, v_active & v_bad_det)) {
            return error.SingularJac;
        }

        const v_safe_det = @select(
            F,
            v_active,
            v_det,
            @as(VecSF, @splat(1.0)),
        );
        const v_delta_x = (-f0 * fwd.j22 + fwd.j12 * f1) / v_safe_det;
        const v_delta_y = (fwd.j21 * f0 - fwd.j11 * f1) / v_safe_det;

        v_x += @select(F, v_active, v_delta_x, @as(VecSF, @splat(0.0)));
        v_y += @select(F, v_active, v_delta_y, @as(VecSF, @splat(0.0)));

        const v_met_delta =
            (@abs(v_delta_x) < v_delta_tol) & (@abs(v_delta_y) < v_delta_tol);
        v_active = v_active & !v_met_delta;
    }

    if (@reduce(.Or, v_active)) {
        return error.DistortionInvFailed;
    }
    return .{ .x = v_x, .y = v_y };
}

fn powSmallSIMD(
    x: VecSF,
    power: u8,
) VecSF {
    var out: VecSF = @splat(1.0);
    for (0..power) |_| {
        out *= x;
    }
    return out;
}

// --------------------------------------------------------------------------------------
// Distortion Unions
// --------------------------------------------------------------------------------------

pub const DistortionModel = common.DistortionModel;

pub fn invDistortionModelSIMD(
    distortion: DistortionModel,
    v_x_d: VecSF,
    v_y_d: VecSF,
    v_lane_active: VecSB,
) !DistortionInvSIMDResult {
    return switch (distortion) {
        .none => .{ .x = v_x_d, .y = v_y_d },
        .brown_conrady => |bc| invDistortionSIMD(
            common.BrownConrady,
            bc,
            v_x_d,
            v_y_d,
            v_lane_active,
        ),
        .brown_conrady_ext => |bc_ext| invDistortionSIMD(
            common.BrownConradyExt,
            bc_ext,
            v_x_d,
            v_y_d,
            v_lane_active,
        ),
        .polynomial => |poly| invPolynomialSIMD(
            poly,
            v_x_d,
            v_y_d,
            v_lane_active,
        ),
        .brown_conrady_polynomial => |chain| blk: {
            const poly_inv = try invPolynomialSIMD(
                chain.polynomial,
                v_x_d,
                v_y_d,
                v_lane_active,
            );
            break :blk try invDistortionSIMD(
                common.BrownConrady,
                chain.brown_conrady,
                poly_inv.x,
                poly_inv.y,
                v_lane_active,
            );
        },
        .brown_conrady_ext_polynomial => |chain| blk: {
            const poly_inv = try invPolynomialSIMD(
                chain.polynomial,
                v_x_d,
                v_y_d,
                v_lane_active,
            );
            break :blk try invDistortionSIMD(
                common.BrownConradyExt,
                chain.brown_conrady_ext,
                poly_inv.x,
                poly_inv.y,
                v_lane_active,
            );
        },
    };
}
