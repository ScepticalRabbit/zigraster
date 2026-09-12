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
const tol = cfg.tol;

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
// Brown Conrady
// --------------------------------------------------------------------------------------

pub const DistortionInvResult = struct {
    x: F,
    y: F,
};

pub const DistortionForwardJacResult = struct {
    x_d: F,
    y_d: F,
    jac: [2][2]F,
};

pub const BrownConrady = struct {
    k1: F = 0,
    k2: F = 0,
    k3: F = 0,
    p1: F = 0,
    p2: F = 0,

    pub fn forward(
        self: BrownConrady,
        x: F,
        y: F,
    ) [2]F {
        const r2 = x * x + y * y;
        const r4 = r2 * r2;
        const r6 = r4 * r2;
        const radial_scale =
            1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6;
        return distortionForwardFromRadialScale(
            x,
            y,
            radial_scale,
            self.p1,
            self.p2,
        );
    }

    pub fn forwardWithJac(
        self: BrownConrady,
        x: F,
        y: F,
    ) DistortionForwardJacResult {
        const r2 = x * x + y * y;
        const r4 = r2 * r2;
        const r6 = r4 * r2;
        const radial_scale =
            1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6;
        const dradial_dr2 =
            self.k1 + 2.0 * self.k2 * r2 + 3.0 * self.k3 * r4;
        return distortionForwardWithJacFromRadialScale(
            x,
            y,
            radial_scale,
            dradial_dr2,
            self.p1,
            self.p2,
        );
    }

    pub fn inv(
        self: BrownConrady,
        x_d: F,
        y_d: F,
    ) !DistortionInvResult {
        return invFromForwardWithJac(BrownConrady, self, x_d, y_d);
    }
};

pub const BrownConradyExt = struct {
    k1: F = 0,
    k2: F = 0,
    k3: F = 0,
    k4: F = 0,
    k5: F = 0,
    k6: F = 0,
    p1: F = 0,
    p2: F = 0,
    s1: F = 0,
    s2: F = 0,
    s3: F = 0,
    s4: F = 0,
    tau_x: F = 0,
    tau_y: F = 0,
    tilt_projection: ?TiltProjection = null,

    pub fn prepare(self: BrownConradyExt) !BrownConradyExt {
        var prepared = self;
        if (prepared.isTiltActive()) {
            const forward_matrix = calcTiltMatrix(self.tau_x, self.tau_y);
            const inverse_matrix = invertMat33(forward_matrix) orelse
                return error.SingularTiltProjection;
            prepared.tilt_projection = .{
                .forward_matrix = forward_matrix,
                .inverse_matrix = inverse_matrix,
            };
        }
        return prepared;
    }

    pub fn isTiltActive(self: BrownConradyExt) bool {
        return @abs(self.tau_x) > tol.distortion.tilt_identity or
            @abs(self.tau_y) > tol.distortion.tilt_identity;
    }

    pub fn getForwardTiltMatrix(self: BrownConradyExt) [3][3]F {
        if (self.tilt_projection) |projection| return projection.forward_matrix;
        return calcTiltMatrix(self.tau_x, self.tau_y);
    }

    pub fn getInverseTiltMatrix(self: BrownConradyExt) ![3][3]F {
        if (self.tilt_projection) |projection| return projection.inverse_matrix;
        return invertMat33(calcTiltMatrix(self.tau_x, self.tau_y)) orelse
            error.SingularTiltProjection;
    }

    pub fn forward(
        self: BrownConradyExt,
        x: F,
        y: F,
    ) [2]F {
        const lens = self.forwardLensWithJac(x, y);
        return self.applyTilt(lens.x_d, lens.y_d).coords;
    }

    pub fn forwardWithJac(
        self: BrownConradyExt,
        x: F,
        y: F,
    ) DistortionForwardJacResult {
        const lens = self.forwardLensWithJac(x, y);
        const tilt = self.applyTilt(lens.x_d, lens.y_d);
        return .{
            .x_d = tilt.coords[0],
            .y_d = tilt.coords[1],
            .jac = mulJac22(tilt.jac, lens.jac),
        };
    }

    pub fn inv(
        self: BrownConradyExt,
        x_d: F,
        y_d: F,
    ) !DistortionInvResult {
        const untilted = try self.removeTilt(x_d, y_d);
        return invBrownConradyExtLens(self, untilted[0], untilted[1]);
    }

    fn applyTilt(self: BrownConradyExt, x: F, y: F) TiltResult {
        if (!self.isTiltActive()) {
            return .{
                .coords = .{ x, y },
                .jac = .{ .{ 1.0, 0.0 }, .{ 0.0, 1.0 } },
            };
        }
        return applyHomography(self.getForwardTiltMatrix(), x, y) catch .{
            .coords = .{ std.math.nan(F), std.math.nan(F) },
            .jac = .{
                .{ std.math.nan(F), std.math.nan(F) },
                .{ std.math.nan(F), std.math.nan(F) },
            },
        };
    }

    fn removeTilt(self: BrownConradyExt, x: F, y: F) ![2]F {
        if (!self.isTiltActive()) return .{ x, y };
        return (try applyHomography(try self.getInverseTiltMatrix(), x, y)).coords;
    }

    fn forwardLensWithJac(
        self: BrownConradyExt,
        x: F,
        y: F,
    ) DistortionForwardJacResult {
        const radial = self.calcRadialScaleAndDerivative(x, y);
        var result = distortionForwardWithJacFromRadialScale(
            x,
            y,
            radial.radial_scale,
            radial.dradial_dr2,
            self.p1,
            self.p2,
        );
        const r2 = x * x + y * y;
        const r4 = r2 * r2;
        result.x_d += self.s1 * r2 + self.s2 * r4;
        result.y_d += self.s3 * r2 + self.s4 * r4;
        result.jac[0][0] += 2.0 * x * (self.s1 + 2.0 * self.s2 * r2);
        result.jac[0][1] += 2.0 * y * (self.s1 + 2.0 * self.s2 * r2);
        result.jac[1][0] += 2.0 * x * (self.s3 + 2.0 * self.s4 * r2);
        result.jac[1][1] += 2.0 * y * (self.s3 + 2.0 * self.s4 * r2);
        return result;
    }

    pub fn calcRadialScaleAndDerivative(
        self: BrownConradyExt,
        x: F,
        y: F,
    ) struct { radial_scale: F, dradial_dr2: F } {
        const r2 = x * x + y * y;
        const r4 = r2 * r2;
        const r6 = r4 * r2;
        const numerator = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6;
        const denominator = 1.0 + self.k4 * r2 + self.k5 * r4 + self.k6 * r6;
        const dnum_dr2 = self.k1 + 2.0 * self.k2 * r2 + 3.0 * self.k3 * r4;
        const dden_dr2 = self.k4 + 2.0 * self.k5 * r2 + 3.0 * self.k6 * r4;
        const radial_scale = numerator / denominator;
        const dradial_dr2 =
            (dnum_dr2 * denominator - numerator * dden_dr2) /
            (denominator * denominator);
        return .{
            .radial_scale = radial_scale,
            .dradial_dr2 = dradial_dr2,
        };
    }
};

const TiltResult = struct {
    coords: [2]F,
    jac: [2][2]F,
};

pub const TiltProjection = struct {
    forward_matrix: [3][3]F,
    inverse_matrix: [3][3]F,
};

pub fn calcTiltMatrix(tau_x: F, tau_y: F) [3][3]F {
    const cos_x = @cos(tau_x);
    const sin_x = @sin(tau_x);
    const cos_y = @cos(tau_y);
    const sin_y = @sin(tau_y);
    const r02 = -sin_y * cos_x;
    const r12 = sin_x;
    const r22 = cos_y * cos_x;
    return .{
        .{ r22 * cos_y - r02 * sin_y, r22 * sin_y * sin_x + r02 * cos_y * sin_x, 0.0 },
        .{ -r12 * sin_y, r22 * cos_x + r12 * cos_y * sin_x, 0.0 },
        .{ sin_y, -cos_y * sin_x, r22 },
    };
}

fn applyHomography(matrix: [3][3]F, x: F, y: F) !TiltResult {
    const numerator_x = matrix[0][0] * x + matrix[0][1] * y + matrix[0][2];
    const numerator_y = matrix[1][0] * x + matrix[1][1] * y + matrix[1][2];
    const denominator = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2];
    if (!std.math.isFinite(denominator) or @abs(denominator) < tol.distortion.det) {
        return error.SingularTiltProjection;
    }
    const inv_denominator = 1.0 / denominator;
    const out_x = numerator_x * inv_denominator;
    const out_y = numerator_y * inv_denominator;
    const inv_denominator_sq = inv_denominator * inv_denominator;
    return .{
        .coords = .{ out_x, out_y },
        .jac = .{
            .{
                (matrix[0][0] * denominator - numerator_x * matrix[2][0]) * inv_denominator_sq,
                (matrix[0][1] * denominator - numerator_x * matrix[2][1]) * inv_denominator_sq,
            },
            .{
                (matrix[1][0] * denominator - numerator_y * matrix[2][0]) * inv_denominator_sq,
                (matrix[1][1] * denominator - numerator_y * matrix[2][1]) * inv_denominator_sq,
            },
        },
    };
}

fn invertMat33(matrix: [3][3]F) ?[3][3]F {
    const det = matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
        matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
        matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
    if (!std.math.isFinite(det) or @abs(det) < tol.distortion.det) return null;
    const inv_det = 1.0 / det;
    return .{
        .{
            (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) * inv_det,
            (matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2]) * inv_det,
            (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]) * inv_det,
        },
        .{
            (matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2]) * inv_det,
            (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]) * inv_det,
            (matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2]) * inv_det,
        },
        .{
            (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]) * inv_det,
            (matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1]) * inv_det,
            (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) * inv_det,
        },
    };
}

fn mulJac22(lhs: [2][2]F, rhs: [2][2]F) [2][2]F {
    return .{
        .{
            lhs[0][0] * rhs[0][0] + lhs[0][1] * rhs[1][0],
            lhs[0][0] * rhs[0][1] + lhs[0][1] * rhs[1][1],
        },
        .{
            lhs[1][0] * rhs[0][0] + lhs[1][1] * rhs[1][0],
            lhs[1][0] * rhs[0][1] + lhs[1][1] * rhs[1][1],
        },
    };
}

fn invBrownConradyExtLens(
    distortion: BrownConradyExt,
    x_d: F,
    y_d: F,
) !DistortionInvResult {
    var x = x_d;
    var y = y_d;
    for (0..cfg.distortion_newton_iter_max) |_| {
        const fwd = distortion.forwardLensWithJac(x, y);
        const f0 = fwd.x_d - x_d;
        const f1 = fwd.y_d - y_d;
        if (@max(@abs(f0), @abs(f1)) < tol.distortion.resid) {
            return .{ .x = x, .y = y };
        }
        const a = fwd.jac[0][0];
        const b = fwd.jac[0][1];
        const c = fwd.jac[1][0];
        const d = fwd.jac[1][1];
        const det = a * d - b * c;
        if (@abs(det) < tol.distortion.det) return error.SingularJac;
        const delta_x = (-f0 * d + b * f1) / det;
        const delta_y = (c * f0 - a * f1) / det;
        x += delta_x;
        y += delta_y;
        if (@max(@abs(delta_x), @abs(delta_y)) < tol.distortion.delta) {
            return .{ .x = x, .y = y };
        }
    }
    return error.DistortionInvFailed;
}

// --------------------------------------------------------------------------------------
// Polynomial Distortion
// --------------------------------------------------------------------------------------

pub const poly_powers_u = [10]u8{ 0, 1, 0, 2, 1, 0, 3, 2, 1, 0 };
pub const poly_powers_v = [10]u8{ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3 };

pub const PolynomialOrder = enum(u8) {
    linear = 1,
    quadratic = 2,
    cubic = 3,

    pub fn termCount(self: PolynomialOrder) usize {
        return switch (self) {
            .linear => 3,
            .quadratic => 6,
            .cubic => 10,
        };
    }
};

pub const PolynomialMap = struct {
    order: PolynomialOrder = .quadratic,
    coeffs_u: [10]F = [_]F{0.0} ** 10,
    coeffs_v: [10]F = [_]F{0.0} ** 10,

    pub fn evaluate(
        self: PolynomialMap,
        x: F,
        y: F,
    ) [2]F {
        const poly = evalPolynomialDisplacement(self, x, y);
        return .{ x + poly.du, y + poly.dv };
    }

    pub fn forwardWithJac(
        self: PolynomialMap,
        x: F,
        y: F,
    ) DistortionForwardJacResult {
        const distorted = self.evaluate(x, y);
        var ddu_dx: F = 0.0;
        var ddu_dy: F = 0.0;
        var ddv_dx: F = 0.0;
        var ddv_dy: F = 0.0;
        const term_count = self.order.termCount();

        for (0..term_count) |ii| {
            const pu = poly_powers_u[ii];
            const pv = poly_powers_v[ii];

            if (pu > 0) {
                const basis_dx = @as(F, @floatFromInt(pu)) *
                    powSmall(x, pu - 1) *
                    powSmall(y, pv);
                ddu_dx += self.coeffs_u[ii] * basis_dx;
                ddv_dx += self.coeffs_v[ii] * basis_dx;
            }
            if (pv > 0) {
                const basis_dy = @as(F, @floatFromInt(pv)) *
                    powSmall(x, pu) *
                    powSmall(y, pv - 1);
                ddu_dy += self.coeffs_u[ii] * basis_dy;
                ddv_dy += self.coeffs_v[ii] * basis_dy;
            }
        }

        return .{
            .x_d = distorted[0],
            .y_d = distorted[1],
            .jac = .{
                .{ 1.0 + ddu_dx, ddu_dy },
                .{ ddv_dx, 1.0 + ddv_dy },
            },
        };
    }

    pub fn inv(
        self: PolynomialMap,
        x_d: F,
        y_d: F,
    ) !DistortionInvResult {
        var x = x_d;
        var y = y_d;

        const max_iters = cfg.distortion_newton_iter_max;
        const tol_resid = tol.distortion.resid;
        const tol_delta = tol.distortion.delta;

        for (0..max_iters) |_| {
            const fwd = self.forwardWithJac(x, y);
            const f0 = fwd.x_d - x_d;
            const f1 = fwd.y_d - y_d;

            if (@max(@abs(f0), @abs(f1)) < tol_resid) {
                return .{ .x = x, .y = y };
            }

            const a = fwd.jac[0][0];
            const b = fwd.jac[0][1];
            const c = fwd.jac[1][0];
            const d = fwd.jac[1][1];
            const det = a * d - b * c;
            if (@abs(det) < tol.distortion.det) {
                return error.SingularJac;
            }

            const delta_x = (-f0 * d + b * f1) / det;
            const delta_y = (c * f0 - a * f1) / det;
            x += delta_x;
            y += delta_y;

            if (@max(@abs(delta_x), @abs(delta_y)) < tol_delta) {
                return .{ .x = x, .y = y };
            }
        }

        return error.DistortionInvFailed;
    }
};

pub const BidirectionalPolynomial = struct {
    forward_map: ?PolynomialMap = null,
    inv_map: ?PolynomialMap = null,

    pub fn forward(
        self: BidirectionalPolynomial,
        x: F,
        y: F,
    ) [2]F {
        if (self.forward_map) |forward_map| {
            return forward_map.evaluate(x, y);
        }
        if (self.inv_map) |inv_map| {
            const solved = inv_map.inv(x, y) catch unreachable;
            return .{ solved.x, solved.y };
        }
        unreachable;
    }

    pub fn inv(
        self: BidirectionalPolynomial,
        x_d: F,
        y_d: F,
    ) !DistortionInvResult {
        if (self.inv_map) |inv_map| {
            const eval = inv_map.evaluate(x_d, y_d);
            return .{ .x = eval[0], .y = eval[1] };
        }
        if (self.forward_map) |forward_map| {
            return try forward_map.inv(x_d, y_d);
        }
        return error.MissingPolynomialMap;
    }
};

pub const BrownConradyPolynomial = struct {
    brown_conrady: BrownConrady = .{},
    polynomial: BidirectionalPolynomial = .{},

    pub fn forward(
        self: BrownConradyPolynomial,
        x: F,
        y: F,
    ) [2]F {
        const brown = self.brown_conrady.forward(x, y);
        return self.polynomial.forward(brown[0], brown[1]);
    }

    pub fn inv(
        self: BrownConradyPolynomial,
        x_d: F,
        y_d: F,
    ) !DistortionInvResult {
        const poly_inv = try self.polynomial.inv(x_d, y_d);
        return try self.brown_conrady.inv(poly_inv.x, poly_inv.y);
    }
};

pub const BrownConradyExtPolynomial = struct {
    brown_conrady_ext: BrownConradyExt = .{},
    polynomial: BidirectionalPolynomial = .{},

    pub fn forward(
        self: BrownConradyExtPolynomial,
        x: F,
        y: F,
    ) [2]F {
        const brown = self.brown_conrady_ext.forward(x, y);
        return self.polynomial.forward(brown[0], brown[1]);
    }

    pub fn inv(
        self: BrownConradyExtPolynomial,
        x_d: F,
        y_d: F,
    ) !DistortionInvResult {
        const poly_inv = try self.polynomial.inv(x_d, y_d);
        return try self.brown_conrady_ext.inv(poly_inv.x, poly_inv.y);
    }
};

// --------------------------------------------------------------------------------------
// Distortion Unions
// --------------------------------------------------------------------------------------

pub const DistortionModel = union(enum) {
    none,
    brown_conrady: BrownConrady,
    brown_conrady_ext: BrownConradyExt,
    polynomial: BidirectionalPolynomial,
    brown_conrady_polynomial: BrownConradyPolynomial,
    brown_conrady_ext_polynomial: BrownConradyExtPolynomial,
};

pub fn prepareDistortionModel(model: DistortionModel) !DistortionModel {
    return switch (model) {
        .brown_conrady_ext => |brown| .{
            .brown_conrady_ext = try brown.prepare(),
        },
        .brown_conrady_ext_polynomial => |chain| .{
            .brown_conrady_ext_polynomial = .{
                .brown_conrady_ext = try chain.brown_conrady_ext.prepare(),
                .polynomial = chain.polynomial,
            },
        },
        else => model,
    };
}

test "BrownConradyExt prepared tilt matches direct evaluation" {
    const direct = BrownConradyExt{
        .k1 = -0.08,
        .s1 = 1.2e-3,
        .tau_x = 0.023,
        .tau_y = -0.031,
    };
    const prepared = try direct.prepare();
    const expected = direct.forward(0.47, -0.29);
    const actual = prepared.forward(0.47, -0.29);
    try std.testing.expectApproxEqAbs(expected[0], actual[0], 1.0e-14);
    try std.testing.expectApproxEqAbs(expected[1], actual[1], 1.0e-14);
    const recovered = try prepared.inv(actual[0], actual[1]);
    try std.testing.expectApproxEqAbs(@as(F, 0.47), recovered.x, 2.0e-5);
    try std.testing.expectApproxEqAbs(@as(F, -0.29), recovered.y, 2.0e-5);
}

test "BrownConradyExt rejects singular prepared tilt" {
    const singular = BrownConradyExt{ .tau_y = std.math.pi / 2.0 };
    try std.testing.expectError(
        error.SingularTiltProjection,
        singular.prepare(),
    );
}

test "BrownConradyExt inverse rejects singular tilt projection" {
    const singular = BrownConradyExt{ .tau_y = std.math.pi / 2.0 };
    try std.testing.expectError(
        error.SingularTiltProjection,
        singular.inv(0.2, -0.3),
    );
}

test "BrownConradyExt rational pole propagates non-finite forward value" {
    const pole = BrownConradyExt{ .k4 = -1.0 };
    const result = pole.forward(1.0, 0.0);
    try std.testing.expect(!std.math.isFinite(result[0]));
}

test "PolynomialMap inverse rejects a singular Jacobian" {
    const singular = PolynomialMap{
        .order = .linear,
        .coeffs_u = .{ 0.0, -1.0, 0.0 } ++ [_]F{0.0} ** 7,
    };
    try std.testing.expectError(error.SingularJac, singular.inv(0.3, -0.2));
}

// --------------------------------------------------------------------------------------
// Point Spread Func
// --------------------------------------------------------------------------------------

pub const SeparablePSF = enum {
    no,
    yes,
};

pub const PixelBoxPSF = struct {
    supp_rad_px: F = 0.5,
};

pub const GaussianPSF = struct {
    sigma_px: F,
    supp_rad_px: F,
    separable: SeparablePSF = .yes,
};

pub const AnisotropicGaussianPSF = struct {
    sigma_x_px: F,
    sigma_y_px: F,
    theta_rad: F = 0.0,
    supp_rad_px: F,
    separable: SeparablePSF = .no,
};

pub const PointSpreadFunc = union(enum) {
    pixel_box: PixelBoxPSF,
    gaussian: GaussianPSF,
    anisotropic_gaussian: AnisotropicGaussianPSF,
};

pub const PreparedPSFMode = enum {
    identity_fast,
    separable,
    nonseparable,
};

pub const PreparedPSF = struct {
    mode: PreparedPSFMode = .identity_fast,
    halo_px: u16 = 0,
    halo_subpx: usize = 0,
    radius_x_subpx: usize = 0,
    radius_y_subpx: usize = 0,
    weights_x: []F = &.{},
    weights_y: []F = &.{},
    weights_2d: []F = &.{},

    pub fn deinit(self: *PreparedPSF, allocator: std.mem.Allocator) void {
        if (self.weights_x.len > 0) allocator.free(self.weights_x);
        if (self.weights_y.len > 0) allocator.free(self.weights_y);
        if (self.weights_2d.len > 0) allocator.free(self.weights_2d);
        self.* = .{};
    }

    pub fn hasFilter(self: PreparedPSF) bool {
        return self.mode != .identity_fast;
    }
};

fn psfKernelValue1D(psf: PointSpreadFunc, dist_px: F) F {
    const abs_dist = @abs(dist_px);
    return switch (psf) {
        .pixel_box => |box| if (abs_dist <= box.supp_rad_px +
            tol.psf.supp_radius_inclusion) 1.0 else 0.0,
        .gaussian => |gauss| if (abs_dist <= gauss.supp_rad_px +
            tol.psf.supp_radius_inclusion)
            @exp(-0.5 * (dist_px * dist_px) / (gauss.sigma_px * gauss.sigma_px))
        else
            0.0,
        .anisotropic_gaussian => unreachable,
    };
}

fn psfKernelValue2D(psf: PointSpreadFunc, dx_px: F, dy_px: F) F {
    return switch (psf) {
        .pixel_box => |box| if (@abs(dx_px) <= box.supp_rad_px +
            tol.psf.supp_radius_inclusion and
            @abs(dy_px) <= box.supp_rad_px +
                tol.psf.supp_radius_inclusion)
            1.0
        else
            0.0,
        .gaussian => |gauss| if (@abs(dx_px) <= gauss.supp_rad_px +
            tol.psf.supp_radius_inclusion and
            @abs(dy_px) <= gauss.supp_rad_px +
                tol.psf.supp_radius_inclusion)
            @exp(-0.5 * (dx_px * dx_px + dy_px * dy_px) /
                (gauss.sigma_px * gauss.sigma_px))
        else
            0.0,
        .anisotropic_gaussian => |gauss| blk: {
            if (@abs(dx_px) > gauss.supp_rad_px +
                tol.psf.supp_radius_inclusion or
                @abs(dy_px) > gauss.supp_rad_px +
                    tol.psf.supp_radius_inclusion)
            {
                break :blk 0.0;
            }
            const c = @cos(gauss.theta_rad);
            const s = @sin(gauss.theta_rad);
            const xr = c * dx_px + s * dy_px;
            const yr = -s * dx_px + c * dy_px;
            break :blk @exp(-0.5 * ((xr * xr) / (gauss.sigma_x_px * gauss.sigma_x_px) +
                (yr * yr) / (gauss.sigma_y_px * gauss.sigma_y_px)));
        },
    };
}

fn normalizeKernel(weights: []F) void {
    var sum: F = 0.0;
    for (weights) |weight| sum += weight;
    if (sum == 0.0) return;
    for (weights) |*weight| weight.* /= sum;
}

fn buildKernel1D(
    allocator: std.mem.Allocator,
    psf: PointSpreadFunc,
    radius_subpx: usize,
    sub_sample: u32,
) ![]F {
    const size = 2 * radius_subpx + 1;
    const weights = try allocator.alloc(F, size);
    const sub_samp_f = @as(F, @floatFromInt(sub_sample));

    for (0..size) |ii| {
        const offset = @as(isize, @intCast(ii)) - @as(isize, @intCast(radius_subpx));
        const dist_px = @as(F, @floatFromInt(offset)) / sub_samp_f;
        weights[ii] = psfKernelValue1D(psf, dist_px);
    }

    normalizeKernel(weights);
    return weights;
}

fn invFromForwardWithJac(
    comptime DistortionType: type,
    distortion: DistortionType,
    x_d: F,
    y_d: F,
) !DistortionInvResult {
    var x = x_d;
    var y = y_d;

    const max_iters = cfg.distortion_newton_iter_max;
    const tol_resid = tol.distortion.resid;
    const tol_delta = tol.distortion.delta;

    for (0..max_iters) |_| {
        const fwd = distortion.forwardWithJac(x, y);
        const f0 = fwd.x_d - x_d;
        const f1 = fwd.y_d - y_d;

        if (@max(@abs(f0), @abs(f1)) < tol_resid) {
            return .{ .x = x, .y = y };
        }

        const a = fwd.jac[0][0];
        const b = fwd.jac[0][1];
        const c = fwd.jac[1][0];
        const d = fwd.jac[1][1];
        const det = a * d - b * c;
        if (@abs(det) < tol.distortion.det) {
            return error.SingularJac;
        }

        const delta_x = (-f0 * d + b * f1) / det;
        const delta_y = (c * f0 - a * f1) / det;

        x += delta_x;
        y += delta_y;

        if (@max(@abs(delta_x), @abs(delta_y)) < tol_delta) {
            return .{ .x = x, .y = y };
        }
    }

    return error.DistortionInvFailed;
}

fn evalPolynomialDisplacement(
    polynomial: PolynomialMap,
    x: F,
    y: F,
) struct { du: F, dv: F } {
    var du: F = 0.0;
    var dv: F = 0.0;
    const term_count = polynomial.order.termCount();

    for (0..term_count) |ii| {
        const basis = powSmall(x, poly_powers_u[ii]) *
            powSmall(y, poly_powers_v[ii]);
        du += polynomial.coeffs_u[ii] * basis;
        dv += polynomial.coeffs_v[ii] * basis;
    }

    return .{ .du = du, .dv = dv };
}

fn powSmall(
    x: F,
    power: u8,
) F {
    var out: F = 1.0;
    for (0..power) |_| {
        out *= x;
    }
    return out;
}

fn distortionForwardFromRadialScale(
    x: F,
    y: F,
    radial_scale: F,
    p1: F,
    p2: F,
) [2]F {
    const r2 = x * x + y * y;
    const x_d =
        x * radial_scale + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
    const y_d =
        y * radial_scale + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
    return .{ x_d, y_d };
}

fn distortionForwardWithJacFromRadialScale(
    x: F,
    y: F,
    radial_scale: F,
    dradial_dr2: F,
    p1: F,
    p2: F,
) DistortionForwardJacResult {
    const distorted = distortionForwardFromRadialScale(
        x,
        y,
        radial_scale,
        p1,
        p2,
    );
    const dradial_dx = dradial_dr2 * 2.0 * x;
    const dradial_dy = dradial_dr2 * 2.0 * y;

    const dx_fwd_dx =
        radial_scale + x * dradial_dx + 2.0 * p1 * y + 6.0 * p2 * x;
    const dx_fwd_dy = x * dradial_dy + 2.0 * p1 * x + 2.0 * p2 * y;
    const dy_fwd_dx = y * dradial_dx + 2.0 * p1 * x + 2.0 * p2 * y;
    const dy_fwd_dy =
        radial_scale + y * dradial_dy + 6.0 * p1 * y + 2.0 * p2 * x;

    return .{
        .x_d = distorted[0],
        .y_d = distorted[1],
        .jac = .{
            .{ dx_fwd_dx, dx_fwd_dy },
            .{ dy_fwd_dx, dy_fwd_dy },
        },
    };
}

fn buildKernel2D(
    allocator: std.mem.Allocator,
    psf: PointSpreadFunc,
    radius_x_subpx: usize,
    radius_y_subpx: usize,
    sub_sample: u32,
) ![]F {
    const width = 2 * radius_x_subpx + 1;
    const height = 2 * radius_y_subpx + 1;
    const weights = try allocator.alloc(F, width * height);
    const sub_samp_f = @as(F, @floatFromInt(sub_sample));

    for (0..height) |yy| {
        const y_off = @as(isize, @intCast(yy)) - @as(isize, @intCast(radius_y_subpx));
        const dy_px = @as(F, @floatFromInt(y_off)) / sub_samp_f;
        for (0..width) |xx| {
            const x_off = @as(isize, @intCast(xx)) - @as(isize, @intCast(radius_x_subpx));
            const dx_px = @as(F, @floatFromInt(x_off)) / sub_samp_f;

            weights[yy * width + xx] = psfKernelValue2D(psf, dx_px, dy_px);
        }
    }

    normalizeKernel(weights);
    return weights;
}

pub fn preparePSF(
    allocator: std.mem.Allocator,
    psf: PointSpreadFunc,
    sub_sample: u32,
) !PreparedPSF {
    switch (psf) {
        .pixel_box => |box| {
            if (box.supp_rad_px <= 0.5 +
                tol.psf.pixel_box_identity_supp_radius)
            {
                return .{};
            }
            const halo_px: u16 = @intCast(@max(
                @as(usize, 0),
                @as(usize, @intFromFloat(@ceil(box.supp_rad_px))),
            ));
            const radius_subpx: usize = @intFromFloat(
                @ceil(box.supp_rad_px * @as(F, @floatFromInt(sub_sample))),
            );
            return .{
                .mode = .separable,
                .halo_px = halo_px,
                .halo_subpx = @as(usize, halo_px) * @as(usize, sub_sample),
                .radius_x_subpx = radius_subpx,
                .radius_y_subpx = radius_subpx,
                .weights_x = try buildKernel1D(allocator, psf, radius_subpx, sub_sample),
                .weights_y = try buildKernel1D(allocator, psf, radius_subpx, sub_sample),
            };
        },
        .gaussian => |gauss| {
            const halo_px: u16 = @intCast(@max(
                @as(usize, 0),
                @as(usize, @intFromFloat(@ceil(gauss.supp_rad_px))),
            ));
            const radius_subpx: usize = @intFromFloat(
                @ceil(gauss.supp_rad_px * @as(F, @floatFromInt(sub_sample))),
            );
            if (gauss.separable == .yes) {
                return .{
                    .mode = .separable,
                    .halo_px = halo_px,
                    .halo_subpx = @as(usize, halo_px) * @as(usize, sub_sample),
                    .radius_x_subpx = radius_subpx,
                    .radius_y_subpx = radius_subpx,
                    .weights_x = try buildKernel1D(allocator, psf, radius_subpx, sub_sample),
                    .weights_y = try buildKernel1D(allocator, psf, radius_subpx, sub_sample),
                };
            }
            return .{
                .mode = .nonseparable,
                .halo_px = halo_px,
                .halo_subpx = @as(usize, halo_px) * @as(usize, sub_sample),
                .radius_x_subpx = radius_subpx,
                .radius_y_subpx = radius_subpx,
                .weights_2d = try buildKernel2D(
                    allocator,
                    psf,
                    radius_subpx,
                    radius_subpx,
                    sub_sample,
                ),
            };
        },
        .anisotropic_gaussian => |gauss| {
            const halo_px: u16 = @intCast(@max(
                @as(usize, 0),
                @as(usize, @intFromFloat(@ceil(gauss.supp_rad_px))),
            ));
            const radius_subpx: usize = @intFromFloat(
                @ceil(gauss.supp_rad_px * @as(F, @floatFromInt(sub_sample))),
            );
            const axis_aligned = @abs(@sin(gauss.theta_rad)) <
                tol.psf.anisotropic_axis_align;
            if (gauss.separable == .yes and axis_aligned) {
                const psf_x = PointSpreadFunc{
                    .gaussian = .{
                        .sigma_px = gauss.sigma_x_px,
                        .supp_rad_px = gauss.supp_rad_px,
                        .separable = .yes,
                    },
                };
                const psf_y = PointSpreadFunc{
                    .gaussian = .{
                        .sigma_px = gauss.sigma_y_px,
                        .supp_rad_px = gauss.supp_rad_px,
                        .separable = .yes,
                    },
                };
                return .{
                    .mode = .separable,
                    .halo_px = halo_px,
                    .halo_subpx = @as(usize, halo_px) * @as(usize, sub_sample),
                    .radius_x_subpx = radius_subpx,
                    .radius_y_subpx = radius_subpx,
                    .weights_x = try buildKernel1D(
                        allocator,
                        psf_x,
                        radius_subpx,
                        sub_sample,
                    ),
                    .weights_y = try buildKernel1D(
                        allocator,
                        psf_y,
                        radius_subpx,
                        sub_sample,
                    ),
                };
            }
            return .{
                .mode = .nonseparable,
                .halo_px = halo_px,
                .halo_subpx = @as(usize, halo_px) * @as(usize, sub_sample),
                .radius_x_subpx = radius_subpx,
                .radius_y_subpx = radius_subpx,
                .weights_2d = try buildKernel2D(
                    allocator,
                    psf,
                    radius_subpx,
                    radius_subpx,
                    sub_sample,
                ),
            };
        },
    }
}
