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

    pub fn forward(
        self: BrownConradyExt,
        x: F,
        y: F,
    ) [2]F {
        const radial = self.calcRadialScaleAndDerivative(x, y);
        return distortionForwardFromRadialScale(
            x,
            y,
            radial.radial_scale,
            self.p1,
            self.p2,
        );
    }

    pub fn forwardWithJac(
        self: BrownConradyExt,
        x: F,
        y: F,
    ) DistortionForwardJacResult {
        const radial = self.calcRadialScaleAndDerivative(x, y);
        return distortionForwardWithJacFromRadialScale(
            x,
            y,
            radial.radial_scale,
            radial.dradial_dr2,
            self.p1,
            self.p2,
        );
    }

    pub fn inv(
        self: BrownConradyExt,
        x_d: F,
        y_d: F,
    ) !DistortionInvResult {
        return invFromForwardWithJac(BrownConradyExt, self, x_d, y_d);
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
