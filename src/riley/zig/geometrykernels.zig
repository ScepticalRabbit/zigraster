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
const S = buildconfig.SimdWidth;
const VecSB = buildconfig.VecSB;
const VecSF = buildconfig.VecSF;
const VecSU8 = buildconfig.VecSU8;
const tol = cfg.tol;

const rops = @import("rasterops.zig");
const newton = @import("newton.zig");
const NewtonSeed = newton.NewtonSeed;
const NewtonSeedSIMD = newton.NewtonSeedSIMD;
const shapefun = @import("shapefun.zig");
const NDArray = @import("ndarray.zig").NDArray;
const Vec3Slices = rops.Vec3Slices;
const rastcfg = @import("rasterconfig.zig");

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const TRI_CENTROID_XI: F = 1.0 / 3.0;
pub const TRI_CENTROID_ETA: F = 1.0 / 3.0;
pub const QUAD_CENTROID_XI: F = 0.0;
pub const QUAD_CENTROID_ETA: F = 0.0;

pub const MeshType = enum(u8) {
    tri3 = 0,
    tri3opt = 1,
    tri6 = 2,
    quad4 = 4,
    quad8 = 5,
    quad9 = 6,

    pub inline fn getNodesNum(self: MeshType) usize {
        return switch (self) {
            .tri3, .tri3opt => 3,
            .tri6 => 6,
            .quad4 => 4,
            .quad8 => 8,
            .quad9 => 9,
        };
    }

    pub inline fn getNumHullPoints(self: MeshType) usize {
        return switch (self) {
            .tri3, .tri3opt => 0,
            .tri6 => 6,
            .quad4 => 4,
            .quad8, .quad9 => 8,
        };
    }
};

pub const SolverKind = enum {
    hyperb,
    newton,
};

pub const CoordSpace = enum {
    raster,
    clip_px_leng,
};

// --------------------------------------------------------------------------------------
// Public Entry-Point Func
// --------------------------------------------------------------------------------------

pub fn GeometryResult(comptime N: usize) type {
    return struct {
        weights: ?[N]F,
        iters: u8,
        status: newton.NewtonStatus = .fail_iter_lim,
        pre_dom_conv: bool = true,
        xi_out: F = 0.0,
        eta_out: F = 0.0,
        xi_final: F = 0.0,
        eta_final: F = 0.0,
    };
}

pub fn GeometryResultSIMD(comptime N: usize) type {
    return struct {
        v_weights: [N]VecSF,
        v_mask: VecSB,
        v_status: VecSU8,
        v_pre_dom_conv: VecSB = @splat(true),
        v_iters: VecSU8,
        v_xi_out: VecSF = undefined,
        v_eta_out: VecSF = undefined,
        v_xi_final: VecSF = undefined,
        v_eta_final: VecSF = undefined,
        v_resid_x: VecSF = undefined,
        v_resid_y: VecSF = undefined,
    };
}

pub inline fn calcInvZClip(
    comptime N: usize,
    nodes: Vec3Slices(F),
    weights: [N]F,
) F {
    var sum_weighted_z: F = 0.0;

    inline for (0..N) |ind| {
        sum_weighted_z += weights[ind] * nodes.z[ind];
    }

    return 1.0 / sum_weighted_z;
}

pub const NewtonParams = struct {
    w_u_coeff: F,
    w_v_coeff: F,
    w_const: F,
};

pub fn Tri3Kernel() type {
    return struct {
        pub const nodes_num = 3;
        pub const hull_nodes_num = 0;
        pub const tess_triangles_num = 0;
        pub const coord_space = .raster;
        pub const solver_kind = .hyperb;

        pub inline fn getInvElemArea(nodes: Vec3Slices(F)) F {
            return 1.0 / rops.edgeFun3(
                nodes.x[0],
                nodes.y[0],
                nodes.x[1],
                nodes.y[1],
                nodes.x[2],
                nodes.y[2],
            );
        }

        pub inline fn solveWeightsHyperb(
            nodes: Vec3Slices(F),
            pixel_x: F,
            pixel_y: F,
            inv_area: F,
        ) GeometryResult(nodes_num) {
            const weights = getWeightsAt(nodes, pixel_x, pixel_y, inv_area);

            if (isInElem(weights)) {
                return .{ .weights = weights, .iters = 1 };
            }

            return .{ .weights = null, .iters = 1 };
        }

        pub inline fn getWeightsAt(
            nodes: Vec3Slices(F),
            pixel_x: F,
            pixel_y: F,
            inv_area: F,
        ) [nodes_num]F {
            return [_]F{
                rops.edgeFun3(
                    nodes.x[1],
                    nodes.y[1],
                    nodes.x[2],
                    nodes.y[2],
                    pixel_x,
                    pixel_y,
                ) * inv_area,
                rops.edgeFun3(
                    nodes.x[2],
                    nodes.y[2],
                    nodes.x[0],
                    nodes.y[0],
                    pixel_x,
                    pixel_y,
                ) * inv_area,
                rops.edgeFun3(
                    nodes.x[0],
                    nodes.y[0],
                    nodes.x[1],
                    nodes.y[1],
                    pixel_x,
                    pixel_y,
                ) * inv_area,
            };
        }

        pub inline fn isInElem(weights: [nodes_num]F) bool {
            const edge_tol = tol.edge.tri_weight_inclusion;

            return weights[0] >= -edge_tol and
                weights[1] >= -edge_tol and
                weights[2] >= -edge_tol;
        }

        pub inline fn calcInvZ(nodes: Vec3Slices(F), weights: [nodes_num]F) F {
            var inv_z: F = 0.0;

            inline for (0..nodes_num) |nn| {
                inv_z += weights[nn] * (1.0 / nodes.z[nn]);
            }

            return inv_z;
        }

        pub inline fn solveWeightsHyperbSIMD(
            nodes: Vec3Slices(F),
            v_pixel_x: VecSF,
            v_pixel_y: VecSF,
            v_inv_area: VecSF,
        ) GeometryResultSIMD(nodes_num) {
            const v_weights = getWeightsAtSIMD(
                nodes,
                v_pixel_x,
                v_pixel_y,
                v_inv_area,
            );
            const v_mask = isInElemSIMD(v_weights);

            return .{
                .v_weights = v_weights,
                .v_mask = v_mask,
                .v_status = @select(
                    u8,
                    v_mask,
                    @as(
                        VecSU8,
                        @splat(
                            @intFromEnum(newton.NewtonStatus.conv_resid),
                        ),
                    ),
                    @as(
                        VecSU8,
                        @splat(
                            @intFromEnum(newton.NewtonStatus.fail_dom),
                        ),
                    ),
                ),
                .v_iters = @splat(1),
            };
        }

        pub inline fn getWeightsAtSIMD(
            nodes: Vec3Slices(F),
            v_pixel_x: VecSF,
            v_pixel_y: VecSF,
            v_inv_area: VecSF,
        ) [nodes_num]VecSF {
            return [_]VecSF{
                rops.edgeFun3SIMD(
                    nodes.x[1],
                    nodes.y[1],
                    nodes.x[2],
                    nodes.y[2],
                    v_pixel_x,
                    v_pixel_y,
                ) * v_inv_area,
                rops.edgeFun3SIMD(
                    nodes.x[2],
                    nodes.y[2],
                    nodes.x[0],
                    nodes.y[0],
                    v_pixel_x,
                    v_pixel_y,
                ) * v_inv_area,
                rops.edgeFun3SIMD(
                    nodes.x[0],
                    nodes.y[0],
                    nodes.x[1],
                    nodes.y[1],
                    v_pixel_x,
                    v_pixel_y,
                ) * v_inv_area,
            };
        }

        pub inline fn isInElemSIMD(v_weights: [nodes_num]VecSF) VecSB {
            const edge_tol = tol.edge.tri_weight_inclusion;
            const v_edge_tol: VecSF = @splat(-edge_tol);

            return (v_weights[0] >= v_edge_tol) &
                (v_weights[1] >= v_edge_tol) &
                (v_weights[2] >= v_edge_tol);
        }

        pub inline fn calcInvZSIMD(
            nodes_inv_z: [nodes_num]VecSF,
            v_weights: [nodes_num]VecSF,
        ) VecSF {
            var v_inv_z: VecSF = @splat(0.0);

            inline for (0..nodes_num) |nn| {
                v_inv_z += v_weights[nn] * nodes_inv_z[nn];
            }

            return v_inv_z;
        }

        pub inline fn getSIMDInvZ(nodes: Vec3Slices(F)) [nodes_num]VecSF {
            var out: [nodes_num]VecSF = undefined;
            inline for (0..nodes_num) |ii| {
                out[ii] = @splat(1.0 / nodes.z[ii]);
            }
            return out;
        }
    };
}

pub fn Tri3OptKernel() type {
    return struct {
        pub const nodes_num = 3;
        pub const hull_nodes_num = 0;
        pub const tess_triangles_num = 0;
        pub const coord_space = .raster;
        pub const solver_kind = .hyperb;

        pub inline fn getSIMDInvZ(nodes: Vec3Slices(F)) [nodes_num]VecSF {
            var out: [nodes_num]VecSF = undefined;
            inline for (0..nodes_num) |ii| {
                out[ii] = @splat(1.0 / nodes.z[ii]);
            }
            return out;
        }
    };
}

pub fn Tri6Kernel() type {
    return struct {
        pub const nodes_num = 6;
        pub const hull_nodes_num = 6;
        pub const tess_triangles_num = 6;
        pub const coord_space = .clip_px_leng;
        pub const solver_kind = .newton;

        pub inline fn initSeed(
            seed_mode: rastcfg.NewtonSeedMode,
            hull_seed: ?NewtonSeed,
        ) NewtonSeed {
            if (seed_mode == .hull) {
                if (hull_seed) |seed| {
                    return seed;
                }
            }
            return .{ .xi = TRI_CENTROID_XI, .eta = TRI_CENTROID_ETA };
        }

        pub inline fn initSeedSIMD(
            seed_mode: rastcfg.NewtonSeedMode,
            hull_seed: ?NewtonSeedSIMD,
        ) NewtonSeedSIMD {
            if (seed_mode == .hull) {
                if (hull_seed) |seed| {
                    return seed;
                }
            }
            return .{
                .v_xi = @splat(TRI_CENTROID_XI),
                .v_eta = @splat(TRI_CENTROID_ETA),
            };
        }

        pub inline fn domViolation(xi: F, eta: F) F {
            return @max(-xi, 0.0) + @max(-eta, 0.0) + @max(xi + eta - 1.0, 0.0);
        }

        pub inline fn solveWeightsNewton(
            nodes: Vec3Slices(F),
            pixel_x: F,
            pixel_y: F,
            x_offset: F,
            y_offset: F,
            xi_seed: F,
            eta_seed: F,
        ) GeometryResult(nodes_num) {
            const targ_x = pixel_x - x_offset;
            const targ_y = pixel_y - y_offset;

            var node_vals: [nodes_num]F = undefined;
            const result = newton.solveScal(
                nodes_num,
                targ_x,
                targ_y,
                nodes.x,
                nodes.y,
                nodes.z,
                xi_seed,
                eta_seed,
                &node_vals,
            );

            if (comptime cfg.newton_solver_mode == .robust) {
                if (result.conv) {
                    return .{
                        .weights = node_vals,
                        .iters = result.iters,
                        .status = result.status,
                        .pre_dom_conv = result.pre_dom_conv,
                        .xi_out = result.xi,
                        .eta_out = result.eta,
                        .xi_final = result.xi,
                        .eta_final = result.eta,
                    };
                }
                return .{
                    .weights = null,
                    .iters = result.iters,
                    .status = result.status,
                    .pre_dom_conv = result.pre_dom_conv,
                    .xi_final = result.xi,
                    .eta_final = result.eta,
                };
            }

            if (result.conv) {
                return .{
                    .weights = node_vals,
                    .iters = result.iters,
                    .status = .conv_resid,
                    .pre_dom_conv = result.pre_dom_conv,
                    .xi_out = result.xi,
                    .eta_out = result.eta,
                    .xi_final = result.xi,
                    .eta_final = result.eta,
                };
            }
            return .{
                .weights = null,
                .iters = result.iters,
                .status = if (result.pre_dom_conv)
                    .fail_dom
                else if (result.hit_iter_lim)
                    .fail_iter_lim
                else
                    .fail_near_singular,
                .pre_dom_conv = result.pre_dom_conv,
                .xi_final = result.xi,
                .eta_final = result.eta,
            };
        }

        pub inline fn solveWeightsNewtonSIMD(
            nodes: Vec3Slices(F),
            v_pixel_x: VecSF,
            v_pixel_y: VecSF,
            v_xi_seed: VecSF,
            v_eta_seed: VecSF,
            x_offset: F,
            y_offset: F,
        ) GeometryResultSIMD(nodes_num) {
            const v_targ_x = v_pixel_x - @as(VecSF, @splat(x_offset));
            const v_targ_y = v_pixel_y - @as(VecSF, @splat(y_offset));

            var v_weights: [nodes_num]VecSF = undefined;
            const res = newton.solveSIMD(
                nodes_num,
                v_targ_x,
                v_targ_y,
                nodes.x,
                nodes.y,
                nodes.z,
                v_xi_seed,
                v_eta_seed,
                &v_weights,
            );

            if (comptime cfg.newton_solver_mode == .robust) {
                return .{
                    .v_weights = v_weights,
                    .v_mask = res.v_conv,
                    .v_status = res.v_status,
                    .v_pre_dom_conv = res.v_pre_dom_conv,
                    .v_iters = res.v_iters,
                    .v_xi_out = res.v_xi,
                    .v_eta_out = res.v_eta,
                    .v_xi_final = res.v_xi,
                    .v_eta_final = res.v_eta,
                    .v_resid_x = res.v_resid_x,
                    .v_resid_y = res.v_resid_y,
                };
            }

            return .{
                .v_weights = v_weights,
                .v_mask = res.v_conv,
                .v_status = blk: {
                    const v_status_fail_iter_lim: VecSU8 =
                        @splat(@intFromEnum(newton.NewtonStatus.fail_iter_lim));
                    const v_status_fail_near_singular: VecSU8 =
                        @splat(@intFromEnum(newton.NewtonStatus.fail_near_singular));
                    const v_status_conv_resid: VecSU8 =
                        @splat(@intFromEnum(newton.NewtonStatus.conv_resid));
                    const v_status_fail_dom: VecSU8 =
                        @splat(@intFromEnum(newton.NewtonStatus.fail_dom));
                    const v_fail_dom = res.v_pre_dom_conv & !res.v_conv;
                    var v_status = v_status_fail_iter_lim;
                    v_status = @select(
                        u8,
                        !res.v_hit_iter_lim & !res.v_pre_dom_conv,
                        v_status_fail_near_singular,
                        v_status,
                    );
                    v_status = @select(
                        u8,
                        res.v_conv,
                        v_status_conv_resid,
                        v_status,
                    );
                    v_status = @select(
                        u8,
                        v_fail_dom,
                        v_status_fail_dom,
                        v_status,
                    );
                    break :blk v_status;
                },
                .v_pre_dom_conv = res.v_pre_dom_conv,
                .v_iters = res.v_iters,
                .v_xi_out = res.v_xi,
                .v_eta_out = res.v_eta,
                .v_xi_final = res.v_xi,
                .v_eta_final = res.v_eta,
                .v_resid_x = res.v_resid_x,
                .v_resid_y = res.v_resid_y,
            };
        }

        pub inline fn calcInvZ(nodes: Vec3Slices(F), weights: [nodes_num]F) F {
            return calcInvZClip(nodes_num, nodes, weights);
        }
    };
}

pub fn Quad4Kernel() type {
    return struct {
        pub const nodes_num = 4;
        pub const hull_nodes_num = 4;
        pub const tess_triangles_num = 2;
        pub const coord_space = .clip_px_leng;
        pub const solver_kind = .newton;

        pub inline fn initSeed(seed_mode: rastcfg.NewtonSeedMode, hull_seed: ?NewtonSeed) NewtonSeed {
            if (seed_mode == .hull) {
                if (hull_seed) |seed| {
                    return seed;
                }
            }
            return .{ .xi = QUAD_CENTROID_XI, .eta = QUAD_CENTROID_ETA };
        }

        pub inline fn initSeedSIMD(
            seed_mode: rastcfg.NewtonSeedMode,
            hull_seed: ?NewtonSeedSIMD,
        ) NewtonSeedSIMD {
            if (seed_mode == .hull) {
                if (hull_seed) |seed| {
                    return seed;
                }
            }
            return .{
                .v_xi = @splat(QUAD_CENTROID_XI),
                .v_eta = @splat(QUAD_CENTROID_ETA),
            };
        }

        pub inline fn domViolation(xi: F, eta: F) F {
            return @max(@abs(xi) - 1.0, 0.0) + @max(@abs(eta) - 1.0, 0.0);
        }

        pub inline fn solveWeightsNewton(
            nodes: Vec3Slices(F),
            pixel_x: F,
            pixel_y: F,
            x_offset: F,
            y_offset: F,
            xi_seed: F,
            eta_seed: F,
        ) GeometryResult(nodes_num) {
            const targ_x = pixel_x - x_offset;
            const targ_y = pixel_y - y_offset;

            var node_vals: [nodes_num]F = undefined;
            const result = newton.solveScal(
                nodes_num,
                targ_x,
                targ_y,
                nodes.x,
                nodes.y,
                nodes.z,
                xi_seed,
                eta_seed,
                &node_vals,
            );
            if (comptime cfg.newton_solver_mode == .robust) {
                if (result.conv) {
                    return .{
                        .weights = node_vals,
                        .iters = result.iters,
                        .status = result.status,
                        .pre_dom_conv = result.pre_dom_conv,
                        .xi_out = result.xi,
                        .eta_out = result.eta,
                        .xi_final = result.xi,
                        .eta_final = result.eta,
                    };
                }
                return .{
                    .weights = null,
                    .iters = result.iters,
                    .status = result.status,
                    .pre_dom_conv = result.pre_dom_conv,
                    .xi_final = result.xi,
                    .eta_final = result.eta,
                };
            }

            if (result.conv) {
                return .{
                    .weights = node_vals,
                    .iters = result.iters,
                    .status = .conv_resid,
                    .pre_dom_conv = result.pre_dom_conv,
                    .xi_out = result.xi,
                    .eta_out = result.eta,
                    .xi_final = result.xi,
                    .eta_final = result.eta,
                };
            }
            return .{
                .weights = null,
                .iters = result.iters,
                .status = if (result.pre_dom_conv)
                    .fail_dom
                else if (result.hit_iter_lim)
                    .fail_iter_lim
                else
                    .fail_near_singular,
                .pre_dom_conv = result.pre_dom_conv,
                .xi_final = result.xi,
                .eta_final = result.eta,
            };
        }

        pub inline fn solveWeightsNewtonSIMD(
            nodes: Vec3Slices(F),
            v_pixel_x: VecSF,
            v_pixel_y: VecSF,
            v_xi_seed: VecSF,
            v_eta_seed: VecSF,
            x_offset: F,
            y_offset: F,
        ) GeometryResultSIMD(nodes_num) {
            const v_targ_x = v_pixel_x - @as(VecSF, @splat(x_offset));
            const v_targ_y = v_pixel_y - @as(VecSF, @splat(y_offset));

            var v_weights: [nodes_num]VecSF = undefined;
            const res = newton.solveSIMD(
                nodes_num,
                v_targ_x,
                v_targ_y,
                nodes.x,
                nodes.y,
                nodes.z,
                v_xi_seed,
                v_eta_seed,
                &v_weights,
            );

            if (comptime cfg.newton_solver_mode == .robust) {
                return .{
                    .v_weights = v_weights,
                    .v_mask = res.v_conv,
                    .v_status = res.v_status,
                    .v_pre_dom_conv = res.v_pre_dom_conv,
                    .v_iters = res.v_iters,
                    .v_xi_out = res.v_xi,
                    .v_eta_out = res.v_eta,
                    .v_xi_final = res.v_xi,
                    .v_eta_final = res.v_eta,
                    .v_resid_x = res.v_resid_x,
                    .v_resid_y = res.v_resid_y,
                };
            }

            return .{
                .v_weights = v_weights,
                .v_mask = res.v_conv,
                .v_status = blk: {
                    const v_status_fail_iter_lim: VecSU8 =
                        @splat(@intFromEnum(newton.NewtonStatus.fail_iter_lim));
                    const v_status_fail_near_singular: VecSU8 =
                        @splat(@intFromEnum(newton.NewtonStatus.fail_near_singular));
                    const v_status_conv_resid: VecSU8 =
                        @splat(@intFromEnum(newton.NewtonStatus.conv_resid));
                    const v_status_fail_dom: VecSU8 =
                        @splat(@intFromEnum(newton.NewtonStatus.fail_dom));
                    const v_fail_dom = res.v_pre_dom_conv & !res.v_conv;
                    var v_status = v_status_fail_iter_lim;
                    v_status = @select(
                        u8,
                        !res.v_hit_iter_lim & !res.v_pre_dom_conv,
                        v_status_fail_near_singular,
                        v_status,
                    );
                    v_status = @select(
                        u8,
                        res.v_conv,
                        v_status_conv_resid,
                        v_status,
                    );
                    v_status = @select(
                        u8,
                        v_fail_dom,
                        v_status_fail_dom,
                        v_status,
                    );
                    break :blk v_status;
                },
                .v_pre_dom_conv = res.v_pre_dom_conv,
                .v_iters = res.v_iters,
                .v_xi_out = res.v_xi,
                .v_eta_out = res.v_eta,
                .v_xi_final = res.v_xi,
                .v_eta_final = res.v_eta,
                .v_resid_x = res.v_resid_x,
                .v_resid_y = res.v_resid_y,
            };
        }

        pub inline fn calcInvZ(nodes: Vec3Slices(F), weights: [nodes_num]F) F {
            return calcInvZClip(nodes_num, nodes, weights);
        }
    };
}

pub fn Quad89Kernel(comptime N: usize) type {
    return struct {
        pub const nodes_num = N;
        pub const hull_nodes_num = 8;
        pub const tess_triangles_num = 8;
        pub const coord_space = .clip_px_leng;
        pub const solver_kind = .newton;

        pub inline fn initSeed(seed_mode: rastcfg.NewtonSeedMode, hull_seed: ?NewtonSeed) NewtonSeed {
            if (seed_mode == .hull) {
                if (hull_seed) |seed| {
                    return seed;
                }
            }
            return .{ .xi = QUAD_CENTROID_XI, .eta = QUAD_CENTROID_ETA };
        }

        pub inline fn initSeedSIMD(
            seed_mode: rastcfg.NewtonSeedMode,
            hull_seed: ?NewtonSeedSIMD,
        ) NewtonSeedSIMD {
            if (seed_mode == .hull) {
                if (hull_seed) |seed| {
                    return seed;
                }
            }
            return .{
                .v_xi = @splat(QUAD_CENTROID_XI),
                .v_eta = @splat(QUAD_CENTROID_ETA),
            };
        }

        pub inline fn domViolation(xi: F, eta: F) F {
            return @max(@abs(xi) - 1.0, 0.0) + @max(@abs(eta) - 1.0, 0.0);
        }

        pub inline fn solveWeightsNewton(
            nodes: Vec3Slices(F),
            pixel_x: F,
            pixel_y: F,
            x_offset: F,
            y_offset: F,
            xi_seed: F,
            eta_seed: F,
        ) GeometryResult(nodes_num) {
            const targ_x = pixel_x - x_offset;
            const targ_y = pixel_y - y_offset;

            var node_vals: [nodes_num]F = undefined;
            const result = newton.solveScal(
                nodes_num,
                targ_x,
                targ_y,
                nodes.x,
                nodes.y,
                nodes.z,
                xi_seed,
                eta_seed,
                &node_vals,
            );

            if (comptime cfg.newton_solver_mode == .robust) {
                if (result.conv) {
                    return .{
                        .weights = node_vals,
                        .iters = result.iters,
                        .status = result.status,
                        .pre_dom_conv = result.pre_dom_conv,
                        .xi_out = result.xi,
                        .eta_out = result.eta,
                        .xi_final = result.xi,
                        .eta_final = result.eta,
                    };
                }
                return .{
                    .weights = null,
                    .iters = result.iters,
                    .status = result.status,
                    .pre_dom_conv = result.pre_dom_conv,
                    .xi_final = result.xi,
                    .eta_final = result.eta,
                };
            }

            if (result.conv) {
                return .{
                    .weights = node_vals,
                    .iters = result.iters,
                    .status = .conv_resid,
                    .pre_dom_conv = result.pre_dom_conv,
                    .xi_out = result.xi,
                    .eta_out = result.eta,
                    .xi_final = result.xi,
                    .eta_final = result.eta,
                };
            }
            return .{
                .weights = null,
                .iters = result.iters,
                .status = if (result.pre_dom_conv)
                    .fail_dom
                else if (result.hit_iter_lim)
                    .fail_iter_lim
                else
                    .fail_near_singular,
                .pre_dom_conv = result.pre_dom_conv,
                .xi_final = result.xi,
                .eta_final = result.eta,
            };
        }

        pub inline fn solveWeightsNewtonSIMD(
            nodes: Vec3Slices(F),
            v_pixel_x: VecSF,
            v_pixel_y: VecSF,
            v_xi_seed: VecSF,
            v_eta_seed: VecSF,
            x_offset: F,
            y_offset: F,
        ) GeometryResultSIMD(nodes_num) {
            const v_targ_x = v_pixel_x - @as(VecSF, @splat(x_offset));
            const v_targ_y = v_pixel_y - @as(VecSF, @splat(y_offset));

            var v_weights: [nodes_num]VecSF = undefined;
            const res = newton.solveSIMD(
                nodes_num,
                v_targ_x,
                v_targ_y,
                nodes.x,
                nodes.y,
                nodes.z,
                v_xi_seed,
                v_eta_seed,
                &v_weights,
            );

            if (comptime cfg.newton_solver_mode == .robust) {
                return .{
                    .v_weights = v_weights,
                    .v_mask = res.v_conv,
                    .v_status = res.v_status,
                    .v_pre_dom_conv = res.v_pre_dom_conv,
                    .v_iters = res.v_iters,
                    .v_xi_out = res.v_xi,
                    .v_eta_out = res.v_eta,
                    .v_xi_final = res.v_xi,
                    .v_eta_final = res.v_eta,
                    .v_resid_x = res.v_resid_x,
                    .v_resid_y = res.v_resid_y,
                };
            }

            return .{
                .v_weights = v_weights,
                .v_mask = res.v_conv,
                .v_status = blk: {
                    const v_status_fail_iter_lim: VecSU8 =
                        @splat(@intFromEnum(newton.NewtonStatus.fail_iter_lim));
                    const v_status_fail_near_singular: VecSU8 =
                        @splat(@intFromEnum(newton.NewtonStatus.fail_near_singular));
                    const v_status_conv_resid: VecSU8 =
                        @splat(@intFromEnum(newton.NewtonStatus.conv_resid));
                    const v_status_fail_dom: VecSU8 =
                        @splat(@intFromEnum(newton.NewtonStatus.fail_dom));
                    const v_fail_dom = res.v_pre_dom_conv & !res.v_conv;
                    var v_status = v_status_fail_iter_lim;
                    v_status = @select(
                        u8,
                        !res.v_hit_iter_lim & !res.v_pre_dom_conv,
                        v_status_fail_near_singular,
                        v_status,
                    );
                    v_status = @select(
                        u8,
                        res.v_conv,
                        v_status_conv_resid,
                        v_status,
                    );
                    v_status = @select(
                        u8,
                        v_fail_dom,
                        v_status_fail_dom,
                        v_status,
                    );
                    break :blk v_status;
                },
                .v_pre_dom_conv = res.v_pre_dom_conv,
                .v_iters = res.v_iters,
                .v_xi_out = res.v_xi,
                .v_eta_out = res.v_eta,
                .v_xi_final = res.v_xi,
                .v_eta_final = res.v_eta,
                .v_resid_x = res.v_resid_x,
                .v_resid_y = res.v_resid_y,
            };
        }

        pub inline fn calcInvZ(nodes: Vec3Slices(F), weights: [nodes_num]F) F {
            return calcInvZClip(nodes_num, nodes, weights);
        }
    };
}
