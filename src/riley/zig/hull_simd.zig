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
const VecSB = buildconfig.VecSB;
const VecSF = buildconfig.VecSF;
const tol = cfg.tol;
const rops = @import("rasterops.zig");
const S = cfg.simd_vec_width;

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const TessTriangle = struct {
    x: [3]F,
    y: [3]F,
    xi: [3]F,
    eta: [3]F,
};

pub const HullResultSIMD = struct {
    v_is_in: VecSB,
    v_seed_xi: VecSF,
    v_seed_eta: VecSF,
};

pub const HullResultScalar = struct {
    is_in: bool,
    seed_xi: F,
    seed_eta: F,
};

// --------------------------------------------------------------------------------------
// Public Entry-Point Func
// --------------------------------------------------------------------------------------

pub fn Tessellation(comptime NT: usize) type {
    return struct {
        triangles: [NT]TessTriangle,

        pub inline fn isInScalar(self: @This(), px: F, py: F) HullResultScalar {
            const eps = tol.hull.scal_inclusion;
            inline for (self.triangles) |tri| {
                const e0 = rops.edgeFun3(tri.x[0], tri.y[0], tri.x[1], tri.y[1], px, py);
                const e1 = rops.edgeFun3(tri.x[1], tri.y[1], tri.x[2], tri.y[2], px, py);
                const e2 = rops.edgeFun3(tri.x[2], tri.y[2], tri.x[0], tri.y[0], px, py);
                if (e0 >= -eps and e1 >= -eps and e2 >= -eps) {
                    const area = rops.edgeFun3(
                        tri.x[0],
                        tri.y[0],
                        tri.x[1],
                        tri.y[1],
                        tri.x[2],
                        tri.y[2],
                    );
                    const inv_area = 1.0 / area;
                    const w0 = e1 * inv_area;
                    const w1 = e2 * inv_area;
                    const w2 = e0 * inv_area;
                    return .{
                        .is_in = true,
                        .seed_xi = w0 * tri.xi[0] + w1 * tri.xi[1] + w2 * tri.xi[2],
                        .seed_eta = w0 * tri.eta[0] + w1 * tri.eta[1] + w2 * tri.eta[2],
                    };
                }
            }
            return .{
                .is_in = false,
                .seed_xi = 0.0,
                .seed_eta = 0.0,
            };
        }

        pub inline fn isInSIMD(
            self: @This(),
            v_px: VecSF,
            v_py: VecSF,
        ) HullResultSIMD {
            const eps = tol.hull.simd_inclusion;

            const v_m_eps: VecSF = @splat(-eps);
            var v_is_in: VecSB = @splat(false);
            var v_seed_xi: VecSF = @splat(0.0);
            var v_seed_eta: VecSF = @splat(0.0);

            inline for (self.triangles) |tri| {
                const v_e0 = rops.edgeFun3SIMD(
                    tri.x[0],
                    tri.y[0],
                    tri.x[1],
                    tri.y[1],
                    v_px,
                    v_py,
                );
                const v_e1 = rops.edgeFun3SIMD(
                    tri.x[1],
                    tri.y[1],
                    tri.x[2],
                    tri.y[2],
                    v_px,
                    v_py,
                );
                const v_e2 = rops.edgeFun3SIMD(
                    tri.x[2],
                    tri.y[2],
                    tri.x[0],
                    tri.y[0],
                    v_px,
                    v_py,
                );

                const v_in_tri = (v_e0 >= v_m_eps) &
                    (v_e1 >= v_m_eps) &
                    (v_e2 >= v_m_eps);

                if (@reduce(.Or, v_in_tri)) {
                    const area = rops.edgeFun3(
                        tri.x[0],
                        tri.y[0],
                        tri.x[1],
                        tri.y[1],
                        tri.x[2],
                        tri.y[2],
                    );

                    const v_inv_area: VecSF = @splat(1.0 / area);
                    const v_w0 = v_e1 * v_inv_area;
                    const v_w1 = v_e2 * v_inv_area;
                    const v_w2 = v_e0 * v_inv_area;

                    const v_tri_xi0: VecSF = @splat(tri.xi[0]);
                    const v_tri_xi1: VecSF = @splat(tri.xi[1]);
                    const v_tri_xi2: VecSF = @splat(tri.xi[2]);
                    const v_tri_eta0: VecSF = @splat(tri.eta[0]);
                    const v_tri_eta1: VecSF = @splat(tri.eta[1]);
                    const v_tri_eta2: VecSF = @splat(tri.eta[2]);

                    const v_curr_xi =
                        v_w0 * v_tri_xi0 + v_w1 * v_tri_xi1 + v_w2 * v_tri_xi2;
                    const v_curr_eta =
                        v_w0 * v_tri_eta0 + v_w1 * v_tri_eta1 + v_w2 * v_tri_eta2;

                    v_seed_xi = @select(F, v_in_tri, v_curr_xi, v_seed_xi);
                    v_seed_eta = @select(F, v_in_tri, v_curr_eta, v_seed_eta);

                    v_is_in = v_is_in | v_in_tri;
                }
            }
            return .{
                .v_is_in = v_is_in,
                .v_seed_xi = v_seed_xi,
                .v_seed_eta = v_seed_eta,
            };
        }
    };
}

pub fn getTessellation(
    comptime N: usize,
    comptime NH: usize,
    comptime NT: usize,
    hull_x: []const F,
    hull_y: []const F,
) Tessellation(NT) {
    var tess = Tessellation(NT){ .triangles = undefined };

    if (N == 4) {
        // Quad4 hull: C0, C1, C2, C3
        // Local para coords: (-1,-1), (1,-1), (1,1), (-1,1)
        tess.triangles[0] = .{
            .x = .{ hull_x[0], hull_x[1], hull_x[2] },
            .y = .{ hull_y[0], hull_y[1], hull_y[2] },
            .xi = .{ -1.0, 1.0, 1.0 },
            .eta = .{ -1.0, -1.0, 1.0 },
        };
        tess.triangles[1] = .{
            .x = .{ hull_x[0], hull_x[2], hull_x[3] },
            .y = .{ hull_y[0], hull_y[2], hull_y[3] },
            .xi = .{ -1.0, 1.0, -1.0 },
            .eta = .{ -1.0, 1.0, 1.0 },
        };
    } else if (N == 6 or N == 8 or N == 9) {
        const node_xi = if (N == 6)
            [_]F{ 0.0, 0.5, 1.0, 0.5, 0.0, 0.0 }
        else
            [_]F{ -1.0, 0.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0 };

        const node_eta = if (N == 6)
            [_]F{ 0.0, 0.0, 0.0, 0.5, 1.0, 0.5 }
        else
            [_]F{ -1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 1.0, 0.0 };

        var cx: F = 0;
        var cy: F = 0;
        var c_xi: F = 0;
        var c_eta: F = 0;

        for (0..NH) |ii| {
            cx += hull_x[ii];
            cy += hull_y[ii];
            c_xi += node_xi[ii];
            c_eta += node_eta[ii];
        }
        cx /= @as(F, @floatFromInt(NH));
        cy /= @as(F, @floatFromInt(NH));
        c_xi /= @as(F, @floatFromInt(NH));
        c_eta /= @as(F, @floatFromInt(NH));

        for (0..NH) |ii| {
            const next = (ii + 1) % NH;
            tess.triangles[ii] = .{
                .x = .{ cx, hull_x[ii], hull_x[next] },
                .y = .{ cy, hull_y[ii], hull_y[next] },
                .xi = .{ c_xi, node_xi[ii], node_xi[next] },
                .eta = .{ c_eta, node_eta[ii], node_eta[next] },
            };
        }
    }

    return tess;
}
