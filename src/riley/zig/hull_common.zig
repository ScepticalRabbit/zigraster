// --------------------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const rops = @import("rasterops.zig");
const cam = @import("camera.zig");
const buildconfig = @import("buildconfig.zig");
const F = buildconfig.F;

const geomkerns = @import("geometrykernels.zig");
const MeshType = geomkerns.MeshType;

const tol = buildconfig.config.tol;

// --------------------------------------------------------------------------------------
// Public Entry-Point Functions
// --------------------------------------------------------------------------------------

pub fn AdaptiveHullPoints(comptime N: usize) type {
    const NH = blk: {
        for (std.enums.values(MeshType)) |mt| {
            if (mt.getNodesNum() == N) {
                if (mt == .tri3) break :blk 0;
                break :blk mt.getNumHullPoints();
            }
        }
        break :blk 0;
    };

    return struct {
        x: [NH]F,
        y: [NH]F,
    };
}

fn calcCornerMidsideCosTheta(
    cx: F,
    cy: F,
    px: F,
    py: F,
    nx: F,
    ny: F,
) ?F {
    const ax = px - cx;
    const ay = py - cy;
    const bx = nx - cx;
    const by = ny - cy;

    const a_mag_sq = ax * ax + ay * ay;
    const b_mag_sq = bx * bx + by * by;
    if (a_mag_sq <= 0.0 or b_mag_sq <= 0.0) {
        return null;
    }

    const inv_a_mag = 1.0 / @sqrt(a_mag_sq);
    const inv_b_mag = 1.0 / @sqrt(b_mag_sq);
    return (ax * inv_a_mag) * (bx * inv_b_mag) + (ay * inv_a_mag) * (by * inv_b_mag);
}

fn calcBezierCtrlPoint(
    c0x: F,
    c0y: F,
    c1x: F,
    c1y: F,
    mx: F,
    my: F,
) [2]F {
    return .{
        2.0 * mx - 0.5 * (c0x + c1x),
        2.0 * my - 0.5 * (c0y + c1y),
    };
}

fn buildAdaptiveHullTri6(
    lx: *const [6]F,
    ly: *const [6]F,
) AdaptiveHullPoints(6) {
    const edges = [3][3]usize{
        .{ 0, 1, 3 },
        .{ 1, 2, 4 },
        .{ 2, 0, 5 },
    };

    var hull: AdaptiveHullPoints(6) = undefined;
    inline for (edges, 0..) |edge, ii| {
        const p0 = edge[0];
        const p1 = edge[1];
        const pm = edge[2];
        const edge_val = rops.edgeFun3(lx[p0], ly[p0], lx[p1], ly[p1], lx[pm], ly[pm]);

        hull.x[ii * 2] = lx[p0];
        hull.y[ii * 2] = ly[p0];
        if (edge_val < 0) {
            const ctrl = calcBezierCtrlPoint(lx[p0], ly[p0], lx[p1], ly[p1], lx[pm], ly[pm]);
            hull.x[ii * 2 + 1] = ctrl[0];
            hull.y[ii * 2 + 1] = ctrl[1];
        } else {
            hull.x[ii * 2 + 1] = lx[pm];
            hull.y[ii * 2 + 1] = ly[pm];
        }
    }
    return hull;
}

fn buildAdaptiveHullQuad(
    comptime N: usize,
    lx: *const [N]F,
    ly: *const [N]F,
) AdaptiveHullPoints(N) {
    const edges = [4][3]usize{
        .{ 0, 1, 4 },
        .{ 1, 2, 5 },
        .{ 2, 3, 6 },
        .{ 3, 0, 7 },
    };

    var hull: AdaptiveHullPoints(N) = undefined;
    inline for (edges, 0..) |edge, ii| {
        const p0 = edge[0];
        const p1 = edge[1];
        const pm = edge[2];
        const edge_val = rops.edgeFun3(lx[p0], ly[p0], lx[p1], ly[p1], lx[pm], ly[pm]);

        hull.x[ii * 2] = lx[p0];
        hull.y[ii * 2] = ly[p0];
        if (edge_val < 0) {
            const ctrl = calcBezierCtrlPoint(lx[p0], ly[p0], lx[p1], ly[p1], lx[pm], ly[pm]);
            hull.x[ii * 2 + 1] = ctrl[0];
            hull.y[ii * 2 + 1] = ctrl[1];
        } else {
            hull.x[ii * 2 + 1] = lx[pm];
            hull.y[ii * 2 + 1] = ly[pm];
        }
    }
    return hull;
}

fn convexifyHullInPlace(comptime NH: usize, hull_x: *[NH]F, hull_y: *[NH]F) void {
    var ii: usize = 0;
    while (ii < 10) : (ii += 1) {
        var changed = false;
        for (0..NH) |nn| {
            const pp = (nn + NH - 1) % NH;
            const mm = (nn + 1) % NH;

            // Vis elems are CW. For CW: convex is val < 0, concave is val > 0.
            const val = rops.edgeFun3(
                hull_x[pp],
                hull_y[pp],
                hull_x[mm],
                hull_y[mm],
                hull_x[nn],
                hull_y[nn],
            );
            if (val > 0) {
                hull_x[nn] = 0.5 * (hull_x[pp] + hull_x[mm]);
                hull_y[nn] = 0.5 * (hull_y[pp] + hull_y[mm]);
                changed = true;
            }
        }
        if (!changed) break;
    }
}

pub fn buildAdaptiveHullPointsFromClip(
    comptime N: usize,
    camera: *const cam.CameraPrepared,
    coords_elem: rops.GatheredElemCoords(N),
    hull_convex_fallback_on: bool,
) AdaptiveHullPoints(N) {
    const x_off = 0.5 * @as(F, @floatFromInt(camera.pixels_num[0]));
    const y_off = 0.5 * @as(F, @floatFromInt(camera.pixels_num[1]));

    var lx: [N]F = undefined;
    var ly: [N]F = undefined;
    for (0..N) |nn| {
        lx[nn] = coords_elem.x[nn] / coords_elem.z[nn] + x_off;
        ly[nn] = coords_elem.y[nn] / coords_elem.z[nn] + y_off;
    }

    if (N == 4) {
        var hull: AdaptiveHullPoints(4) = undefined;
        inline for (0..4) |nn| {
            hull.x[nn] = lx[nn];
            hull.y[nn] = ly[nn];
        }
        return hull;
    }

    const cos_lower = @cos(std.math.degreesToRadians(tol.hull.corner_midside_ang_lower_deg));
    const cos_upper = @cos(std.math.degreesToRadians(tol.hull.corner_midside_ang_upper_deg));

    if (N == 6) {
        var hull = buildAdaptiveHullTri6(&lx, &ly);

        if (hull_convex_fallback_on) {
            const corner_midsides = [3][2]usize{ .{ 5, 3 }, .{ 3, 4 }, .{ 4, 5 } };
            var trigger = false;

            for (corner_midsides, 0..) |mids, nn| {
                const cos_theta = calcCornerMidsideCosTheta(
                    lx[nn],
                    ly[nn],
                    lx[mids[0]],
                    ly[mids[0]],
                    lx[mids[1]],
                    ly[mids[1]],
                ) orelse -2.0;
                if (cos_theta >= cos_lower or cos_theta <= cos_upper) {
                    trigger = true;
                    break;
                }
            }

            if (trigger) convexifyHullInPlace(6, &hull.x, &hull.y);
        }
        return hull;
    }

    if (N == 8 or N == 9) {
        var hull = buildAdaptiveHullQuad(N, &lx, &ly);

        if (hull_convex_fallback_on) {
            const corner_midsides = [4][2]usize{ .{ 7, 4 }, .{ 4, 5 }, .{ 5, 6 }, .{ 6, 7 } };
            var trigger = false;
            for (corner_midsides, 0..) |mids, nn| {
                const cos_theta = calcCornerMidsideCosTheta(
                    lx[nn],
                    ly[nn],
                    lx[mids[0]],
                    ly[mids[0]],
                    lx[mids[1]],
                    ly[mids[1]],
                ) orelse -2.0;
                if (cos_theta >= cos_lower or cos_theta <= cos_upper) {
                    trigger = true;
                    break;
                }
            }

            if (trigger) convexifyHullInPlace(8, &hull.x, &hull.y);
        }
        return hull;
    }

    unreachable;
}
