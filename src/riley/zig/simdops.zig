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
const VecSB = buildconfig.VecSB;
const VecSF = buildconfig.VecSF;

// --------------------------------------------------------------------------------------
// Public Entry-Point Func
// --------------------------------------------------------------------------------------

pub inline fn loadVecSF(subpx_vals: []const F, start_u: usize) VecSF {
    const lane_vals: [S]F = subpx_vals[start_u..][0..S].*;
    return @as(VecSF, lane_vals);
}

pub inline fn storeVecSF(subpx_vals: []F, start_u: usize, v_vals: VecSF) void {
    const lane_vals: [S]F = @as([S]F, v_vals);
    subpx_vals[start_u..][0..S].* = lane_vals;
}

pub inline fn storeMaskedVecSF(
    subpx_vals: []F,
    start_u: usize,
    v_mask_active: VecSB,
    v_vals: VecSF,
) void {
    const v_old_vals = loadVecSF(subpx_vals, start_u);
    const v_new_vals = @select(F, v_mask_active, v_vals, v_old_vals);
    storeVecSF(subpx_vals, start_u, v_new_vals);
}
