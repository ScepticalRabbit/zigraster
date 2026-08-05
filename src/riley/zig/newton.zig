// --------------------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const buildconfig = @import("buildconfig.zig");
const common = @import("newton_common.zig");
const scal = @import("newton_scalar.zig");
const simd = @import("newton_simd.zig");

// --------------------------------------------------------------------------------------
// Public Entry-Point Func
// --------------------------------------------------------------------------------------

pub fn solveScal(
    comptime N: usize,
    targ_x: buildconfig.F,
    targ_y: buildconfig.F,
    elem_node_x: []const buildconfig.F,
    elem_node_y: []const buildconfig.F,
    elem_node_w: []const buildconfig.F,
    xi_seed: buildconfig.F,
    eta_seed: buildconfig.F,
    node_vals: *[N]buildconfig.F,
) if (buildconfig.config.newton_solver_mode == .robust)
    common.NewtonResRobustScal
else
    common.NewtonResFastScal {
    if (comptime buildconfig.config.newton_solver_mode == .robust) {
        return scal.solveRobustScal(
            N,
            targ_x,
            targ_y,
            elem_node_x,
            elem_node_y,
            elem_node_w,
            xi_seed,
            eta_seed,
            node_vals,
        );
    }

    return scal.solveFastScal(
        N,
        targ_x,
        targ_y,
        elem_node_x,
        elem_node_y,
        elem_node_w,
        xi_seed,
        eta_seed,
        node_vals,
    );
}

pub fn solveSIMD(
    comptime N: usize,
    v_targ_x: buildconfig.VecSF,
    v_targ_y: buildconfig.VecSF,
    elem_node_x: []const buildconfig.F,
    elem_node_y: []const buildconfig.F,
    elem_node_w: []const buildconfig.F,
    v_xi_seed: buildconfig.VecSF,
    v_eta_seed: buildconfig.VecSF,
    v_node_vals: *[N]buildconfig.VecSF,
) if (buildconfig.config.newton_solver_mode == .robust)
    common.NewtonResRobustSIMD
else
    common.NewtonResFastSIMD {
    if (comptime buildconfig.config.newton_solver_mode == .robust) {
        return simd.solveRobustSIMD(
            N,
            v_targ_x,
            v_targ_y,
            elem_node_x,
            elem_node_y,
            elem_node_w,
            v_xi_seed,
            v_eta_seed,
            v_node_vals,
        );
    }

    return simd.solveFastSIMD(
        N,
        v_targ_x,
        v_targ_y,
        elem_node_x,
        elem_node_y,
        elem_node_w,
        v_xi_seed,
        v_eta_seed,
        v_node_vals,
    );
}

pub const reportSolve = scal.reportSolve;
pub const selectSeed = common.selectSeed;
pub const isSeedFinite = common.isSeedFinite;
pub const updateSeedState = common.updateSeedState;
pub const applySeedReuseInPlace = common.applySeedReuseInPlace;
pub const updateSeedStateFromSIMDResult = common.updateSeedStateFromSIMDResult;
pub const evaluateSeedQuality = common.evaluateSeedQuality;
pub const evaluateSolveState = common.evaluateSolveState;
pub const calcJacDet2D = common.calcJacDet2D;
pub const isConvStatus = common.isConvStatus;
pub const isPreDomConvStatus = common.isPreDomConvStatus;
pub const hitIterLimitStatus = common.hitIterLimitStatus;
pub const statusLabel = common.statusLabel;

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const NewtonSeed = common.NewtonSeed;
pub const NewtonSeedSIMD = common.NewtonSeedSIMD;
pub const NewtonSeedState = common.NewtonSeedState;
pub const NewtonSeedQuality = common.NewtonSeedQuality;
pub const NewtonEvalState = common.NewtonEvalState;
pub const NewtonStatus = common.NewtonStatus;
pub const NewtonResFastScal = common.NewtonResFastScal;
pub const NewtonResRobustScal = common.NewtonResRobustScal;
pub const NewtonResFastSIMD = common.NewtonResFastSIMD;
pub const NewtonResRobustSIMD = common.NewtonResRobustSIMD;
