// --------------------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const cfg = @import("buildconfig.zig").config;
const impl = if (cfg.simd == .on)
    @import("hull_simd.zig")
else
    @import("hull_scalar.zig");
const simd_impl = @import("hull_simd.zig");

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const TessTriangle = impl.TessTriangle;
pub const HullResultSIMD = simd_impl.HullResultSIMD;
pub const Tessellation = impl.Tessellation;
pub const getTessellation = impl.getTessellation;

pub const AdaptiveHullPoints = @import("hull_common.zig").AdaptiveHullPoints;
pub const buildAdaptiveHullPointsFromClip =
    @import("hull_common.zig").buildAdaptiveHullPointsFromClip;
