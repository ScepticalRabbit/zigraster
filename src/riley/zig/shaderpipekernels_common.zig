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
const MatSlice = @import("matslice.zig").MatSlice;
const shaderops = @import("shaderops.zig");
const shaderpipe = @import("shaderpipe.zig");
const CoordSpace = @import("geometrykernels.zig").CoordSpace;
const texops = @import("textureops.zig");

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub fn MonoShaderPipeKern(comptime N: usize) type {
    const scal = @import("shaderpipekernels_scalar.zig");
    const simd = @import("shaderpipekernels_simd.zig");
    const impl = if (buildconfig.config.simd == .on) simd else scal;
    return impl.MonoShaderPipeKern(N);
}

pub fn RgbShaderPipeKern(comptime N: usize) type {
    const scal = @import("shaderpipekernels_scalar.zig");
    const simd = @import("shaderpipekernels_simd.zig");
    const impl = if (buildconfig.config.simd == .on) simd else scal;
    return impl.RgbShaderPipeKern(N);
}

pub fn MultiShaderPipeKern(comptime N: usize, comptime C: usize) type {
    const scal = @import("shaderpipekernels_scalar.zig");
    const simd = @import("shaderpipekernels_simd.zig");
    const impl = if (buildconfig.config.simd == .on) simd else scal;
    return impl.MultiShaderPipeKern(N, C);
}
