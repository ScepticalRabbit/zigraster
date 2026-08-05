// --------------------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const cfg = @import("buildconfig.zig").config;
const common_impl = @import("textureops_common.zig");
const simd_impl = @import("textureops_simd.zig");
const impl = if (cfg.simd == .on) simd_impl else common_impl;

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const TexSamp = common_impl.TexSamp;
pub const TexSampMode = common_impl.TexSampMode;
pub const TexSampConfig = common_impl.TexSampConfig;
pub const lanczos2Coeff = common_impl.lanczos2Coeff;
pub const TextureSampConfig = TexSampConfig;
pub const TexSample = TexSamp;
pub const TexSampleMode = TexSampMode;
pub const TexSampleConfig = TexSampConfig;
pub const TextureSampleConfig = TextureSampConfig;
pub const texelToFloat = common_impl.texelToFloat;
pub const Tex = impl.Tex;

// --------------------------------------------------------------------------------------
// Public Entry-Point Func
// --------------------------------------------------------------------------------------

pub const sampScal = impl.sampScal;
pub const sampGrey = impl.sampGrey;
pub const sampOneLane = simd_impl.sampOneLane;
pub const sampWide = simd_impl.sampWide;
pub const sampLanes = simd_impl.sampLanes;
pub const sampLanesTri3 = simd_impl.sampLanesTri3;
pub const sampleScal = sampScal;
pub const sampleGreyscale = sampGrey;
pub const sampleOneLane = sampOneLane;
pub const sampleWide = sampWide;
pub const sampleLanes = sampLanes;
pub const sampleLanesTri3 = sampLanesTri3;
