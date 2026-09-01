// --------------------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const comm = @import("shaderpipekernels_common.zig");

// --------------------------------------------------------------------------------------
// Public Entry-Point Types
// --------------------------------------------------------------------------------------

pub const MonoShaderPipeKern = comm.MonoShaderPipeKern;
pub const RgbShaderPipeKern = comm.RgbShaderPipeKern;
pub const MultiShaderPipeKern = comm.MultiShaderPipeKern;
