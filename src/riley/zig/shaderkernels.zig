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
    @import("shaderkernels_simd.zig")
else
    @import("shaderkernels_scalar.zig");

// --------------------------------------------------------------------------------------
// Public Entry-Point Func
// --------------------------------------------------------------------------------------

pub const NodalKernel = impl.NodalKernel;
pub const TexKernel = impl.TexKernel;
pub const FuncKernel = impl.FuncKernel;
