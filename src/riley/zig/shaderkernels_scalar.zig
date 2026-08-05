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
const comm = @import("shaderkernels_common.zig");
const shaderops = @import("shaderops.zig");
const MatSlice = @import("matslice.zig").MatSlice;
const CoordSpace = @import("geometrykernels.zig").CoordSpace;

// --------------------------------------------------------------------------------------
// Nodal Interp Shader
// --------------------------------------------------------------------------------------

pub fn NodalKernel(comptime N: usize) type {
    return struct {
        pub inline fn shade(
            comptime coord_space: CoordSpace,
            ctx_shade: shaderops.ShadeContext,
            interp: shaderops.InterpData(N),
            shader_buf: *const shaderops.LocalShaderBuff(N),
            shader: *const shaderops.NodalPrepared,
            ctx_report: anytype,
            spx_image_scratch: *MatSlice(F),
        ) void {
            comm.shadeNodalScalComm(
                N,
                coord_space,
                ctx_shade,
                interp,
                shader_buf,
                shader,
                ctx_report,
                spx_image_scratch,
            );
        }
    };
}

// --------------------------------------------------------------------------------------
// Texture Shader
// --------------------------------------------------------------------------------------

pub fn TexKernel(
    comptime N: usize,
    comptime T: type,
    comptime C: usize,
) type {
    return struct {
        pub inline fn shade(
            comptime coord_space: CoordSpace,
            ctx_shade: shaderops.ShadeContext,
            interp: shaderops.InterpData(N),
            shader_buf: *const shaderops.LocalShaderBuff(N),
            shader: *const shaderops.TexPrepared(T, C),
            ctx_report: anytype,
            spx_image_scratch: *MatSlice(F),
        ) void {
            comm.shadeTexScalComm(
                N,
                T,
                C,
                coord_space,
                ctx_shade,
                interp,
                shader_buf,
                shader,
                ctx_report,
                spx_image_scratch,
            );
        }
    };
}

// --------------------------------------------------------------------------------------
// Function Shader
// --------------------------------------------------------------------------------------

pub fn FuncKernel(
    comptime N: usize,
    comptime C: usize,
) type {
    return struct {
        pub inline fn shade(
            comptime coord_space: CoordSpace,
            ctx_shade: shaderops.ShadeContext,
            interp: shaderops.InterpData(N),
            shader_buf: *const shaderops.LocalShaderBuff(N),
            shader: *const shaderops.FuncPrepared,
            ctx_report: anytype,
            spx_image_scratch: *MatSlice(F),
        ) void {
            comm.shadeFuncScalComm(
                N,
                C,
                coord_space,
                ctx_shade,
                interp,
                shader_buf,
                shader,
                ctx_report,
                spx_image_scratch,
            );
        }
    };
}
