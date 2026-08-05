// --------------------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const cfg = @import("buildconfig.zig").config;
const comm = @import("shaderops_common.zig");
const scal = @import("shaderops_scalar.zig");
const simd_impl = @import("shaderops_simd.zig");

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const ScaleOver = comm.ScaleOver;
pub const NormalType = comm.NormalType;
pub const FuncCoordMode = comm.FuncCoordMode;
pub const FuncShaderBuiltin = comm.FuncShaderBuiltin;
pub const FuncShaderParams = comm.FuncShaderParams;
pub const LocalShaderBuff = comm.LocalShaderBuff;
pub const NodalInput = comm.NodalInput;
pub const NodalPrepared = comm.NodalPrepared;
pub const TexInput = comm.TexInput;
pub const TexPrepared = comm.TexPrepared;
pub const FuncInput = comm.FuncInput;
pub const FuncPrepared = comm.FuncPrepared;
pub const ShadeContext = comm.ShadeContext;
pub const InterpData = comm.InterpData;
pub const ShaderInput = comm.ShaderInput;
pub const normFuncShaderParams = comm.normFuncShaderParams;
pub const NodalStatic = comm.NodalStatic;
pub const TexStatic = comm.TexStatic;
pub const FuncStatic = comm.FuncStatic;
pub const ShaderStatic = comm.ShaderStatic;
pub const ShaderPrepared = comm.ShaderPrepared;

// --------------------------------------------------------------------------------------
// Public Entry-Point Func
// --------------------------------------------------------------------------------------

pub const fillNodalClipScal = scal.fillNodalClipScal;
pub const fillNodalPerspScal = scal.fillNodalPerspScal;
pub const fillTexClipScal = scal.fillTexClipScal;
pub const fillTexPerspScal = scal.fillTexPerspScal;
pub const fillFuncClipScal = scal.fillFuncClipScal;
pub const fillFuncPerspScal = scal.fillFuncPerspScal;
pub const fillNodalClip = fillNodalClipScal;
pub const fillNodalPersp = fillNodalPerspScal;
pub const fillTexClip = fillTexClipScal;
pub const fillTexPersp = fillTexPerspScal;
pub const fillFuncClip = fillFuncClipScal;
pub const fillFuncPersp = fillFuncPerspScal;
pub const fillNodalClipSIMD = simd_impl.fillNodalClipSIMD;
pub const fillNodalPerspSIMD = simd_impl.fillNodalPerspSIMD;
pub const fillTexClipSIMD = simd_impl.fillTexClipSIMD;
pub const fillTexPerspSIMD = simd_impl.fillTexPerspSIMD;
pub const fillFuncClipSIMD = simd_impl.fillFuncClipSIMD;
pub const fillFuncPerspSIMD = simd_impl.fillFuncPerspSIMD;
