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

const mono = @import("shaderpipe_mono.zig");
const rgb = @import("shaderpipe_rgb.zig");
const multi = @import("shaderpipe_multi.zig");
const shaderops_common = @import("shaderops_common.zig");

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const ShaderPipeKind = enum {
    monochrome,
    rgb,
    multi_channel,
    infrared,
};

pub const ShaderPipeRequest = union(ShaderPipeKind) {
    monochrome: void,
    rgb: void,
    multi_channel: usize,
    infrared: void,

    pub fn getFieldsNum(self: ShaderPipeRequest) u8 {
        return switch (self) {
            .monochrome => 1,
            .rgb => 3,
            .multi_channel => |channel_count| @as(u8, @intCast(channel_count)),
            .infrared => 1,
        };
    }
};

pub const ScaleOver = shaderops_common.ScaleOver;
pub const NormalType = shaderops_common.NormalType;
pub const FuncCoordMode = shaderops_common.FuncCoordMode;
pub const FuncShaderBuiltin = shaderops_common.FuncShaderBuiltin;
pub const FuncShaderParams = shaderops_common.FuncShaderParams;

pub const MonoPipeState = mono.MonoPipeState;
pub const MonoPipeStateSIMD = mono.MonoPipeStateSIMD;
pub const MonoShaderStage = mono.MonoShaderStage;
pub const MonoTerminal = mono.MonoTerminal;
pub const LocalMonoPipeBuff = mono.LocalMonoPipeBuff;
pub const MonoNodalInput = mono.MonoNodalInput;
pub const MonoNodalStatic = mono.MonoNodalStatic;
pub const MonoNodalPrepared = mono.MonoNodalPrepared;
pub const MonoTexPayload = mono.MonoTexPayload;
pub const MonoTexInput = mono.MonoTexInput;
pub const MonoTexStatic = mono.MonoTexStatic;
pub const MonoTexPrepared = mono.MonoTexPrepared;
pub const MonoFuncInput = mono.MonoFuncInput;
pub const MonoFuncStatic = mono.MonoFuncStatic;
pub const MonoFuncPrepared = mono.MonoFuncPrepared;
pub const MonoShaderSourceInput = mono.MonoShaderSourceInput;
pub const MonoShaderSourceStatic = mono.MonoShaderSourceStatic;
pub const MonoShaderTransformInput = mono.MonoShaderTransformInput;
pub const MonoShaderPipeInput = mono.MonoShaderPipeInput;
pub const MonoShaderPipeStatic = mono.MonoShaderPipeStatic;
pub const MonoShaderPipePrepared = mono.MonoShaderPipePrepared;

pub const RgbPipeState = rgb.RgbPipeState;
pub const RgbPipeStateSIMD = rgb.RgbPipeStateSIMD;
pub const RgbShaderStage = rgb.RgbShaderStage;
pub const RgbTerminal = rgb.RgbTerminal;
pub const LocalRgbPipeBuff = rgb.LocalRgbPipeBuff;
pub const RgbConstantInput = rgb.RgbConstantInput;
pub const RgbNodalInput = rgb.RgbNodalInput;
pub const RgbNodalStatic = rgb.RgbNodalStatic;
pub const RgbNodalPrepared = rgb.RgbNodalPrepared;
pub const RgbTexPayload = rgb.RgbTexPayload;
pub const RgbTexInput = rgb.RgbTexInput;
pub const RgbTexStatic = rgb.RgbTexStatic;
pub const RgbTexPrepared = rgb.RgbTexPrepared;
pub const RgbFuncInput = rgb.RgbFuncInput;
pub const RgbFuncStatic = rgb.RgbFuncStatic;
pub const RgbFuncPrepared = rgb.RgbFuncPrepared;
pub const RgbShaderSourceInput = rgb.RgbShaderSourceInput;
pub const RgbShaderSourceStatic = rgb.RgbShaderSourceStatic;
pub const RgbShaderTransformInput = rgb.RgbShaderTransformInput;
pub const RgbShaderPipeInput = rgb.RgbShaderPipeInput;
pub const RgbShaderPipeStatic = rgb.RgbShaderPipeStatic;
pub const RgbShaderPipePrepared = rgb.RgbShaderPipePrepared;

pub const MultiPipeState = multi.MultiPipeState;
pub const MultiPipeStateSIMD = multi.MultiPipeStateSIMD;
pub const MultiShaderStage = multi.MultiShaderStage;
pub const MultiTerminal = multi.MultiTerminal;
pub const LocalMultiPipeBuff = multi.LocalMultiPipeBuff;
pub const MultiNodalInput = multi.MultiNodalInput;
pub const MultiNodalStatic = multi.MultiNodalStatic;
pub const MultiNodalPrepared = multi.MultiNodalPrepared;
pub const MultiShaderPipeInput = multi.MultiShaderPipeInput;
pub const MultiShaderPipeStatic = multi.MultiShaderPipeStatic;
pub const MultiShaderPipePrepared = multi.MultiShaderPipePrepared;

pub const MeshShaderPipesInput = struct {
    mono: ?MonoShaderPipeInput = null,
    rgb: ?RgbShaderPipeInput = null,
    multi: ?MultiShaderPipeInput = null,
};

pub const MeshShaderPipesStatic = struct {
    mono: ?MonoShaderPipeStatic = null,
    rgb: ?RgbShaderPipeStatic = null,
    multi: ?MultiShaderPipeStatic = null,
};

pub const MeshShaderPipesPrepared = struct {
    mono: ?MonoShaderPipePrepared = null,
    rgb: ?RgbShaderPipePrepared = null,
    multi: ?MultiShaderPipePrepared = null,
};
