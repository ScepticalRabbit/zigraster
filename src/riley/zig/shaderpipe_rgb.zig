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
const cfg = buildconfig.config;
const F = buildconfig.F;
const S = buildconfig.SimdWidth;
const VecSF = buildconfig.VecSF;
const VecSB = buildconfig.VecSB;

const ndarray = @import("ndarray.zig");
const imageops = @import("imageops.zig");
const texops = @import("textureops.zig");
const meshio = @import("meshio.zig");
const shaderops = @import("shaderops.zig");

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const ScaleOver = shaderops.ScaleOver;
pub const NormalType = shaderops.NormalType;
pub const FuncCoordMode = shaderops.FuncCoordMode;
pub const FuncShaderBuiltin = shaderops.FuncShaderBuiltin;
pub const FuncShaderParams = shaderops.FuncShaderParams;

pub const RgbPipeState = struct {
    value: [3]F,
};

pub const RgbPipeStateSIMD = struct {
    r: VecSF,
    g: VecSF,
    b: VecSF,
};

pub const RgbTerminal = struct {
    pub inline fn eval(
        self: *const RgbTerminal,
        state: *const RgbPipeState,
    ) [3]F {
        _ = self;
        return state.value;
    }

    pub inline fn evalSIMD(
        self: *const RgbTerminal,
        state: *const RgbPipeStateSIMD,
    ) [3]VecSF {
        _ = self;
        return .{ state.r, state.g, state.b };
    }
};

pub fn LocalRgbPipeBuff(comptime N: usize) type {
    return struct {
        nodal_data: [3 * N]F = undefined,
        uv_data: [2 * N]F = undefined,
        func_coords: [3 * N]F = undefined,
        normals: [3 * N]F = undefined,
        actual_fields: u8 = 0,
        actual_func_coords: u8 = 0,

        const Self = @This();

        pub inline fn loadNodal(
            self: *Self,
            array: ndarray.NDArray(F),
            start_idx: usize,
            fields_num: u8,
        ) void {
            std.debug.assert(fields_num <= 3);
            self.actual_fields = fields_num;
            const count = @as(usize, fields_num) * N;
            @memcpy(self.nodal_data[0..count], array.slice[start_idx .. start_idx + count]);
        }

        pub inline fn loadUVs(
            self: *Self,
            array: ndarray.NDArray(F),
            start_idx: usize,
        ) void {
            @memcpy(self.uv_data[0 .. 2 * N], array.slice[start_idx .. start_idx + 2 * N]);
        }

        pub inline fn loadNormals(
            self: *Self,
            array: ndarray.NDArray(F),
            start_idx: usize,
        ) void {
            const count = 3 * N;
            @memcpy(self.normals[0..count], array.slice[start_idx .. start_idx + count]);
        }

        pub inline fn loadFuncCoords(
            self: *Self,
            array: ndarray.NDArray(F),
            start_idx: usize,
            coords_num: u8,
        ) void {
            std.debug.assert(coords_num <= 3);
            self.actual_func_coords = coords_num;
            const count = @as(usize, coords_num) * N;
            @memcpy(
                self.func_coords[0..count],
                array.slice[start_idx .. start_idx + count],
            );
        }

        pub inline fn interpNodal(
            self: *const Self,
            field_idx: usize,
            weights: [N]F,
        ) F {
            const base = field_idx * N;
            var sum: F = 0.0;
            inline for (0..N) |nn| {
                sum += weights[nn] * self.nodal_data[base + nn];
            }
            return sum;
        }

        pub inline fn interpNormal(
            self: *const Self,
            weights: [N]F,
        ) [3]F {
            var norm = [3]F{ 0.0, 0.0, 0.0 };
            inline for (0..N) |nn| {
                norm[0] += weights[nn] * self.normals[0 * N + nn];
                norm[1] += weights[nn] * self.normals[1 * N + nn];
                norm[2] += weights[nn] * self.normals[2 * N + nn];
            }
            return norm;
        }

        pub inline fn interpFuncCoord(
            self: *const Self,
            coord_idx: usize,
            weights: [N]F,
        ) F {
            const base = coord_idx * N;
            var sum: F = 0.0;
            inline for (0..N) |nn| {
                sum += weights[nn] * self.func_coords[base + nn];
            }
            return sum;
        }
    };
}

pub const ScaleRgbStage = struct {
    factor: [3]F,
};

pub const OffsetRgbStage = struct {
    offset: [3]F,
};

pub const LinearMapRgbStage = struct {
    scale_mul: [3]F,
    scale_add: [3]F,
};

pub const ClampRgbStage = struct {
    min: [3]F,
    max: [3]F,
};

pub const LutRgbStage = struct {
    lut_r: []const F,
    lut_g: []const F,
    lut_b: []const F,
};

pub const RgbTexPayload = union(enum) {
    u8: texops.Tex(u8, 3),
    u16: texops.Tex(u16, 3),
    f: texops.Tex(F, 3),
};

pub const RgbConstantInput = struct {
    value: [3]F = .{ 0.5, 0.5, 0.5 },
};

pub const RgbConstantPrepared = struct {
    value: [3]F,
};

pub const RgbNodalInput = struct {
    field: meshio.Field,
    bits: ?u8 = 8,
    scaling: imageops.ScaleStrategy = .none,
    scale_over: ScaleOver = .over_frames,
    normal_type: NormalType = .none,
};

pub const RgbNodalStatic = struct {
    field: meshio.Field,
    bits: ?u8 = 8,
    scaling: imageops.ScaleStrategy = .none,
    scale_over: ScaleOver = .over_frames,
    normal_type: NormalType = .none,
};

pub const RgbNodalPrepared = struct {
    elem_field: ndarray.NDArray(F),
    bits: ?u8 = 8,
    scaling: imageops.ScaleStrategy = .none,
    scale_over: ScaleOver = .over_frames,
    scale_mul: F = 1.0,
    scale_add: F = 0.0,
    normal_type: NormalType = .none,
    elem_normals: ?ndarray.MappedNDArray(F) = null,
};

pub const RgbTexInput = struct {
    uvs: ndarray.NDArray(F),
    tex: RgbTexPayload,
    samp_cfg: texops.TexSampConfig = .{
        .sample = .cubic_catmull_rom,
        .mode = .lut_lerp,
    },
    bits: ?u8 = 8,
    scaling: imageops.ScaleStrategy = .none,
    normal_type: NormalType = .none,
};

pub const RgbTexStatic = struct {
    elem_uvs: ndarray.NDArray(F),
    tex: RgbTexPayload,
    samp_cfg: texops.TexSampConfig = .{
        .sample = .cubic_catmull_rom,
        .mode = .lut_lerp,
    },
    bits: ?u8 = 8,
    scaling: imageops.ScaleStrategy = .none,
    normal_type: NormalType = .none,
};

pub const RgbTexPrepared = struct {
    elem_uvs: ndarray.NDArray(F),
    tex: RgbTexPayload,
    samp_cfg: texops.TexSampConfig = .{
        .sample = .cubic_catmull_rom,
        .mode = .lut_lerp,
    },
    bits: ?u8 = 8,
    scaling: imageops.ScaleStrategy = .none,
    scale_mul: F = 1.0,
    scale_add: F = 0.0,
    normal_type: NormalType = .none,
    elem_normals: ?ndarray.MappedNDArray(F) = null,
};

pub const RgbFuncInput = struct {
    uvs: ?ndarray.NDArray(F) = null,
    coord_mode: FuncCoordMode = .para,
    builtin: FuncShaderBuiltin,
    params: FuncShaderParams = .{},
    bits: ?u8 = 8,
    scaling: imageops.ScaleStrategy = .none,
    normal_type: NormalType = .none,
};

pub const RgbFuncStatic = struct {
    elem_uvs: ?ndarray.NDArray(F),
    coord_mode: FuncCoordMode = .para,
    builtin: FuncShaderBuiltin,
    params: FuncShaderParams = .{},
    bits: ?u8 = 8,
    scaling: imageops.ScaleStrategy = .none,
    normal_type: NormalType = .none,
};

pub const RgbFuncPrepared = struct {
    elem_uvs: ?ndarray.NDArray(F),
    elem_world_ref: ?ndarray.NDArray(F) = null,
    elem_world_def: ?ndarray.NDArray(F) = null,
    coord_mode: FuncCoordMode = .para,
    builtin: FuncShaderBuiltin,
    params: FuncShaderParams = .{},
    bits: ?u8 = 8,
    scaling: imageops.ScaleStrategy = .none,
    scale_mul: F = 1.0,
    scale_add: F = 0.0,
    normal_type: NormalType = .none,
    elem_normals: ?ndarray.MappedNDArray(F) = null,
};

pub const RgbShaderStage = union(enum) {
    constant: RgbConstantPrepared,
    nodal: RgbNodalPrepared,
    texture: RgbTexPrepared,
    function: RgbFuncPrepared,
    scale: ScaleRgbStage,
    offset: OffsetRgbStage,
    clamp: ClampRgbStage,
    linear_map: LinearMapRgbStage,
    lut: LutRgbStage,
};

pub const RgbShaderSourceInput = union(enum) {
    constant: RgbConstantInput,
    nodal: RgbNodalInput,
    tex_rgb_u8: shaderops.TexInput(u8, 3),
    tex_rgb_u16: shaderops.TexInput(u16, 3),
    tex_rgb_f: shaderops.TexInput(F, 3),
    tex: RgbTexInput,
    func_rgb: RgbFuncInput,
    func: RgbFuncInput,
};

pub const RgbShaderTransformInput = union(enum) {
    scale: ScaleRgbStage,
    offset: OffsetRgbStage,
    clamp: ClampRgbStage,
    linear_map: LinearMapRgbStage,
    lut: LutRgbStage,
};

pub const RgbShaderPipeInput = struct {
    source: RgbShaderSourceInput,
    transforms: []const RgbShaderTransformInput = &.{},
    terminal: RgbTerminal = .{},
};

pub const RgbShaderSourceStatic = union(enum) {
    constant: RgbConstantInput,
    nodal: RgbNodalStatic,
    tex: RgbTexStatic,
    func: RgbFuncStatic,
};

pub const RgbShaderPipeStatic = struct {
    source: RgbShaderSourceStatic,
    transforms: []const RgbShaderTransformInput = &.{},
    terminal: RgbTerminal = .{},
};

pub const RgbShaderPipePrepared = struct {
    stages: []const RgbShaderStage,
    terminal: RgbTerminal = .{},
};
