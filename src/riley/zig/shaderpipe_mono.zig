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

pub const MonoPipeState = struct {
    value: F,
};

pub const MonoPipeStateSIMD = struct {
    value: VecSF,
};

pub const MonoTerminal = struct {
    pub inline fn eval(
        self: *const MonoTerminal,
        state: *const MonoPipeState,
    ) F {
        _ = self;
        return state.value;
    }

    pub inline fn evalSIMD(
        self: *const MonoTerminal,
        state: *const MonoPipeStateSIMD,
    ) VecSF {
        _ = self;
        return state.value;
    }
};

pub fn LocalMonoPipeBuff(comptime N: usize) type {
    return struct {
        nodal_data: [N]F = undefined,
        uv_data: [2 * N]F = undefined,
        func_coords: [3 * N]F = undefined,
        normals: [3 * N]F = undefined,
        actual_func_coords: u8 = 0,

        const Self = @This();

        pub inline fn loadNodal(
            self: *Self,
            array: ndarray.NDArray(F),
            start_idx: usize,
        ) void {
            @memcpy(self.nodal_data[0..N], array.slice[start_idx .. start_idx + N]);
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
            weights: [N]F,
        ) F {
            var sum: F = 0.0;
            inline for (0..N) |nn| {
                sum += weights[nn] * self.nodal_data[nn];
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

pub const ScaleStage = struct {
    factor: F,
};

pub const OffsetStage = struct {
    offset: F,
};

pub const LinearMapStage = struct {
    scale_mul: F,
    scale_add: F,
};

pub const MapRangeStage = struct {
    in_min: F,
    in_max: F,
    out_min: F,
    out_max: F,
};

pub const ClampStage = struct {
    min: F,
    max: F,
};

pub const InvertStage = struct {
    max_val: F = 1.0,
};

pub const LutStage = struct {
    lut: []const F,
};

pub const MonoTexPayload = union(enum) {
    u8: texops.Tex(u8, 1),
    u16: texops.Tex(u16, 1),
    f: texops.Tex(F, 1),
};

pub const MonoNodalInput = struct {
    field: meshio.Field,
    bits: ?u8 = 8,
    scaling: imageops.ScaleStrategy = .none,
    scale_over: ScaleOver = .over_frames,
    normal_type: NormalType = .none,
};

pub const MonoNodalStatic = struct {
    field: meshio.Field,
    bits: ?u8 = 8,
    scaling: imageops.ScaleStrategy = .none,
    scale_over: ScaleOver = .over_frames,
    normal_type: NormalType = .none,
};

pub const MonoNodalPrepared = struct {
    elem_field: ndarray.NDArray(F),
    bits: ?u8 = 8,
    scaling: imageops.ScaleStrategy = .none,
    scale_over: ScaleOver = .over_frames,
    scale_mul: F = 1.0,
    scale_add: F = 0.0,
    normal_type: NormalType = .none,
    elem_normals: ?ndarray.MappedNDArray(F) = null,
};

pub const MonoTexInput = struct {
    uvs: ndarray.NDArray(F),
    tex: MonoTexPayload,
    samp_cfg: texops.TexSampConfig = .{
        .sample = .cubic_catmull_rom,
        .mode = .lut_lerp,
    },
    bits: ?u8 = 8,
    scaling: imageops.ScaleStrategy = .none,
    normal_type: NormalType = .none,
};

pub const MonoTexStatic = struct {
    elem_uvs: ndarray.NDArray(F),
    tex: MonoTexPayload,
    samp_cfg: texops.TexSampConfig = .{
        .sample = .cubic_catmull_rom,
        .mode = .lut_lerp,
    },
    bits: ?u8 = 8,
    scaling: imageops.ScaleStrategy = .none,
    normal_type: NormalType = .none,
};

pub const MonoTexPrepared = struct {
    elem_uvs: ndarray.NDArray(F),
    tex: MonoTexPayload,
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

pub const MonoFuncInput = struct {
    uvs: ?ndarray.NDArray(F) = null,
    coord_mode: FuncCoordMode = .para,
    builtin: FuncShaderBuiltin,
    params: FuncShaderParams = .{},
    bits: ?u8 = 8,
    scaling: imageops.ScaleStrategy = .none,
    normal_type: NormalType = .none,
};

pub const MonoFuncStatic = struct {
    elem_uvs: ?ndarray.NDArray(F),
    coord_mode: FuncCoordMode = .para,
    builtin: FuncShaderBuiltin,
    params: FuncShaderParams = .{},
    bits: ?u8 = 8,
    scaling: imageops.ScaleStrategy = .none,
    normal_type: NormalType = .none,
};

pub const MonoFuncPrepared = struct {
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

pub const MonoShaderStage = union(enum) {
    nodal: MonoNodalPrepared,
    texture: MonoTexPrepared,
    function: MonoFuncPrepared,
    scale: ScaleStage,
    offset: OffsetStage,
    linear_map: LinearMapStage,
    map_range: MapRangeStage,
    clamp: ClampStage,
    invert: InvertStage,
    lut: LutStage,
};

pub const MonoShaderSourceInput = union(enum) {
    nodal: MonoNodalInput,
    tex_u8: shaderops.TexInput(u8, 1),
    tex_u16: shaderops.TexInput(u16, 1),
    tex_f: shaderops.TexInput(F, 1),
    tex: MonoTexInput,
    func: MonoFuncInput,
};

pub const MonoShaderTransformInput = union(enum) {
    scale: ScaleStage,
    offset: OffsetStage,
    linear_map: LinearMapStage,
    map_range: MapRangeStage,
    clamp: ClampStage,
    invert: InvertStage,
    lut: LutStage,
};

pub const MonoShaderPipeInput = struct {
    source: MonoShaderSourceInput,
    transforms: []const MonoShaderTransformInput = &.{},
    terminal: MonoTerminal = .{},
};

pub const MonoShaderSourceStatic = union(enum) {
    nodal: MonoNodalStatic,
    tex: MonoTexStatic,
    func: MonoFuncStatic,
};

pub const MonoShaderPipeStatic = struct {
    source: MonoShaderSourceStatic,
    transforms: []const MonoShaderTransformInput = &.{},
    terminal: MonoTerminal = .{},
};

pub const MonoShaderPipePrepared = struct {
    stages: []const MonoShaderStage,
    terminal: MonoTerminal = .{},
};
