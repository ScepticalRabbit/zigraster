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
const meshio = @import("meshio.zig");
const shaderops = @import("shaderops.zig");

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const ScaleOver = shaderops.ScaleOver;
pub const NormalType = shaderops.NormalType;

pub fn MultiPipeState(comptime C: usize) type {
    return struct {
        value: [C]F,
    };
}

pub fn MultiPipeStateSIMD(comptime C: usize) type {
    return struct {
        channels: [C]VecSF,
    };
}

pub fn MultiTerminal(comptime C: usize) type {
    return struct {
        pub inline fn eval(
            self: *const @This(),
            state: *const MultiPipeState(C),
        ) [C]F {
            _ = self;
            return state.value;
        }

        pub inline fn evalSIMD(
            self: *const @This(),
            state: *const MultiPipeStateSIMD(C),
        ) [C]VecSF {
            _ = self;
            return state.channels;
        }
    };
}

pub fn LocalMultiPipeBuff(comptime N: usize, comptime C: usize) type {
    return struct {
        nodal_data: [C * N]F = undefined,
        normals: [3 * N]F = undefined,
        actual_fields: u8 = 0,

        const Self = @This();

        pub inline fn loadNodal(
            self: *Self,
            array: ndarray.NDArray(F),
            start_idx: usize,
            fields_num: u8,
        ) void {
            std.debug.assert(fields_num <= C);
            self.actual_fields = fields_num;
            const count = @as(usize, fields_num) * N;
            @memcpy(self.nodal_data[0..count], array.slice[start_idx .. start_idx + count]);
        }

        pub inline fn loadNormals(
            self: *Self,
            array: ndarray.NDArray(F),
            start_idx: usize,
        ) void {
            const count = 3 * N;
            @memcpy(self.normals[0..count], array.slice[start_idx .. start_idx + count]);
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
    };
}

pub const MultiNodalInput = struct {
    field: meshio.Field,
    bits: ?u8 = 8,
    scaling: imageops.ScaleStrategy = .none,
    scale_over: ScaleOver = .over_frames,
    normal_type: NormalType = .none,
};

pub const MultiNodalStatic = struct {
    field: meshio.Field,
    bits: ?u8 = 8,
    scaling: imageops.ScaleStrategy = .none,
    scale_over: ScaleOver = .over_frames,
    normal_type: NormalType = .none,
};

pub const MultiNodalPrepared = struct {
    elem_field: ndarray.NDArray(F),
    bits: ?u8 = 8,
    scaling: imageops.ScaleStrategy = .none,
    scale_over: ScaleOver = .over_frames,
    scale_mul: F = 1.0,
    scale_add: F = 0.0,
    normal_type: NormalType = .none,
    elem_normals: ?ndarray.MappedNDArray(F) = null,
};

pub fn MultiShaderStage(comptime C: usize) type {
    _ = C;
    return union(enum) {
        nodal: MultiNodalPrepared,
    };
}

pub const MultiShaderPipeInput = struct {
    nodal: MultiNodalInput,
};

pub const MultiShaderPipeStatic = struct {
    nodal: MultiNodalStatic,
};

pub const MultiShaderPipePrepared = struct {
    nodal: MultiNodalPrepared,
};
