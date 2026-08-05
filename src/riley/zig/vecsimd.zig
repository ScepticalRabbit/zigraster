// --------------------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const print = std.debug.print;

const matstack = @import("matstack.zig");
const Mat44Ops = matstack.Mat44Ops;
const Mat44T = matstack.Mat44T;

const NDArray = @import("ndarray.zig").NDArray;

// --------------------------------------------------------------------------------------
// Public Entry-Point Func
// --------------------------------------------------------------------------------------

pub fn Vec3SIMD(comptime N: usize, comptime T: type) type {
    return struct {
        x: @Vector(N, T),
        y: @Vector(N, T),
        z: @Vector(N, T),

        const Self = @This();

        pub fn init(x: []const T, y: []const T, z: []const T) Self {
            return .{
                .x = x[0..N].*,
                .y = y[0..N].*,
                .z = z[0..N].*,
            };
        }

        pub fn initSplat(x_splat: T, y_splat: T, z_splat: T) Self {
            return .{
                .x = @splat(x_splat),
                .y = @splat(y_splat),
                .z = @splat(z_splat),
            };
        }

        pub fn add(self: Self, other: Self) Self {
            return .{
                .x = self.x + other.x,
                .y = self.y + other.y,
                .z = self.z + other.z,
            };
        }

        pub fn sub(self: Self, other: Self) Self {
            return .{
                .x = self.x - other.x,
                .y = self.y - other.y,
                .z = self.z - other.z,
            };
        }

        pub fn scale(self: Self, scale_val: T) Self {
            const scale_vec: @Vector(N, T) = @splat(scale_val);
            return .{
                .x = self.x * scale_vec,
                .y = self.y * scale_vec,
                .z = self.z * scale_vec,
            };
        }

        pub fn dot(self: Self, other: Self) @Vector(N, T) {
            return (self.x * other.x) + (self.y * other.y) + (self.z * other.z);
        }

        pub fn vecLenSquare(self: Self) @Vector(N, T) {
            return self.dot(self);
        }

        pub fn vecLen(self: Self) @Vector(N, T) {
            return @sqrt(self.dot(self));
        }

        pub fn cross(self: Self, other: Self) Self {
            return .{
                .x = (self.y * other.z) - (self.z * other.y),
                .y = (self.z * other.x) - (self.x * other.z),
                .z = (self.x * other.y) - (self.y * other.x),
            };
        }
    };
}

// NOTE: this is not a general func, it only works with NDArrays representing a
// mesh with dims=(elems_num,3,nodes_per_elem) where 3 is the coord[x,y,z].
// N should be nodes_per_elem.
pub fn loadElemVec3SIMD(
    comptime N: usize,
    comptime T: type,
    elem_array: *const NDArray(T),
    elem_idx: usize,
) !Vec3SIMD(N, T) {
    var start_slice: usize = elem_array.getFlatIdx(&[_]usize{ elem_idx, 0, 0 });
    // if coords then stride=3, if fields then stride=fields_num
    const stride: usize = elem_array.strides[1];

    const x_slice = elem_array.slice[start_slice .. start_slice + N];
    start_slice += stride;
    const y_slice = elem_array.slice[start_slice .. start_slice + N];
    start_slice += stride;
    const z_slice = elem_array.slice[start_slice .. start_slice + N];

    return Vec3SIMD(N, T).init(x_slice, y_slice, z_slice);
}

// NOTE: this is not a general func, it only works with NDArrays representing a
// mesh with dims=(elems_num,3,nodes_per_elem) where 3 is the coord[x,y,z].
// N should be nodes_per_elem.
pub fn saveElemVec3SIMD(
    comptime N: usize,
    comptime T: type,
    elem_array: *NDArray(T),
    elem_idx: usize,
    vec: Vec3SIMD(N, T),
) !void {
    var start_slice: usize = elem_array.getFlatIdx(&[_]usize{ elem_idx, 0, 0 });
    // if coords then stride=3, if fields then stride=fields_num
    const stride: usize = elem_array.strides[1];

    elem_array.slice[start_slice..][0..N].* = vec.x;
    start_slice += stride;
    elem_array.slice[start_slice..][0..N].* = vec.y;
    start_slice += stride;
    elem_array.slice[start_slice..][0..N].* = vec.z;
}

pub fn mat44Mul(
    comptime N: usize,
    comptime T: type,
    mat: Mat44T(T),
    vec: Vec3SIMD(N, T),
) Vec3SIMD(N, T) {
    var vec_res: Vec3SIMD(N, T) = undefined;

    var mat_row = Vec3SIMD(N, T).initSplat(mat.get(0, 0), mat.get(0, 1), mat.get(0, 2));
    vec_res.x = mat_row.dot(vec) + @as(@Vector(N, T), @splat(mat.get(0, 3)));

    mat_row = Vec3SIMD(N, T).initSplat(mat.get(1, 0), mat.get(1, 1), mat.get(1, 2));
    vec_res.y = mat_row.dot(vec) + @as(@Vector(N, T), @splat(mat.get(1, 3)));

    mat_row = Vec3SIMD(N, T).initSplat(mat.get(2, 0), mat.get(2, 1), mat.get(2, 2));
    vec_res.z = mat_row.dot(vec) + @as(@Vector(N, T), @splat(mat.get(2, 3)));

    return vec_res;
}

pub fn VecSIMD(comptime D: usize, comptime N: usize, comptime T: type) type {
    return struct {
        slice: [D]@Vector(N, T),

        const Self = @This();

        pub fn init(slices: [D][]const T) Self {
            var out: Self = undefined;
            inline for (0..D) |ii| {
                out.slice[ii] = slices[ii][0..N].*;
            }
            return out;
        }

        pub fn initSplat(splat_vals: [D]T) Self {
            var out: Self = undefined;
            inline for (0..D) |ii| {
                out.slice[ii] = @splat(splat_vals[ii]);
            }
            return out;
        }

        pub fn add(self: Self, other: Self) Self {
            var out: Self = undefined;
            inline for (0..D) |ii| {
                out.slice[ii] = self.slice[ii] + other.slice[ii];
            }
            return out;
        }

        pub fn sub(self: Self, other: Self) Self {
            var out: Self = undefined;
            inline for (0..D) |ii| {
                out.slice[ii] = self.slice[ii] - other.slice[ii];
            }
            return out;
        }

        pub fn scale(self: Self, scale_val: T) Self {
            var out: Self = undefined;
            const scale_vec: @Vector(N, T) = @splat(scale_val);
            inline for (0..D) |ii| {
                out.slice[ii] = scale_vec * self.slice[ii];
            }
            return out;
        }

        pub fn dot(self: Self, other: Self) @Vector(N, T) {
            var out: @Vector(N, T) = @splat(0);
            inline for (0..D) |ii| {
                out += self.slice[ii] * other.slice[ii];
            }
            return out;
        }

        pub fn vecLenSquare(self: Self) @Vector(N, T) {
            return self.dot(self);
        }

        pub fn vecLen(self: Self) @Vector(N, T) {
            return @sqrt(self.dot(self));
        }
    };
}

pub fn loadVecSIMDFromElemArray(
    comptime D: usize,
    comptime N: usize,
    comptime T: type,
    elem_array: *const NDArray(T),
    elem_idx: usize,
) !VecSIMD(D, N, T) {
    const stride = elem_array.strides[1];
    const flat_start = elem_array.getFlatIdx(&[_]usize{ elem_idx, 0, 0 });

    var out: VecSIMD(D, N, T) = undefined;
    inline for (0..D) |ii| {
        const start_slice = flat_start + (ii * stride);
        out.slice[ii] = elem_array.slice[start_slice .. start_slice + N].*;
    }
    return out;
}

pub fn saveVecSIMDToElemArray(
    comptime D: usize,
    comptime N: usize,
    comptime T: type,
    elem_array: *NDArray(T),
    elem_idx: usize,
    vec: VecSIMD(D, N, T),
) !void {
    const stride = elem_array.strides[1];
    const flat_start = elem_array.getFlatIdx(&[_]usize{ elem_idx, 0, 0 });

    inline for (0..D) |ii| {
        const start_slice = flat_start + (ii * stride);
        elem_array.slice[start_slice .. start_slice + N].* = vec.slice[ii];
    }
}
