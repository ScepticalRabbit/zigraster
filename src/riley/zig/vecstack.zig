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
const assert = std.debug.assert;
const buildconfig = @import("buildconfig.zig");
const F = buildconfig.F;

const SliceOps = @import("sliceops.zig");
const ValIdx = SliceOps.ValIdx;

const EType = buildconfig.F;
pub const Vec2f = Vec2T(EType);
pub const Vec3f = Vec3T(EType);

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub fn VecStack(comptime elem_n: comptime_int, comptime T: type) type {
    return struct {
        slice: [elem_n]T,

        const Self: type = @This();

        pub fn initFill(fill_val: T) Self {
            return .{ .slice = [_]T{fill_val} ** elem_n };
        }

        pub fn initOnes() Self {
            return initFill(1);
        }

        pub fn initZeros() Self {
            return initFill(0);
        }

        pub fn initSlice(slice_in: []const T) Self {
            return .{ .slice = slice_in[0..elem_n].* };
        }

        pub fn get(self: *const Self, ind: usize) T {
            return self.slice[ind];
        }

        pub fn set(self: *Self, ind: usize, val: T) void {
            self.slice[ind] = val;
        }

        pub fn x(self: Self) T {
            return self.slice[0];
        }

        pub fn y(self: Self) T {
            return self.slice[1];
        }

        pub fn z(self: Self) T {
            return self.slice[2];
        }

        pub fn w(self: Self) T {
            return self.slice[3];
        }

        pub fn add(self: *const Self, to_add: Self) Self {
            var vec_out: Self = undefined;
            for (0..elem_n) |ii| {
                vec_out.slice[ii] = self.slice[ii] + to_add.slice[ii];
            }
            return vec_out;
        }

        pub fn sub(self: *const Self, to_sub: Self) Self {
            var vec_out: Self = undefined;
            for (0..elem_n) |ii| {
                vec_out.slice[ii] = self.slice[ii] - to_sub.slice[ii];
            }
            return vec_out;
        }

        pub fn mulScal(self: *const Self, scal: T) Self {
            var vec_out: Self = undefined;
            for (0..elem_n) |ii| {
                vec_out.slice[ii] = scal * self.slice[ii];
            }
            return vec_out;
        }

        pub fn dot(self: *const Self, to_dot: Self) T {
            var dot_prod: T = 0;
            for (0..elem_n) |ii| {
                dot_prod += self.slice[ii] * to_dot.slice[ii];
            }
            return dot_prod;
        }

        pub fn norm(self: *const Self) T {
            var norm_out: T = 0;
            for (0..elem_n) |ii| {
                norm_out += self.slice[ii] * self.slice[ii];
            }
            return norm_out;
        }

        pub fn vecLen(self: *const Self) T {
            return @sqrt(self.norm());
        }

        pub fn max(self: *const Self) ValIdx(T) {
            return SliceOps.max(T, &self.slice);
        }

        pub fn min(self: *const Self) ValIdx(T) {
            return SliceOps.min(T, &self.slice);
        }

        pub fn sum(self: *const Self) T {
            return SliceOps.sum(T, &self.slice);
        }

        pub fn mean(self: *const Self) T {
            return SliceOps.mean(T, &self.slice);
        }

        pub fn apply(
            self: *const Self,
            comptime func: anytype,
        ) Self {
            var applied: Self = undefined;
            for (self.slice, 0..) |elem, ii| {
                applied.slice[ii] = func(elem);
            }
            return applied;
        }

        pub fn vecPrint(self: *const Self) void {
            print("[", .{});
            for (0..elem_n) |ii| {
                print("{e:.6},", .{self.slice[ii]});
            }
            print("]\n", .{});
        }
    };
}

pub fn Vec2T(comptime T: type) type {
    return VecStack(2, T);
}

pub fn Vec3T(comptime T: type) type {
    return VecStack(3, T);
}

pub fn initVec2(comptime T: type, x_in: T, y_in: T) Vec2T(T) {
    return Vec2T(T){
        .slice = [2]T{ x_in, y_in },
    };
}

pub fn initVec3(comptime T: type, x_in: T, y_in: T, z_in: T) Vec3T(T) {
    return Vec3T(T){
        .slice = [3]T{ x_in, y_in, z_in },
    };
}

pub const Vec3Ops = struct {
    pub fn cross(comptime T: type, vec0: Vec3T(T), vec1: Vec3T(T)) Vec3T(T) {
        var vec_out: Vec3T(T) = undefined;
        vec_out.slice[0] = vec0.slice[1] * vec1.slice[2] - vec0.slice[2] * vec1.slice[1];
        vec_out.slice[1] = vec0.slice[0] * vec1.slice[2] - vec0.slice[2] * vec1.slice[0];
        vec_out.slice[2] = vec0.slice[0] * vec1.slice[1] - vec0.slice[1] * vec1.slice[0];
        return vec_out;
    }
};

pub const Vec3SliceOps = struct {
    pub fn max(comptime T: type, vec: []Vec3T(T), ind: usize) T {
        assert(vec.len > 0);
        assert(ind < 3);

        var val: T = vec[0].get(ind);
        for (vec[1..]) |vv| {
            if (vv.get(ind) > val) {
                val = vv.get(ind);
            }
        }

        return val;
    }

    pub fn min(comptime T: type, vec: []Vec3T(T), ind: usize) T {
        assert(vec.len > 0);
        assert(ind < 3);

        var val: T = vec[0].get(ind);
        for (vec[1..]) |vv| {
            if (vv.get(ind) < val) {
                val = vv.get(ind);
            }
        }

        return val;
    }
};

// --------------------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------------------

const expectEqual = std.testing.expectEqual;
const expectEqualSlices = std.testing.expectEqualSlices;

test "VecSliceOps.max" {
    var vec_slice: [3]Vec3f = undefined;
    vec_slice[0] = initVec3(F, -1.0, 2.0, 7.0);
    vec_slice[1] = initVec3(F, 2.0, -2.0, 7.0);
    vec_slice[2] = initVec3(F, 5.0, -10.0, 0.0);

    const max_x: F = Vec3SliceOps.max(F, vec_slice[0..], 0);
    const max_y: F = Vec3SliceOps.max(F, vec_slice[0..], 1);
    const max_z: F = Vec3SliceOps.max(F, vec_slice[0..], 2);

    const exp_max_x: F = 5.0;
    const exp_max_y: F = 2.0;
    const exp_max_z: F = 7.0;

    try expectEqual(exp_max_x, max_x);
    try expectEqual(exp_max_y, max_y);
    try expectEqual(exp_max_z, max_z);
}

test "Vec.apply" {
    const vec_len: usize = 7;
    const vec0 = VecStack(vec_len, EType).initOnes();

    const vec_exp_ones = VecStack(vec_len, EType).initOnes();
    const vec_exp_zeros = VecStack(vec_len, EType).initZeros();

    const vec_sqrt = vec0.apply(std.math.sqrt);

    try expectEqualSlices(EType, &vec_exp_ones.slice, &vec_sqrt.slice);

    const vec1 = VecStack(vec_len, EType).initZeros();
    const vec_atan = vec1.apply(std.math.atan);

    try expectEqualSlices(EType, &vec_exp_zeros.slice, &vec_atan.slice);

    const vec_e = vec1.apply(SliceOps.exp);

    try expectEqualSlices(EType, &vec_exp_ones.slice, &vec_e.slice);
}

test "Vec.max" {
    const v0 = [_]EType{ 1, 3, 6, 7, 8, 1, -2, -3, 0, 5 };
    const vec0 = VecStack(v0.len, EType).initSlice(&v0);

    const exp_val = ValIdx(EType){
        .val = 8,
        .idx = 4,
    };

    try expectEqual(exp_val, vec0.max());
}

test "Vec.min" {
    const v0 = [_]EType{ 1, 3, 6, 7, 8, 1, -2, -3, 0, 5 };
    const vec0 = VecStack(v0.len, EType).initSlice(&v0);

    const exp_val = ValIdx(EType){
        .val = -3,
        .idx = 7,
    };

    try expectEqual(exp_val, vec0.min());
}

test "Vec.sum" {
    const v0 = [_]EType{ 1, 3, 6, 7, 8, 1, -2, -3, 0, 5 };
    const vec0 = VecStack(v0.len, EType).initSlice(&v0);

    const exp_val: EType = 26;

    try expectEqual(exp_val, vec0.sum());
}

test "Vec.mean" {
    const v0 = [_]EType{ 1, 3, 6, 7, 8, 1, -2, -3, 0, 5 };
    const vec0 = VecStack(v0.len, EType).initSlice(&v0);

    const exp_val: EType = 2.6;

    try expectEqual(exp_val, vec0.mean());
}

test "Vec3f.add" {
    var vec0 = Vec3f.initOnes();
    const vec1 = Vec3f.initFill(2);
    const vec_exp = Vec3f.initFill(3);

    try expectEqualSlices(EType, &vec0.add(vec1).slice, &vec_exp.slice);
}

test "Vec3f.sub" {
    var vec0 = Vec3f.initOnes();
    const vec1 = Vec3f.initFill(7);
    const vec_exp = Vec3f.initFill(-6);

    try expectEqualSlices(EType, &vec0.sub(vec1).slice, &vec_exp.slice);
}

test "Vec3f.mulScal" {
    var vec0 = Vec3f.initOnes();
    const scal: EType = 1.23;
    const vec_exp = Vec3f.initFill(scal);

    try expectEqualSlices(EType, &vec0.mulScal(scal).slice, &vec_exp.slice);
}

test "Vec3f.dot" {
    const fill: EType = 7.0;
    var vec0 = Vec3f.initFill(fill);
    var vec1 = Vec3f.initFill(fill);
    const dot_exp: EType = 3 * fill * fill;

    try expectEqual(vec0.dot(vec1), dot_exp);
    try expectEqual(vec1.dot(vec0), dot_exp);
}

test "Vec3f.norm" {
    const fill: EType = 2.0;
    var vec0 = Vec3f.initFill(fill);
    const norm_exp: EType = 3 * fill * fill;

    try expectEqual(vec0.norm(), norm_exp);
}

test "Vec3f.length" {
    const arr = [_]EType{ 2.0, -1.0, 3.0 };
    const leng_exp = @sqrt(arr[0] * arr[0] + arr[1] * arr[1] + arr[2] * arr[2]);
    var vec = Vec3f.initSlice(&arr);

    try expectEqual(vec.vecLen(), leng_exp);
}

test "Vec3Ops.cross" {
    const v0 = [_]EType{ 1.0, 0.0, 0.0 };
    const vec0 = Vec3f.initSlice(&v0);
    const v1 = [_]EType{ 0.0, 1.0, 0.0 };
    const vec1 = Vec3f.initSlice(&v1);
    const v2 = [_]EType{ 0.0, 0.0, 1.0 };
    const cross_exp = Vec3f.initSlice(&v2);

    try expectEqual(Vec3Ops.cross(F, vec0, vec1), cross_exp);
}
