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
const print = std.debug.print;
const assert = std.debug.assert;

const SliceOps = @import("sliceops.zig");
const ValIdx = SliceOps.ValIdx;

const TestType = F;

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub fn VecSlice(comptime T: type) type {
    return struct {
        slice: []T,

        const Self: type = @This();

        pub fn init(slice: []T) Self {
            return .{
                .slice = slice,
            };
        }

        pub fn fill(self: *const Self, fill_val: T) void {
            @memset(self.slice, fill_val);
        }

        pub fn get(self: *const Self, ind: usize) T {
            return self.slice[ind];
        }

        pub fn set(self: *const Self, ind: usize, val: T) void {
            self.slice[ind] = val;
        }

        pub fn addInPlace(self: *const Self, to_add: *const Self) void {
            for (0..self.slice.len) |ii| {
                self.slice[ii] += to_add.slice[ii];
            }
        }

        pub fn subInPlace(self: *const Self, to_sub: *const Self) void {
            for (0..self.slice.len) |ii| {
                self.slice[ii] -= to_sub.slice[ii];
            }
        }

        pub fn mulInPlace(self: *const Self, to_mul: *const Self) void {
            for (0..self.slice.len) |ii| {
                self.slice[ii] *= to_mul.slice[ii];
            }
        }

        pub fn divInPlace(self: *const Self, to_div: *const Self) void {
            for (0..self.slice.len) |ii| {
                self.slice[ii] /= to_div.slice[ii];
            }
        }

        pub fn mulScalInPlace(self: *const Self, scal: T) void {
            for (0..self.slice.len) |ii| {
                self.slice[ii] = scal * self.slice[ii];
            }
        }

        pub fn applyInPlace(
            self: *const Self,
            comptime func: anytype,
        ) void {
            for (self.slice, 0..) |elem, ii| {
                self.slice[ii] = func(elem);
            }
        }

        pub fn dot(self: *const Self, to_dot: Self) T {
            var dot_prod: T = 0;
            for (0..self.slice.len) |ii| {
                dot_prod += self.slice[ii] * to_dot.slice[ii];
            }
            return dot_prod;
        }

        pub fn norm(self: *const Self) T {
            var norm_out: T = 0;
            for (0..self.slice.len) |ii| {
                norm_out += self.slice[ii] * self.slice[ii];
            }
            return norm_out;
        }

        pub fn vecLen(self: *const Self) T {
            return @sqrt(self.norm());
        }

        pub fn max(self: *const Self) ValIdx(T) {
            return SliceOps.max(T, self.slice);
        }

        pub fn min(self: *const Self) ValIdx(T) {
            return SliceOps.min(T, self.slice);
        }

        pub fn sum(self: *const Self) T {
            return SliceOps.sum(T, self.slice);
        }

        pub fn mean(self: *const Self) T {
            return SliceOps.mean(T, self.slice);
        }

        pub fn vecPrint(self: *const Self) void {
            print("[", .{});
            for (0..self.slice.len) |ii| {
                print("{e:.3},", .{self.slice[ii]});
            }
            print("]\n", .{});
        }
    };
}

// --------------------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------------------

const testing = std.testing;
const expectEqual = std.testing.expectEqual;
const expectApproxEqAbs = testing.expectApproxEqAbs;
const expectEqualSlices = testing.expectEqualSlices;

test "VecSlice.addInPlace" {
    const vec_len: usize = 10;

    var v0 = [_]F{0.0} ** vec_len;
    const vec0 = VecSlice(TestType).init(v0[0..]);

    var v1 = [_]F{0.0} ** vec_len;
    const vec1 = VecSlice(TestType).init(v1[0..]);

    var ve = [_]F{0.0} ** vec_len;
    const vec_exp = VecSlice(TestType).init(ve[0..]);

    vec0.fill(1.0);
    vec1.fill(1.0);
    vec_exp.fill(2.0);

    vec0.addInPlace(&vec1);

    try expectEqualSlices(TestType, vec_exp.slice, vec0.slice);
}

test "VecSlice.subInPlace" {
    const vec_len: usize = 10;

    var v0 = [_]F{0.0} ** vec_len;
    const vec0 = VecSlice(TestType).init(v0[0..]);

    var v1 = [_]F{0.0} ** vec_len;
    const vec1 = VecSlice(TestType).init(v1[0..]);

    var ve = [_]F{0.0} ** vec_len;
    const vec_exp = VecSlice(TestType).init(ve[0..]);

    vec0.fill(1.0);
    vec1.fill(1.0);
    vec_exp.fill(0.0);

    vec0.subInPlace(&vec1);

    try expectEqualSlices(TestType, vec_exp.slice, vec0.slice);
}

test "VecSlice.mulInPlace" {
    const vec_len: usize = 10;

    var v0 = [_]F{0.0} ** vec_len;
    const vec0 = VecSlice(TestType).init(v0[0..]);

    var v1 = [_]F{0.0} ** vec_len;
    const vec1 = VecSlice(TestType).init(v1[0..]);

    var ve = [_]F{0.0} ** vec_len;
    const vec_exp = VecSlice(TestType).init(ve[0..]);

    vec0.fill(1.0);
    vec1.fill(1.0);
    vec_exp.fill(1.0);

    vec0.mulInPlace(&vec1);

    try expectEqualSlices(TestType, vec_exp.slice, vec0.slice);
}

test "VecSlice.divInPlace" {
    const vec_len: usize = 10;

    var v0 = [_]F{0.0} ** vec_len;
    const vec0 = VecSlice(TestType).init(v0[0..]);

    var v1 = [_]F{0.0} ** vec_len;
    const vec1 = VecSlice(TestType).init(v1[0..]);

    var ve = [_]F{0.0} ** vec_len;
    const vec_exp = VecSlice(TestType).init(ve[0..]);

    vec0.fill(1.0);
    vec1.fill(1.0);
    vec_exp.fill(1.0);

    vec0.divInPlace(&vec1);

    try expectEqualSlices(TestType, vec_exp.slice, vec0.slice);
}

test "VecSlice.mulScalInPlace" {
    const vec_len: usize = 10;

    var v0 = [_]F{0.0} ** vec_len;
    const vec0 = VecSlice(TestType).init(v0[0..]);

    var ve = [_]F{0.0} ** vec_len;
    const vec_exp = VecSlice(TestType).init(ve[0..]);

    vec0.fill(1.0);
    vec_exp.fill(2.0);

    const scal: TestType = 2.0;

    vec0.mulScalInPlace(scal);

    try expectEqualSlices(TestType, vec_exp.slice, vec0.slice);
}

test "VecSlice.max" {
    var v0 = [_]F{0.0} ** 10;
    const vec0 = VecSlice(TestType).init(v0[0..]);

    const exp_idx: usize = 4;
    const exp_val: F = 8.0;

    vec0.fill(0.0);
    vec0.set(exp_idx, exp_val);

    const exp_val_idx = ValIdx(TestType){
        .val = exp_val,
        .idx = exp_idx,
    };

    try expectEqual(exp_val_idx, vec0.max());
}

test "VecSlice.min" {
    var v0 = [_]F{0.0} ** 10;
    const vec0 = VecSlice(TestType).init(v0[0..]);

    const exp_idx: usize = 7;
    const exp_val: F = -3.0;

    vec0.fill(0.0);
    vec0.set(exp_idx, exp_val);

    const exp_val_idx = ValIdx(TestType){
        .val = exp_val,
        .idx = exp_idx,
    };

    try expectEqual(exp_val_idx, vec0.min());
}

test "VecSlice.sum" {
    var v0 = [_]F{0.0} ** 12;
    const vec0 = VecSlice(TestType).init(v0[0..]);

    vec0.fill(1.0);
    vec0.set(0, 0.0);

    const exp_val: TestType = 11;

    try expectEqual(exp_val, vec0.sum());
}

test "VecSlice.mean" {
    var v0 = [_]F{0.0} ** 10;
    const vec0 = VecSlice(TestType).init(v0[0..]);

    vec0.fill(1.0);
    vec0.set(2, 0.0);
    vec0.set(7, 0.0);

    const exp_val: TestType = 0.8;

    try expectEqual(exp_val, vec0.mean());
}

test "VecSlice.apply" {
    const vec_len: usize = 7;

    var v0 = [_]F{0.0} ** vec_len;
    var vec0 = VecSlice(TestType).init(v0[0..]);

    var ve1 = [_]F{1.0} ** vec_len;
    const vec_exp_ones = VecSlice(TestType).init(ve1[0..]);

    var ve0 = [_]F{0.0} ** vec_len;
    const vec_exp_zeros = VecSlice(TestType).init(ve0[0..]);

    vec0.fill(1.0);
    vec_exp_ones.fill(1.0);
    vec_exp_zeros.fill(0.0);

    vec0.applyInPlace(std.math.sqrt);

    try expectEqualSlices(TestType, vec_exp_ones.slice, vec0.slice);

    var v1 = [_]F{0.0} ** vec_len;
    var vec1 = VecSlice(TestType).init(v1[0..]);

    vec1.fill(0.0);
    vec1.applyInPlace(std.math.atan);

    try expectEqualSlices(TestType, vec_exp_zeros.slice, vec1.slice);

    vec1.applyInPlace(SliceOps.exp);

    try expectEqualSlices(TestType, vec_exp_ones.slice, vec1.slice);
}
