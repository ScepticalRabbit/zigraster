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

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub fn ValIdx(ValType: type) type {
    return struct {
        val: ValType,
        idx: usize,
    };
}

pub fn max(comptime T: type, slice: []const T) ValIdx(T) {
    assert(slice.len > 0);

    var val_idx = ValIdx(T){
        .val = slice[0],
        .idx = 0,
    };

    for (slice[1..], 1..) |elem, ii| {
        if (elem > val_idx.val) {
            val_idx.idx = ii;
            val_idx.val = elem;
        }
    }

    return val_idx;
}

pub fn min(comptime T: type, slice: []const T) ValIdx(T) {
    assert(slice.len > 0);

    var val_idx = ValIdx(T){
        .val = slice[0],
        .idx = 0,
    };

    for (slice[1..], 1..) |elem, ii| {
        if (elem < val_idx.val) {
            val_idx.idx = ii;
            val_idx.val = elem;
        }
    }

    return val_idx;
}

pub fn sum(comptime T: type, slice: []const T) T {
    assert(slice.len > 0);

    var sum_out: T = 0;
    for (slice[0..]) |elem| {
        sum_out += elem;
    }
    return sum_out;
}

pub fn mean(comptime T: type, slice: []const T) T {
    return sum(T, slice) / @as(T, @floatFromInt(slice.len));
}

// Removing inline from the stdlib version for use with 'apply'
pub fn exp(value: anytype) @TypeOf(value) {
    return @exp(value);
}

// Based on copy forwards in std.mem
pub fn apply(
    comptime T: type,
    dest: []T,
    source: []const T,
    comptime func: anytype,
) void {
    for (dest[0..source.len], source) |*dd, ss| {
        dd.* = func(ss);
    }
}

pub fn rangeLen(start: F, stop: F, step: F) usize {
    const range: F = @ceil((stop - start) / step);
    const range_length: usize = @as(usize, @intFromFloat(range));
    return range_length;
}

pub fn dot(comptime T: type, slice0: []const T, slice1: []const T) T {
    assert(slice0.len == slice1.len);

    var dot_prod: T = 0;
    for (0..slice0.len) |ii| {
        dot_prod += slice0[ii] * slice1[ii];
    }
    return dot_prod;
}

pub fn norm(comptime T: type, vec: []const T) T {
    var norm_out: T = 0;

    for (0..vec.len) |ii| {
        norm_out += vec[ii] * vec[ii];
    }

    return norm_out;
}

pub fn vecLen(comptime T: type, vec: []const T) T {
    return @sqrt(norm(T, vec));
}

pub fn add(comptime T: type, vec0: []const T, vec1: []const T, vec_out: []T) !void {
    assert(vec0.len == vec1.len);
    assert(vec0.len == vec_out.len);

    for (0..vec0.len) |ii| {
        vec_out[ii] = vec0[ii] + vec1[ii];
    }
}

pub fn sub(comptime T: type, vec0: []const T, vec1: []const T, vec_out: []T) !void {
    assert(vec0.len == vec1.len);
    assert(vec0.len == vec_out.len);

    for (0..vec0.len) |ii| {
        vec_out[ii] = vec0[ii] - vec1[ii];
    }
}

pub fn mul(
    comptime T: type,
    vec0: []const T,
    vec1: []const T,
    vec_out: []T,
) !void {
    assert(vec0.len == vec1.len);
    assert(vec0.len == vec_out.len);

    for (0..vec0.len) |ii| {
        vec_out[ii] = vec0[ii] * vec1[ii];
    }
}

pub fn div(comptime T: type, vec0: []const T, vec1: []const T, vec_out: []T) !void {
    assert(vec0.len == vec1.len);

    for (0..vec0.len) |ii| {
        vec_out[ii] = vec0[ii] / vec1[ii];
    }
}

pub fn mulScal(comptime T: type, vec0: []const T, scal: T, vec_out: []T) !void {
    assert(vec0.len == vec_out.len);

    for (0..vec0.len) |ii| {
        vec_out[ii] = scal * vec0[ii];
    }
}

pub fn slicePrint(comptime T: type, slice: []const T) void {
    print("[", .{});
    for (0..slice.len) |ii| {
        print("{},", .{slice[ii]});
    }
    print("]\n", .{});
}



// --------------------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------------------
const TestType = F;

const testing = std.testing;
const expectEqual = testing.expectEqual;
const expectApproxEqAbs = testing.expectApproxEqAbs;
const expectEqualSlices = testing.expectEqualSlices;

test "slice.add" {
    const vec_len: usize = 10;

    var vec0 = [_]F{1.0} ** vec_len;
    var vec1 = [_]F{1.0} ** vec_len;
    var vec_exp = [_]F{2.0} ** vec_len;

    var vec_op = [_]F{0.0} ** vec_len;

    try add(TestType, vec0[0..], vec1[0..], vec_op[0..]);

    try expectEqualSlices(TestType, vec_exp[0..], vec_op[0..]);
}

test "slice.sub" {
    const vec_len: usize = 10;

    var vec0 = [_]F{1.0} ** vec_len;
    var vec1 = [_]F{1.0} ** vec_len;
    var vec_exp = [_]F{0.0} ** vec_len;

    var vec_op = [_]F{-1.0} ** vec_len;

    try sub(TestType, vec0[0..], vec1[0..], vec_op[0..]);

    try expectEqualSlices(TestType, vec_exp[0..], vec_op[0..]);
}

test "slice.mul" {
    const vec_len: usize = 10;

    var vec0 = [_]F{1.0} ** vec_len;
    var vec1 = [_]F{1.0} ** vec_len;
    var vec_exp = [_]F{1.0} ** vec_len;

    var vec_op = [_]F{0.0} ** vec_len;

    try mul(TestType, vec0[0..], vec1[0..], vec_op[0..]);

    try expectEqualSlices(TestType, vec_exp[0..], vec_op[0..]);
}

test "slice.div" {
    const vec_len: usize = 10;

    var vec0 = [_]F{1.0} ** vec_len;
    var vec1 = [_]F{1.0} ** vec_len;
    var vec_exp = [_]F{1.0} ** vec_len;

    var vec_op = [_]F{0.0} ** vec_len;

    try div(TestType, vec0[0..], vec1[0..], vec_op[0..]);

    try expectEqualSlices(TestType, vec_exp[0..], vec_op[0..]);
}

test "slice.mulScal" {
    const vec_len: usize = 10;

    var vec0 = [_]F{1.0} ** vec_len;
    var vec_exp = [_]F{2.0} ** vec_len;
    const scal: TestType = 2.0;

    var vec_op = [_]F{0.0} ** vec_len;

    try mulScal(TestType, vec0[0..], scal, vec_op[0..]);

    try expectEqualSlices(TestType, vec_exp[0..], vec_op[0..]);
}

test "slice.apply" {
    const arr_ones = [_]TestType{1} ** 7;
    const arr_zeros = [_]TestType{0} ** 7;

    var arr_out = [_]TestType{-1} ** 7;

    apply(TestType, &arr_out, &arr_ones, std.math.sqrt);

    try expectEqual(arr_ones, arr_out);

    arr_out = [_]TestType{-1} ** 7;
    apply(TestType, &arr_out, &arr_zeros, std.math.atan);

    try expectEqual(arr_zeros, arr_out);

    arr_out = [_]TestType{-1} ** 7;
    apply(TestType, &arr_out, &arr_zeros, exp);

    try expectEqual(arr_ones, arr_out);
}

test "slice.max" {
    const array = [_]TestType{ 1, 2, 3, 7, 0, -3, 1 };
    const max_idx = max(TestType, &array);

    const max_idx_exp = ValIdx(TestType){
        .val = 7,
        .idx = 3,
    };

    try expectEqual(max_idx_exp, max_idx);
}

test "slice.min" {
    const array = [_]TestType{ 1, 2, 3, 7, 0, -3, 1 };
    const min_idx = min(TestType, &array);

    const min_idx_exp = ValIdx(TestType){
        .val = -3,
        .idx = 5,
    };

    try expectEqual(min_idx_exp, min_idx);
}

test "slice.sum" {
    const array = [_]TestType{ 1, 2, 3, 7, 0, -3, 1 };
    const sum_exp: TestType = 11;
    const sum_arr = sum(TestType, &array);

    try expectEqual(sum_exp, sum_arr);
}

test "slice.mean" {
    const array = [_]TestType{ 1, 2, 3, 7, 0, -3, 1 };
    const mean_exp: TestType = 11.0 / 7.0;
    const mean_arr = mean(TestType, &array);

    try expectEqual(mean_exp, mean_arr);
}
