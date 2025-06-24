const std = @import("std");
const print = std.debug.print;

const testing = std.testing;
const assert = std.debug.assert;
const expectEqual = testing.expectEqual;
const expectApproxEqAbs = testing.expectApproxEqAbs;
const expectEqualSlices = testing.expectEqualSlices;

pub fn ValInd(ValType: type) type {
    return struct {
        val: ValType,
        ind: usize,
    };
}

pub fn max(comptime EType: type, slice: []const EType) ValInd(EType) {
    assert(slice.len > 0);

    var val_ind = ValInd(EType){
        .val = slice[0],
        .ind = 0,
    };

    for (slice[1..], 1..) |elem, ii| {
        if (elem > val_ind.val) {
            val_ind.ind = ii;
            val_ind.val = elem;
        }
    }

    return val_ind;
}

pub fn min(comptime EType: type, slice: []const EType) ValInd(EType) {
    assert(slice.len > 0);

    var val_ind = ValInd(EType){
        .val = slice[0],
        .ind = 0,
    };

    for (slice[1..], 1..) |elem, ii| {
        if (elem < val_ind.val) {
            val_ind.ind = ii;
            val_ind.val = elem;
        }
    }

    return val_ind;
}

pub fn sum(comptime EType: type, slice: []const EType) EType {
    assert(slice.len > 0);

    var sum_out: EType = 0;
    for (slice[0..]) |elem| {
        sum_out += elem;
    }
    return sum_out;
}

pub fn mean(comptime EType: type, slice: []const EType) EType {
    return sum(EType, slice) / @as(EType, @floatFromInt(slice.len));
}

// Removing inline from from the stdlib version for use with 'apply'
pub fn exp(value: anytype) @TypeOf(value) {
    return @exp(value);
}

// Based on copy forwards in std.mem
pub fn apply(comptime EType: type, dest: []EType, source: []const EType, func: *const fn (val: anytype) EType) void {
    for (dest[0..source.len], source) |*dd, ss| {
        dd.* = func(ss);
    }
}

pub fn rangeLen(start: f64, stop: f64, step: f64) usize {
    const range: f64 = @ceil((stop - start) / step);
    const range_length: usize = @as(usize, @intFromFloat(range));
    return range_length;
}

pub fn dot(comptime EType: type, slice0: []const EType, slice1: []const EType) EType {
    assert(slice0.len == slice1.len);

    var dot_prod: EType = 0;
    for (0..slice0.len) |ii| {
        dot_prod += slice0[ii] * slice1[ii];
    }
    return dot_prod;
}

pub fn norm(comptime EType: type, vec: []const EType) EType {
    var norm_out: EType = 0;

    for (0..vec.len) |ii|{
        norm_out += vec[ii] * vec[ii];
    }

    return norm_out;
}

pub fn vecLen(comptime EType: type, vec: []const EType) EType {
    return @sqrt(norm(EType,vec));
}

pub fn add(comptime EType: type, vec0: []const EType, vec1: []const EType, vec_out: []EType) !void{
    assert(vec0.len == vec1.len);
    assert(vec0.len == vec_out.len);

    for (0..vec0.len) |ii| {
        vec_out[ii] = vec0[ii] + vec1[ii];
    }
}

pub fn sub(comptime EType: type, vec0: []const EType, vec1: []const EType, vec_out: []EType) !void{
    assert(vec0.len == vec1.len);
    assert(vec0.len == vec_out.len);

    for (0..vec0.len) |ii| {
        vec_out[ii] = vec0[ii] - vec1[ii];
    }
}

pub fn mul(comptime EType: type, vec0: []const EType, vec1: []const EType, vec_out: []EType) !void{
    assert(vec0.len == vec1.len);
    assert(vec0.len == vec_out.len);

    for (0..vec0.len) |ii| {
        vec_out[ii] = vec0[ii] * vec1[ii];
    }
}


pub fn div(comptime EType: type, vec0: []const EType, vec1: []const EType, vec_out: []EType) !void{
    assert(vec0.len == vec1.len);

    for (0..vec0.len) |ii| {
        vec_out[ii] = vec0[ii] / vec1[ii];
    }
}

pub fn mulScalar(comptime EType: type, vec0: []const EType, scalar: EType, vec_out: []EType) !void{
    assert(vec0.len == vec_out.len);

    for (0..vec0.len) |ii| {
        vec_out[ii] = scalar*vec0[ii];
    }
}

pub fn slicePrint(comptime EType: type, slice: []const EType) void {
    print("[", .{});
    for (0..slice.len) |ii| {
        print("{},", .{slice[ii]});
    }
    print("]\n", .{});
}


// TODO: add tests for
// - norm
// - vecLen

const TestType = f64;

test "slice.add" {
    const vec_len: usize = 10;

    var vec0 = [_]f64{1.0} ** vec_len;
    var vec1 = [_]f64{1.0} ** vec_len;
    var vec_exp = [_]f64{2.0} ** vec_len;

    var vec_op = [_]f64{0.0} ** vec_len;

    try add(TestType,vec0[0..], vec1[0..], vec_op[0..]);

    try expectEqualSlices(TestType, vec_exp[0..], vec_op[0..]);

}

test "slice.sub" {
    const vec_len: usize = 10;

    var vec0 = [_]f64{1.0} ** vec_len;
    var vec1 = [_]f64{1.0} ** vec_len;
    var vec_exp = [_]f64{0.0} ** vec_len;

    var vec_op = [_]f64{-1.0} ** vec_len;

    try sub(TestType,vec0[0..], vec1[0..], vec_op[0..]);

    try expectEqualSlices(TestType, vec_exp[0..], vec_op[0..]);

}

test "slice.mul" {
    const vec_len: usize = 10;

    var vec0 = [_]f64{1.0} ** vec_len;
    var vec1 = [_]f64{1.0} ** vec_len;
    var vec_exp = [_]f64{1.0} ** vec_len;

    var vec_op = [_]f64{0.0} ** vec_len;

    try mul(TestType,vec0[0..], vec1[0..], vec_op[0..]);

    try expectEqualSlices(TestType, vec_exp[0..], vec_op[0..]);

}

test "slice.div" {
    const vec_len: usize = 10;

    var vec0 = [_]f64{1.0} ** vec_len;
    var vec1 = [_]f64{1.0} ** vec_len;
    var vec_exp = [_]f64{1.0} ** vec_len;

    var vec_op = [_]f64{0.0} ** vec_len;

    try div(TestType,vec0[0..], vec1[0..], vec_op[0..]);

    try expectEqualSlices(TestType, vec_exp[0..], vec_op[0..]);
}

test "slice.mulScalar" {
    const vec_len: usize = 10;

    var vec0 = [_]f64{1.0} ** vec_len;
    var vec_exp = [_]f64{2.0} ** vec_len;
    const scalar: TestType = 2.0;

    var vec_op = [_]f64{0.0} ** vec_len;

    try mulScalar(TestType,vec0[0..],scalar,vec_op[0..]);

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
    const max_ind = max(TestType, &array);

    const max_ind_exp = ValInd(TestType){
        .val = 7,
        .ind = 3,
    };

    try expectEqual(max_ind_exp, max_ind);
}

test "slice.min" {
    const array = [_]TestType{ 1, 2, 3, 7, 0, -3, 1 };
    const min_ind = min(TestType, &array);

    const min_ind_exp = ValInd(TestType){
        .val = -3,
        .ind = 5,
    };

    try expectEqual(min_ind_exp, min_ind);
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
