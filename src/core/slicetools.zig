const std = @import("std");
const print = std.debug.print;
const assert = std.debug.assert;
const expectEqual = std.testing.expectEqual;

pub fn ValInd(ValType: type) type {
    return struct {
        val: ValType,
        ind: usize,
    };
}

pub fn max(comptime EType: type, slice: []const EType) ValInd(EType) {
    assert(slice.len > 0);

    var val_ind = ValInd(EType) {
        .val = slice[0],
        .ind = 0,
    };

    for (slice[1..], 1..) |elem,ii| {
        if (elem > val_ind.val) {
            val_ind.ind = ii;
            val_ind.val = elem;
        }
    }

    return val_ind;
}

pub fn min(comptime EType: type, slice: []const EType) ValInd(EType) {
    assert(slice.len > 0);

    var val_ind = ValInd(EType) {
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
    return sum(EType, slice) / @as(EType,@floatFromInt(slice.len));
}

const TestType = f64;

test "slice.max" {
    const array = [_]TestType{1,2,3,7,0,-3,1};
    const max_ind = max(TestType,&array);

    const max_ind_exp = ValInd(TestType) {
        .val = 7,
        .ind = 3,
    };

    try expectEqual(max_ind_exp,max_ind);
}

test "slice.min" {
    const array = [_]TestType{1,2,3,7,0,-3,1};
    const min_ind = min(TestType,&array);

    const min_ind_exp = ValInd(TestType) {
        .val = -3,
        .ind = 5,
    };

    try expectEqual(min_ind_exp,min_ind);
}

test "slice.sum" {
    const array = [_]TestType{1,2,3,7,0,-3,1};
    const sum_exp: TestType = 11;
    const sum_arr = sum(TestType,&array);

    try expectEqual(sum_exp, sum_arr);
}

test "slice.mean" {
    const array = [_]TestType{1,2,3,7,0,-3,1};
    const mean_exp: TestType = 11.0/7.0;
    const mean_arr = mean(TestType,&array);

    try expectEqual(mean_exp, mean_arr);
}
