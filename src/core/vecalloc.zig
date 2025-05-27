const std = @import("std");
const print = std.debug.print;

const testing = std.testing;
const assert = std.debug.assert;
const expectEqual = std.testing.expectEqual;
const expectApproxEqAbs = testing.expectApproxEqAbs;
const expectEqualSlices = testing.expectEqualSlices;

const slice = @import("slicetools.zig");
const ValInd = slice.ValInd;

const EType = f64;

pub fn VecAlloc(comptime ElemType: type) type {
    return struct {
        elems: []ElemType,
        alloc: std.mem.Allocator,

        const Self: type = @This();

        pub fn init(allocator: std.mem.Allocator, elem_n: usize) !Self {
            return .{
                .elems = try allocator.alloc(ElemType,elem_n),
                .alloc = allocator,
            };
        }

        pub fn deinit(self: Self) void {
            self.alloc.free(self.elems);
        }

        pub fn fill(self: *const Self, fill_val: ElemType) void {
            for (0..self.elems.len) |ii| {
                self.elems[ii] = fill_val;
            }
        }

        pub fn get(self: *const Self, ind: usize) ElemType {
            return self.elems[ind];
        }

        pub fn set(self: *const Self, ind: usize, val: ElemType) void {
            self.elems[ind] = val;
        }

        pub fn addInPlace(self: *const Self, to_add: Self) void {
            for (0..self.elems.len) |ii| {
                self.elems[ii] += to_add.elems[ii];
            }
        }

        pub fn subInPlace(self: *const Self, to_sub: Self) void {
            for (0..self.elems.len) |ii| {
                self.elems[ii] -= to_sub.elems[ii];
            }
        }

        pub fn mulInPlace(self: *const Self, scalar: ElemType) void {
            for (0..self.elems.len) |ii| {
                self.elems[ii] = scalar * self.elems[ii];
            }
        }

        pub fn applyInPlace(self: *const Self, func: *const fn(val: anytype) ElemType) void {
            for (self.elems,0..) |elem,ii| {
                self.elems[ii] = func(elem);
            }
        }

        pub fn dot(self: *const Self, to_dot: Self) ElemType {
            var dot_prod: ElemType = 0;
            for (0..self.elems.len) |ii| {
                dot_prod += self.elems[ii] * to_dot.elems[ii];
            }
            return dot_prod;
        }

        pub fn norm(self: *const Self) ElemType {
            var norm_out: ElemType = 0;
            for (0..self.elems.len) |ii| {
                norm_out += self.elems[ii] * self.elems[ii];
            }
            return norm_out;
        }

        pub fn vecLen(self: *const Self) ElemType {
            return @sqrt(self.norm());
        }

        pub fn max(self: *const Self) ValInd(ElemType) {
            return slice.max(ElemType, self.elems);
        }

        pub fn min(self: *const Self) ValInd(ElemType) {
            return slice.min(ElemType, self.elems);
        }

        pub fn sum(self: *const Self) ElemType {
            return slice.sum(ElemType, self.elems);
        }

        pub fn mean(self: *const Self) ElemType {
            return slice.mean(ElemType, self.elems);
        }

        pub fn vecPrint(self: *const Self) void {
            print("[", .{});
            for (0..self.elems.len) |ii| {
                print("{e:.3},", .{self.elems[ii]});
            }
            print("]\n", .{});
        }
    };
}



// TODO: test in place maths

test "VecAlloc.max" {
    const vec0 = try VecAlloc(EType).init(testing.allocator,10);
    defer vec0.deinit();

    const exp_ind: usize = 4;
    const exp_val: f64 = 8.0;

    vec0.fill(0.0);
    vec0.set(exp_ind,exp_val);

    const exp_val_ind = ValInd(EType){
        .val = exp_val,
        .ind = exp_ind,
    };

    try expectEqual(exp_val_ind, vec0.max());
}

test "VecAlloc.min" {
    const vec0 = try VecAlloc(EType).init(testing.allocator,10);
    defer vec0.deinit();

    const exp_ind: usize = 7;
    const exp_val: f64 = -3.0;

    vec0.fill(0.0);
    vec0.set(exp_ind,exp_val);

    const exp_val_ind = ValInd(EType){
        .val = exp_val,
        .ind = exp_ind,
    };

    try expectEqual(exp_val_ind, vec0.min());
}

test "VecAlloc.sum" {
    const vec0 = try VecAlloc(EType).init(testing.allocator,12);
    defer vec0.deinit();

    vec0.fill(1.0);
    vec0.set(0,0.0);

    const exp_val: EType = 11;

    try expectEqual(exp_val, vec0.sum());
}

test "VecAlloc.mean" {
    const vec0 = try VecAlloc(EType).init(testing.allocator,10);
    defer vec0.deinit();

    vec0.fill(1.0);
    vec0.set(2,0.0);
    vec0.set(7,0.0);

    const exp_val: EType = 0.8;

    try expectEqual(exp_val, vec0.mean());
}

test "VecAlloc.apply" {
    const vec_len: usize = 7;
    var vec0 = try VecAlloc(EType).init(testing.allocator, vec_len);
    defer vec0.deinit();

    const vec_exp_ones = try VecAlloc(EType).init(testing.allocator, vec_len);
    defer vec_exp_ones.deinit();

    const vec_exp_zeros = try VecAlloc(EType).init(testing.allocator, vec_len);
    defer vec_exp_zeros.deinit();

    vec0.fill(1.0);
    vec_exp_ones.fill(1.0);
    vec_exp_zeros.fill(0.0);

    vec0.applyInPlace(std.math.sqrt);

    try expectEqualSlices(EType,vec_exp_ones.elems, vec0.elems);

    var vec1 = try VecAlloc(EType).init(testing.allocator, vec_len);
    defer vec1.deinit();

    vec1.fill(0.0);
    vec1.applyInPlace(std.math.atan);

    try expectEqualSlices(EType,vec_exp_zeros.elems, vec1.elems);

    vec1.applyInPlace(slice.exp);

    try expectEqualSlices(EType, vec_exp_ones.elems, vec1.elems);
}
