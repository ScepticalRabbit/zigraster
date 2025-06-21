const std = @import("std");
const print = std.debug.print;

const testing = std.testing;
const assert = std.debug.assert;
const expectEqual = std.testing.expectEqual;
const expectApproxEqAbs = testing.expectApproxEqAbs;
const expectEqualSlices = testing.expectEqualSlices;

const SliceOps = @import("sliceops.zig");
const ValInd = SliceOps.ValInd;

const TestType = f64;

pub fn VecSlice(comptime ElemType: type) type {
    return struct {
        elems: []ElemType,

        const Self: type = @This();

        pub fn init(elems: []ElemType) Self {
            return .{
                .elems = elems,
            };
        }

        pub fn fill(self: *const Self, fill_val: ElemType) void {
            @memset(self.elems,fill_val);
        }

        pub fn get(self: *const Self, ind: usize) ElemType {
            return self.elems[ind];
        }

        pub fn set(self: *const Self, ind: usize, val: ElemType) void {
            self.elems[ind] = val;
        }

        pub fn addInPlace(self: *const Self, to_add: *const Self) void {
            for (0..self.elems.len) |ii| {
                self.elems[ii] += to_add.elems[ii];
            }
        }

        pub fn subInPlace(self: *const Self, to_sub: *const Self) void {
            for (0..self.elems.len) |ii| {
                self.elems[ii] -= to_sub.elems[ii];
            }
        }

        pub fn mulInPlace(self: *const Self, to_mul: *const Self) void {
            for (0..self.elems.len) |ii| {
                self.elems[ii] *= to_mul.elems[ii];
            }
        }

        pub fn divInPlace(self: *const Self, to_div: *const Self) void {
            for (0..self.elems.len) |ii| {
                self.elems[ii] /= to_div.elems[ii];
            }
        }

        pub fn mulScalarInPlace(self: *const Self, scalar: ElemType) void {
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
            return SliceOps.max(ElemType, self.elems);
        }

        pub fn min(self: *const Self) ValInd(ElemType) {
            return SliceOps.min(ElemType, self.elems);
        }

        pub fn sum(self: *const Self) ElemType {
            return SliceOps.sum(ElemType, self.elems);
        }

        pub fn mean(self: *const Self) ElemType {
            return SliceOps.mean(ElemType, self.elems);
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

pub fn VecSliceOps(comptime ElemType: type) type {
    return struct {
        pub fn add(vec0: []const ElemType, vec1: []const ElemType, vec_out: []ElemType) !void{
            assert(vec0.len == vec1.len);
            assert(vec0.len == vec_out.len);

            for (0..vec0.len) |ii| {
                vec_out[ii] = vec0[ii] + vec1[ii];
            }
        }

        pub fn sub(vec0: []const ElemType, vec1: []const ElemType, vec_out: []ElemType) !void{
            assert(vec0.len == vec1.len);
            assert(vec0.len == vec_out.len);

            for (0..vec0.len) |ii| {
                vec_out[ii] = vec0[ii] - vec1[ii];
            }
        }

        pub fn mul(vec0: []const ElemType, vec1: []const ElemType, vec_out: []ElemType) !void{
            assert(vec0.len == vec1.len);
            assert(vec0.len == vec_out.len);

            for (0..vec0.len) |ii| {
                vec_out[ii] = vec0[ii] * vec1[ii];
            }
        }


        pub fn div(vec0: []const ElemType, vec1: []const ElemType, vec_out: []ElemType) !void{
            assert(vec0.len == vec1.len);

            for (0..vec0.len) |ii| {
                vec_out[ii] = vec0[ii] / vec1[ii];
            }
        }

        pub fn mulScalar(vec0: []const ElemType, scalar: ElemType, vec_out: []ElemType) !void{
            assert(vec0.len == vec_out.len);

            for (0..vec0.len) |ii| {
                vec_out[ii] = scalar*vec0[ii];
            }
        }
    };
}


// test "VecSliceOps.add" {
//     const vec_len: usize = 10;

//     const vec0 = try VecSlice(TestType).init(testing.allocator,vec_len);
//     defer vec0.deinit();

//     const vec1 = try VecSlice(TestType).init(testing.allocator,vec_len);
//     defer vec1.deinit();

//     const vec_exp = try VecSlice(TestType).init(testing.allocator,vec_len);
//     defer vec_exp.deinit();

//     vec0.fill(1.0);
//     vec1.fill(1.0);
//     vec_exp.fill(2.0);
//     const VecOps = VecSliceOps(TestType);

//     const vec_add = try VecOps.add(testing.allocator, &vec0, &vec1);
//     defer vec_add.deinit();

//     try expectEqualSlices(TestType, vec_exp.elems, vec_add.elems);
// }

// test "VecSliceOps.sub" {
//     const vec_len: usize = 10;

//     const vec0 = try VecSlice(TestType).init(testing.allocator,vec_len);
//     defer vec0.deinit();

//     const vec1 = try VecSlice(TestType).init(testing.allocator,vec_len);
//     defer vec1.deinit();

//     const vec_exp = try VecSlice(TestType).init(testing.allocator,vec_len);
//     defer vec_exp.deinit();

//     vec0.fill(1.0);
//     vec1.fill(1.0);
//     vec_exp.fill(0.0);
//     const VecOps = VecSliceOps(TestType);

//     const vec_add = try VecOps.sub(testing.allocator, &vec0, &vec1);
//     defer vec_add.deinit();

//     try expectEqualSlices(TestType, vec_exp.elems, vec_add.elems);
// }

// test "VecSliceOps.mul" {
//     const vec_len: usize = 10;

//     const vec0 = try VecSlice(TestType).init(testing.allocator,vec_len);
//     defer vec0.deinit();

//     const vec1 = try VecSlice(TestType).init(testing.allocator,vec_len);
//     defer vec1.deinit();

//     const vec_exp = try VecSlice(TestType).init(testing.allocator,vec_len);
//     defer vec_exp.deinit();

//     vec0.fill(1.0);
//     vec1.fill(1.0);
//     vec_exp.fill(1.0);
//     const VecOps = VecSliceOps(TestType);

//     const vec_add = try VecOps.mul(testing.allocator, &vec0, &vec1);
//     defer vec_add.deinit();

//     try expectEqualSlices(TestType, vec_exp.elems, vec_add.elems);
// }

// test "VecSliceOps.div" {
//     const vec_len: usize = 10;

//     const vec0 = try VecSlice(TestType).init(testing.allocator,vec_len);
//     defer vec0.deinit();

//     const vec1 = try VecSlice(TestType).init(testing.allocator,vec_len);
//     defer vec1.deinit();

//     const vec_exp = try VecSlice(TestType).init(testing.allocator,vec_len);
//     defer vec_exp.deinit();

//     vec0.fill(1.0);
//     vec1.fill(1.0);
//     vec_exp.fill(1.0);
//     const VecOps = VecSliceOps(TestType);

//     const vec_add = try VecOps.div(testing.allocator, &vec0, &vec1);
//     defer vec_add.deinit();

//     try expectEqualSlices(TestType, vec_exp.elems, vec_add.elems);
// }

test "VecSliceOps.mulScalar" {
    const vec_len: usize = 10;

    var v0 = [_]f64{0.0} ** vec_len;
    const vec0 = VecSlice(TestType).init(v0[0..]);

    var ve = [_]f64{0.0} ** vec_len;
    const vec_exp = VecSlice(TestType).init(ve[0..]);

    vec0.fill(1.0);
    vec_exp.fill(2.0);

    const scalar: TestType = 2.0;
    const VecOps = VecSliceOps(TestType);

    const vec_add = try VecOps.mulScalar(testing.allocator, &vec0, scalar);
    defer vec_add.deinit();

    try expectEqualSlices(TestType, vec_exp.elems, vec_add.elems);
}

test "VecSliceOps.addInPlace" {
    const vec_len: usize = 10;

    var v0 = [_]f64{0.0} ** vec_len;
    const vec0 = VecSlice(TestType).init(v0[0..]);

    var v1 = [_]f64{0.0} ** vec_len;
    const vec1 = VecSlice(TestType).init(v1[0..]);

    var ve = [_]f64{0.0} ** vec_len;
    const vec_exp = VecSlice(TestType).init(ve[0..]);

    vec0.fill(1.0);
    vec1.fill(1.0);
    vec_exp.fill(2.0);

    vec0.addInPlace(&vec1);

    try expectEqualSlices(TestType, vec_exp.elems, vec0.elems);
}

test "VecSliceOps.subInPlace" {
    const vec_len: usize = 10;

    var v0 = [_]f64{0.0} ** vec_len;
    const vec0 = VecSlice(TestType).init(v0[0..]);

    var v1 = [_]f64{0.0} ** vec_len;
    const vec1 = VecSlice(TestType).init(v1[0..]);

    var ve = [_]f64{0.0} ** vec_len;
    const vec_exp = VecSlice(TestType).init(ve[0..]);

    vec0.fill(1.0);
    vec1.fill(1.0);
    vec_exp.fill(0.0);

    vec0.subInPlace(&vec1);

    try expectEqualSlices(TestType, vec_exp.elems, vec0.elems);
}

test "VecSliceOps.mulInPlace" {
    const vec_len: usize = 10;

    var v0 = [_]f64{0.0} ** vec_len;
    const vec0 = VecSlice(TestType).init(v0[0..]);

    var v1 = [_]f64{0.0} ** vec_len;
    const vec1 = VecSlice(TestType).init(v1[0..]);

    var ve = [_]f64{0.0} ** vec_len;
    const vec_exp = VecSlice(TestType).init(ve[0..]);

    vec0.fill(1.0);
    vec1.fill(1.0);
    vec_exp.fill(1.0);

    vec0.mulInPlace(&vec1);

    try expectEqualSlices(TestType, vec_exp.elems, vec0.elems);
}

test "VecSliceOps.divInPlace" {
    const vec_len: usize = 10;

    var v0 = [_]f64{0.0} ** vec_len;
    const vec0 = VecSlice(TestType).init(v0[0..]);

    var v1 = [_]f64{0.0} ** vec_len;
    const vec1 = VecSlice(TestType).init(v1[0..]);

    var ve = [_]f64{0.0} ** vec_len;
    const vec_exp = VecSlice(TestType).init(ve[0..]);

    vec0.fill(1.0);
    vec1.fill(1.0);
    vec_exp.fill(1.0);

    vec0.divInPlace(&vec1);

    try expectEqualSlices(TestType, vec_exp.elems, vec0.elems);
}

test "VecSliceOps.mulScalarInPlace" {
    const vec_len: usize = 10;

    var v0 = [_]f64{0.0} ** vec_len;
    const vec0 = VecSlice(TestType).init(v0[0..]);

    var ve = [_]f64{0.0} ** vec_len;
    const vec_exp = VecSlice(TestType).init(ve[0..]);

    vec0.fill(1.0);
    vec_exp.fill(2.0);

    const scalar: TestType = 2.0;

    vec0.mulScalarInPlace(scalar);

    try expectEqualSlices(TestType, vec_exp.elems, vec0.elems);
}

test "VecSliceOps.max" {
    var v0 = [_]f64{0.0} ** 10;
    const vec0 = VecSlice(TestType).init(v0[0..]);

    const exp_ind: usize = 4;
    const exp_val: f64 = 8.0;

    vec0.fill(0.0);
    vec0.set(exp_ind,exp_val);

    const exp_val_ind = ValInd(TestType){
        .val = exp_val,
        .ind = exp_ind,
    };

    try expectEqual(exp_val_ind, vec0.max());
}

test "VecSliceOps.min" {
    var v0 = [_]f64{0.0} ** 10;
    const vec0 = VecSlice(TestType).init(v0[0..]);

    const exp_ind: usize = 7;
    const exp_val: f64 = -3.0;

    vec0.fill(0.0);
    vec0.set(exp_ind,exp_val);

    const exp_val_ind = ValInd(TestType){
        .val = exp_val,
        .ind = exp_ind,
    };

    try expectEqual(exp_val_ind, vec0.min());
}

test "VecSliceOps.sum" {
    var v0 = [_]f64{0.0} ** 12;
    const vec0 = VecSlice(TestType).init(v0[0..]);

    vec0.fill(1.0);
    vec0.set(0,0.0);

    const exp_val: TestType = 11;

    try expectEqual(exp_val, vec0.sum());
}

test "VecSliceOps.mean" {
    var v0 = [_]f64{0.0} ** 10;
    const vec0 = VecSlice(TestType).init(v0[0..]);

    vec0.fill(1.0);
    vec0.set(2,0.0);
    vec0.set(7,0.0);

    const exp_val: TestType = 0.8;

    try expectEqual(exp_val, vec0.mean());
}

test "VecSliceOps.apply" {
    const vec_len: usize = 7;

    var v0 = [_]f64{0.0} ** vec_len;
    var vec0 = VecSlice(TestType).init(v0[0..]);

    var ve1 = [_]f64{1.0} ** vec_len;
    const vec_exp_ones = VecSlice(TestType).init(ve1[0..]);

    var ve0 = [_]f64{0.0} ** vec_len;
    const vec_exp_zeros = VecSlice(TestType).init(ve0[0..]);

    vec0.fill(1.0);
    vec_exp_ones.fill(1.0);
    vec_exp_zeros.fill(0.0);

    vec0.applyInPlace(std.math.sqrt);

    try expectEqualSlices(TestType,vec_exp_ones.elems, vec0.elems);

    var v1 = [_]f64{0.0} ** vec_len;
    var vec1 = VecSlice(TestType).init(v1[0..]);

    vec1.fill(0.0);
    vec1.applyInPlace(std.math.atan);

    try expectEqualSlices(TestType,vec_exp_zeros.elems, vec1.elems);

    vec1.applyInPlace(SliceOps.exp);

    try expectEqualSlices(TestType, vec_exp_ones.elems, vec1.elems);
}
