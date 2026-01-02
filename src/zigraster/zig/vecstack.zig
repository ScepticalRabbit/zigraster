const std = @import("std");
const print = std.debug.print;
const assert = std.debug.assert;
const expectEqual = std.testing.expectEqual;
const expectEqualSlices = std.testing.expectEqualSlices;

const SliceOps = @import("sliceops.zig");
const ValInd = SliceOps.ValInd;

const EType = f64;
pub const Vec2f = Vec2T(EType);
pub const Vec3f = Vec3T(EType);

// TODO:
// - Tests for VecSliceOps

pub fn VecStack(comptime elem_n: comptime_int, comptime ElemType: type) type {
    return struct {
        elems: [elem_n]ElemType,

        const Self: type = @This();

        pub fn initFill(fill_val: ElemType) Self {
            return .{ .elems = [_]ElemType{fill_val} ** elem_n };
        }

        pub fn initOnes() Self {
            return initFill(1);
        }

        pub fn initZeros() Self {
            return initFill(0);
        }

        pub fn initSlice(slice_in: []const ElemType) Self {
            return .{ .elems = slice_in[0..elem_n].* };
        }

        pub fn get(self: *const Self, ind: usize) ElemType {
            return self.elems[ind];
        }

        pub fn set(self: *Self, ind: usize, val: ElemType) void {
            self.elems[ind] = val;
        }

        pub fn x(self: Self) ElemType {
            return self.elems[0];
        }
        
        pub fn y(self: Self) ElemType {
            return self.elems[1];
        }

        pub fn z(self: Self) ElemType {
            return self.elems[2];
        }

        pub fn w(self: Self) ElemType {
            return self.elems[2];
        }

        pub fn add(self: *const Self, to_add: Self) Self {
            var vec_out: Self = undefined;
            for (0..elem_n) |ii| {
                vec_out.elems[ii] = self.elems[ii] + to_add.elems[ii];
            }
            return vec_out;
        }

        pub fn sub(self: *const Self, to_sub: Self) Self {
            var vec_out: Self = undefined;
            for (0..elem_n) |ii| {
                vec_out.elems[ii] = self.elems[ii] - to_sub.elems[ii];
            }
            return vec_out;
        }

        pub fn mulScalar(self: *const Self, scalar: ElemType) Self {
            var vec_out: Self = undefined;
            for (0..elem_n) |ii| {
                vec_out.elems[ii] = scalar * self.elems[ii];
            }
            return vec_out;
        }

        pub fn dot(self: *const Self, to_dot: Self) ElemType {
            var dot_prod: ElemType = 0;
            for (0..elem_n) |ii| {
                dot_prod += self.elems[ii] * to_dot.elems[ii];
            }
            return dot_prod;
        }

        pub fn norm(self: *const Self) ElemType {
            var norm_out: ElemType = 0;
            for (0..elem_n) |ii| {
                norm_out += self.elems[ii] * self.elems[ii];
            }
            return norm_out;
        }

        pub fn vecLen(self: *const Self) ElemType {
            return @sqrt(self.norm());
        }

        pub fn max(self: *const Self) ValInd(ElemType) {
            return SliceOps.max(ElemType, &self.elems);
        }

        pub fn min(self: *const Self) ValInd(ElemType) {
            return SliceOps.min(ElemType, &self.elems);
        }

        pub fn sum(self: *const Self) ElemType {
            return SliceOps.sum(ElemType, &self.elems);
        }

        pub fn mean(self: *const Self) ElemType {
            return SliceOps.mean(ElemType, &self.elems);
        }

        pub fn apply(self: *const Self, func: *const fn(val: anytype) ElemType) Self {
            var applied: Self = undefined;
            for (self.elems,0..) |elem,ii| {
                applied.elems[ii] = func(elem);
            }
            return applied;
        }

        pub fn vecPrint(self: *const Self) void {
            print("[", .{});
            for (0..elem_n) |ii| {
                print("{e:.3},", .{self.elems[ii]});
            }
            print("]\n", .{});
        }
    };
}

pub fn Vec2T(comptime ElemType: type) type {
    return VecStack(2, ElemType);
}

pub fn Vec3T(comptime ElemType: type) type {
    return VecStack(3, ElemType);
}

pub fn initVec2(comptime ElemType: type, x: ElemType, y: ElemType) Vec2T(ElemType) {
    return Vec2T(ElemType){
        .elems = [3]ElemType{ x, y },
    };
}

pub fn initVec3(comptime ElemType: type, x: ElemType, y: ElemType, z: ElemType) Vec3T(ElemType) {
    return Vec3T(ElemType){
        .elems = [3]ElemType{ x, y, z },
    };
}

pub const Vec3Ops = struct {
    pub fn cross(ElemType: type, vec0: Vec3T(ElemType), vec1: Vec3T(ElemType)) Vec3T(ElemType) {
        var vec_out: Vec3T(ElemType) = undefined;
        vec_out.elems[0] = vec0.elems[1] * vec1.elems[2] - vec0.elems[2] * vec1.elems[1];
        vec_out.elems[1] = vec0.elems[0] * vec1.elems[2] - vec0.elems[2] * vec1.elems[0];
        vec_out.elems[2] = vec0.elems[0] * vec1.elems[1] - vec0.elems[1] * vec1.elems[0];
        return vec_out;
    }
};

pub const Vec3SliceOps = struct {
    pub fn max(ElemType: type, vec: []Vec3T(ElemType), ind: usize) ElemType {
        assert(vec.len > 0);
        assert(ind < 3);

        var val: ElemType = vec[0].get(ind);
        for (vec[1..]) |vv| {

            if (vv.get(ind) > val) {
                val = vv.get(ind);
            }
        }

        return val;
    }

    pub fn min(ElemType: type, vec: []Vec3T(ElemType), ind: usize) ElemType {
        assert(vec.len > 0);
        assert(ind < 3);

        var val: ElemType = vec[0].get(ind);
        for (vec[1..]) |vv| {

            if (vv.get(ind) < val) {
                val = vv.get(ind);
            }
        }

        return val;
    }
};

test "VecSliceOps.max" {
    var vec_slice: [3]Vec3f = undefined;
    vec_slice[0] = initVec3(f64, -1.0, 2.0, 7.0);
    vec_slice[1] = initVec3(f64, 2.0, -2.0, 7.0);
    vec_slice[2] = initVec3(f64, 5.0, -10.0, 0.0);

    const max_x: f64 = Vec3SliceOps.max(f64,vec_slice[0..],0);
    const max_y: f64 = Vec3SliceOps.max(f64,vec_slice[0..],1);
    const max_z: f64 = Vec3SliceOps.max(f64,vec_slice[0..],2);

    const exp_max_x: f64 = 5.0;
    const exp_max_y: f64 = 2.0;
    const exp_max_z: f64 = 7.0;

    try expectEqual(exp_max_x, max_x);
    try expectEqual(exp_max_y, max_y);
    try expectEqual(exp_max_z, max_z);
}


test "Vec.apply" {
    const vec_len: usize = 7;
    const vec0 = VecStack(vec_len,EType).initOnes();

    const vec_exp_ones = VecStack(vec_len,EType).initOnes();
    const vec_exp_zeros = VecStack(vec_len,EType).initZeros();

    const vec_sqrt = vec0.apply(std.math.sqrt);

    try expectEqualSlices(EType, &vec_exp_ones.elems, &vec_sqrt.elems);

    const vec1 = VecStack(vec_len,EType).initZeros();
    const vec_atan = vec1.apply(std.math.atan);

    try expectEqualSlices(EType, &vec_exp_zeros.elems, &vec_atan.elems);

    const vec_e = vec1.apply(SliceOps.exp);

    try expectEqualSlices(EType, &vec_exp_ones.elems, &vec_e.elems);
}

test "Vec.max" {
    const v0 = [_]EType{ 1, 3, 6, 7, 8, 1, -2, -3, 0, 5 };
    const vec0 = VecStack(v0.len, EType).initSlice(&v0);

    const exp_val = ValInd(EType){
        .val = 8,
        .ind = 4,
    };

    try expectEqual(exp_val, vec0.max());
}

test "Vec.min" {
    const v0 = [_]EType{ 1, 3, 6, 7, 8, 1, -2, -3, 0, 5 };
    const vec0 = VecStack(v0.len, EType).initSlice(&v0);

    const exp_val = ValInd(EType){
        .val = -3,
        .ind = 7,
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

    try expectEqualSlices(EType,&vec0.add(vec1).elems, &vec_exp.elems);
}

test "Vec3f.sub" {
    var vec0 = Vec3f.initOnes();
    const vec1 = Vec3f.initFill(7);
    const vec_exp = Vec3f.initFill(-6);

    try expectEqualSlices(EType, &vec0.sub(vec1).elems, &vec_exp.elems);
}

test "Vec3f.mulScalar" {
    var vec0 = Vec3f.initOnes();
    const scalar: EType = 1.23;
    const vec_exp = Vec3f.initFill(scalar);

    try expectEqualSlices(EType,&vec0.mulScalar(scalar).elems, &vec_exp.elems);
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

    try expectEqual(Vec3Ops.cross(EType, vec0, vec1), cross_exp);
}
