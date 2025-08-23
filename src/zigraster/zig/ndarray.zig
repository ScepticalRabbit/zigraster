const std = @import("std");
const print = std.debug.print;

const testing = std.testing;
const assert = std.debug.assert;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const expectApproxEqAbs = testing.expectApproxEqAbs;
const expectEqualSlices = testing.expectEqualSlices;

const sliceops = @import("sliceops.zig");

const NDArrayError = error{
    ElemsWrongLenForDims,
    IndexOutOfBounds,
    DimOutOfBounds,
    IndicesWrongLenForDims,
    DimMismatch,
};

pub fn NDArray(comptime EType: type) type {
    return struct {
        elems: []EType,
        dims: []usize,

        const Self: type = @This();

        pub fn init(elems: []EType, dims: []usize) !Self {
            var dim_prod: usize = dims[0];
            for (1..dims.len) |dd| {
                dim_prod *= dims[dd];
            }

            if (elems.len != dim_prod) {
                return NDArrayError.ElemsWrongLenForDims;
            }

            return .{
                .elems = elems,
                .dims = dims,
            };
        }

        pub fn set(self: *Self, indices: []const usize, in_val: EType) !void {
            const ind: usize = self.flatInd(indices);
            self.elems[ind] = in_val;
        }

        pub fn get(self: *Self, indices: []const usize) !EType {
            const ind: usize = self.flatInd(indices);
            return self.elems[ind];
        }

        pub fn flatInd(self: *const Self, indices: []const usize) !usize {
            // Row-major (C format) for flat NDArrays, last dimension is
            // consistent in memory.
            if (indices.len != self.dims.len) {
                return NDArrayError.IndicesWrongLenForDims;
            }

            var flat_ind: usize = 0;
            var mult: usize = 1;

            var ii: usize = self.dims.len - 1;
            while (ii > 0) : (ii -= 1) {
                if (indices[ii] >= self.dims[ii]) {
                    return NDArrayError.IndexOutOfBounds;
                }

                flat_ind += indices[ii] * mult;
                mult *= self.dims[ii];
            }

            return flat_ind;
        }

        pub fn flatStride(self: *const Self, dim_ind: usize) !usize {
            // Row-major (C format) for flat NDArrays
            // stride = product of all dimensions to the right of the selected
            // dimension.
            if (dim_ind >= self.dims.len) {
                return NDArrayError.DimOutOfBounds;
            }

            var stride: usize = 1;

            var ii: usize = dim_ind + 1;
            while (ii < self.dims.len) : (ii += 1) {
                stride *= self.dims[ii];
            }

            return stride;
        }

        pub fn fill(self: *const Self, fill_val: EType) void {
            @memset(self.elems[0..], fill_val);
        }

        pub fn addInPlace(self: *const Self, to_add: *const Self) void {
            if (!matchArrayDims(EType, self, to_add)) {
                return NDArrayError.DimMismatch;
            }

            for (0..self.elems.len) |ii| {
                self.elems[ii] += to_add.elems[ii];
            }
        }

        pub fn subInPlace(self: *const Self, to_sub: *const Self) void {
            if (!matchArrayDims(EType, self, to_sub)) {
                return NDArrayError.DimMismatch;
            }

            for (0..self.elems.len) |ii| {
                self.elems[ii] -= to_sub.elems[ii];
            }
        }

        pub fn mulInPlace(self: *const Self, to_mul: *const Self) void {
            if (!matchArrayDims(EType, self, to_mul)) {
                return NDArrayError.DimMismatch;
            }

            for (0..self.elems.len) |ii| {
                self.elems[ii] *= to_mul.elems[ii];
            }
        }

        pub fn divInPlace(self: *const Self, to_div: *const Self) void {
            if (!matchArrayDims(EType, self, to_div)) {
                return NDArrayError.DimMismatch;
            }

            for (0..self.elems.len) |ii| {
                self.elems[ii] /= to_div.elems[ii];
            }
        }

        pub fn mulScalarInPlace(self: *const Self, scalar: EType) void {
            for (0..self.elems.len) |ii| {
                self.elems[ii] *= scalar;
            }
        }
    };
}

pub fn NDArrayOps(comptime EType: type) type {
    return struct {
        pub fn add(arr0: *const NDArray(EType), arr1: *const NDArray(EType), arr_out: *NDArray(EType)) !void {
            if (!matchArrayDims(EType, arr0, arr1) or !matchArrayDims(EType, arr0, arr_out)){
                return NDArray.DimMismatch;
            }
            sliceops.add(EType, arr0.elems, arr1.elems, arr_out.elems);
        }

        pub fn sub(arr0: *const NDArray(EType), arr1: *const NDArray(EType), arr_out: *NDArray(EType)) !void {
            if (!matchArrayDims(EType, arr0, arr1) or !matchArrayDims(EType, arr0, arr_out)){
                return NDArray.DimMismatch;
            }
            sliceops.sub(EType, arr0.elems, arr1.elems, arr_out.elems);
        }

        pub fn mulElemWise(arr0: *const NDArray(EType), arr1: *const NDArray(EType), arr_out: *NDArray(EType)) !void {
            if (!matchArrayDims(EType, arr0, arr1) or !matchArrayDims(EType, arr0, arr_out)){
                return NDArray.DimMismatch;
            }
            sliceops.mul(EType, arr0.elems, arr1.elems, arr_out.elems);
        }

        pub fn divElemWise(arr0: *const NDArray(EType), arr1: *const NDArray(EType), arr_out: *NDArray(EType)) !void {
            if (!matchArrayDims(EType, arr0, arr1) or !matchArrayDims(EType, arr0, arr_out)){
                return NDArray.DimMismatch;
            }
            sliceops.div(EType, arr0.elems, arr1.elems, arr_out.elems);
        }
    };
}

pub fn matchArrayDims(comptime EType: type, arr0: *const NDArray(EType), arr1: *const NDArray(EType)) bool {
    if (arr0.dims.len != arr1.dims.len) {
        return false;
    }

    for (0..arr0.dims.len) |ii| {
        if (arr0.dims[ii] != arr1.dims[ii]) {
            return false;
        }
    }

    return true;
}

test "matchArrayDims" {
    var dims0 = [_]usize{3,3,2};
    var elems0 = [_]f64{0.0} ** 18;
    const arr0 = try NDArray(f64).init(elems0[0..],dims0[0..]);

    var dims1 = [_]usize{3,3,2};
    var elems1 = [_]f64{1.0} ** 18;
    const arr1 = try NDArray(f64).init(elems1[0..],dims1[0..]);

    var dims2 = [_]usize{2,3,3};
    var elems2 = [_]f64{1.0} ** 18;
    const arr2 = try NDArray(f64).init(elems2[0..],dims2[0..]);

    var dims3 = [_]usize{2,2,2};
    var elems3 = [_]f64{0.0} ** 8;
    const arr3 = try NDArray(f64).init(elems3[0..],dims3[0..]);

    try expect(matchArrayDims(f64,&arr0,&arr1));
    try expect(!matchArrayDims(f64,&arr0,&arr2));
    try expect(!matchArrayDims(f64,&arr0,&arr3));
}
