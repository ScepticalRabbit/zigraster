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

const MatSlice = @import("matslice.zig").MatSlice;
const sliceops = @import("sliceops.zig");

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub fn NDArray(comptime T: type) type {
    return struct {
        slice: []T,
        dims: []usize,
        strides: []usize,

        const Self: type = @This();

        pub fn init(allocator: std.mem.Allocator, slice: []T, dims: []const usize) !Self {
            var dim_prod: usize = dims[0];
            for (1..dims.len) |dd| {
                dim_prod *= dims[dd];
            }

            assert(slice.len >= dim_prod);

            const dims_heap = try allocator.dupe(usize, dims);
            const strides_heap = try allocator.alloc(usize, dims.len);

            var ndarray = NDArray(T){
                .slice = slice,
                .dims = dims_heap,
                .strides = strides_heap,
            };

            for (0..dims.len) |dd| {
                ndarray.strides[dd] = ndarray.calcFlatStride(dd);
            }

            return ndarray;
        }

        pub fn initFlat(allocator: std.mem.Allocator, dims: []const usize) !Self {
            var dim_prod: usize = 1;
            for (dims) |dd| {
                dim_prod *= dd;
            }

            const slice = try allocator.alloc(T, dim_prod);

            return try Self.init(allocator, slice, dims);
        }

        pub fn deinit(self: *const Self, allocator: std.mem.Allocator) void {
            allocator.free(self.dims);
            allocator.free(self.strides);
        }

        pub fn set(self: *Self, indices: []const usize, in_val: T) void {
            const idx: usize = self.getFlatIdx(indices);
            self.slice[idx] = in_val;
        }

        pub fn get(self: *const Self, indices: []const usize) T {
            const idx: usize = self.getFlatIdx(indices);
            return self.slice[idx];
        }

        pub fn getFlat(self: *const Self, flat_idx: usize) T {
            assert(flat_idx < self.slice.len);
            return self.slice[flat_idx];
        }

        pub fn setFlat(self: *Self, flat_idx: usize, in_val: T) void {
            assert(flat_idx < self.slice.len);
            self.slice[flat_idx] = in_val;
        }

        pub fn getFlatIdx(self: *const Self, indices: []const usize) usize {
            assert(indices.len == self.dims.len);

            var flat: usize = 0;
            for (indices, 0..) |idx, dim| {
                assert(idx < self.dims[dim]);

                flat += idx * self.strides[dim];
            }

            return flat;
        }

        pub fn planeBase(self: *const Self, idx0: usize) usize {
            assert(self.dims.len >= 1);
            assert(idx0 < self.dims[0]);
            return idx0 * self.strides[0];
        }

        pub fn subBase2(self: *const Self, idx0: usize, idx1: usize) usize {
            assert(self.dims.len >= 2);
            assert(idx0 < self.dims[0]);
            assert(idx1 < self.dims[1]);
            return idx0 * self.strides[0] + idx1 * self.strides[1];
        }

        pub fn offset2(
            self: *const Self,
            idx0: usize,
            idx1: usize,
        ) usize {
            assert(self.dims.len == 2);
            return self.subBase2(idx0, idx1);
        }

        pub fn offset3(
            self: *const Self,
            idx0: usize,
            idx1: usize,
            idx2: usize,
        ) usize {
            assert(self.dims.len == 3);
            assert(idx2 < self.dims[2]);
            return self.subBase2(idx0, idx1) + idx2;
        }

        pub fn offset4(
            self: *const Self,
            idx0: usize,
            idx1: usize,
            idx2: usize,
            idx3: usize,
        ) usize {
            assert(self.dims.len == 4);
            assert(idx0 < self.dims[0]);
            assert(idx1 < self.dims[1]);
            assert(idx2 < self.dims[2]);
            assert(idx3 < self.dims[3]);
            return idx0 * self.strides[0] +
                idx1 * self.strides[1] +
                idx2 * self.strides[2] +
                idx3;
        }

        pub fn calcFlatStride(self: *const Self, dim_idx: usize) usize {
            assert(dim_idx < self.dims.len);

            var stride: usize = 1;

            var ii: usize = dim_idx + 1;
            while (ii < self.dims.len) : (ii += 1) {
                stride *= self.dims[ii];
            }

            return stride;
        }

        pub fn fill(self: *const Self, fill_val: T) void {
            @memset(self.slice[0..], fill_val);
        }

        pub fn addInPlace(self: *const Self, to_add: *const Self) void {
            assert(matchArrayDims(T, self, to_add));

            for (0..self.slice.len) |ii| {
                self.slice[ii] += to_add.slice[ii];
            }
        }

        pub fn subInPlace(self: *const Self, to_sub: *const Self) void {
            assert(matchArrayDims(T, self, to_sub));

            for (0..self.slice.len) |ii| {
                self.slice[ii] -= to_sub.slice[ii];
            }
        }

        pub fn mulInPlace(self: *const Self, to_mul: *const Self) void {
            assert(matchArrayDims(T, self, to_mul));

            for (0..self.slice.len) |ii| {
                self.slice[ii] *= to_mul.slice[ii];
            }
        }

        pub fn divInPlace(self: *const Self, to_div: *const Self) void {
            assert(matchArrayDims(T, self, to_div));

            for (0..self.slice.len) |ii| {
                self.slice[ii] /= to_div.slice[ii];
            }
        }

        pub fn mulScalInPlace(self: *const Self, scal: T) void {
            for (0..self.slice.len) |ii| {
                self.slice[ii] *= scal;
            }
        }

        pub fn getSlice(
            self: *const Self,
            fixed_idx: []const usize,
            fixed_dim: usize,
        ) []T {
            assert(fixed_idx.len == self.dims.len);
            assert((fixed_dim + 1) < self.dims.len);

            const start = self.getFlatIdx(fixed_idx);
            const stride = self.strides[fixed_dim];
            return self.slice[start .. start + stride];
        }

        pub fn getPlaneSlice(self: *const Self, chan: usize) []T {
            const start = chan * self.strides[0];
            return self.slice[start..];
        }

        pub fn fixedPrefixView(
            self: *const Self,
            allocator: std.mem.Allocator,
            fixed_prefix: []const usize,
        ) !Self {
            assert(fixed_prefix.len < self.dims.len);

            var start: usize = 0;
            for (fixed_prefix, 0..) |idx, dim| {
                assert(idx < self.dims[dim]);
                start += idx * self.strides[dim];
            }

            const prefix_last_dim = fixed_prefix.len - 1;
            const view_len = self.strides[prefix_last_dim];
            return try Self.init(
                allocator,
                self.slice[start .. start + view_len],
                self.dims[fixed_prefix.len..],
            );
        }

        pub fn dupe(self: *const Self, allocator: std.mem.Allocator) !Self {
            const new_slice = try allocator.dupe(T, self.slice);
            errdefer allocator.free(new_slice);
            return try Self.init(allocator, new_slice, self.dims);
        }
    };
}

pub fn MappedNDArray(comptime T: type) type {
    return struct {
        array: NDArray(T),
        map: []usize,

        const Self: type = @This();

        pub fn deinit(self: *const Self, allocator: std.mem.Allocator) void {
            self.array.deinit(allocator);
            allocator.free(self.map);
        }
    };
}

pub fn NDArrayOps(comptime T: type) type {
    return struct {
        pub fn add(
            arr0: *const NDArray(T),
            arr1: *const NDArray(T),
            arr_out: *NDArray(T),
        ) void {
            assert(matchArrayDims(T, arr0, arr1));
            assert(matchArrayDims(T, arr0, arr_out));

            sliceops.add(T, arr0.slice, arr1.slice, arr_out.slice);
        }

        pub fn sub(
            arr0: *const NDArray(T),
            arr1: *const NDArray(T),
            arr_out: *NDArray(T),
        ) void {
            assert(matchArrayDims(T, arr0, arr1));
            assert(matchArrayDims(T, arr0, arr_out));

            sliceops.sub(T, arr0.slice, arr1.slice, arr_out.slice);
        }

        pub fn mulElemWise(
            arr0: *const NDArray(T),
            arr1: *const NDArray(T),
            arr_out: *NDArray(T),
        ) void {
            assert(matchArrayDims(T, arr0, arr1));
            assert(matchArrayDims(T, arr0, arr_out));

            sliceops.mul(T, arr0.slice, arr1.slice, arr_out.slice);
        }

        pub fn divElemWise(
            arr0: *const NDArray(T),
            arr1: *const NDArray(T),
            arr_out: *NDArray(T),
        ) void {
            assert(matchArrayDims(T, arr0, arr1));
            assert(matchArrayDims(T, arr0, arr_out));

            sliceops.div(T, arr0.slice, arr1.slice, arr_out.slice);
        }

        pub fn extractMat(
            allocator: std.mem.Allocator,
            arr: *const NDArray(T),
            fixed_idxs: []const usize,
            row_ext: usize,
            col_ext: usize,
            mat: *MatSlice(T),
        ) !void {
            assert(fixed_idxs.len == arr.dims.len);
            assert(row_ext <= arr.dims.len);
            assert(col_ext <= arr.dims.len);
            const num_elems: usize = arr.dims[row_ext] * arr.dims[col_ext];
            assert(num_elems == mat.slice.len);

            var get_dims_slice: []usize = try allocator.dupe(usize, fixed_idxs);
            defer allocator.free(get_dims_slice);

            for (0..arr.dims[row_ext]) |rr| {
                for (0..arr.dims[col_ext]) |cc| {
                    get_dims_slice[row_ext] = rr;
                    get_dims_slice[col_ext] = cc;

                    mat.set(rr, cc, arr.get(get_dims_slice));
                }
            }
        }
    };
}

pub fn matchArrayDims(
    comptime T: type,
    arr0: *const NDArray(T),
    arr1: *const NDArray(T),
) bool {
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

// --------------------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------------------

const testing = std.testing;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const expectApproxEqAbs = testing.expectApproxEqAbs;
const expectEqualSlices = testing.expectEqualSlices;
const talloc = testing.allocator;

test "matchArrayDims" {
    var dims0 = [_]usize{ 3, 3, 2 };
    const slice0 = try talloc.alloc(F, 18);
    defer talloc.free(slice0);
    var arr0 = try NDArray(F).init(talloc, slice0, dims0[0..]);
    defer arr0.deinit(talloc);

    var dims1 = [_]usize{ 3, 3, 2 };
    const slice1 = try talloc.alloc(F, 18);
    defer talloc.free(slice1);
    var arr1 = try NDArray(F).init(talloc, slice1, dims1[0..]);
    defer arr1.deinit(talloc);

    try expect(matchArrayDims(F, &arr0, &arr1));
}

test "getFlatIdx" {
    var dims0 = [_]usize{ 2, 3, 3 };
    const slice0 = try talloc.alloc(F, 18);
    defer talloc.free(slice0);
    var arr0 = try NDArray(F).init(talloc, slice0, dims0[0..]);
    defer arr0.deinit(talloc);

    const idxs0 = [_]usize{ 1, 2, 1 };
    const exp_idx0: usize = 16;
    const flat_idx0 = arr0.getFlatIdx(idxs0[0..]);

    try expectEqual(flat_idx0, exp_idx0);
}

test "calcFlatStride" {
    var dims0 = [_]usize{ 2, 3, 3 };
    const slice0 = try talloc.alloc(F, 18);
    defer talloc.free(slice0);
    var arr0 = try NDArray(F).init(talloc, slice0, dims0[0..]);
    defer arr0.deinit(talloc);
    const check0 = [_]usize{ 9, 3, 1 };

    for (0..3) |aa| {
        try expectEqual(arr0.strides[aa], check0[aa]);
    }
}

test "getSlice" {
    var dims0 = [_]usize{ 3, 2, 2 };
    const slice0 = try talloc.alloc(F, 12);
    defer talloc.free(slice0);
    var arr0 = try NDArray(F).init(talloc, slice0, dims0[0..]);
    defer arr0.deinit(talloc);

    var set_idxs = [_]usize{ 1, 0, 0 };
    for (0..dims0[1]) |ii| {
        for (0..dims0[2]) |jj| {
            set_idxs[1] = ii;
            set_idxs[2] = jj;
            arr0.set(set_idxs[0..], 7);
        }
    }

    var fixed_idxs = [_]usize{ 1, 0, 0 };
    const ext_slice0 = arr0.getSlice(fixed_idxs[0..], 0);

    try expectEqual(ext_slice0.len, 4);
    try expectEqual(ext_slice0[0], 7);
}

test "fixedPrefixView" {
    var dims0 = [_]usize{ 2, 3, 2, 2 };
    const slice0 = try talloc.alloc(F, 24);
    defer talloc.free(slice0);
    for (0..slice0.len) |ii| slice0[ii] = @floatFromInt(ii);
    var arr0 = try NDArray(F).init(talloc, slice0, dims0[0..]);
    defer arr0.deinit(talloc);

    const fixed = [_]usize{ 1, 2 };
    var view = try arr0.fixedPrefixView(talloc, fixed[0..]);
    defer view.deinit(talloc);

    try expectEqual(@as(usize, 2), view.dims.len);
    try expectEqual(@as(usize, 2), view.dims[0]);
    try expectEqual(@as(usize, 2), view.dims[1]);
    try expectEqual(arr0.get(&[_]usize{ 1, 2, 0, 0 }), view.get(&[_]usize{ 0, 0 }));
    try expectEqual(arr0.get(&[_]usize{ 1, 2, 1, 1 }), view.get(&[_]usize{ 1, 1 }));
}

test "offset helpers" {
    var dims3 = [_]usize{ 2, 3, 4 };
    const slice3 = try talloc.alloc(F, 2 * 3 * 4);
    defer talloc.free(slice3);
    var arr3 = try NDArray(F).init(talloc, slice3, dims3[0..]);
    defer arr3.deinit(talloc);

    var dims0 = [_]usize{ 2, 3, 4, 5 };
    const slice0 = try talloc.alloc(F, 2 * 3 * 4 * 5);
    defer talloc.free(slice0);
    var arr0 = try NDArray(F).init(talloc, slice0, dims0[0..]);
    defer arr0.deinit(talloc);

    const base0 = arr0.planeBase(1);
    const base1 = arr0.subBase2(1, 2);
    const off3 = arr3.offset3(1, 2, 3);
    const off4 = arr0.offset4(1, 2, 3, 4);

    try expectEqual(@as(usize, 60), base0);
    try expectEqual(@as(usize, 100), base1);
    try expectEqual(@as(usize, 23), off3);
    try expectEqual(@as(usize, 119), off4);
}
