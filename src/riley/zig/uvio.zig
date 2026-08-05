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
const assert = std.debug.assert;
const NDArray = @import("ndarray.zig").NDArray;
const csvio = @import("csvio.zig");

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const UVMap = struct {
    array: NDArray(F),
    buff: []F,

    const Self = @This();

    pub fn init(outer_alloc: std.mem.Allocator, nodes_num: usize) !Self {
        const buff = try outer_alloc.alloc(F, nodes_num * 2);
        @memset(buff, 0.0);

        const dims = [_]usize{ nodes_num, 2 }; // u, v

        const array = try NDArray(F).init(outer_alloc, buff, dims[0..]);

        return .{
            .array = array,
            .buff = buff,
        };
    }

    pub fn deinit(self: *Self, outer_alloc: std.mem.Allocator) void {
        self.array.deinit(outer_alloc);
        outer_alloc.free(self.buff);
    }

    pub fn getU(self: *const Self, node_idx: usize) F {
        assert(node_idx < self.array.dims[0]);
        return self.array.get(&[_]usize{ node_idx, 0 });
    }

    pub fn getV(self: *const Self, node_idx: usize) F {
        assert(node_idx < self.array.dims[0]);
        return self.array.get(&[_]usize{ node_idx, 1 });
    }

    pub fn getUV(self: *const Self, node_idx: usize) []F {
        assert(node_idx < self.array.dims[0]);
        const start = node_idx * 2;
        return self.buff[start .. start + 2];
    }

    pub fn setUV(self: *Self, node_idx: usize, u: F, v: F) void {
        assert(node_idx < self.array.dims[0]);
        self.array.set(&[_]usize{ node_idx, 0 }, u);
        self.array.set(&[_]usize{ node_idx, 1 }, v);
    }
};

// --------------------------------------------------------------------------------------
// Public Entry-Point Functions
// --------------------------------------------------------------------------------------

pub fn loadUVs(
    outer_alloc: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
) !NDArray(F) {
    const uv_arr = try csvio.loadScalarCsv2D(outer_alloc, io, path);

    if (uv_arr.dims.len != 2 or uv_arr.dims[1] != 2) {
        outer_alloc.free(uv_arr.slice);
        var tmp = uv_arr;
        tmp.deinit(outer_alloc);
        return error.InvalidCsvFormat;
    }

    return uv_arr;
}

pub fn loadUVMap(outer_alloc: std.mem.Allocator, io: std.Io, path: []const u8) !UVMap {
    const uv_arr = try loadUVs(outer_alloc, io, path);
    return UVMap{
        .array = uv_arr,
        .buff = uv_arr.slice,
    };
}

// --------------------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------------------

const testing = std.testing;

test "Load UVMap from committed tri3 sphere fixture" {
    const allocator = testing.allocator;
    const io = std.testing.io;

    const path = "data/min/tri3_sphere200/uvs.csv";
    var uv_map = try loadUVMap(allocator, io, path);
    defer uv_map.deinit(allocator);

    try testing.expectEqual(@as(usize, 256), uv_map.array.dims[0]);

    // First row: 0.4, 0.4
    try testing.expectApproxEqAbs(@as(F, 0.4), uv_map.getU(0), 1e-8);
    try testing.expectApproxEqAbs(@as(F, 0.4), uv_map.getV(0), 1e-8);

    // Last row: 0.6, 0.6
    try testing.expectApproxEqAbs(@as(F, 0.6), uv_map.getU(255), 1e-8);
    try testing.expectApproxEqAbs(@as(F, 0.6), uv_map.getV(255), 1e-8);

    const uv = uv_map.getUV(0);
    try testing.expectApproxEqAbs(@as(F, 0.4), uv[0], 1e-8);
    try testing.expectApproxEqAbs(@as(F, 0.4), uv[1], 1e-8);
}

test "Load UVMap from committed tri6 sphere fixture" {
    const allocator = testing.allocator;
    const io = std.testing.io;

    const path = "data/min/tri6_sphere200/uvs.csv";
    var uv_map = try loadUVMap(allocator, io, path);
    defer uv_map.deinit(allocator);

    try testing.expectEqual(@as(usize, 961), uv_map.array.dims[0]);
}

test "Load UVMap from committed tri3 twoelems fixture" {
    const allocator = testing.allocator;
    const io = std.testing.io;

    const path = "data/min/tri3_twoelems/uvs.csv";
    var uv_map = try loadUVMap(allocator, io, path);
    defer uv_map.deinit(allocator);

    try testing.expectEqual(@as(usize, 4), uv_map.array.dims[0]);
}

test "Load UVMap from committed tri6 twoelems fixture" {
    const allocator = testing.allocator;
    const io = std.testing.io;

    const path = "data/min/tri6_twoelems/uvs.csv";
    var uv_map = try loadUVMap(allocator, io, path);
    defer uv_map.deinit(allocator);

    try testing.expectEqual(@as(usize, 10), uv_map.array.dims[0]);
}
