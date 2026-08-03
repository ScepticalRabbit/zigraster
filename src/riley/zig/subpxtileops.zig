// --------------------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const rops = @import("rasterops.zig");

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const SubpxTile = struct {
    subx_min: i32,
    subx_max: i32,
    suby_min: i32,
    suby_max: i32,
    overlap_start: usize = 0,
    overlap_count: usize = 0,
};

pub const SubpxOverlap = struct {
    mesh_idx: usize,
    elem_idx: usize,
    subx_min: i32,
    subx_max: i32,
    suby_min: i32,
    suby_max: i32,
};

pub const SubpxTilingOverlaps = struct {
    tiles: []SubpxTile,
    overlaps: []SubpxOverlap,

    pub fn deinit(
        self: *SubpxTilingOverlaps,
        outer_alloc: std.mem.Allocator,
    ) void {
        outer_alloc.free(self.tiles);
        outer_alloc.free(self.overlaps);
        self.* = undefined;
    }
};

// --------------------------------------------------------------------------------------
// Public Entry-Point Func
// --------------------------------------------------------------------------------------

pub fn calcTiles(
    outer_alloc: std.mem.Allocator,
    subx_min: i32,
    subx_max: i32,
    suby_min: i32,
    suby_max: i32,
    tile_size_subpx: u16,
) ![]SubpxTile {
    std.debug.assert(subx_min < subx_max);
    std.debug.assert(suby_min < suby_max);
    std.debug.assert(tile_size_subpx > 0);

    const tile_size: i32 = tile_size_subpx;
    const tiles_x = try std.math.divCeil(
        usize,
        @intCast(subx_max - subx_min),
        tile_size_subpx,
    );
    const tiles_y = try std.math.divCeil(
        usize,
        @intCast(suby_max - suby_min),
        tile_size_subpx,
    );
    const tiles_num = try std.math.mul(usize, tiles_x, tiles_y);
    const tiles = try outer_alloc.alloc(SubpxTile, tiles_num);

    for (0..tiles_y) |yy| {
        const tile_suby_min = suby_min + @as(i32, @intCast(yy)) * tile_size;
        const tile_suby_max = @min(tile_suby_min + tile_size, suby_max);
        for (0..tiles_x) |xx| {
            const tile_subx_min = subx_min + @as(i32, @intCast(xx)) * tile_size;
            const tile_subx_max = @min(tile_subx_min + tile_size, subx_max);
            tiles[yy * tiles_x + xx] = .{
                .subx_min = tile_subx_min,
                .subx_max = tile_subx_max,
                .suby_min = tile_suby_min,
                .suby_max = tile_suby_max,
            };
        }
    }

    return tiles;
}

pub fn elemBBoxToSubpx(
    elem_bbox: rops.ElemBBox,
    sub_samp: u32,
) !SubpxOverlap {
    const sub_samp_i: i32 = @intCast(sub_samp);
    return .{
        .mesh_idx = 0,
        .elem_idx = elem_bbox.elem_idx,
        .subx_min = try std.math.mul(i32, elem_bbox.x_min, sub_samp_i),
        .subx_max = try std.math.mul(i32, elem_bbox.x_max, sub_samp_i),
        .suby_min = try std.math.mul(i32, elem_bbox.y_min, sub_samp_i),
        .suby_max = try std.math.mul(i32, elem_bbox.y_max, sub_samp_i),
    };
}

pub fn binElemOverlaps(
    outer_alloc: std.mem.Allocator,
    tiles_inp: []const SubpxTile,
    elem_bboxes_by_mesh: []const []const rops.ElemBBox,
    sub_samp: u32,
) !SubpxTilingOverlaps {
    std.debug.assert(sub_samp > 0);
    const tiles = try outer_alloc.dupe(SubpxTile, tiles_inp);
    errdefer outer_alloc.free(tiles);

    var overlaps_num: usize = 0;
    for (tiles) |*tile| {
        var tile_overlaps: usize = 0;
        for (elem_bboxes_by_mesh) |elem_bboxes| {
            for (elem_bboxes) |elem_bbox| {
                const overlap = try elemBBoxToSubpx(elem_bbox, sub_samp);
                if (rangesIntersect(tile, overlap)) {
                    tile_overlaps += 1;
                }
            }
        }
        tile.overlap_start = overlaps_num;
        tile.overlap_count = tile_overlaps;
        overlaps_num = try std.math.add(usize, overlaps_num, tile_overlaps);
    }

    const overlaps = try outer_alloc.alloc(SubpxOverlap, overlaps_num);
    errdefer outer_alloc.free(overlaps);
    for (tiles) |tile| {
        var overlap_idx = tile.overlap_start;
        for (elem_bboxes_by_mesh, 0..) |elem_bboxes, mesh_idx| {
            for (elem_bboxes) |elem_bbox| {
                var overlap = try elemBBoxToSubpx(elem_bbox, sub_samp);
                if (!rangesIntersect(&tile, overlap)) continue;

                overlap.mesh_idx = mesh_idx;
                overlap.subx_min = @max(overlap.subx_min, tile.subx_min);
                overlap.subx_max = @min(overlap.subx_max, tile.subx_max);
                overlap.suby_min = @max(overlap.suby_min, tile.suby_min);
                overlap.suby_max = @min(overlap.suby_max, tile.suby_max);
                overlaps[overlap_idx] = overlap;
                overlap_idx += 1;
            }
        }
        std.debug.assert(overlap_idx == tile.overlap_start + tile.overlap_count);
    }

    return .{ .tiles = tiles, .overlaps = overlaps };
}

fn rangesIntersect(
    tile: *const SubpxTile,
    overlap: SubpxOverlap,
) bool {
    return overlap.subx_min < tile.subx_max and
        overlap.subx_max > tile.subx_min and
        overlap.suby_min < tile.suby_max and
        overlap.suby_max > tile.suby_min;
}

// --------------------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------------------

test "calcTiles partitions a sub-pixel rectangle without halo" {
    const tiles = try calcTiles(std.testing.allocator, -2, 7, -1, 5, 4);
    defer std.testing.allocator.free(tiles);

    try std.testing.expectEqual(@as(usize, 6), tiles.len);
    try std.testing.expectEqual(SubpxTile{
        .subx_min = -2,
        .subx_max = 2,
        .suby_min = -1,
        .suby_max = 3,
    }, tiles[0]);
    try std.testing.expectEqual(SubpxTile{
        .subx_min = 6,
        .subx_max = 7,
        .suby_min = 3,
        .suby_max = 5,
    }, tiles[5]);
}

test "elemBBoxToSubpx scales conservative pixel bounds" {
    const overlap = try elemBBoxToSubpx(.{
        .elem_idx = 3,
        .x_min = -2,
        .x_max = 7,
        .y_min = 1,
        .y_max = 4,
    }, 8);

    try std.testing.expectEqual(@as(i32, -16), overlap.subx_min);
    try std.testing.expectEqual(@as(i32, 56), overlap.subx_max);
    try std.testing.expectEqual(@as(i32, 8), overlap.suby_min);
    try std.testing.expectEqual(@as(i32, 32), overlap.suby_max);
}

test "binElemOverlaps gives each halo-free tile clipped overlaps" {
    const tiles = try calcTiles(std.testing.allocator, 0, 8, 0, 4, 4);
    defer std.testing.allocator.free(tiles);
    const mesh0 = [_]rops.ElemBBox{
        .{ .elem_idx = 0, .x_min = 1, .x_max = 3, .y_min = 0, .y_max = 2 },
        .{ .elem_idx = 1, .x_min = 3, .x_max = 5, .y_min = 1, .y_max = 3 },
    };
    const mesh1 = [_]rops.ElemBBox{
        .{ .elem_idx = 0, .x_min = 6, .x_max = 8, .y_min = 0, .y_max = 4 },
    };
    const meshes = [_][]const rops.ElemBBox{ &mesh0, &mesh1 };
    var tiling = try binElemOverlaps(
        std.testing.allocator,
        tiles,
        &meshes,
        1,
    );
    defer tiling.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 4), tiling.overlaps.len);
    try std.testing.expectEqual(@as(usize, 2), tiling.tiles[0].overlap_count);
    try std.testing.expectEqual(@as(usize, 2), tiling.tiles[1].overlap_count);
    try std.testing.expectEqual(@as(i32, 4), tiling.overlaps[1].subx_max);
    try std.testing.expectEqual(@as(usize, 1), tiling.overlaps[3].mesh_idx);
}
