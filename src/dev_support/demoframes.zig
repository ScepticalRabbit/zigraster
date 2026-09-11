// --------------------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");

const meshio = @import("../riley/zig/meshio.zig");

// --------------------------------------------------------------------------------------
// Public Entry-Point Functions
// --------------------------------------------------------------------------------------

pub fn firstLastIndices(
    outer_alloc: std.mem.Allocator,
    frames_num: usize,
) ![]usize {
    if (frames_num == 0) return error.NoFrames;

    const selected_num: usize = if (frames_num == 1) 1 else 2;
    const indices = try outer_alloc.alloc(usize, selected_num);
    indices[0] = 0;
    if (selected_num == 2) indices[1] = frames_num - 1;
    return indices;
}

pub fn evenlySpacedIndices(
    outer_alloc: std.mem.Allocator,
    frames_num: usize,
    frames_max: usize,
) ![]usize {
    if (frames_num == 0) return error.NoFrames;
    if (frames_max == 0) return error.ZeroFrameLimit;

    const selected_num = @min(frames_num, frames_max);
    const indices = try outer_alloc.alloc(usize, selected_num);
    if (selected_num == 1) {
        indices[0] = 0;
        return indices;
    }

    for (0..selected_num) |ii| {
        indices[ii] = ii * (frames_num - 1) / (selected_num - 1);
    }
    return indices;
}

pub fn selectFieldFrames(
    outer_alloc: std.mem.Allocator,
    field: *const meshio.Field,
    frame_indices: []const usize,
) !meshio.Field {
    if (frame_indices.len == 0) return error.NoFrames;

    var selected = try meshio.Field.initAlloc(
        outer_alloc,
        frame_indices.len,
        field.getCoordN(),
        field.getFieldsN(),
    );
    errdefer selected.deinit(outer_alloc);

    for (frame_indices, 0..) |source_frame, target_frame| {
        if (source_frame >= field.getTimeN()) return error.FrameOutOfBounds;
        for (0..field.getCoordN()) |nn| {
            for (0..field.getFieldsN()) |ff| {
                const value = field.array.get(
                    &[_]usize{ source_frame, nn, ff },
                );
                selected.array.set(
                    &[_]usize{ target_frame, nn, ff },
                    value,
                );
            }
        }
    }
    return selected;
}

// --------------------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------------------

test "first and last indices retain both endpoints" {
    const alloc = std.testing.allocator;
    const indices = try firstLastIndices(alloc, 64);
    defer alloc.free(indices);

    try std.testing.expectEqualSlices(usize, &[_]usize{ 0, 63 }, indices);
}

test "even frame selection caps and spans the source sequence" {
    const alloc = std.testing.allocator;
    const indices = try evenlySpacedIndices(alloc, 100, 8);
    defer alloc.free(indices);

    try std.testing.expectEqualSlices(
        usize,
        &[_]usize{ 0, 14, 28, 42, 56, 70, 84, 99 },
        indices,
    );
}

test "even frame selection retains short sequences" {
    const alloc = std.testing.allocator;
    const indices = try evenlySpacedIndices(alloc, 3, 8);
    defer alloc.free(indices);

    try std.testing.expectEqualSlices(usize, &[_]usize{ 0, 1, 2 }, indices);
}

test "field selection copies the requested source frames" {
    const alloc = std.testing.allocator;
    var field = try meshio.Field.initAlloc(alloc, 4, 2, 1);
    defer field.deinit(alloc);
    for (0..4) |frame_idx| {
        for (0..2) |node_idx| {
            field.array.set(
                &[_]usize{ frame_idx, node_idx, 0 },
                @floatFromInt(10 * frame_idx + node_idx),
            );
        }
    }

    var selected = try selectFieldFrames(alloc, &field, &[_]usize{ 0, 3 });
    defer selected.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 2), selected.getTimeN());
    try std.testing.expectEqual(@as(f64, 0.0), selected.array.get(&.{ 0, 0, 0 }));
    try std.testing.expectEqual(@as(f64, 1.0), selected.array.get(&.{ 0, 1, 0 }));
    try std.testing.expectEqual(@as(f64, 30.0), selected.array.get(&.{ 1, 0, 0 }));
    try std.testing.expectEqual(@as(f64, 31.0), selected.array.get(&.{ 1, 1, 0 }));
}
