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
const NDArray = @import("ndarray.zig").NDArray;


// --------------------------------------------------------------------------------------
// Public Entry-Point Func
// --------------------------------------------------------------------------------------

pub fn readCsvToList(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
) !std.ArrayList([]const u8) {
    const cwd = std.Io.Dir.cwd();
    var file = try cwd.openFile(io, path, .{ .mode = .read_only });
    defer file.close(io);

    var read_buf: [8 * 1024 * 1024]u8 = undefined;
    var file_reader = file.reader(io, &read_buf);
    const reader = &file_reader.interface;

    var lines: std.ArrayList([]const u8) = .empty;
    while (try reader.takeDelimiter('\n')) |line| {
        const line_trimmed = std.mem.trim(u8, line, " \r\t");
        if (line_trimmed.len == 0) continue;
        const line_dup = try allocator.dupe(u8, line_trimmed);
        try lines.append(allocator, line_dup);
    }

    return lines;
}

pub fn freeCsvLines(
    allocator: std.mem.Allocator,
    lines: *std.ArrayList([]const u8),
) void {
    for (lines.items) |line| allocator.free(line);
    lines.deinit(allocator);
}

pub fn countCsvCols(line: []const u8) usize {
    var cols_num: usize = 0;
    var iter = std.mem.splitScalar(u8, line, ',');
    while (iter.next()) |cell| {
        if (cell.len > 0) cols_num += 1;
    }
    return cols_num;
}

pub fn hasPackedChannels(line: []const u8) bool {
    var iter = std.mem.splitScalar(u8, line, ',');
    while (iter.next()) |cell| {
        if (cell.len == 0) continue;
        return std.mem.indexOfScalar(u8, cell, ':') != null;
    }
    return false;
}


pub fn loadScalarCsv2D(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
) !NDArray(F) {
    var lines = try readCsvToList(allocator, io, path);
    defer freeCsvLines(allocator, &lines);
    return loadScalarCsv2DFromLines(allocator, lines.items);
}

pub fn loadScalarCsv2DFromLines(
    allocator: std.mem.Allocator,
    lines: []const []const u8,
) !NDArray(F) {
    if (lines.len == 0) return error.EmptyCsv;

    const rows_num = lines.len;
    const cols_num = countCsvCols(lines[0]);
    var array = try NDArray(F).initFlat(
        allocator,
        &[_]usize{ rows_num, cols_num },
    );
    errdefer {
        allocator.free(array.slice);
        array.deinit(allocator);
    }

    for (lines, 0..) |line, rr| {
        var iter = std.mem.splitScalar(u8, line, ',');
        var cc: usize = 0;
        while (iter.next()) |cell| {
            if (cell.len == 0) continue;
            if (cc >= cols_num) return error.CSVColsMismatch;
            array.set(&[_]usize{ rr, cc }, try parseCellFloat(cell));
            cc += 1;
        }
        if (cc != cols_num) return error.CSVColsMismatch;
    }

    return array;
}

fn parseCellFloat(cell: []const u8) !F {
    return std.fmt.parseFloat(F, std.mem.trim(u8, cell, " \r\n\t"));
}

pub fn loadPackedCsv2D(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    channels: usize,
) !NDArray(F) {
    var lines = try readCsvToList(allocator, io, path);
    defer freeCsvLines(allocator, &lines);
    return loadPackedCsv2DFromLines(allocator, lines.items, channels);
}

pub fn loadPackedCsv2DFromLines(
    allocator: std.mem.Allocator,
    lines: []const []const u8,
    channels: usize,
) !NDArray(F) {
    if (lines.len == 0) return error.EmptyCsv;

    const rows_num = lines.len;
    const cols_num = countCsvCols(lines[0]);
    var array = try NDArray(F).initFlat(
        allocator,
        &[_]usize{ rows_num, cols_num, channels },
    );
    errdefer {
        allocator.free(array.slice);
        array.deinit(allocator);
    }

    for (lines, 0..) |line, rr| {
        var cell_iter = std.mem.splitScalar(u8, line, ',');
        var cc: usize = 0;
        while (cell_iter.next()) |cell| {
            if (cell.len == 0) continue;
            if (cc >= cols_num) return error.CSVColsMismatch;

            var ch_iter = std.mem.splitScalar(u8, cell, ':');
            for (0..channels) |ch| {
                const ch_cell = ch_iter.next() orelse
                    return error.CSVChannelsMismatch;
                array.set(
                    &[_]usize{ rr, cc, ch },
                    try parseCellFloat(ch_cell),
                );
            }
            cc += 1;
        }
        if (cc != cols_num) return error.CSVColsMismatch;
    }

    return array;
}

pub fn saveScalarGridCSV(
    io: std.Io,
    out_dir: std.Io.Dir,
    file_name: []const u8,
    rows_num: usize,
    cols_num: usize,
    ctx: anytype,
    comptime getVal: anytype,
) !void {
    const csv_file = try out_dir.createFile(io, file_name, .{});
    defer csv_file.close(io);

    var write_buf: [4096]u8 = undefined;
    var file_writer = csv_file.writer(io, &write_buf);
    const writer = &file_writer.interface;

    for (0..rows_num) |rr| {
        for (0..cols_num) |cc| {
            try writer.print("{d}", .{getVal(ctx, rr, cc)});
            try writer.writeAll(",");
        }
        try writer.writeAll("\n");
    }

    try file_writer.flush();
}

pub fn savePackedGridCSV(
    io: std.Io,
    out_dir: std.Io.Dir,
    file_name: []const u8,
    rows_num: usize,
    cols_num: usize,
    channels_num: usize,
    ctx: anytype,
    comptime getVal: anytype,
) !void {
    const csv_file = try out_dir.createFile(io, file_name, .{});
    defer csv_file.close(io);

    var write_buf: [4096]u8 = undefined;
    var file_writer = csv_file.writer(io, &write_buf);
    const writer = &file_writer.interface;

    for (0..rows_num) |rr| {
        for (0..cols_num) |cc| {
            for (0..channels_num) |ch| {
                try writer.print("{d}", .{getVal(ctx, rr, cc, ch)});
                if (ch + 1 < channels_num) try writer.writeAll(":");
            }
            if (cc + 1 < cols_num) try writer.writeAll(",");
        }
        try writer.writeAll("\n");
    }

    try file_writer.flush();
}
