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
const time = std.time;
const assert = std.debug.assert;

const Vec3f = @import("vecstack.zig").Vec3f;
const slice = @import("sliceops.zig");

const MatSlice = @import("matslice.zig").MatSlice;
const NDArray = @import("ndarray.zig").NDArray;
const csvio = @import("csvio.zig");

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const Coords = struct {
    mat: MatSlice(F),
    mem: []F,

    const Self: type = @This();

    pub fn init(mem: []F, coords_num: usize) Self {
        assert(mem.len == coords_num * 3);
        const mat_coords = MatSlice(F).init(mem, coords_num, 3);

        return .{
            .mat = mat_coords,
            .mem = mem,
        };
    }

    pub fn initAlloc(outer_alloc: std.mem.Allocator, coords_num: usize) !Self {
        const mat_mem = try outer_alloc.alloc(F, coords_num * 3);

        return init(mat_mem, coords_num);
    }

    pub inline fn x(self: *const Self, ind: usize) F {
        return self.mat.get(ind, 0);
    }

    pub inline fn y(self: *const Self, ind: usize) F {
        return self.mat.get(ind, 1);
    }

    pub inline fn z(self: *const Self, ind: usize) F {
        return self.mat.get(ind, 2);
    }

    pub fn getVecSlice(self: *const Self, ind: usize) []F {
        return self.mat.getSlice(ind);
    }

    pub fn getVec3(self: *const Coords, ind: usize) Vec3f {
        const vec_slice = self.mat.getSlice(ind);
        const vec = Vec3f.initSlice(vec_slice);
        return vec;
    }
};

pub const Connect = struct {
    table: MatSlice(usize),
    table_mem: []usize,

    const Self: type = @This();

    pub fn init(mem: []usize, elems_num: usize, nodes_per_elem: usize) Self {
        assert(mem.len == elems_num * nodes_per_elem);

        const mat_table = MatSlice(usize).init(mem, elems_num, nodes_per_elem);

        return .{
            .table = mat_table,
            .table_mem = mem,
        };
    }

    pub fn initAlloc(
        outer_alloc: std.mem.Allocator,
        elems_num: usize,
        nodes_per_elem: usize,
    ) !Self {
        const mat_mem = try outer_alloc.alloc(usize, elems_num * nodes_per_elem);

        return init(mat_mem, elems_num, nodes_per_elem);
    }

    pub inline fn getElemsNum(self: Self) usize {
        return self.table.rows_num;
    }

    pub inline fn getNodesPerElem(self: Self) usize {
        return self.table.cols_num;
    }

    pub fn deinit(self: *Self, outer_alloc: std.mem.Allocator) void {
        outer_alloc.free(self.table_mem);
    }

    pub fn getElem(self: *const Self, elem_num: usize) []usize {
        const ind_start: usize = elem_num * self.getNodesPerElem();
        const ind_end: usize = ind_start + self.getNodesPerElem();
        return self.table_mem[ind_start..ind_end];
    }
};

pub const Field = struct {
    array: NDArray(F),
    array_mem: []F,

    const Self = @This();

    pub fn initAlloc(
        outer_alloc: std.mem.Allocator,
        time_n: usize,
        coord_n: usize,
        fields_n: u8,
    ) !Self {
        const mem_array = try outer_alloc.alloc(F, time_n * coord_n * fields_n);
        @memset(mem_array, 0.0);

        const mem_dims = [3]usize{ time_n, coord_n, @as(usize, fields_n) };
        const arr = try NDArray(F).init(outer_alloc, mem_array, mem_dims[0..]);

        return .{
            .array = arr,
            .array_mem = mem_array,
        };
    }

    pub inline fn getTimeN(self: *const Self) usize {
        return self.array.dims[0];
    }
    pub inline fn getCoordN(self: *const Self) usize {
        return self.array.dims[1];
    }
    pub inline fn getFieldsN(self: *const Self) u8 {
        std.debug.assert(self.array.dims[2] <= std.math.maxInt(u8));
        return @intCast(self.array.dims[2]);
    }

    pub fn deinit(self: *Self, outer_alloc: std.mem.Allocator) void {
        outer_alloc.free(self.array_mem);
        self.array.deinit(outer_alloc);
    }
};

// --------------------------------------------------------------------------------------
// Public Entry-Point Func
// --------------------------------------------------------------------------------------

pub fn parseCoords(
    outer_alloc: std.mem.Allocator,
    csv_lines: *const std.ArrayList([]const u8),
) !Coords {
    const coord_count: usize = csv_lines.items.len;
    var coords = try Coords.initAlloc(outer_alloc, coord_count);

    const num_coords: u8 = 3;
    var num_count: u8 = 0;

    for (csv_lines.items, 0..) |line_str, ii| {
        //print("\nParsing line: {}\n", .{ii});
        var split_iter = std.mem.splitScalar(u8, line_str, ',');

        while (split_iter.next()) |num_str| {
            const num: F = try std.fmt.parseFloat(F, num_str);

            coords.mat.set(ii, num_count, num);

            num_count += 1;
            if (num_count >= num_coords) {
                num_count = 0;
                break;
            }
        }
    }

    return coords;
}

pub fn parseConnect(
    outer_alloc: std.mem.Allocator,
    csv_lines: *const std.ArrayList([]const u8),
) !Connect {
    const elem_count = csv_lines.items.len;

    var split_iter = std.mem.splitScalar(u8, csv_lines.items[0], ',');
    var nodes_per_elem: u8 = 0;
    while (split_iter.next()) |num_str| {
        _ = num_str;
        nodes_per_elem += 1;
    }

    const connect = try Connect.initAlloc(outer_alloc, elem_count, nodes_per_elem);

    var elem: usize = 0;
    var node: usize = 0;
    for (csv_lines.items, 0..) |line_str, ii| {
        elem = ii;
        node = 0;

        split_iter = std.mem.splitScalar(u8, line_str, ',');

        while (split_iter.next()) |num_str| {
            const num_f: F = try std.fmt.parseFloat(F, num_str);
            const num_i: usize = @intFromFloat(num_f);

            connect.table_mem[elem * nodes_per_elem + node] = num_i;

            node += 1;
        }
    }
    return connect;
}

pub fn getFieldTimeN(csv_lines: *const std.ArrayList([]const u8)) usize {
    var split_iter = std.mem.splitScalar(u8, csv_lines.items[0], ',');
    var time_n: usize = 0;
    while (split_iter.next()) |num_str| {
        _ = num_str;
        time_n += 1;
    }

    return time_n;
}

pub fn parseField(
    csv_lines: *const std.ArrayList([]const u8),
    field: *Field,
    field_n: u8,
) !void {

    // Each row is a coordinate
    // Each field csv has row where each column in the row is a time step
    var inds = [_]usize{ 0, 0, 0 }; // time_n,coord_n,field_n
    inds[2] = field_n;

    for (csv_lines.items, 0..) |line_str, ii| {
        inds[0] = 0; // time_n
        inds[1] = ii; // coord_n, each row is a new coord

        var split_iter = std.mem.splitScalar(u8, line_str, ',');

        while (split_iter.next()) |num_str| {
            const num_f: F = try std.fmt.parseFloat(F, num_str);

            field.array.set(inds[0..], num_f);

            inds[0] += 1; // increment time_n as we step along the row
        }
    }
}

const default_field_files = &[_][]const u8{
    "field_disp_x.csv",
    "field_disp_y.csv",
    "field_disp_z.csv",
};

pub const SimDataFiles = struct {
    coord_file: []const u8 = "coords.csv",
    connect_file: []const u8 = "connectivity.csv",
    field_files: ?[]const []const u8 = default_field_files,
    disp_files: ?[]const []const u8 = default_field_files,
};

pub const SimData = struct {
    coords: Coords,
    connect: Connect,
    field: ?Field,
    disp: ?Field,

    pub fn deinit(self: *SimData, outer_alloc: std.mem.Allocator) void {
        outer_alloc.free(self.coords.mem);
        self.connect.deinit(outer_alloc);
        if (self.field) |*ff| ff.deinit(outer_alloc);
        if (self.disp) |*dd| dd.deinit(outer_alloc);
    }
};

pub fn loadSimData(
    outer_alloc: std.mem.Allocator,
    io: std.Io,
    coord_path: []const u8,
    connect_path: []const u8,
    field_paths: ?[]const []const u8,
    disp_paths: ?[]const []const u8,
) !SimData {
    var arena = std.heap.ArenaAllocator.init(outer_alloc);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    //--------------------------------------------------------------------------
    // Read and parse coordinates csv file
    var lines = try csvio.readCsvToList(arena_alloc, io, coord_path);
    const coords = try parseCoords(outer_alloc, &lines);
    lines.clearRetainingCapacity();

    //--------------------------------------------------------------------------
    // Read and parse the connectivity table csv file
    lines = try csvio.readCsvToList(arena_alloc, io, connect_path);
    const connect = try parseConnect(outer_alloc, &lines);
    lines.clearRetainingCapacity();

    //--------------------------------------------------------------------------
    // Parse field (optional)
    var field: ?Field = null;
    if (field_paths) |fp| {
        if (fp.len > 0) {
            lines = try csvio.readCsvToList(arena_alloc, io, fp[0]);
            const time_n: usize = getFieldTimeN(&lines);
            const coord_n: usize = lines.items.len;
            std.debug.assert(fp.len <= std.math.maxInt(u8));
            field = try Field.initAlloc(outer_alloc, time_n, coord_n, @intCast(fp.len));
            try parseField(&lines, &field.?, 0);
            lines.clearRetainingCapacity();

            for (fp[1..], 1..) |path, ii| {
                lines = try csvio.readCsvToList(arena_alloc, io, path);
                try parseField(&lines, &field.?, @intCast(ii));
                lines.clearRetainingCapacity();
            }
        }
    }

    //--------------------------------------------------------------------------
    // Parse displacement (optional)
    var disp: ?Field = null;
    if (disp_paths) |dp| {
        if (dp.len > 0) {
            lines = try csvio.readCsvToList(arena_alloc, io, dp[0]);
            const time_n: usize = getFieldTimeN(&lines);
            const coord_n: usize = lines.items.len;
            std.debug.assert(dp.len <= std.math.maxInt(u8));
            disp = try Field.initAlloc(outer_alloc, time_n, coord_n, @intCast(dp.len));
            try parseField(&lines, &disp.?, 0);
            lines.clearRetainingCapacity();

            for (dp[1..], 1..) |path, ii| {
                lines = try csvio.readCsvToList(arena_alloc, io, path);
                try parseField(&lines, &disp.?, @intCast(ii));
                lines.clearRetainingCapacity();
            }
        }
    }

    return .{
        .coords = coords,
        .connect = connect,
        .field = field,
        .disp = disp,
    };
}

pub fn loadMultiSimData(
    outer_alloc: std.mem.Allocator,
    io: std.Io,
    dir_paths: []const []const u8,
    files: SimDataFiles,
) ![]SimData {
    var sim_data_slice = try outer_alloc.alloc(SimData, dir_paths.len);
    var loaded_count: usize = 0;
    errdefer {
        for (0..loaded_count) |ii| {
            sim_data_slice[ii].deinit(outer_alloc);
        }
        outer_alloc.free(sim_data_slice);
    }

    for (dir_paths, 0..) |dir_path, ii| {
        const path_coords = try std.fmt.allocPrint(
            outer_alloc,
            "{s}{s}",
            .{ dir_path, files.coord_file },
        );
        defer outer_alloc.free(path_coords);

        const path_connect = try std.fmt.allocPrint(
            outer_alloc,
            "{s}{s}",
            .{ dir_path, files.connect_file },
        );
        defer outer_alloc.free(path_connect);

        var field_paths: ?[][]const u8 = null;
        if (files.field_files) |ff| {
            field_paths = try outer_alloc.alloc([]const u8, ff.len);
            for (ff, 0..) |suffix, jj| {
                field_paths.?[jj] = try std.fmt.allocPrint(
                    outer_alloc,
                    "{s}{s}",
                    .{ dir_path, suffix },
                );
            }
        }
        defer if (field_paths) |fp| {
            for (fp) |pp| outer_alloc.free(pp);
            outer_alloc.free(fp);
        };

        var disp_paths: ?[][]const u8 = null;
        if (files.disp_files) |df| {
            disp_paths = try outer_alloc.alloc([]const u8, df.len);
            for (df, 0..) |suffix, jj| {
                disp_paths.?[jj] = try std.fmt.allocPrint(
                    outer_alloc,
                    "{s}{s}",
                    .{ dir_path, suffix },
                );
            }
        }
        defer if (disp_paths) |dp| {
            for (dp) |pp| outer_alloc.free(pp);
            outer_alloc.free(dp);
        };

        sim_data_slice[ii] = try loadSimData(
            outer_alloc,
            io,
            path_coords,
            path_connect,
            field_paths,
            disp_paths,
        );
        loaded_count += 1;
    }
    return sim_data_slice;
}

test "loadMultiSimData twoelems" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const dir_paths = [_][]const u8{
        "data/min/tri3_twoelems/",
        "data/min/tri6_twoelems/",
        "data/min/quad4_twoelems/",
        "data/min/quad8_twoelems/",
        "data/min/quad9_twoelems/",
    };

    const sim_datas = try loadMultiSimData(allocator, io, &dir_paths, .{});
    defer {
        for (sim_datas) |*sim_data| sim_data.deinit(allocator);
        allocator.free(sim_datas);
    }

    const expected_nodes = [_]usize{ 3, 6, 4, 8, 9 };

    for (sim_datas, 0..) |sim_data, ii| {
        try std.testing.expectEqual(@as(usize, 2), sim_data.connect.getElemsNum());
        try std.testing.expectEqual(
            expected_nodes[ii],
            sim_data.connect.getNodesPerElem(),
        );
        if (sim_data.field) |field| {
            try std.testing.expectEqual(sim_data.coords.mat.rows_num, field.getCoordN());
        } else {
            return error.TestExpectedField;
        }
    }
}
