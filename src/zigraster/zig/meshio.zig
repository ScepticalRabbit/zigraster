const std = @import("std");
const print = std.debug.print;
const Vec3f = @import("vecstack.zig").Vec3f;
const slice = @import("sliceops.zig");

const MatSlice = @import("matslice.zig").MatSlice;

pub const Coords = struct {
    x: []f64,
    y: []f64,
    z: []f64,
    len: usize,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, coord_n: usize) !Coords {
        return .{
            .x = try allocator.alloc(f64, coord_n),
            .y = try allocator.alloc(f64, coord_n),
            .z = try allocator.alloc(f64, coord_n),
            .len = coord_n,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Coords) void {
        self.allocator.free(self.x);
        self.allocator.free(self.y);
        self.allocator.free(self.z);
    }

    pub fn getVec3(self: *const Coords, ind: usize) Vec3f {
        var vec: Vec3f = undefined;
        vec.set(0, self.x[ind]);
        vec.set(1, self.y[ind]);
        vec.set(2, self.z[ind]);
        return vec;
    }
};

pub const Connect = struct {
    nodes_per_elem: u8,
    elem_n: usize,
    table: []usize,

    const Self: type = @This();

    pub fn getElem(self: *const Self, elem_num: usize) []usize {
        const ind_start: usize = elem_num * self.nodes_per_elem;
        const ind_end: usize = ind_start + self.nodes_per_elem;
        return self.table[ind_start..ind_end];
    }

    pub fn getInd(self: *const Self, elem_num: usize, node_num: usize) usize {
        return self.table[elem_num * self.nodes_per_elem + node_num];
    }
};

// TODO:
// - Need to have this store frames with all components
// - Maybe we use an array list of MatAlloc?

pub const Field = struct {
    coord_n: usize,
    time_n: usize,
    data: MatSlice(f64),
    buffer: []f64,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(alloc: std.mem.Allocator, coord_n: usize, time_n: usize) !Self {
        const buff = try alloc.alloc(f64, coord_n*time_n);

        return .{
            .coord_n = coord_n,
            .time_n = time_n,
            .data = try MatSlice(f64).init(buff, coord_n, time_n),
            .buffer = buff,
            .allocator = alloc,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.buffer);
    }
};

pub fn readCsvToList(allocator: std.mem.Allocator, path: []const u8) !std.ArrayList([]const u8) {
    var file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    var buffered_reader = std.io.bufferedReader(file.reader());
    var reader = buffered_reader.reader();
    var buffer: [4096]u8 = undefined;

    var lines = std.ArrayList([]const u8).init(allocator);

    while (true) {
        const line = try reader.readUntilDelimiterOrEof(&buffer, '\n');

        if (line) |line_str| {
            const line_copy = try allocator.alloc(u8, line_str.len);
            std.mem.copyForwards(u8, line_copy, line_str);
            try lines.append(line_copy);
        } else {
            break;
        }
    }

    return lines;
}

pub fn parseCoords(csv_lines: *const std.ArrayList([]const u8), coords: *Coords) !void {
    const num_coords: u8 = 3;
    var num_count: u8 = 0;

    for (csv_lines.items, 0..) |line_str, ii| {
        //print("\nParsing line: {}\n", .{ii});
        var split_iter = std.mem.splitScalar(u8, line_str, ',');

        while (split_iter.next()) |num_str| {
            const num: f64 = try std.fmt.parseFloat(f64, num_str);

            if (num_count == 0) {
                coords.x[ii] = num;
            } else if (num_count == 1) {
                coords.y[ii] = num;
            } else if (num_count == 2) {
                coords.z[ii] = num;
            }

            num_count += 1;
            if (num_count >= num_coords) {
                num_count = 0;
                break;
            }
        }
    }
}

// TODO: fix this so that connect has an allocator and passes back a reference and does not return the
// large connectivity table by copying!
pub fn parseConnect(allocator: std.mem.Allocator, csv_lines: *const std.ArrayList([]const u8)) !Connect {
    // Get the number of elements and the number of nodes per element
    const elem_count = csv_lines.items.len;

    var split_iter = std.mem.splitScalar(u8, csv_lines.items[0], ',');
    var nodes_per_elem: u8 = 0;
    while (split_iter.next()) |num_str| {
        _ = num_str;
        nodes_per_elem += 1;
    }
    // print("Connect: total elements = {}\n", .{elem_count});
    // print("Connect: nodes per element = {}\n",.{nodes_per_elem});

    const connect = Connect{
        .elem_n = elem_count,
        .nodes_per_elem = nodes_per_elem,
        .table = try allocator.alloc(usize, elem_count * nodes_per_elem),
    };

    // print("Connect: connect.len={}\n",.{connect.table.len});

    var elem: usize = 0;
    var node: usize = 0;
    for (csv_lines.items, 0..) |line_str, ii| {
        elem = ii;
        node = 0;

        split_iter = std.mem.splitScalar(u8, line_str, ',');

        while (split_iter.next()) |num_str| {
            const num_f: f64 = try std.fmt.parseFloat(f64, num_str);
            const num_i: usize = @intFromFloat(num_f);

            connect.table[elem * nodes_per_elem + node] = num_i;

            // print("Line: {}, num str: {s}, usize: {}\n", .{ii,num_str,num_i});

            node += 1;
        }
        //print("\n",.{});

    }

    return connect;
}

pub fn parseField(allocator: std.mem.Allocator, csv_lines: *const std.ArrayList([]const u8)) !Field {
    const coord_n = csv_lines.items.len;

    var split_iter = std.mem.splitScalar(u8, csv_lines.items[0], ',');
    var time_n: usize = 0;
    while (split_iter.next()) |num_str| {
        _ = num_str;
        time_n += 1;
    }

    // print("Field: total pts = {}\n", .{coord_n});
    // print("Field: total time steps = {}\n",.{time_n});

    var field = try Field.init(allocator, coord_n, time_n);

    var coord: usize = 0;
    var time_step: usize = 0;
    for (csv_lines.items, 0..) |line_str, ii| {
        coord = ii;
        time_step = 0;

        split_iter = std.mem.splitScalar(u8, line_str, ',');

        while (split_iter.next()) |num_str| {
            const num_f: f64 = try std.fmt.parseFloat(f64, num_str);
            field.data.set(coord, time_step, num_f);
            //field.data[coord * time_n + time_step] = num_f;
            time_step += 1;
        }
    }

    return field;
}
