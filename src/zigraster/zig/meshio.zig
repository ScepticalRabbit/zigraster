const std = @import("std");
const print = std.debug.print;
const Vec3f = @import("vecstack.zig").Vec3f;
const slice = @import("sliceops.zig");

const MatSlice = @import("matslice.zig").MatSlice;
const NDArray = @import("ndarray.zig").NDArray;


// TODO: this should wrap a MatSlice and allocate a buffer
pub const Coords = struct {
    x: []f64,
    y: []f64,
    z: []f64,
    len: usize,

    pub fn init(allocator: std.mem.Allocator, coord_n: usize) !Coords {
        return .{
            .x = try allocator.alloc(f64, coord_n),
            .y = try allocator.alloc(f64, coord_n),
            .z = try allocator.alloc(f64, coord_n),
            .len = coord_n,
        };
    }

    pub fn deinit(self: *Coords, allocator: std.mem.Allocator) void {
        allocator.free(self.x);
        allocator.free(self.y);
        allocator.free(self.z);
    }

    pub fn getVec3(self: *const Coords, ind: usize) Vec3f {
        var vec: Vec3f = undefined;
        vec.set(0, self.x[ind]);
        vec.set(1, self.y[ind]);
        vec.set(2, self.z[ind]);
        return vec;
    }
};

// TODO; this should wrap a MatSlice and allocate a buffer. Note: Row major so 
// we need to have dims=[elem_num,node_nums]
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


pub const Field = struct {
    array: NDArray(f64),
    buffer_dims: []usize,
    buffer_array: []f64,

    const Self = @This();

    pub fn init(alloc: std.mem.Allocator, time_n: usize, coord_n: usize,
                fields_n: usize) !Self {

        const buff_array = try alloc.alloc(f64, time_n*coord_n*fields_n);
        @memset(buff_array,0.0);
        
        var buff_dims = try alloc.alloc(usize,3);
        buff_dims[0] = time_n;
        buff_dims[1] = coord_n;
        buff_dims[2] = fields_n;

        const arr = try NDArray(f64).init(alloc,buff_array,buff_dims);
        
        return .{
            .array = arr,
            .buffer_dims = buff_dims, 
            .buffer_array = buff_array,
        };
    }

    pub fn getTimeN(self: *const Self) usize {return self.buffer_dims[0];}
    pub fn getCoordN(self: *const Self) usize {return self.buffer_dims[1];}
    pub fn getFieldsN(self: *const Self) usize {return self.buffer_dims[2];}
   
    pub fn deinit(self: *Self, alloc: std.mem.Allocator) void {
        self.array.deinit(alloc);
        alloc.free(self.buffer);
    }
};

// TODO: should probably pass in an io struct here
// NOTE: fixed for 0.16-dev to init io
pub fn readCsvToList(allocator: std.mem.Allocator, 
                     path: []const u8
                    ) !std.ArrayList([]const u8) {

    var file = try std.fs.cwd().openFile(path, .{ .mode = .read_only});
    defer file.close();

    var single_thread_io: std.Io.Threaded = .init_single_threaded;
    const io = single_thread_io.io();

    var read_buff: [4096]u8 = undefined;    
    var reader = file.reader(io,&read_buff); // type: std.fs.File.Reader
	
    var lines: std.ArrayList([] const u8)  = .{};

    // Read lines without the trailing '\n' (exclusive).
    while (try reader.interface.takeDelimiter('\n')) |line| {
        // Optional: trim Windows '\r'
        const clean = if (@import("builtin").os.tag == .windows)
            std.mem.trimRight(u8, line, "\r")
        else
            line;

        const copy = try allocator.dupe(u8, clean);
        try lines.append(allocator,copy);
    } 
    
    return lines;
}

pub fn parseCoords(csv_lines: *const std.ArrayList([]const u8), 
                   coords: *Coords) !void {
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

// TODO: fix this so that connect has an allocator and passes back a reference 
// and does not return the large connectivity table by copying!
pub fn parseConnect(allocator: std.mem.Allocator, 
                    csv_lines: *const std.ArrayList([]const u8)) !Connect {
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

pub fn getFieldTimeN(csv_lines: *const std.ArrayList([]const u8)) usize {

    var split_iter = std.mem.splitScalar(u8, csv_lines.items[0], ',');
    var time_n: usize = 0;
    while (split_iter.next()) |num_str| {
        _ = num_str;
        time_n += 1;
    }

    return time_n;
}

pub fn parseField(csv_lines: *const std.ArrayList([]const u8), 
                  field: *Field,
                  field_n: usize) !void {

    // Each row is a coordinate
    // Each field csv has row where each column in the row is a time step
    var inds = [_]usize{0,0,0}; // time_n,coord_n,field_n
    inds[2] = field_n;

    for (csv_lines.items, 0..) |line_str, ii| {
        inds[0] = 0;     // time_n
        inds[1] = ii;    // coord_n, each row is a new coord

        var split_iter = std.mem.splitScalar(u8, line_str, ',');

        while (split_iter.next()) |num_str| {
            
            const num_f: f64 = try std.fmt.parseFloat(f64, num_str);
            
            try field.array.set(inds[0..],num_f);
          
            inds[0] += 1; // increment time_n as we step along the row
        }
    }
}
