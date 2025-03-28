const std = @import("std");
const print = std.debug.print;
const time = std.time;
const Instant = time.Instant;

const meshio = @import("core/meshio.zig");
const Coords = meshio.Coords;
const vector = @import("core/vector.zig");
const matrix = @import("core/matrix.zig");

const Rotation = @import("core/rotation.zig").Rotation;
const Vec3f = vector.Vec3f;
const Mat44f = matrix.Mat44f;
const Mat44Ops = matrix.Mat44Ops;

const Camera = @import("core/camera.zig").Camera;
const CameraOps = @import("core/camera.zig").CameraOps;

const Rasteriser = struct {};

pub fn main() !void {
    const print_break = [_]u8{'-'} ** 80;
    print("{s}\nZig Rasteriser\n{s}\n", .{ print_break, print_break });

    //--------------------------------------------------------------------------
    // USER INPUT VARIABLES

    // Paths to data files
    const path_data = "data/";
    const path_coords = path_data ++ "coords.csv";
    const path_connect = path_data ++ "connectivity.csv";
    const path_field = path_data ++ "field_disp_y.csv";

    print("Data paths:\n", .{});
    print("Coords: {s}\n", .{path_coords});
    print("Connect: {s}\n", .{path_connect});
    print("Field: {s}\n\n", .{path_field});

    // Camera Parameters
    const pixel_num = [_]u32{ 960, 1280 };
    const pixel_size = [_]f64{ 5.3e-3, 5.3e-3 };
    const focal_leng: f64 = 50.0;
    const cam_rot = Rotation.init(0.0, -30.0, 0.0);
    const fov_scale_factor: f64 = 1.1;
    const subsample: u8 = 2;

    //--------------------------------------------------------------------------
    // MEMORY ALLOCATORS
    const page_alloc = std.heap.page_allocator;
    //var page_alloc = std.heap.PageAllocator(std.heap.page_allocator);
    var arena = std.heap.ArenaAllocator.init(page_alloc);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    //--------------------------------------------------------------------------
    // READ DATA FROM CSV FILES
    var time_start = try Instant.now();
    var time_end = try Instant.now();

    // Read the csv files with the element connectivity table, nodal coords and
    // the field to render
    print("Reading coords, connectivity and field paths:\n{s}\n{s}\n{s}\n", .{ path_coords, path_connect, path_field });

    // Read the csv file into an array list
    time_start = try Instant.now();
    var lines = try meshio.readCsvToList(page_alloc, path_coords);
    time_end = try Instant.now();
    const time_read_coords: f64 = @floatFromInt(time_end.since(time_start));

    // Print the array list line by line
    // for (lines.items,0..) |line_str,line_num|{
    //     print("Line {}: {s}\n", .{line_num,line_str});
    // }
    print("\nCoords: read {} lines from csv.\n", .{lines.items.len});
    print("Coords: read time = {d:.3}ms\n", .{time_read_coords / time.ns_per_ms});

    // Pass the coords into a series of arrays
    const coord_count: usize = lines.items.len;
    //var coords = try arena_alloc.alloc(Vec3f, coord_count);
    var coords = try Coords.init(page_alloc, coord_count);
    defer coords.deinit();

    time_start = try Instant.now();
    try meshio.parseCoords(&lines, &coords);
    time_end = try Instant.now();
    const time_parse_coords: f64 = @floatFromInt(time_end.since(time_start));
    print("Coords: parse time = {d:.3}ms\n", .{time_parse_coords / time.ns_per_ms});

    // print("COORDS:\n",.{});
    // for (0..coords.len) |cc| {
    //     coords[cc].vecPrint();
    // }
    // print("\n",.{});

    // Parse the connectivity table into a 2D array - first clear the lines array
    lines.clearRetainingCapacity();

    // Read the csv file into an array list
    time_start = try Instant.now();
    lines = try meshio.readCsvToList(page_alloc, path_connect);
    time_end = try Instant.now();
    const time_read_connect: f64 = @floatFromInt(time_end.since(time_start));
    print("\nConnect: read {} lines from csv.\n", .{lines.items.len});
    print("Connect: read time = {d:.3}ms\n", .{time_read_connect / time.ns_per_ms});

    time_start = try Instant.now();
    const connect = try meshio.parseConnect(arena_alloc, &lines);
    time_end = try Instant.now();
    const time_parse_connect: f64 = @floatFromInt(time_end.since(time_start));
    print("Connect: elements={}, nodes per element={}\n", .{ connect.elem_n, connect.nodes_per_elem });
    print("Connect: parse time = {d:.3}ms\n", .{time_parse_connect / time.ns_per_ms});

    // print("\nCONNECT TABLE\n",.{});
    // var ii: usize = 0;
    // for (0..connect.elem_count) |ee| {
    //     print("{} : ", .{ee});
    //     for (0..connect.nodes_per_elem) |nn| {
    //         print("{}," , .{connect.table[ee*connect.nodes_per_elem+nn]});
    //         ii += 1;
    //     }
    //     print("\n",.{});
    // }

    // Parse the field data
    lines.clearRetainingCapacity();

    // Read the csv file into an array list
    time_start = try Instant.now();
    lines = try meshio.readCsvToList(page_alloc, path_field);
    time_end = try Instant.now();
    const time_read_field: f64 = @floatFromInt(time_end.since(time_start));
    print("\nField: read {} lines from csv.\n", .{lines.items.len});
    print("Field: read time = {d:.3}ms\n", .{time_read_field / time.ns_per_ms});

    time_start = try Instant.now();
    const field = try meshio.parseField(arena_alloc, &lines);
    time_end = try Instant.now();
    const time_parse_field: f64 = @floatFromInt(time_end.since(time_start));
    print("Field: coords={}, time steps={}\n", .{ field.coord_n, field.time_n });
    print("Field: parse time = {d:.3}ms\n", .{time_parse_field / time.ns_per_ms});

    //--------------------------------------------------------------------------
    // Build Camera
    print("{s}\n", .{print_break});
    const cam_pos = CameraOps.pos_fill_frame_from_rot(&coords, pixel_num, pixel_size, focal_leng, cam_rot, fov_scale_factor);
    print("Camera position:\n", .{});
    cam_pos.vecPrint();

    const roi_pos = CameraOps.roi_cent_from_coords(&coords);
    print("\nROI center position:\n", .{});
    roi_pos.vecPrint();

    const cam_data = Camera.init(pixel_num, pixel_size, cam_pos, cam_rot, roi_pos, focal_leng, subsample);
    print("\nWorld to camera matrix:\n", .{});
    cam_data.world_to_cam_mat.matPrint();

    print("{s}\n", .{print_break});

    //--------------------------------------------------------------------------
    //
    


}
