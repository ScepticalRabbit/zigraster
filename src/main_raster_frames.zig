const std = @import("std");
const print = std.debug.print;
const time = std.time;
const Instant = time.Instant;

const meshio = @import("zigraster/zig/meshio.zig");

const Coords = meshio.Coords;
const Connect = meshio.Connect;
const Field = meshio.Field;
const VecStack = @import("zigraster/zig/vecstack.zig");
const MatStack = @import("zigraster/zig/matstack.zig");

const Rotation = @import("zigraster/zig/rotation.zig").Rotation;
const Vec3f = VecStack.Vec3f;
const Mat44f = MatStack.Mat44f;
const Mat44Ops = MatStack.Mat44Ops;

const matslice = @import("zigraster/zig/matslice.zig");
const MatSlice = matslice.MatSlice;
const MatSliceOps = matslice.MatSliceOps;

const Camera = @import("zigraster/zig/camera.zig").Camera;
const CameraOps = @import("zigraster/zig/camera.zig").CameraOps;

const Raster = @import("zigraster/zig/raster.zig").Raster;

pub fn main() !void {
    const print_break = [_]u8{'-'} ** 80;
    print("{s}\nZig Rasteriser\n{s}\n", .{ print_break, print_break });

    //--------------------------------------------------------------------------
    // USER INPUT VARIABLES

    // Paths to data files
    const path_data = "data/block/";

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
    const alpha_z: f64 = std.math.degreesToRadians(0.0);
    const beta_y: f64 = std.math.degreesToRadians(-30.0);
    const gamma_x: f64 = std.math.degreesToRadians(-10.0);
    const cam_rot = Rotation.init(alpha_z, beta_y, gamma_x);
    const fov_scale_factor: f64 = 1.1;
    const subsample: u8 = 2;

    //--------------------------------------------------------------------------
    // MEMORY ALLOCATORS
    const page_alloc = std.heap.page_allocator;

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
    defer coords.deinit(page_alloc);

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
    print("Field: parse time = {d:.3}ms\n\n", .{time_parse_field / time.ns_per_ms});

    //--------------------------------------------------------------------------
    // Build Camera
    print("{s}\n", .{print_break});
    const roi_pos = CameraOps.roi_cent_from_coords(&coords);
    print("\nROI center position:\n", .{});
    roi_pos.vecPrint();

    const cam_pos = CameraOps.pos_fill_frame_from_rot(&coords, pixel_num, pixel_size, focal_leng, cam_rot, fov_scale_factor);
    print("Camera position:\n", .{});
    cam_pos.vecPrint();

    const camera = Camera.init(pixel_num, pixel_size, cam_pos, cam_rot, roi_pos, focal_leng, subsample);

    print("\nWorld to camera matrix:\n", .{});
    camera.world_to_cam_mat.matPrint();

    print("{s}\n", .{print_break});

    //--------------------------------------------------------------------------
    // Raster Frame
    // print("\n",.{});
    // print("connect.elem_n={any}\n",.{connect.elem_n});
    // print("connect.nodes_per_elem={any}\n",.{connect.nodes_per_elem});
    // print("\n",.{});

    const cwd = std.fs.cwd();

    const dir_name = "raster-out";

    cwd.makeDir(dir_name) catch |err| switch (err) {
        error.PathAlreadyExists => {}, // Path exists do nothing
        else => return err, // Propagate any other error
    };

    var out_dir = try cwd.openDir(dir_name, .{});
    defer out_dir.close();

    time_start = try Instant.now();

    const image_array = try Raster.rasterAllFrames(arena_alloc, out_dir, &coords, &connect, &field, &camera);

    time_end = try Instant.now();
    const time_raster: f64 = @floatFromInt(time_end.since(time_start));
    print("Total raster time = {d:.3}ms\n\n", .{time_raster / time.ns_per_ms});

    // Print diagnostics to console to see if there is an image
    const image_max = std.mem.max(f64, image_array.elems);
    const image_min = std.mem.min(f64, image_array.elems);
    print("Image: [max, min] = [{}, {}]\n\n", .{ image_max, image_min });

    //--------------------------------------------------------------------------
    // Save csv of image file for analysis
    // const cwd = std.fs.cwd();

    // const dir_name = "raster-out";
    // const image_name = "image.csv";

    // cwd.makeDir(dir_name) catch |err| switch (err) {
    //     error.PathAlreadyExists => {}, // Path exists do nothing
    //     else => return err, // Propagate any other error
    // };

    // var out_dir = try cwd.openDir(dir_name, .{});
    // defer out_dir.close();

    // print("Saving output image to: {s}\n", .{dir_name});

    // time_start = try Instant.now();
    // try image_out_buff.saveCSV(out_dir, image_name);
    // time_end = try Instant.now();

    // const time_save_image: f64 = @floatFromInt(time_end.since(time_start));
    // print("Image buffer save time = {d:.3} ms\n\n", .{
    //     time_save_image / time.ns_per_ms,
    // });

    //--------------------------------------------------------------------------
    // Save csv files of subpx buffers for analysis
    // const image_subpx_name = "image_subpx.csv";
    // const depth_name = "depth.csv";

    // time_start = try Instant.now();
    // try image_subpx.image.saveCSV(out_dir, image_subpx_name);
    // time_end = try Instant.now();

    // const time_save_subimage: f64 = @floatFromInt(time_end.since(time_start));

    // time_start = try Instant.now();
    // try image_subpx.depth.saveCSV(out_dir, depth_name);
    // time_end = try Instant.now();

    // const time_save_depth: f64 = @floatFromInt(time_end.since(time_start));
    // print("Image, depth subpx save time = {d:.3}, {d:.3} ms\n", .{time_save_subimage / time.ns_per_ms, time_save_depth / time.ns_per_ms});

}
