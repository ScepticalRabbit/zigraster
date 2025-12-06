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

const ndarray = @import("zigraster/zig/ndarray.zig");
const NDArray = ndarray.NDArray;

const Camera = @import("zigraster/zig/camera.zig").Camera;
const CameraOps = @import("zigraster/zig/camera.zig").CameraOps;

const Raster = @import("zigraster/zig/raster.zig").Raster;

pub const SimData = struct {
    coords: Coords,
    connect: Connect,
    field: Field,
};

pub fn main() !void {
    const print_break = [_]u8{'-'} ** 80;
    print("{s}\nZig Software Rasteriser\n{s}\n", .{ print_break, print_break });

    var time_start = try Instant.now();
    var time_end = try Instant.now();

    //==========================================================================
    // MEMORY ALLOCATORS
    const page_alloc = std.heap.page_allocator;
    
    var sim_arena = std.heap.ArenaAllocator.init(page_alloc);
    defer sim_arena.deinit();
    const sim_alloc = sim_arena.allocator();

    // TODO: convert setup to a function that returns a SimData object - this 
    // setup function can then be reused between all the different scripts.
    //==========================================================================
    // SETUP: load simulation data from file
    const sim_data = setup: {
        var setup_arena = std.heap.ArenaAllocator.init(page_alloc);
        defer setup_arena.deinit();
        const setup_alloc = setup_arena.allocator();
    
        //----------------------------------------------------------------------
        // USER INPUT VARIABLES
        // Paths to data files
        const path_data = "data/block/";

        const path_coords = path_data ++ "coords.csv";
        const path_connect = path_data ++ "connectivity.csv";

        const path_field_x = path_data ++ "field_disp_x.csv";
        const path_field_y = path_data ++ "field_disp_y.csv";
        const path_field_z = path_data ++ "field_disp_z.csv";
        const field_n: usize = 3; // VECTOR FIELD 

        print("Data paths:\n", .{});
        print("Coords: {s}\n", .{path_coords});
        print("Connect: {s}\n", .{path_connect});
        print("Field, x: {s}\n", .{path_field_x});
        print("Field, y: {s}\n", .{path_field_y});
        print("Field, z: {s}\n", .{path_field_z});
        
        //----------------------------------------------------------------------
        // 1. Read and parse coordinates csv file

        // Read the csv file into an array list
        time_start = try Instant.now();
        var lines = try meshio.readCsvToList(setup_alloc, path_coords);
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
        var coords = try Coords.init(sim_alloc, coord_count);
        
        time_start = try Instant.now();
        try meshio.parseCoords(&lines, &coords);
        time_end = try Instant.now();
        const time_parse_coords: f64 = @floatFromInt(time_end.since(time_start));
        print("Coords: parse time = {d:.3}ms\n", .{time_parse_coords / time.ns_per_ms});

        // print("COORDS:\n",.{});
        // for (0..coords.len) |cc| {
        //     coords.getVec3(cc).vecPrint();
        // }
        // print("\n",.{});

        // Clear the lines array for next read
        lines.clearRetainingCapacity();

        //----------------------------------------------------------------------
        // 2. Read and parse the connectivity table

        // Read the csv file into an array list
        time_start = try Instant.now();
        lines = try meshio.readCsvToList(setup_alloc, path_connect);
        time_end = try Instant.now();
        const time_read_connect: f64 = @floatFromInt(time_end.since(time_start));
        print("\nConnect: read {} lines from csv.\n", .{lines.items.len});
        print("Connect: read time = {d:.3}ms\n", .{time_read_connect / time.ns_per_ms});

        time_start = try Instant.now();
        const connect = try meshio.parseConnect(sim_alloc, &lines);
        time_end = try Instant.now();
        const time_parse_connect: f64 = @floatFromInt(time_end.since(time_start));
        print("Connect: elements={}, nodes per element={}\n", 
           .{ connect.elem_n, connect.nodes_per_elem });
        print("Connect: parse time = {d:.3}ms\n", .{time_parse_connect / time.ns_per_ms});

        // print("\nCONNECT TABLE\n",.{});
        // var ii: usize = 0;
        // for (0..connect.elem_n) |ee| {
        //     print("{} : ", .{ee});
        //     for (0..connect.nodes_per_elem) |nn| {
        //         print("{}," , .{connect.table[ee*connect.nodes_per_elem+nn]});
        //         ii += 1;
        //     }
        //     print("\n",.{});
        // }

        lines.clearRetainingCapacity();

        //----------------------------------------------------------------------
        // 3.1 Parse X displacement field

        // Read the csv for the first field as this will tell us how many time steps
        // we have and how many coords to pre-alloc our field struct
        time_start = try Instant.now();
        lines = try meshio.readCsvToList(setup_alloc, path_field_x);
        time_end = try Instant.now();
        var time_read_field: f64 = @floatFromInt(time_end.since(time_start));
        print("\nField.x: read {} lines from csv.\n", .{lines.items.len});
        print("Field.x: read time = {d:.3}ms\n", .{time_read_field / time.ns_per_ms});
                         
        // Create the field struct to hold all the data
        const time_n: usize = meshio.getFieldTimeN(&lines);
        const coord_n: usize = lines.items.len;
        var field = try meshio.Field.init(sim_alloc,time_n,coord_n,field_n);   

        // Parse the first field 
        time_start = try Instant.now();
        try meshio.parseField(&lines,&field,0);
        time_end = try Instant.now();
        var time_parse_field: f64 = @floatFromInt(time_end.since(time_start));
        print("Field.x: coords={}, time steps={}\n", .{ field.getCoordN(), field.getTimeN() });
        print("Field.x: parse time = {d:.3}ms\n", .{time_parse_field / time.ns_per_ms});

        lines.clearRetainingCapacity();

        //----------------------------------------------------------------------
        // 3.2 Parse Y displacement field 

        time_start = try Instant.now();
        lines = try meshio.readCsvToList(setup_alloc, path_field_y);
        time_end = try Instant.now();
        time_read_field = @floatFromInt(time_end.since(time_start));
        print("\nField.y: read {} lines from csv.\n", .{lines.items.len});
        print("Field.y: read time = {d:.3}ms\n", .{time_read_field / time.ns_per_ms});

        time_start = try Instant.now();
        try meshio.parseField(&lines,&field,1);
        time_end = try Instant.now();
        time_parse_field = @floatFromInt(time_end.since(time_start));
        print("Field.y: parse time = {d:.3}ms\n", .{time_parse_field / time.ns_per_ms});

        lines.clearRetainingCapacity();

        //----------------------------------------------------------------------
        // 3.3 Parse Z displacement fields

        lines.clearRetainingCapacity();
        time_start = try Instant.now();
        lines = try meshio.readCsvToList(setup_alloc, path_field_z);
        time_end = try Instant.now();
        time_read_field = @floatFromInt(time_end.since(time_start));
        print("\nField.z: read {} lines from csv.\n", .{lines.items.len});
        print("Field.z: read time = {d:.3}ms\n", .{time_read_field / time.ns_per_ms});

        time_start = try Instant.now();
        try meshio.parseField(&lines,&field,2);
        time_end = try Instant.now();
        time_parse_field = @floatFromInt(time_end.since(time_start));
        print("Field.z: parse time = {d:.3}ms\n", .{time_parse_field / time.ns_per_ms});

        break :setup SimData{
            .coords = coords,
            .connect = connect,
            .field = field,
        };
    }; // setup, end


    //--------------------------------------------------------------------------
    // CHECK FIELD LOADED CORRECTLY
    const field_coord_n = sim_data.field.getCoordN();
    const field_time_n = sim_data.field.getTimeN();
    const field_fields_n = sim_data.field.getFieldsN();
    
    var fixed_inds = [_]usize{8,0,0};
    const field_slice = try sim_data.field.array.getSlice(fixed_inds[0..],0);
    const field_mat = try MatSlice(f64).init(field_slice,
                                            field_coord_n,
                                            field_fields_n);

    print("\nfield: time_n = {d}\n",.{field_time_n});
    print("field: coord_n = {d}\n",.{field_coord_n});
    print("field: fields_n = {d}\n\n",.{field_fields_n});
    print("field: mat = \n",.{});
    field_mat.matPrint();

    { // Raster block
        //======================================================================
        // 4. Build Camera
        const pixel_num = [_]u32{960,1280};//[_]u32{ 960, 1280 };
        const pixel_size = [_]f64{ 5.3e-3, 5.3e-3 };
        const focal_leng: f64 = 50.0;
        const alpha_z: f64 = std.math.degreesToRadians(0.0);
        const beta_y: f64 = std.math.degreesToRadians(-30.0);
        const gamma_x: f64 = std.math.degreesToRadians(-10.0);
        const cam_rot = Rotation.init(alpha_z, beta_y, gamma_x);
        const fov_scale_factor: f64 = 1.1;
        const subsample: u8 = 2;
        
        print("{s}\n", .{print_break});
        const roi_pos = CameraOps.roi_cent_from_coords(&sim_data.coords);
        
        print("\nROI center position:\n", .{});
        roi_pos.vecPrint();
        
        const cam_pos = CameraOps.pos_fill_frame_from_rot(&sim_data.coords, 
                                                          pixel_num, 
                                                          pixel_size, 
                                                          focal_leng, 
                                                          cam_rot, 
                                                          fov_scale_factor);
        
        print("\nCamera position:\n", .{});
        cam_pos.vecPrint();
        
        const camera = Camera.init(pixel_num, 
                                   pixel_size, 
                                   cam_pos, 
                                   cam_rot, 
                                   roi_pos, 
                                   focal_leng, 
                                   subsample);
        
        print("\nWorld to camera matrix:\n", .{});
        camera.world_to_cam_mat.matPrint();
        
        print("{s}\n", .{print_break});
        
        //======================================================================
        // 5. Raster Frame
        print("Rastering Image...\n", .{});
        const frame_ind: usize = 8;
        const num_fields = sim_data.field.getFieldsN();
        
        const images_mem = try sim_alloc.alloc(f64, 
                                               num_fields
                                               * camera.pixels_num[1]
                                               * camera.pixels_num[0]);
        @memset(images_mem,0.0);
        
        var images_dims = [_]usize{num_fields,
                            	   camera.pixels_num[1],
                            	   camera.pixels_num[0]};
        var images_arr = try NDArray(f64).init(sim_alloc,
                                               images_mem,
                                               images_dims[0..]);
        
        time_start = try Instant.now();

        // Creates own arena for temporary render buffers which should be 
        // cleared after rendering a frame.
        try Raster.rasterOneFrame(page_alloc, 
                                  frame_ind, 
                                  &sim_data.coords, 
                                  &sim_data.connect, 
                                  &sim_data.field, 
                                  &camera, 
                                  &images_arr);
                               
        time_end = try Instant.now();
        const time_raster: f64 = @floatFromInt(time_end.since(time_start));
        print("Raster time = {d:.3}ms\n\n", .{time_raster / time.ns_per_ms});
        
        // Print diagnostics to console to see if there is an image
        const image_max = std.mem.max(f64, images_arr.elems);
        const image_min = std.mem.min(f64, images_arr.elems);
        print("Image: [max, min] = [{d:.6}, {d:.6}]\n\n", .{ image_max, image_min });
        
        //======================================================================
        // 6. Save image to disk
        const cwd = std.fs.cwd();
        
        const dir_name = "raster-out";
        var name_buff: [1024]u8 = undefined;
        
        cwd.makeDir(dir_name) catch |err| switch (err) {
            error.PathAlreadyExists => {}, // Path exists do nothing
            else => return err, // Propagate any other error
        };
        
        var out_dir = try cwd.openDir(dir_name, .{});
        defer out_dir.close();
        
        print("Saving output images to: {s}\n", .{dir_name});
        
        var image_slice_inds = [_]usize{0,0,0};
        for (0..num_fields) |ff|{
            image_slice_inds[0] = ff;
            
            const file_name = try std.fmt.bufPrint(name_buff[0..], 
                                           "image_out_field{d}_frame{d}.csv", 
                                           .{ ff,frame_ind });
        
            // Grab a matrix slice of the field images
            const image_slice = try images_arr.getSlice(image_slice_inds[0..],0); 
            const image_mat = try MatSlice(f64).init(image_slice,
                                                     camera.pixels_num[1],
                                                     camera.pixels_num[0]);
            
            time_start = try Instant.now();
            try image_mat.saveCSV(out_dir, file_name);
            time_end = try Instant.now();
        
            const time_save_image: f64 = @floatFromInt(time_end.since(time_start));
            print("Field {d} image save time = {d:.3} ms\n", 
                .{ff,time_save_image / time.ns_per_ms,});
        }        
    } // raster, end         
} // main, end
