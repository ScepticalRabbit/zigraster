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
    print("{s}\nZig Rasteriser\n{s}\n", .{ print_break, print_break });

    var time_start = try Instant.now();
    var time_end = try Instant.now();

    //==========================================================================
    // MEMORY ALLOCATORS
    const page_alloc = std.heap.page_allocator;
    
    //==========================================================================
    // SETUP: load simulation data from file
    const path_data = "data/block/";

    const path_coords = path_data ++ "coords.csv";
    const path_connect = path_data ++ "connectivity.csv";

    const path_fields = [_][]const u8{ 
        path_data ++ "field_disp_x.csv",
        path_data ++ "field_disp_y.csv",
        path_data ++ "field_disp_z.csv",
    };

    const sim_data = try meshio.load_sim_data(page_alloc,
                                              path_coords,
                                              path_connect,
                                              path_fields[0..]);

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

    //==========================================================================
    // Build Camera
    
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
    
    //==========================================================================    
    // Raster All Frames

    const cwd = std.fs.cwd();

    const dir_name = "raster-out";

    cwd.makeDir(dir_name) catch |err| switch (err) {
        error.PathAlreadyExists => {}, // Path exists do nothing
        else => return err, // Propagate any other error
    };

    var out_dir = try cwd.openDir(dir_name, .{});
    defer out_dir.close();

    time_start = try Instant.now();

    const image_array = try Raster.rasterAllFrames(page_alloc, 
                                                   out_dir, 
                                                   &sim_data.coords, 
                                                   &sim_data.connect, 
                                                   &sim_data.field, 
                                                   &camera);

    time_end = try Instant.now();
    const time_raster: f64 = @floatFromInt(time_end.since(time_start));
    print("Total raster time = {d:.3}ms\n\n", .{time_raster / time.ns_per_ms});

    // Print diagnostics to console to see if there is an image
    const image_max = std.mem.max(f64, image_array.elems);
    const image_min = std.mem.min(f64, image_array.elems);
    print("Image: [max, min] = [{}, {}]\n\n", .{ image_max, image_min });

} // main, end
