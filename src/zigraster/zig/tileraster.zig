const std = @import("std");
const print = std.debug.print;
const time = std.time;
const Instant = time.Instant;

const Vec3f = @import("vecstack.zig").Vec3f;
const Vec3SliceOps = @import("vecstack.zig").Vec3SliceOps;

const Mat44Ops = @import("matstack.zig").Mat44Ops;

const VecSlice = @import("vecslice.zig").VecSlice;
const MatSlice = @import("matslice.zig").MatSlice;
const NDArray = @import("ndarray.zig").NDArray;

const sliceops = @import("sliceops.zig");

const Coords = @import("meshio.zig").Coords;
const Connect = @import("meshio.zig").Connect;
const Field = @import("meshio.zig").Field;

const Camera = @import("camera.zig").Camera;


pub fn rasterOneFrame(allocator: std.mem.Allocator, 
                      frame_ind: usize, 
                      coords: *const Coords, 
                      connect: *const Connect, 
                      field: *const Field, 
                      camera: *const Camera, 
                      image_out_arr: *NDArray(f64),
                      ) !void {
    // TEMP
    _ = camera;
    @memset(image_out_arr.elems,0.0);

    //-----------------------------------------------------------------------------------------
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    //-----------------------------------------------------------------------------------------
    // **ELEMENT WISE PRE-TRANSFORM**
    
    const elems_num: usize = connect.elem_n;
    const nodes_per_elem: usize = connect.nodes_per_elem;
    const coords_num: usize = 3;
    const fields_num: usize = field.getFieldsN();

    var elems_coord_arr_dims = [_]usize{elems_num,nodes_per_elem,coords_num};
    const elems_coord_arr_size: usize = elems_num*nodes_per_elem*coords_num;
    const elems_coord_arr_mem = try arena_alloc.alloc(f64, elems_coord_arr_size);
    var elems_coord_arr = try NDArray(f64).init(allocator, 
                                                elems_coord_arr_mem, 
                                                elems_coord_arr_dims[0..]);

    var elems_field_arr_dims = [_]usize{elems_num,nodes_per_elem,fields_num};
    const elems_field_arr_size: usize = elems_num*nodes_per_elem*fields_num;
    const elems_field_arr_mem = try arena_alloc.alloc(f64, elems_field_arr_size);
    var elems_field_arr = try NDArray(f64).init(allocator, 
                                                elems_field_arr_mem, 
                                                elems_field_arr_dims[0..]);
        
    var elems_inds = [_]usize{0,0,0};
    var field_inds = [_]usize{frame_ind,0,0}; 
    for (0..elems_num) |ee| {
        elems_inds[0] = ee;
        const coord_inds: []usize = connect.getElem(ee);

        for (0..connect.nodes_per_elem) |nn| {
            elems_inds[1] = coord_inds[nn];
            field_inds[1] = coord_inds[nn];
            
            elems_inds[2] = 0;            
            try elems_coord_arr.set(elems_inds[0..],coords.x[coord_inds[nn]]);
            elems_inds[2] = 1;            
            try elems_coord_arr.set(elems_inds[0..],coords.y[coord_inds[nn]]);
            elems_inds[2] = 2;            
            try elems_coord_arr.set(elems_inds[0..],coords.z[coord_inds[nn]]);
            
            for (0..fields_num) |ff| {
                elems_inds[2] = ff;
                field_inds[2] = ff;
                const field_val = try field.array.get(field_inds[0..]);
                try elems_field_arr.set(elems_inds[0..],field_val);
            }
        } 
    }


}
                      
