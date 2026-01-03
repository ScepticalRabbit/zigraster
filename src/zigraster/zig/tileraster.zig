const std = @import("std");
const print = std.debug.print;
const time = std.time;
const Instant = time.Instant;

const Vec3T = @import("vecstack.zig").Vec3T;
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

const rops = @import("rasterops.zig");


const BoundBox = struct {
    x_min_i: usize,
    x_max_i: usize,
    y_min_i: usize,
    y_max_i: usize,
};

pub fn boundBoxOverlap(E_min_x: i32, E_max_x: i32, E_min_y: i32, E_max_y: i32,
                T_min_x: i32, T_max_x: i32, T_min_y: i32, T_max_y: i32) bool {
    if (E_max_x < T_min_x or E_min_x > T_max_x) {
        return false;
    }
    
    if (E_max_y < T_min_y or E_min_y > T_max_y) {
        return false;
    }
    return true;
}

pub fn rasterOneFrame(allocator: std.mem.Allocator, 
                      frame_ind: usize, 
                      coords: *const Coords, 
                      connect: *const Connect, 
                      field: *const Field, 
                      camera: *const Camera, 
                      image_out_arr: *NDArray(f64),
                      ) !void {
    // TEMP
    @memset(image_out_arr.elems,0.0);
    print("DEBUG\n",.{});
    
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

    // dims=(elems_num,coord[x,y,z],nodes_per_elem)
    var elem_coord_arr_dims = [_]usize{elems_num,coords_num,nodes_per_elem};
    const elem_coord_arr_size: usize = elems_num*nodes_per_elem*coords_num;
    const elem_coord_arr_mem = try arena_alloc.alloc(f64, elem_coord_arr_size);
    var elem_coord_arr = try NDArray(f64).init(allocator, 
                                                elem_coord_arr_mem, 
                                                elem_coord_arr_dims[0..]);

    // dims=(elems_num,fields_num,nodes_per_elem)
    var elems_field_arr_dims = [_]usize{elems_num,fields_num,nodes_per_elem};
    const elems_field_arr_size: usize = elems_num*nodes_per_elem*fields_num;
    const elems_field_arr_mem = try arena_alloc.alloc(f64, elems_field_arr_size);
    var elems_field_arr = try NDArray(f64).init(allocator, 
                                                elems_field_arr_mem, 
                                                elems_field_arr_dims[0..]);
    // dims=(elems_num,coord[x,y,z],nodes_per_elem)    
    var elem_inds = [_]usize{0,0,0};
    // dims=(times_num,nodes_num,field_num)
    var field_inds = [_]usize{frame_ind,0,0}; 
    
    for (0..elems_num) |ee| {
        elem_inds[0] = ee;
        const coord_inds: []usize = connect.getElem(ee);

        for (0..connect.nodes_per_elem) |nn| {
            elem_inds[2] = nn;
            field_inds[1] = coord_inds[nn];

            // print("ee={d}, nn={d}\n",.{ee,nn});
            // print("elems_coord_dims=[{d},{d},{d}]\n",.{elem_coord_arr_dims[0],
            //                                            elem_coord_arr_dims[1],
            //                                            elem_coord_arr_dims[2]});
            // print("elem_inds=[{d},{d},{d}]\n",.{elem_inds[0],elem_inds[1],elem_inds[2]});
            // print("\n",.{});
                        
            elem_inds[1] = 0;            
            try elem_coord_arr.set(elem_inds[0..],coords.x[coord_inds[nn]]);
            elem_inds[1] = 1;            
            try elem_coord_arr.set(elem_inds[0..],coords.y[coord_inds[nn]]);
            elem_inds[1] = 2;            
            try elem_coord_arr.set(elem_inds[0..],coords.z[coord_inds[nn]]);
            
            for (0..fields_num) |ff| {
                elem_inds[1] = ff;
                field_inds[2] = ff;
                const field_val = try field.array.get(field_inds[0..]);
                try elems_field_arr.set(elem_inds[0..],field_val);
            }
        } 
    }

    print("\n",.{});
    print("elem_coord_arr:\n",.{});
    print("    dims=[{d},{d},{d}]\n",
          .{elem_coord_arr.dims[0],elem_coord_arr.dims[1],elem_coord_arr.dims[2]});
    print("    strides=[{d},{d},{d}]\n",
          .{elem_coord_arr.strides[0],elem_coord_arr.strides[1],elem_coord_arr.strides[2]});
    print("\n",.{});

    //-----------------------------------------------------------------------------------------
    // World to Raster Coords 
    
    // dims=(elems_num,coord[x,y,z],nodes_per_elem)    
    elem_inds = .{0,0,0}; 
    var node_flat: usize = 0;
    
    for (0..elem_coord_arr.dims[0]) |ee| {
        elem_inds[0] = ee;
        
        for (0..elem_coord_arr.dims[1]) |nn| {
            elem_inds[1] = nn;
            
            node_flat = try elem_coord_arr.getFlatInd(elem_inds[0..]);

            const coord_world = Vec3T(f64).initSlice(elem_coord_arr.elems[node_flat..]);
            const coord_raster = rops.worldToRasterCoords(coord_world,camera);
            // dest, source
            @memcpy(elem_coord_arr.elems[node_flat..node_flat+3],coord_raster.elems[0..]);
// 
//             print("ee={d}, nn={d}\n",.{ee,nn});
//             print("coord_world=[{d},{d},{d}]\n",
//                   .{coord_world.x(),coord_world.y(),coord_world.z()});
//             print("coord_raster=[{d:.3},{d:.3},{d:.3}]\n",
//                   .{coord_raster.x(),coord_raster.y(),coord_raster.z()});
//             print("node_flat={d}\n",.{node_flat});
//             print("elem_coord_arr[{d}]={d}\n",
//                   .{node_flat,elem_coord_arr.elems[node_flat]});
//             print("elem_coord_arr[{d}]={d}\n",
//                   .{node_flat+1,elem_coord_arr.elems[node_flat+1]});
//             print("elem_coord_arr[{d}]={d}\n",
//                   .{node_flat+2,elem_coord_arr.elems[node_flat+2]});
//             print("\n",.{});   
        }        
    }

    //-----------------------------------------------------------------------------------------
    // Element Bounding Boxes
//     elem_inds = .{0,0,0}; 
//     var elem_flat: usize = 0; 
//         
//     for (0..elem_coord_arr.dims[0]) |ee| {
//         elem_inds[0] = ee;
//         elem_flat = try elem_coord_arr.getFlatInd(elem_inds[0..]);
// 
//         const elem_end = elem_flat+elem_coord_arr.strides[0];
//         const elem_slice = elem_coord_arr.elems[elem_flat..elem_end];
// 
//         var x_min = elem_slice[0];
//         var x_max = elem_slice[0];
//         var y_min = elem_slice[1];
//         var y_max = elem_slice[1];
// 
//         // nodes_per_elem = elem_coord_arr.dims[1]
//         for (1..nodes_per_elem) |nn| {
//             // x min and x max in raster coords
//             if (elem_slice[nn*3] < x_min) {
//                 x_min = elem_slice[nn*3];    
//             } else if (elem_slice[nn*3] > x_max) {
//                 x_max = elem_slice[nn*3];                 
//             }
// 
//             // y min and y max in raster coords
//             if (elem_slice[nn*3+1] < y_min) {
//                 y_min = elem_slice[nn*3+1];    
//             } else if (elem_slice[nn*3+1] > y_max) {
//                 y_max = elem_slice[nn*3+1];                 
//             } 
//         }
// 
//         print("elem_slice=\n",.{});
//         for (0..elem_slice.len) |ii| {
//             print("{d:.3},",.{elem_slice[ii]});
//         }
//         print("\n",.{});
//         print("x_min={d:.3},x_max={d:.3}\n",.{x_min,x_max});
//         print("y_min={d:.3},y_max={d:.3}\n",.{y_min,y_max});
//         print("\n\n",.{});    
//     }
// 
//     //-----------------------------------------------------------------------------------------
//     // Element Tile Overlap Sort: Pass 1, How many element in each tile?
// 
//     const tile_size: usize = 16;
//     const tiles_num_x: usize = try std.math.divCeil(usize,sub_px_num_x,tile_size);
//     const tiles_num_y: usize = try std.math.divCeil(usize,sub_px_num_y,tile_size);
//     const tiles_num: usize = tiles_num_x*tiles_num_y;    
// 
//      print("Tiles:\n    tile_size={d}, tiles_num_x={d}, tiles_num_y={d}, tiles_num={}\n",
//           .{tile_size,tiles_num_x,tiles_num_y,tiles_num});
// 
//     // - Loop over elements
//     //  - If not on screen, CONTINUE
//     //  - If on screen work out which tile it overlaps and increment the count for that tile
//     // Need to allocate a slice of memory to store the `tile_counts`
// 
//     // Needs to be allocate because we could have a high res camera with 1000s of tiles
//     const tile_elem_counts: []usize = try alloc.alloc(usize,tiles_num); 

    //-----------------------------------------------------------------------------------------
    // **TILE SPLIT**

    // TODO
    // - **BACK FACE CULLING**
    // - Project all elements onto screen space
    // - Work out which elements are in which tile and allocate a buffer to store this
    // - Need bounding boxes of all elements, bounding boxes of all tiles

    const sub_px_num_x: usize = camera.pixels_num[0]*camera.sub_sample; 
    const sub_px_num_y: usize = camera.pixels_num[1]*camera.sub_sample;

    print("\n",.{});
    print("Camera:\n    pixels_x={d}, pixels_y={d}\n",
          .{camera.pixels_num[0],camera.pixels_num[1]});
    print("    sub_sample={}, sub_pixels_x={d}, sub_pixels_y={d}\n\n",
          .{camera.sub_sample,sub_px_num_x,sub_px_num_y});

    // **LOOP** over elements, calculate raster coords and crop
    //const cam_px_x_f = @as(f64, @floatFromInt(camera.pixels_num[0] - 1));
    //const cam_px_y_f = @as(f64, @floatFromInt(camera.pixels_num[1] - 1));
}
                      
