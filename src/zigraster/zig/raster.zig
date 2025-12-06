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


pub const Raster = struct {
    fn worldToRasterCoords(coord_world: Vec3f, camera: *const Camera) Vec3f {
        // TODO: simplify this to a matrix mult
        var coord_raster: Vec3f = Mat44Ops.mulVec3(f64, 
        										   camera.world_to_cam_mat, 
        										   coord_world);

        coord_raster.elems[0] = camera.image_dist 
                                * coord_raster.elems[0] 
                                / (-coord_raster.elems[2]);
        coord_raster.elems[1] = camera.image_dist 
                                * coord_raster.elems[1] 
                                / (-coord_raster.elems[2]);

        coord_raster.elems[0] = 2.0 * coord_raster.elems[0] 
                                / camera.image_dims[0];
        coord_raster.elems[1] = 2.0 * coord_raster.elems[1] 
                                / camera.image_dims[1];

        coord_raster.elems[0] = (coord_raster.elems[0] + 1.0) 
        	/ 2.0 * @as(f64, @floatFromInt(camera.pixels_num[0]));
        coord_raster.elems[1] = (1.0 - coord_raster.elems[1]) 
        	/ 2.0 * @as(f64, @floatFromInt(camera.pixels_num[1]));
        coord_raster.elems[2] = -1.0 * coord_raster.elems[2];

        return coord_raster;
    }

    fn edgeFun3(vert_0: Vec3f, vert_1: Vec3f, vert_2: Vec3f) f64 {
        return ((vert_2.get(0) - vert_0.get(0)) 
              * (vert_1.get(1) - vert_0.get(1)) 
              - (vert_2.get(1) - vert_0.get(1)) 
              * (vert_1.get(0) - vert_0.get(0)));
    }

    fn boundIndexMin(min_val: f64) usize {
        var min_ind: usize = @as(usize, @intFromFloat(@floor(min_val)));
        if (min_ind < 0) {
            min_ind = 0;
        }
        return min_ind;
    }

    fn boundIndexMax(max_val: f64, pixels_num: usize) usize {
        var max_ind: usize = @as(usize, @intFromFloat(@ceil(max_val)));
        if (max_ind > (pixels_num - 1)) {
            max_ind = (pixels_num - 1);
        }
        return max_ind;
    }

    pub fn averageImage(image_subpx: *const MatSlice(f64), 
                        sub_samp: u8, 
                        image_avg: *MatSlice(f64)) void {
                        
        const num_px_x: usize = (image_subpx.cols_n) / @as(usize, sub_samp);
        const num_px_y: usize = (image_subpx.rows_n) / @as(usize, sub_samp);
        const sub_samp_us: usize = @as(usize, sub_samp);
        const sub_samp_f: f64 = @as(f64, @floatFromInt(sub_samp));
        const subpx_per_px: f64 = sub_samp_f * sub_samp_f;

        // TODO: do some error checking on the Matrices here to check dims agree
        // with the variables above

        var px_sum: f64 = 0.0;

        for (0..num_px_y) |iy| {
            for (0..num_px_x) |ix| {
                px_sum = 0.0;
                for (0..sub_samp_us) |sy| {
                    for (0..sub_samp_us) |sx| {
                        px_sum += image_subpx.get(sub_samp_us * iy + sy, 
                                                  sub_samp_us * ix + sx);
                    }
                }
                image_avg.set(iy, ix, px_sum / subpx_per_px);
            }
        }
    }

    pub fn rasterOneFrame(allocator: std.mem.Allocator, 
                          frame_ind: usize, 
                          coords: *const Coords, 
                          connect: *const Connect, 
                          field: *const Field, 
                          camera: *const Camera, 
                          image_out_arr: *NDArray(f64)) !void {

        // We allocate all temporary buffers on our arena so no need to defer
        // free any temporary buffers in this function
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const arena_alloc = arena.allocator();

        const tol: f64 = 1e-12;
        var elems_in_image: usize = 0;
        const num_fields: usize = field.getFieldsN();

        var nodes_raster_buff: []Vec3f = try arena_alloc.alloc(
            Vec3f, connect.nodes_per_elem);

		// Stores N weights, one for each node in the element
        var weights_buff: []f64 = try arena_alloc.alloc(
            f64, connect.nodes_per_elem);

		// Stores all F field values at the N nodes per element
		var field_inds = [_]usize{frame_ind,0,0};
        const field_buff: []f64 = try arena_alloc.alloc(
            f64, num_fields*connect.nodes_per_elem);

		var field_raster_mat = try MatSlice(f64).init(field_buff,
		                                              connect.nodes_per_elem,
		                                              num_fields);

		// Stores field value at the pixel
		var px_field: f64 = 0.0;

		// Sub-pixel image buffers
        const subpx_x: usize = @as(usize, camera.pixels_num[0]) 
        					   * @as(usize, camera.sub_sample);
        const subpx_y: usize = @as(usize, camera.pixels_num[1]) 
                               * @as(usize, camera.sub_sample);

        
        var depth_subpx_inds = [_]usize{0,0};
		var image_subpx_inds = [_]usize{0,0,0};

        // Sub-pixel image buffer
        var image_subpx_dims = [_]usize{num_fields,subpx_y,subpx_x};
		const image_subpx_mem = try arena_alloc.alloc(
		    f64,subpx_y*subpx_x*num_fields);
		
		var image_subpx = try NDArray(f64).init(arena_alloc,
		                                        image_subpx_mem,
		                                        image_subpx_dims[0..]);
		
		// Sub-pixel depth buffer
		var depth_subpx_dims = [_]usize{subpx_y,subpx_x};
	    const depth_subpx_mem = try arena_alloc.alloc(f64,subpx_y*subpx_x);
		var depth_subpx = try NDArray(f64).init(arena_alloc,
		                                        depth_subpx_mem,
		                                        depth_subpx_dims[0..]);

		// Set image background to 0.0 and depth buffer to large value.
        image_subpx.fill(0.0);
        depth_subpx.fill(1e6);

        var px_coord_buff: Vec3f = Vec3f.initZeros();

        // Lifted constants out of loop
        const sub_samp_f: f64 = @as(f64, @floatFromInt(camera.sub_sample));
        const coord_step: f64 = 1.0 / sub_samp_f;
        const coord_offset: f64 = 1.0 / (2.0 * sub_samp_f);

		//----------------------------------------------------------------------
		// Raster Loop
        for (0..connect.elem_n) |ee| {
            const coord_inds: []usize = connect.getElem(ee);

            for (0..connect.nodes_per_elem) |nn| {
                nodes_raster_buff[nn] = worldToRasterCoords(
                	coords.getVec3(coord_inds[nn]), camera);
            }

            const elem_area: f64 = edgeFun3(nodes_raster_buff[0], 
                                            nodes_raster_buff[1], 
                                            nodes_raster_buff[2]);

            if (elem_area < -tol) {
                continue;
            }

            // print("Node 0:",.{});
            // nodes_raster[0].vecPrint();
            // print("Node 1:",.{});
            // nodes_raster[1].vecPrint();
            // print("Node 2:", .{});
            // nodes_raster[2].vecPrint();
            // print("{} ELEM AREA : {d:.4}\n\n",.{ee,elem_area});

            const x_min: f64 = Vec3SliceOps.min(f64, nodes_raster_buff, 0);
            const x_max: f64 = Vec3SliceOps.max(f64, nodes_raster_buff, 0);

            if ((x_min > @as(f64, @floatFromInt(camera.pixels_num[0] - 1))) or (x_max < 0.0)) {
                continue;
            }

            const y_min: f64 = Vec3SliceOps.min(f64, nodes_raster_buff, 1);
            const y_max: f64 = Vec3SliceOps.max(f64, nodes_raster_buff, 1);

            if ((y_min > @as(f64, @floatFromInt(camera.pixels_num[1] - 1))) or (y_max < 0.0)) {
                continue;
            }

            elems_in_image += 1;

            // print("Elem {}: x, min {}\n", .{ ee, x_min });
            // print("Elem {}: x, max {}\n", .{ ee, x_max });
            // print("Elem {}: x, min {}\n", .{ ee, y_min });
            // print("Elem {}: x, max {}\n\n", .{ ee, y_max });

            const xi_min: usize = boundIndexMin(x_min);
            const xi_max: usize = boundIndexMax(x_max, @as(usize, camera.pixels_num[0]));
            const yi_min: usize = boundIndexMin(y_min);
            const yi_max: usize = boundIndexMax(y_max, @as(usize, camera.pixels_num[1]));

            // print("Elem {}: xi, min {}\n", .{ ee, xi_min });
            // print("Elem {}: xi, max {}\n", .{ ee, xi_max });
            // print("Elem {}: yi, min {}\n", .{ ee, yi_min });
            // print("Elem {}: yi, max {}\n", .{ ee, yi_max });
            // print("\n",.{});

            const xi_min_f: f64 = @as(f64, @floatFromInt(xi_min));
            const xi_max_f: f64 = @as(f64, @floatFromInt(xi_max));
            const yi_min_f: f64 = @as(f64, @floatFromInt(yi_min));
            const yi_max_f: f64 = @as(f64, @floatFromInt(yi_max));

            var bound_coord_x: f64 = xi_min_f + 1.0 / (2.0 * sub_samp_f);
            var bound_coord_y: f64 = yi_min_f + 1.0 / (2.0 * sub_samp_f);
            var bound_ind_x: usize = @as(usize, camera.sub_sample) * xi_min;
            var bound_ind_y: usize = @as(usize, camera.sub_sample) * yi_min;

            const num_bound_x: usize = sliceops.rangeLen(xi_min_f, xi_max_f, coord_step);
            const num_bound_y: usize = sliceops.rangeLen(yi_min_f, yi_max_f, coord_step);

            var inv_buff: f64 = 0.0;
            for (0..connect.nodes_per_elem) |nn| {
                inv_buff = 1.0 / nodes_raster_buff[nn].get(2);
                nodes_raster_buff[nn].set(2, inv_buff);
            }

            // print("Elem {}: bound_coord_x={d}\n",.{ee,bound_coord_x});
            // print("Elem {}: bound_coord_y={d}\n",.{ee,bound_coord_y});
            // print("Elem {}: bound_ind_x={}\n",.{ee,bound_ind_x});
            // print("Elem {}: bound_ind_y={}\n",.{ee,bound_ind_y});
            // print("Elem {}: coord_step={d}\n",.{ee,coord_step});
            // print("Elem {}: num_bound_x={}\n",.{ee,num_bound_x});
            // print("Elem {}: num_bound_y={}\n\n",.{ee,num_bound_y});

            for (0..num_bound_y) |jj| {
                _ = jj;

                bound_coord_x = xi_min_f + coord_offset;
                bound_ind_x = camera.sub_sample * xi_min;

                for (0..num_bound_x) |ii| {
                    _ = ii;

                    px_coord_buff.set(0, bound_coord_x);
                    px_coord_buff.set(1, bound_coord_y);

                    weights_buff[0] = edgeFun3(nodes_raster_buff[1], 
                                               nodes_raster_buff[2], 
                                               px_coord_buff);
                    if (weights_buff[0] < -tol) {
                        bound_coord_x += coord_step;
                        bound_ind_x += 1;
                        continue;
                    }

                    weights_buff[1] = edgeFun3(nodes_raster_buff[2], 
                    						   nodes_raster_buff[0], 
                    						   px_coord_buff);
                    if (weights_buff[1] < -tol) {
                        bound_coord_x += coord_step;
                        bound_ind_x += 1;
                        continue;
                    }

                    weights_buff[2] = edgeFun3(nodes_raster_buff[0], 
                    						   nodes_raster_buff[1], 
                    						   px_coord_buff);
                    if (weights_buff[2] < -tol) {
                        bound_coord_x += coord_step;
                        bound_ind_x += 1;
                        continue;
                    }

                    // if ((ee % 10) == 0){
                    //     print("Elem: {}\n",.{ee});
                    //     print("x bound ind={}, coord={d}\n",
                    //         .{bound_ind_x,bound_coord_x});
                    //     print("y bound ind={}, coord={d}\n",
                    //         .{bound_ind_y,bound_coord_y});
                    //     print("weights=[{d},{d},{d}]\n",
                    //         .{weights_buff[0],weights_buff[1],weights_buff[2]});
                    //     print("\n",.{});
                    // }

                    var weight_dot_nodes: f64 = 0.0;
                    for (0..connect.nodes_per_elem) |nn| {
                        weights_buff[nn] = weights_buff[nn] / elem_area;
                        weight_dot_nodes += weights_buff[nn] 
                        			        * nodes_raster_buff[nn].get(2);
                    }

                    // Calculate the depth for this sub-pixel
                    const px_coord_z: f64 = 1.0 / weight_dot_nodes;

                    // If this pixel is behind another we move on
					depth_subpx_inds[0] = bound_ind_y;
					depth_subpx_inds[1] = bound_ind_x;
					
					const depth_arr_z: f64 = try depth_subpx.get(depth_subpx_inds[0..]); 
					
                    if (px_coord_z >= depth_arr_z) {
                        bound_coord_x += coord_step;
                        bound_ind_x += 1;
                        continue;
                    }

                    try depth_subpx.set(depth_subpx_inds[0..], px_coord_z);

                    // if ((ee % 10) == 0) {
                    //     print("Elem: {}\n", .{ee});
                    //     print("x bound ind={}, coord={d}\n", 
                    //     .{ bound_ind_x, bound_coord_x });
                    //     print("y bound ind={}, coord={d}\n", 
                    //     .{ bound_ind_y, bound_coord_y });
                    //     print("weight_dot_nodes={d}\n", .{weight_dot_nodes});
                    //     print("depth_arr_z={d}\n",. {depth_arr_z});
                    //     print("px_coord_z={d}\n", .{px_coord_z});
                    //     print("\n", .{});
                    // }

					//----------------------------------------------------------
					// BROKEN: PSEUDO CODE FROM HERE
					// - Update to deal with rendering F fields in a single pass

					var field_val: f64 = 0.0;
                    for (0..connect.nodes_per_elem) |nn| {
                        // NOTE:
                        // field.array, shape=(time_n,coord_n,field_n)
                        // field_raster_mat, shape=(field_n,nodes_per_elem)
						for (0..num_fields) |ff|{
							field_inds[1] = coord_inds[nn]; // This is scattered
							field_inds[2] = ff;

							field_val = try	field.array.get(field_inds[0..]);

	                        field_raster_mat.set(ff,nn,field_val);
	                    }

                    }
                    
                    // TODO: works up to here getting field values                        
					image_subpx_inds[1] = bound_ind_y;
					image_subpx_inds[2] = bound_ind_x;

					for (0..num_fields) |ff| {
						const field_slice = try field_raster_mat.getSlice(ff);
                    	px_field = sliceops.dot(f64, field_slice, weights_buff);

	                    px_field = px_field * px_coord_z;

	                    // print("\nind_y={} , ind_x={}, px_field={}\n",
	                    //      .{bound_ind_y,bound_ind_x,px_field});

						image_subpx_inds[0] = ff;
					    try image_subpx.set(image_subpx_inds[0..], px_field);
					}

					//----------------------------------------------------------
                    // End for(x) - increment the x coords
                    bound_coord_x += coord_step;
                    bound_ind_x += 1;
                }
				//--------------------------------------------------------------
                // End for(y) - increment the y coords
                bound_coord_y += coord_step;
                bound_ind_y += 1;
            }
        }

        const image_subpx_max = std.mem.max(f64,image_subpx.elems);
        const image_subpx_min = std.mem.min(f64,image_subpx.elems);
        const depth_subpx_max = std.mem.max(f64,depth_subpx.elems);
        const depth_subpx_min = std.mem.min(f64,depth_subpx.elems);
        print("\nimage_subpx_max,min=[{d:.6},{d:.6}]\n",.{image_subpx_max,image_subpx_min});
        print("depth_subpx_max,min=[{d:.6},{d:.6}]\n",.{depth_subpx_max,depth_subpx_min});


        var out_slice_inds = [_]usize{0,0,0};
        for (0..num_fields) |ff| {
            out_slice_inds[0] = ff;

            // 1) Create MatSlice for sub-pixel image for given field ff
            const image_subpx_slice = try image_subpx.getSlice(
                out_slice_inds[0..],0);
            const image_subpx_mat = try MatSlice(f64).init(image_subpx_slice,
                                                          subpx_y,
                                                          subpx_x);

            // 2) Create wrapper MatSlice for actual images dims from last
            // two dims of the image_out_arr using getSlice()
            // Need to get it from image_out_arr
            const image_out_slice = try image_out_arr.getSlice(
                out_slice_inds[0..],0);
            var image_out_mat = try MatSlice(f64).init(image_out_slice,
                                                      camera.pixels_num[1],
                                                      camera.pixels_num[0]);

            averageImage(&image_subpx_mat, camera.sub_sample, &image_out_mat);
        }
    
        //----------------------------------------------------------------------
        // DEBUG: SAVE SUB-PIXEL IMAGES TO DISK
//         const cwd = std.fs.cwd();
//         const dir_name = "raster-out";
// 
//         cwd.makeDir(dir_name) catch |err| switch (err) {
//          error.PathAlreadyExists => {}, // Path exists do nothing
//          else => return err, // Propagate any other error
//         };
//         var out_dir = try cwd.openDir(dir_name, .{});
//         defer out_dir.close();
// 
// 
//         var name_buff: [1024]u8 = undefined;
//         var file_name = try std.fmt.bufPrint(name_buff[0..], 
//                                            "depthsp_frame{d}.csv", 
//                                            .{ frame_ind });
//         const depth_mat = try MatSlice(f64).init(depth_subpx.elems[0..],
//                                              subpx_y,
//                                              subpx_x);
//         try depth_mat.saveCSV(out_dir,file_name);                                        
// 
//         for (0..num_fields) |ff| {
//             out_slice_inds[0] = ff;
//             
//             file_name = try std.fmt.bufPrint(name_buff[0..], 
//                                              "imagesp_field{d}_frame{d}.csv", 
//                                              .{ ff,frame_ind });
//             const imagesp_slice = try image_subpx.getSlice(out_slice_inds[0..],0);
//             const imagesp_mat = try MatSlice(f64).init(imagesp_slice,subpx_y,subpx_x);
//             try imagesp_mat.saveCSV(out_dir,file_name);
// 
//             file_name = try std.fmt.bufPrint(name_buff[0..], 
//                                              "image_field{d}_frame{d}.csv", 
//                                              .{ ff,frame_ind });
//             const image_slice = try image_out_arr.getSlice(out_slice_inds[0..],0);
//             const image_mat = try MatSlice(f64).init(image_slice,
//                                                      camera.pixels_num[1],
//                                                      camera.pixels_num[0]);
//             try image_mat.saveCSV(out_dir,file_name);                                         
//         }
        		
    }

    pub fn rasterAllFrames(allocator: std.mem.Allocator, 
                           out_dir: std.fs.Dir, 
                           coords: *const Coords, 
                           connect: *const Connect, 
                           field: *const Field, 
                           camera: *const Camera) !NDArray(f64) {

        // We allocate all temporary buffers on our arena so no need to defer
        // free any temporary buffers in this function
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const arena_alloc = arena.allocator();

        const num_fields: usize = field.getFieldsN();
        const num_time: usize = field.getTimeN();
                           
        const frame_arr_size: usize = num_time 
									   * num_fields
                                       * camera.pixels_num[0] 
                                       * camera.pixels_num[1];

        // We are going to return this so we use the input allocator instead of
        // the arena.
        const frame_arr_mem = try allocator.alloc(f64, frame_arr_size);
		
        var frame_arr_dims = [_]usize{ num_time, 
									   num_fields,
                                       camera.pixels_num[1], 
                                       camera.pixels_num[0],};
                                        
        // We are going to return this so we use the input allocator instead of
        // the arena. It also owns the memory slice we just allocated.
        var frame_arr = try NDArray(f64).init(allocator, 
                                              frame_arr_mem, 
                                              frame_arr_dims[0..]);

        const image_stride: usize = frame_arr.strides[0];
        var image_inds = [_]usize{ 0, 0, 0 ,0}; // frame,field,px_y,px_x
		var field_inds = [_]usize{0,0,0}; // field,px_y_px_x

        var time_start = try Instant.now();
        var time_end = try Instant.now();
        var time_raster: f64 = 0.0;

        var name_buff: [1024]u8 = undefined;

        print("Starting rastering frames.\n", .{});

        for (0..num_time) |tt| {
            time_start = try Instant.now();

            image_inds[0] = tt;
            const start_ind = try frame_arr.getFlatInd(image_inds[0..]);
            const end_ind = start_ind + image_stride;

            const images_mem = frame_arr.elems[start_ind..end_ind];
            // This is only temporary so we use our arena - the memory lives on
            // in the input allocator
            var images_arr = try NDArray(f64).init(arena_alloc,
                                                   images_mem, 
            									   frame_arr_dims[1..]);

            // This will create it's own arena for temporary storage so we pass
            // through the input allocator for this.
            try rasterOneFrame(allocator, tt, coords, connect, 
                               field, camera, &images_arr);

            for (0..num_fields) |ff| {
		    	const file_name = try std.fmt.bufPrint(name_buff[0..], 
                                                   "all_field{d}_frame{d}.csv", 
                                                   .{ ff,tt });

				field_inds[0] = ff;
				const field_slice = try images_arr.getSlice(field_inds[0..],0);
				
			    const image_mat = try MatSlice(f64).init(field_slice,
			                                         camera.pixels_num[1],
			                                         camera.pixels_num[0]);
            	try image_mat.saveCSV(out_dir, file_name);
			}
			
            time_end = try Instant.now();
            time_raster = @floatFromInt(time_end.since(time_start));

            print("Frame {}, raster time = {d:.3}ms\n", 
                  .{ tt, time_raster / time.ns_per_ms });
        }

        print("Rastering complete.\n\n", .{});

        return frame_arr;
    }
};
