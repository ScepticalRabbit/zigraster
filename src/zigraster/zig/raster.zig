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

// NOTE: Used for debugging and passing back sub-pixel buffers
pub const ImageAlloc = struct {
    alloc: std.mem.Allocator,
    image: MatSlice(f64),
    depth: MatSlice(f64),
    im_buff: []f64,
    d_buff: []f64,

    const Self = @This();

    pub fn init(alloc: std.mem.Allocator, pixels_x: usize, pixels_y: usize) !Self {
        const image_buff = try alloc.alloc(f64, pixels_x * pixels_y);
        const depth_buff = try alloc.alloc(f64, pixels_x * pixels_y);

        return .{
            .alloc = alloc,
            .image = try MatSlice(f64).init(image_buff, pixels_y, pixels_x),
            .depth = try MatSlice(f64).init(depth_buff, pixels_y, pixels_x),
            .im_buff = image_buff,
            .d_buff = depth_buff,
        };
    }

    pub fn deinit(self: *const Self) void {
        self.alloc.free(self.im_buff);
        self.alloc.free(self.d_buff);
    }
};

pub const Raster = struct {
    fn worldToRasterCoords(coord_world: Vec3f, camera: *const Camera) Vec3f {
        // TODO: simplify this to a matrix mult
        var coord_raster: Vec3f = Mat44Ops.mulVec3(f64, camera.world_to_cam_mat, coord_world);

        coord_raster.elems[0] = camera.image_dist * coord_raster.elems[0] / (-coord_raster.elems[2]);
        coord_raster.elems[1] = camera.image_dist * coord_raster.elems[1] / (-coord_raster.elems[2]);

        coord_raster.elems[0] = 2.0 * coord_raster.elems[0] / camera.image_dims[0];
        coord_raster.elems[1] = 2.0 * coord_raster.elems[1] / camera.image_dims[1];

        coord_raster.elems[0] = (coord_raster.elems[0] + 1.0) / 2.0 * @as(f64, @floatFromInt(camera.pixels_num[0]));
        coord_raster.elems[1] = (1.0 - coord_raster.elems[1]) / 2.0 * @as(f64, @floatFromInt(camera.pixels_num[1]));
        coord_raster.elems[2] = -1.0 * coord_raster.elems[2];

        return coord_raster;
    }

    fn edgeFun3(vert_0: Vec3f, vert_1: Vec3f, vert_2: Vec3f) f64 {
        return ((vert_2.get(0) - vert_0.get(0)) * (vert_1.get(1) - vert_0.get(1)) - (vert_2.get(1) - vert_0.get(1)) * (vert_1.get(0) - vert_0.get(0)));
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

    pub fn averageImage(image_subpx: *const MatSlice(f64), sub_samp: u8, image_avg: *MatSlice(f64)) void {
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
                        px_sum += image_subpx.get(sub_samp_us * iy + sy, sub_samp_us * ix + sx);
                    }
                }
                image_avg.set(iy, ix, px_sum / subpx_per_px);
            }
        }
    }

    pub fn rasterOneFrame(alloc: std.mem.Allocator, frame_ind: usize, coords: *const Coords, connect: *const Connect, field: *const Field, camera: *const Camera, image_out_buff: *MatSlice(f64)) !void {
        const tol: f64 = 1e-12;
        var elems_in_image: usize = 0;

        // NOTE: speed test shows this is way faster than stack allocation.
        // 71ms per frame vs 115ms per frame for equivalent setup
        var nodes_raster_buff: []Vec3f = try alloc.alloc(Vec3f, connect.nodes_per_elem);
        defer alloc.free(nodes_raster_buff);
        var weights_buff: []f64 = try alloc.alloc(f64, connect.nodes_per_elem);
        defer alloc.free(weights_buff);
        const field_raster_buff: []f64 = try alloc.alloc(f64, connect.nodes_per_elem);
        defer alloc.free(field_raster_buff);

        // Allocate buffers on the stack, assuming max nodes is 4
        // NOTE: this is slower than heap allocation
        // const max_nodes: usize = 4;
        // var nodes_raster_arr: [max_nodes]Vec3f = undefined;
        // var nodes_raster_buff = nodes_raster_arr[0..nodes_raster_arr.len];
        // //var weights_buff_arr = [_]f64{0.0} ** max_nodes;
        // var weights_buff_arr: [max_nodes]f64 = undefined;
        // var weights_buff = weights_buff_arr[0..weights_buff_arr.len];
        // //var field_raster_arr = [_]f64{0.0} ** max_nodes;
        // var field_raster_arr: [max_nodes]f64 = undefined;
        // var field_raster_buff = field_raster_arr[0..field_raster_arr.len];

        const subpx_x: usize = @as(usize, camera.pixels_num[0]) * @as(usize, camera.sub_sample);
        const subpx_y: usize = @as(usize, camera.pixels_num[1]) * @as(usize, camera.sub_sample);

        // var image_buff_subpx = try MatAlloc(f64).init(alloc,subpx_y,subpx_x);
        // var depth_buff_subpx = try MatAlloc(f64).init(alloc,subpx_y, subpx_x);

        // NOTE: we keep this instead of 2x MatSlice buffers so we can return it
        // for debugging purposes.
        var image_subpx = try ImageAlloc.init(alloc, subpx_x, subpx_y);
        defer image_subpx.deinit();
        image_subpx.image.fill(0.0);
        image_subpx.depth.fill(1e6);

        // TODO: this will need to be de-allocated instead of passed back.

        var px_coord_buff: Vec3f = Vec3f.initZeros();

        // Lifted constants out of loop
        const sub_samp_f: f64 = @as(f64, @floatFromInt(camera.sub_sample));
        const coord_step: f64 = 1.0 / sub_samp_f;
        const coord_offset: f64 = 1.0 / (2.0 * sub_samp_f);

        for (0..connect.elem_n) |ee| {
            const coord_inds: []usize = connect.getElem(ee);

            for (0..connect.nodes_per_elem) |nn| {
                nodes_raster_buff[nn] = worldToRasterCoords(coords.getVec3(coord_inds[nn]), camera);
            }

            const elem_area: f64 = edgeFun3(nodes_raster_buff[0], nodes_raster_buff[1], nodes_raster_buff[2]);

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

                    weights_buff[0] = edgeFun3(nodes_raster_buff[1], nodes_raster_buff[2], px_coord_buff);
                    if (weights_buff[0] < -tol) {
                        bound_coord_x += coord_step;
                        bound_ind_x += 1;
                        continue;
                    }

                    weights_buff[1] = edgeFun3(nodes_raster_buff[2], nodes_raster_buff[0], px_coord_buff);
                    if (weights_buff[1] < -tol) {
                        bound_coord_x += coord_step;
                        bound_ind_x += 1;
                        continue;
                    }

                    weights_buff[2] = edgeFun3(nodes_raster_buff[0], nodes_raster_buff[1], px_coord_buff);
                    if (weights_buff[2] < -tol) {
                        bound_coord_x += coord_step;
                        bound_ind_x += 1;
                        continue;
                    }

                    // if ((ee % 10) == 0){
                    //     print("Elem: {}\n",.{ee});
                    //     print("x bound ind={}, coord={d}\n",.{bound_ind_x,bound_coord_x});
                    //     print("y bound ind={}, coord={d}\n",.{bound_ind_y,bound_coord_y});
                    //     print("weights=[{d},{d},{d}]\n",.{weights_buff[0],weights_buff[1],weights_buff[2]});
                    //     print("\n",.{});
                    // }

                    var weight_dot_nodes: f64 = 0.0;
                    for (0..connect.nodes_per_elem) |nn| {
                        weights_buff[nn] = weights_buff[nn] / elem_area;
                        weight_dot_nodes += weights_buff[nn] * nodes_raster_buff[nn].get(2);
                    }

                    // Calculate the depth for this sub-pixel
                    const px_coord_z: f64 = 1.0 / weight_dot_nodes;

                    // If this pixel is behind another we move on
                    if (px_coord_z >= image_subpx.depth.get(bound_ind_y, bound_ind_x)) {
                        bound_coord_x += coord_step;
                        bound_ind_x += 1;
                        continue;
                    }

                    image_subpx.depth.set(bound_ind_y, bound_ind_x, px_coord_z);

                    // if ((ee % 10) == 0) {
                    //     print("Elem: {}\n", .{ee});
                    //     print("x bound ind={}, coord={d}\n", .{ bound_ind_x, bound_coord_x });
                    //     print("y bound ind={}, coord={d}\n", .{ bound_ind_y, bound_coord_y });
                    //     print("weight_dot_nodes={d}\n", .{weight_dot_nodes});
                    //     print("px_coord_z={d}\n", .{px_coord_z});
                    //     print("\n", .{});
                    // }

                    for (0..connect.nodes_per_elem) |nn| {
                        // print("nn={}\n",.{nn});
                        // print("coord_inds[nn]={}\n",.{coord_inds[nn]});
                        // print("field.data.rows_n={}\n",.{field.data.rows_n});
                        // print("field.data.cols_n={}\n",.{field.data.cols_n});
                        field_raster_buff[nn] = field.data.get(coord_inds[nn], frame_ind);
                    }

                    var px_field: f64 = sliceops.dot(f64, field_raster_buff, weights_buff);
                    px_field = px_field * px_coord_z;

                    //print("\nind_y={} , ind_x={}, px_field={}\n",.{bound_ind_y,bound_ind_x,px_field});

                    image_subpx.image.set(bound_ind_y, bound_ind_x, px_field);

                    // End for(x) - increment the x coords
                    bound_coord_x += coord_step;
                    bound_ind_x += 1;
                }

                // End for(y) - increment the y coords
                bound_coord_y += coord_step;
                bound_ind_y += 1;
            }
        }

        // NOTE: image buffer is now passed into the function instead of being
        // allocated here
        //var image = try ImageAlloc.init(alloc, camera.pixels_num[0], camera.pixels_num[1]);

        averageImage(&image_subpx.image, camera.sub_sample, image_out_buff);

        // NOTE: only need to do this for debugging - this can be discarded here
        // averageImage(&image_subpx.depth, camera.sub_sample, &image.depth);

        // NOTE: could return the sub_px buffers here, but if we don't free the
        // memory then we have a leak. Need to defer deinit() them to avoid this.
        // return image_subpx;
    }

    pub fn rasterAllFrames(alloc: std.mem.Allocator, out_dir: std.fs.Dir, coords: *const Coords, connect: *const Connect, field: *const Field, camera: *const Camera) !NDArray(f64) {
        const frame_buff_size: usize = field.time_n * camera.pixels_num[0] * camera.pixels_num[1];

        const frame_buff_mem = try alloc.alloc(f64, frame_buff_size);
        var frame_buff_dims = [_]usize{ field.time_n, camera.pixels_num[1], camera.pixels_num[0] };

        var frame_buff = try NDArray(f64).init(alloc, frame_buff_mem, frame_buff_dims[0..]);

        const image_stride: usize = frame_buff.strides[0];
        var image_inds = [_]usize{ 0, 0, 0 }; // frame,px_y,px_x

        var time_start = try Instant.now();
        var time_end = try Instant.now();
        var time_raster: f64 = 0.0;

        var name_buff: [512]u8 = undefined;
        const field_name = "field";

        print("Starting rastering frames.\n", .{});

        for (0..field.time_n) |ff| {
            time_start = try Instant.now();

            image_inds[0] = ff;
            const start_ind = try frame_buff.getFlatInd(image_inds[0..]);
            const end_ind = start_ind + image_stride;

            const image_out_mem = frame_buff.elems[start_ind..end_ind];
            var image_out_buff = try MatSlice(f64).init(image_out_mem, camera.pixels_num[1], camera.pixels_num[0]);
            try rasterOneFrame(alloc, ff, coords, connect, field, camera, &image_out_buff);

            const file_name = try std.fmt.bufPrint(name_buff[0..], "{s}_frame{d}.csv", .{ field_name, ff });

            try image_out_buff.saveCSV(out_dir, file_name);

            time_end = try Instant.now();
            time_raster = @floatFromInt(time_end.since(time_start));

            print("Frame {}, raster time = {d:.3}ms\n", .{ ff, time_raster / time.ns_per_ms });
        }

        print("Rastering complete.\n\n", .{});

        return frame_buff;
    }
};
