const std = @import("std");
const print = std.debug.print;

const Vec3f = @import("vector.zig").Vec3f;
const Vec3SliceOps = @import("vector.zig").Vec3SliceOps;

const Mat44Ops = @import("matrix.zig").Mat44Ops;

const VecAlloc = @import("vecalloc.zig").VecAlloc;
const MatAlloc = @import("matalloc.zig").MatAlloc;

const SliceTools = @import("slicetools.zig");

const Coords = @import("meshio.zig").Coords;
const Connect = @import("meshio.zig").Connect;
const Field = @import("meshio.zig").Field;

const Camera = @import("camera.zig").Camera;

pub const Image = struct {
    allocator: std.mem.Allocator,
    buffer: MatAlloc(f64),
    depth: MatAlloc(f64),

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, pixels_x: usize, pixels_y: usize) !Self {
        return .{
            .allocator = allocator,
            .buffer = try MatAlloc(f64).init(allocator, pixels_y, pixels_x),
            .depth = try MatAlloc(f64).init(allocator, pixels_y, pixels_x),
        };
    }

    pub fn deinit(self: Self) void {
        self.allocator.free(self.buffer);
        self.allocator.free(self.depth);
    }
};

pub const Raster = struct {
    pub fn world_to_raster_coords(coord_world: Vec3f, camera: *const Camera) Vec3f {
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

    pub fn edge_fun3(vert_0: Vec3f, vert_1: Vec3f, vert_2: Vec3f) f64 {
        return ((vert_2.get(0) - vert_0.get(0)) * (vert_1.get(1) - vert_0.get(1)) - (vert_2.get(1) - vert_0.get(1)) * (vert_1.get(0) - vert_0.get(0)));
    }

    fn bound_index_min(min_val: f64) usize {
        var min_ind: usize = @as(usize, @intFromFloat(@floor(min_val)));
        if (min_ind < 0) {
            min_ind = 0;
        }
        return min_ind;
    }

    fn bound_index_max(max_val: f64, pixels_num: usize) usize {
        var max_ind: usize = @as(usize, @intFromFloat(@ceil(max_val)));
        if (max_ind > (pixels_num - 1)) {
            max_ind = (pixels_num - 1);
        }
        return max_ind;
    }

    pub fn raster_frame(allocator: std.mem.Allocator, frame_ind: usize, coords: *const Coords, connect: *const Connect, field: *const Field, camera: *const Camera) !Image {

        const tol: f64 = 1e-12;
        var elems_in_image: usize = 0;

        var nodes_raster_buff: []Vec3f = try allocator.alloc(Vec3f, connect.nodes_per_elem);
        defer allocator.free(nodes_raster_buff);

        var weights_buff: []f64 = try allocator.alloc(f64, connect.nodes_per_elem);
        defer allocator.free(weights_buff);

        const field_raster_buff: []f64 = try allocator.alloc(f64, connect.nodes_per_elem);
        defer allocator.free(field_raster_buff);

        const pixels_x: usize = @as(usize, camera.pixels_num[0]) * @as(usize, camera.sub_sample);
        const pixels_y: usize = @as(usize, camera.pixels_num[1]) * @as(usize, camera.sub_sample);

        var image_subpx = try Image.init(allocator, pixels_x, pixels_y);

        var px_coord_buff: Vec3f = Vec3f.initZeros();

        // Lifted constants out of loop
        const sub_samp_f: f64 = @as(f64, @floatFromInt(camera.sub_sample));
        const coord_step: f64 = 1.0 / sub_samp_f;
        const coord_offset: f64 = 1.0 / (2.0 * sub_samp_f);

        for (0..connect.elem_n) |ee| {
            const coord_inds: []usize = connect.getElem(ee);

            for (0..connect.nodes_per_elem) |nn| {
                nodes_raster_buff[nn] = world_to_raster_coords(coords.getVec3(coord_inds[nn]), camera);
            }

            const elem_area: f64 = edge_fun3(nodes_raster_buff[0], nodes_raster_buff[1], nodes_raster_buff[2]);

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

            const xi_min: usize = bound_index_min(x_min);
            const xi_max: usize = bound_index_max(x_max, @as(usize, camera.pixels_num[0]));
            const yi_min: usize = bound_index_min(y_min);
            const yi_max: usize = bound_index_max(y_max, @as(usize, camera.pixels_num[1]));

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

            const num_bound_x: usize = SliceTools.range_len(xi_min_f, xi_max_f, coord_step);
            const num_bound_y: usize = SliceTools.range_len(yi_min_f, yi_max_f, coord_step);

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

                    weights_buff[0] = edge_fun3(nodes_raster_buff[1], nodes_raster_buff[2], px_coord_buff);
                    if (weights_buff[0] < -tol) {
                        bound_coord_x += coord_step;
                        bound_ind_x += 1;
                        continue;
                    }

                    weights_buff[1] = edge_fun3(nodes_raster_buff[2], nodes_raster_buff[0], px_coord_buff);
                    if (weights_buff[1] < -tol) {
                        bound_coord_x += coord_step;
                        bound_ind_x += 1;
                        continue;
                    }

                    weights_buff[2] = edge_fun3(nodes_raster_buff[0], nodes_raster_buff[1], px_coord_buff);
                    if (weights_buff[2] < -tol) {
                        bound_coord_x += coord_step;
                        bound_ind_x += 1;
                        continue;
                    }

                    // if ((ee % 6) == 0){
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

                    image_subpx.depth.set(bound_ind_y,bound_ind_x, px_coord_z);

                    // if ((ee % 6) == 0) {
                    //     print("Elem: {}\n", .{ee});
                    //     print("x bound ind={}, coord={d}\n", .{ bound_ind_x, bound_coord_x });
                    //     print("y bound ind={}, coord={d}\n", .{ bound_ind_y, bound_coord_y });
                    //     print("weight_dot_nodes={d}\n", .{weight_dot_nodes});
                    //     print("px_coord_z={d}\n", .{px_coord_z});
                    //     print("\n", .{});
                    // }

                    for (0..connect.nodes_per_elem) |nn| {
                        field_raster_buff[nn] = field.data.get(coord_inds[nn],frame_ind);
                    }

                    var px_field: f64 = SliceTools.dot(f64,field_raster_buff,weights_buff);
                    px_field = px_field*px_coord_z;

                    image_subpx.buffer.set(bound_ind_y,bound_ind_x, px_field);

                    // End for(x) - increment the x coords
                    bound_coord_x += coord_step;
                    bound_ind_x += 1;
                }

                // End for(y) - increment the y coords
                bound_coord_y += coord_step;
                bound_ind_y += 1;
            }
        }

        return image_subpx;
    }
};
