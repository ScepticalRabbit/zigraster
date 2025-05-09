const std = @import("std");
const print = std.debug.print;

const Vec3f = @import("vector.zig").Vec3f;
const Vec3SliceOps = @import("vector.zig").Vec3SliceOps;

const Mat44Ops = @import("matrix.zig").Mat44Ops;

const Coords = @import("meshio.zig").Coords;
const Connect = @import("meshio.zig").Connect;
const Field = @import("meshio.zig").Field;

const Camera = @import("camera.zig").Camera;

pub const Raster = struct {
    // camera: Camera,
    // coords: Coords,
    // connect: Connect,
    // field: []Field,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
        };
    }

    pub fn world_to_raster_coords(coord_world: Vec3f, camera: Camera) Vec3f {
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
        var min_ind: usize = @as(usize,@intFromFloat(@floor(min_val)));
        if (min_ind < 0){
            min_ind = 0;
        }
        return min_ind;
    }

    fn bound_index_max(max_val: f64, pixels_num: usize) usize {
        var max_ind: usize = @as(usize,@intFromFloat(@ceil(max_val)));
        if (max_ind > (pixels_num-1)){
            max_ind = (pixels_num-1);
        }
        return max_ind;
    }

    pub fn raster_frame(self: *const Self, coords: Coords, connect: Connect, field: Field, camera: Camera) !void {
        const tol: f64 = 1e-12;
        const elems_in_image: usize = 0;

        var nodes_raster: []Vec3f = try self.allocator.alloc(Vec3f, connect.nodes_per_elem);
        const field_raster: []f64 = try self.allocator.alloc(f64, connect.nodes_per_elem);
        const px_coord: []f64 = try self.allocator.alloc(f64, connect.nodes_per_elem);
        const weights: []f64 = try self.allocator.alloc(f64, connect.nodes_per_elem);

        _ = field;
        _ = elems_in_image;
        _ = field_raster;
        _ = px_coord;
        _ = weights;

        for (0..connect.elem_n) |ee| {
            const coord_inds: []usize = connect.getElem(ee);

            for (0..connect.nodes_per_elem) |nn| {
                nodes_raster[nn] = world_to_raster_coords(coords.getVec3(coord_inds[nn]), camera);
            }

            const elem_area: f64 = edge_fun3(nodes_raster[0], nodes_raster[1], nodes_raster[2]);

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

            const x_min: f64 = Vec3SliceOps.min(f64, nodes_raster, 0);
            const x_max: f64 = Vec3SliceOps.max(f64, nodes_raster, 0);

            if ((x_min > @as(f64,@floatFromInt(camera.pixels_num[0]-1))) or (x_max < 0.0)) {
                continue;
            }

            const y_min: f64 = Vec3SliceOps.min(f64, nodes_raster, 1);
            const y_max: f64 = Vec3SliceOps.max(f64, nodes_raster, 1);

            if ((y_min > @as(f64,@floatFromInt(camera.pixels_num[1]-1))) or (y_max < 0.0)) {
                continue;
            }

            // print("Elem {}: x, min {}\n", .{ ee, x_min });
            // print("Elem {}: x, max {}\n", .{ ee, x_max });
            // print("Elem {}: x, min {}\n", .{ ee, y_min });
            // print("Elem {}: x, max {}\n\n", .{ ee, y_max });

            const xi_min: usize = bound_index_min(x_min);
            const xi_max: usize = bound_index_max(x_max, @as(usize,camera.pixels_num[0]));
            const yi_min: usize = bound_index_min(y_min);
            const yi_max: usize = bound_index_max(y_max, @as(usize,camera.pixels_num[1]));

            // print("Elem {}: xi, min {}\n", .{ ee, xi_min });
            // print("Elem {}: xi, max {}\n", .{ ee, xi_max });
            // print("Elem {}: yi, min {}\n", .{ ee, yi_min });
            // print("Elem {}: yi, max {}\n", .{ ee, yi_max });
            // print("\n",.{});

            const xi_min_f: f64 = @as(f64,@floatFromInt(xi_min));
            const xi_max_f: f64 = @as(f64,@floatFromInt(xi_max));
            const yi_min_f: f64 = @as(f64,@floatFromInt(yi_min));
            const yi_max_f: f64 = @as(f64,@floatFromInt(yi_max));
            const sub_samp_f: f64 = @as(f64,@floatFromInt(camera.sub_sample));

            // These will need to be var later
            const bound_coord_x: f64 = xi_min_f + 1.0/(2.0*sub_samp_f);
            const bound_coord_y: f64 = yi_min_f + 1.0/(2.0*sub_samp_f);
            const bound_ind_x: usize = @as(usize,camera.sub_sample)*xi_min;
            const bound_ind_y: usize = @as(usize,camera.sub_sample)*yi_min;
            
            const coord_step: f64 = 1.0 / sub_samp_f;






        }
    }
};
