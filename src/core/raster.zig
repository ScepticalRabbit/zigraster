const std = @import("std");
const print = std.debug.print;
const Vec3f = @import("vector.zig").Vec3f;
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


        coord_raster.elems[0] = 2.0*coord_raster.elems[0] / camera.image_dims[0];
        coord_raster.elems[1] = 2.0*coord_raster.elems[1] / camera.image_dims[1];

        coord_raster.elems[0] = (coord_raster.elems[0] + 1.0)/2.0 * @as(f64,@floatFromInt(camera.pixels_num[0]));
        coord_raster.elems[1] = (1.0 - coord_raster.elems[1])/2.0 * @as(f64,@floatFromInt(camera.pixels_num[1]));
        coord_raster.elems[2] = -1.0*coord_raster.elems[2];

        print("WORLD=  [{d:.3},{d:.3},{d:.3}]\n",.{coord_world.elems[0],coord_world.elems[1],coord_world.elems[2]});
        print("RASTER= [{d:.3},{d:.3},{d:.3}]\n\n",.{coord_raster.elems[0],coord_raster.elems[1],coord_raster.elems[2]});

        return coord_raster;
    }

    pub fn raster_frame(self: *const Self, coords: Coords, connect: Connect, field: Field, camera: Camera) !void {


        const elems_in_image: usize = 0;

        var nodes_raster: []Vec3f = try self.allocator.alloc(Vec3f, connect.nodes_per_elem);
        const field_raster: []f64 = try self.allocator.alloc(f64, connect.nodes_per_elem);
        const px_coord: []f64 = try self.allocator.alloc(f64, connect.nodes_per_elem);
        const weights: []f64 = try self.allocator.alloc(f64, connect.nodes_per_elem);
        const elem_area: f64 = 0.0;

        _ = field;
        _ = elems_in_image;
        _ = field_raster;
        _ = px_coord;
        _ = weights;
        _ = elem_area;


        for (0..connect.elem_n) |ee| {

            const coord_inds: []usize = connect.getElem(ee);

            //print("\nee={any}\n",.{ee});
            for (0..connect.nodes_per_elem) |nn|{
                nodes_raster[nn] = world_to_raster_coords(coords.getVec3(coord_inds[nn]), camera);

                //print("nodes_raster[{any}]=",.{nn});
                //nodes_raster[nn].vecPrint();
                //print("\n",.{});
            }




        }
    }



};